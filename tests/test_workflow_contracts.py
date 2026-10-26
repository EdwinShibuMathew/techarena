"""Public synthetic checks for chronology, snapshots, gates and compute ownership."""
import copy
import hashlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from ml_workflow.contracts import CaseSpec,ExperimentSpec,JAN20,DAY,validate_events,first_indices,canonical
from ml_workflow.storage import read,write_new,extract_snapshot
from ml_workflow.budget import Budget
from ml_workflow import engine
spec=importlib.util.spec_from_file_location("workflow_fixture",ROOT/"experiments/workflow-smoke/prepare_fixture.py")
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


class WorkflowContractTests(unittest.TestCase):
    def test_spec_requires_cross_prefix_development_and_fixed_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            value=read(fixture.create(Path(tmp)))
            ExperimentSpec.parse(value)
            for mutate in (
                lambda d:d["development"][0].update(target_cutoff=d["development"][0]["meta_cutoff"]),
                lambda d:d["development"][0].update(target_end=JAN20+6*DAY),
                lambda d:d["promotion"]["primary"].update(target_cutoff=JAN20+4*DAY),
                lambda d:d["bootstrap"].update(resamples=10),
                lambda d:d.update(baseline_sha256="0"*64),
                lambda d:d.update(adapter="../outside.py"),
                lambda d:d.update(purpose="research")):
                bad=copy.deepcopy(value);mutate(bad)
                with self.assertRaises(ValueError):ExperimentSpec.parse(bad)

    def test_history_boundaries_labels_and_first_users(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);value=read(fixture.create(root))
            with np.load(root/value["inputs"]["train"]["path"],allow_pickle=False) as d:raw={k:d[k] for k in d.files}
            validate_events(raw,True)
            for timestamp in (raw["t"][0],raw["t"][0]+1):
                bad={k:v.copy() for k,v in raw.items()};bad["h"][0,0]=1;bad["ht"][0,0]=timestamp
                with self.assertRaisesRegex(ValueError,"History crosses"):validate_events(bad,True)
            with self.assertRaisesRegex(ValueError,"Evaluation labels"):validate_events(raw,False)
            users=np.array([1,1,2,2]);times=np.array([1,2,2,3])
            np.testing.assert_array_equal(first_indices(users,times,2,4),[1,2])

    def test_snapshot_rejects_traversal_links_and_content_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i,name in enumerate(("../escape","/absolute","source/../escape","source/link","source/good")):
                archive=root/f"bad{i}.tar.gz"
                with tarfile.open(archive,"w:gz") as tar:
                    entry=tarfile.TarInfo(name)
                    if name.endswith("link"):entry.type=tarfile.SYMTYPE;entry.linkname="/outside";tar.addfile(entry)
                    else:entry.size=3;tar.addfile(entry,io.BytesIO(b"bad"))
                with self.assertRaises(ValueError):extract_snapshot(archive,root/f"out{i}",{"good":hashlib.sha256(b"good").hexdigest()})

    def test_budget_exclusive_retry_and_timeout_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"compute.json"
            with Budget(path,"first",.1,2.,carry=.2,carry_identity="prior") as budget:
                with self.assertRaises(BlockingIOError):
                    with Budget(path,"second",.1,2.,carry=.2,carry_identity="prior"):pass
                with self.assertRaises(subprocess.TimeoutExpired):budget.run([sys.executable,"-c","import time; time.sleep(10)"],tmp)
            state=read(path)
            self.assertGreaterEqual(state["used_seconds"],.29)
            with self.assertRaisesRegex(ValueError,"retries prohibited"):
                with Budget(path,"first",.1,2.,carry=.2,carry_identity="prior"):pass
            with self.assertRaisesRegex(ValueError,"lineage changed"):
                with Budget(path,"third",.1,2.,carry=.2,carry_identity="reset"):pass

    def test_contended_phase_can_run_after_shared_budget_lock_is_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/"artifacts/experiments/run";work.mkdir(parents=True)
            manifest={"spec_sha256":"f"*64,"spec":{"purpose":"research",
                "budget":{"phase_seconds":1,"total_seconds":10}},
                "carried_seconds":0.,"carry_identity":None,"source":{"commit":"fixture"}}
            ledger=root/"artifacts/experiments/compute-ledger.json"
            claim=work/"develop.attempt.json"
            def launch(budget,command,cwd):
                self.assertTrue(claim.is_file(),"Claim must be committed before launching")
                self.assertEqual(read(claim)["spec_sha256"],manifest["spec_sha256"])
                self.assertEqual(read(ledger)["attempts"][-1]["owner"],"run/develop")
                self.assertEqual(read(ledger)["attempts"][-1]["status"],"running")
                return 0
            with patch.object(engine,"locate",return_value=work), patch.object(engine,"load",return_value=manifest), \
                 patch.object(Budget,"run",autospec=True,side_effect=launch) as child:
                with Budget(ledger,"other-run/develop",1,10):
                    before=ledger.read_bytes()
                    with self.assertRaises(BlockingIOError):engine.run(root,"run","develop")
                    self.assertFalse(claim.exists())
                    self.assertFalse((work/"develop.ledger.json").exists())
                    self.assertEqual(ledger.read_bytes(),before)
                    child.assert_not_called()
                self.assertEqual(engine.run(root,"run","develop"),0)
                child.assert_called_once()
                self.assertEqual(read(work/"develop.ledger.json")["exit_code"],0)
                attempts=read(ledger)["attempts"]
                self.assertEqual([a["owner"] for a in attempts],["other-run/develop","run/develop"])
                self.assertEqual(attempts[-1]["status"],"complete")
                with self.assertRaisesRegex(ValueError,"Phase attempted"):engine.run(root,"run","develop")
                child.assert_called_once()

    def test_final_accounting_write_failure_prevents_publication(self):
        from ml_workflow import budget as budget_module
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/"artifacts/experiments/run";work.mkdir(parents=True)
            manifest={"spec_sha256":"f"*64,"spec":{"purpose":"research","release_name":"v8",
                "budget":{"phase_seconds":1,"total_seconds":10}},
                "carried_seconds":0.,"carry_identity":None,"source":{"commit":"fixture"}}
            ledger=root/"artifacts/experiments/compute-ledger.json"
            original_write=budget_module.write_json
            def failing_accounting(path,value):
                if path==ledger and value["attempts"][-1]["status"]=="complete":
                    raise OSError("injected final accounting failure")
                original_write(path,value)
            with patch.object(engine,"locate",return_value=work), patch.object(engine,"load",return_value=manifest), \
                 patch.object(engine,"checked",return_value={"passed":True}), \
                 patch.object(Budget,"run",return_value=0) as child, \
                 patch.object(budget_module,"write_json",side_effect=failing_accounting), \
                 patch.object(engine,"publish") as publish:
                with self.assertRaisesRegex(OSError,"final accounting failure"):
                    engine.run(root,"run","finalize")
                child.assert_called_once()
                publish.assert_not_called()
            self.assertEqual(read(ledger)["attempts"][-1]["status"],"running")
            self.assertEqual(read(ledger)["used_seconds"],1.)
            self.assertTrue((work/"finalize.attempt.json").exists())
            self.assertFalse((work/"finalize.ledger.json").exists())
            self.assertFalse((work/"published.json").exists())
            self.assertFalse((root/"configs/v8.json").exists())
            self.assertFalse((root/"artifacts/submissions").exists())

    def test_killed_controller_keeps_its_full_compute_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"compute.json"
            code="from pathlib import Path; import os; from ml_workflow.budget import Budget; "
            code+="budget=Budget(Path("+repr(str(path))+"),'crashed',1,3,carry=.5,carry_identity='prior'); budget.__enter__(); os._exit(9)"
            result=subprocess.run([sys.executable,"-c","import sys; sys.path.insert(0,"+repr(str(ROOT/"tools"))+"); "+code])
            self.assertEqual(result.returncode,9)
            self.assertEqual(read(path)["used_seconds"],1.5)
            with Budget(path,"next",.1,3,carry=.5,carry_identity="prior") as budget:
                self.assertEqual(budget.state["used_seconds"],1.6)

    def test_all_promotion_failure_paths_are_recomputed(self):
        def metric(gain=.01,low=.001):
            return {"baseline_auc":.8,"candidate_auc":.8+gain,"auc_diff":gain,"ci_lower":low,"ci_upper":.02,"resamples":1000}
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);write_new(work/"manifest.json",{})
            def report(stage,results):
                path=work/(stage+".json")
                if path.exists():path.unlink();(work/(stage+".seal.json")).unlink()
                write_new(path,{"complete":True,"passed":True,"files":{},"manifest_sha256":engine.digest(work/"manifest.json"),"results":results})
                write_new(work/(stage+".seal.json"),{"sha256":engine.digest(path)})
            for results in ({"primary":metric(low=-.001),"secondary":metric()},
                            {"primary":metric(),"secondary":metric(gain=-.001)}):
                report("screen",results)
                with self.assertRaisesRegex(ValueError,"gate failed"):engine.checked(work,"screen")
            for low,late in ((-.001,.01),(.001,-.01)):
                row=metric(low=low);row["late"]={"auc_diff":late}
                report("stress",{"five_day":row})
                with self.assertRaisesRegex(ValueError,"gate failed"):engine.checked(work,"stress")
            row=metric();row["resamples"]=1
            report("screen",{"primary":row,"secondary":metric()})
            with self.assertRaisesRegex(ValueError,"confidence"):engine.checked(work,"screen")

    def test_synthetic_runs_cannot_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,"Synthetic runs cannot"):
                engine.publish(Path(tmp),Path(tmp),{"spec":{"purpose":"synthetic_validation"}})
