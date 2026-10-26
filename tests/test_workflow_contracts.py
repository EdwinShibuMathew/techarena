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
