"""Public end-to-end CLI campaign with real Torch/CatBoost fitting and reload."""
import importlib.util
import json
import time
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from ml_workflow.budget import Budget
AVAILABLE=all(importlib.util.find_spec(name) is not None for name in ("torch","catboost","sklearn"))


@unittest.skipUnless(AVAILABLE,"CPU ML dependencies required; enforced by model-contracts CI")
class WorkflowIntegrationTests(unittest.TestCase):
    def make_repository(self,root):
        names=subprocess.check_output(["git","ls-files"],cwd=ROOT,text=True).splitlines()
        for name in names:
            destination=root/name;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,destination)
        def git(*args):return subprocess.run(["git",*args],cwd=root,check=True,capture_output=True)
        git("init","-q");git("config","user.name","Workflow test");git("config","user.email","test@example.invalid")
        git("add",".");git("commit","-qm","Public synthetic fixture")
        return git

    def cli(self,root,*args,code=0):
        result=subprocess.run([sys.executable,str(root/"tools/run_experiment.py"),*args],cwd=root,
                              capture_output=True,text=True,timeout=150)
        self.assertEqual(result.returncode,code,result.stdout+result.stderr)
        return result.stdout

    def prepare(self,root,name):
        result=subprocess.run([sys.executable,str(root/"experiments/workflow-smoke/prepare_fixture.py"),"--fixture-id",name],
                              cwd=root,check=True,capture_output=True,text=True)
        spec=Path(result.stdout.strip())
        self.cli(root,"prepare","--run-id",name,"--spec",str(spec))
        return spec,root/"artifacts/validation"/name

    def test_all_stages_and_independent_final_replay_without_campaign_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_repository(root)
            _,work=self.prepare(root,"success")
            # Contention must not close a phase that never launched a worker.
            with Budget(work/"compute.json","other-run/develop",120,600):
                before=(work/"compute.json").read_bytes()
                self.cli(root,"develop","--run-id","success",code=1)
                self.assertFalse((work/"develop.attempt.json").exists())
                self.assertFalse((work/"models").exists())
                self.assertEqual((work/"compute.json").read_bytes(),before)
            # A later maintenance edit cannot change the frozen experiment implementation.
            with (root/"tools/predict_v6.py").open("a") as stream:stream.write("\n# unrelated maintenance\n")
            for stage in ("develop","screen","stress","finalize"):
                self.cli(root,stage,"--run-id","success")
            report=json.loads(self.cli(root,"verify","--run-id","success"))
            self.assertTrue(report["validation_complete"]);self.assertFalse(report["submission_ready"])
            self.assertEqual(report["rows"],480)
            self.assertEqual((work/"candidate.csv").read_bytes(),(work/"independent-replay.csv").read_bytes())
            self.assertFalse((root/"configs/v8.json").exists());self.assertFalse((root/"artifacts/submissions").exists())
            receipts=[json.loads(p.read_text()) for p in (work/"cache-receipts").glob("*.json")]
            self.assertTrue(any(r["hit"] for r in receipts));self.assertTrue(any(not r["hit"] for r in receipts))
            self.cli(root,"develop","--run-id","success",code=1)
            (work/"candidate.csv").write_bytes(b"corrupt")
            self.cli(root,"verify","--run-id","success",code=1)

    def test_worker_stops_when_budget_controller_is_killed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);git=self.make_repository(root)
            adapter=root/"experiments/workflow-smoke/adapter.py"
            adapter.write_text(adapter.read_text().replace('    torch.manual_seed(metadata["seed"])',
                '    import time\n    (path.parent/"entered").touch()\n    time.sleep(2)\n    (path.parent/"escaped").touch()\n    torch.manual_seed(metadata["seed"])'))
            git("add",str(adapter));git("commit","-qm","Synthetic controller failure fixture")
            _,work=self.prepare(root,"crash")
            process=subprocess.Popen([sys.executable,str(root/"tools/run_experiment.py"),"develop","--run-id","crash"],cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:
                deadline=time.monotonic()+30
                while not list((work/"models").glob("*/*/*/entered")):
                    self.assertIsNone(process.poll(),"Controller exited before fixture entered training")
                    self.assertLess(time.monotonic(),deadline)
                    time.sleep(.05)
                process.kill();process.wait(timeout=5)
                time.sleep(2.2)
                self.assertEqual(list((work/"models").glob("*/*/*/escaped")),[])
                ledger=json.loads((work/"compute.json").read_text())
                self.assertEqual(ledger["used_seconds"],120)
                self.cli(root,"develop","--run-id","crash",code=1)
            finally:
                if process.poll() is None:process.kill();process.wait(timeout=5)

    def test_failed_gate_is_terminal_and_inputs_and_worker_are_guarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);git=self.make_repository(root)
            # Change only this disposable adapter before snapshotting: actual fitting still runs,
            # but constant predictions must fail development against the constant comparator.
            adapter=root/"experiments/workflow-smoke/adapter.py"
            text=adapter.read_text().replace('p.reshape(-1,10)','np.full(rowids.shape,.05)')
            adapter.write_text(text);git("add",str(adapter));git("commit","-qm","Deliberately non-improving synthetic model")
            spec,work=self.prepare(root,"failure")
            self.cli(root,"develop","--run-id","failure","--worker",code=1)
            self.cli(root,"develop","--run-id","failure",code=2)
            for stage in ("develop","screen","stress","finalize"):
                self.cli(root,stage,"--run-id","failure",code=1)
            self.assertFalse((work/"screen.attempt.json").exists())
            raw=json.loads(spec.read_text())["inputs"]["train"]["path"]
            with (root/raw).open("ab") as stream:stream.write(b"changed")
            self.cli(root,"verify","--run-id","failure",code=1)
