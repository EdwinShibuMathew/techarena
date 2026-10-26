"""Publishing includes experiment code without making real network calls."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import publish_private as p


class ExperimentPublicationTests(unittest.TestCase):
    def test_real_index_includes_experiment_and_report_and_ignores_weights(self):
        for with_experiment in (False,True):
            with self.subTest(experiment=with_experiment),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                subprocess.run(["git","init","-q","-b","fixture"],cwd=root,check=True)
                subprocess.run(["git","remote","add","origin",f"https://github.com/{p.OWNER}/{p.NAME}.git"],cwd=root,check=True)
                files={"README.md":"fixture",".gitignore":"*.pt\n",".gitattributes":"",".github/test.yml":"fixture",
                    "requirements.txt":"","requirements-ci.txt":"","docs/report.md":"trial","tools/stub.py":"pass",
                    "tests/test_stub.py":"pass","legacy/frozen.py":"pass"}
                if with_experiment:files.update({"experiments/trial/train.py":"pass","experiments/trial/weights.pt":"private"})
                for name,value in files.items():
                    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
                path=root/"configs/source_manifest.json";path.parent.mkdir()
                path.write_text(json.dumps({"files":{"legacy/frozen.py":hashlib.sha256(b"pass").hexdigest()}}))
                original=p.run;pushes=[]
                def fake(args,capture=True):
                    if args==["gh","api","user"]:return json.dumps({"login":p.OWNER,"id":1})
                    if args[:3]==["gh","repo","view"]:return json.dumps({"isPrivate":True,"nameWithOwner":f"{p.OWNER}/{p.NAME}","url":f"https://github.com/{p.OWNER}/{p.NAME}"})
                    if args[:2]==["git","push"]:pushes.append(args);return ""
                    if "commit" in args:return ""
                    return original(args,capture)
                with patch.object(p,"ROOT",root),patch.object(p,"run",side_effect=fake),contextlib.redirect_stdout(io.StringIO()):p.main()
                staged=subprocess.check_output(["git","ls-files"],cwd=root,text=True).splitlines()
                self.assertIn("docs/report.md",staged)
                self.assertEqual("experiments/trial/train.py" in staged,with_experiment)
                self.assertNotIn("experiments/trial/weights.pt",staged)
                self.assertEqual(len(pushes),1)
