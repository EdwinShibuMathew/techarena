"""Bundle representation compatibility and concurrent metadata publication."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import consolidate as c


class PortableReceiptTests(unittest.TestCase):
    def test_only_uniform_line_endings_are_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "configs").mkdir()
            text = b'{\n  "files": {}\n}\n'
            path = root / "configs/source_manifest.json"
            for original in (text, text.replace(b"\n", b"\r\n")):
                manifest = {"source_manifest_sha256": hashlib.sha256(original).hexdigest()}
                for checkout in (text, text.replace(b"\n", b"\r\n")):
                    path.write_bytes(checkout)
                    binding = c.source_manifest_binding(root, manifest)
                    self.assertEqual(binding["checkout_sha256"], c.digest(path))
                    self.assertEqual(binding["method"], "exact" if original == checkout else "lf-crlf-only")
                for bad in (text.replace(b"{}", b'{"altered": true}'), text + b" ", text.replace(b"\n", b"\r\n", 1)):
                    path.write_bytes(bad)
                    with self.assertRaises(RuntimeError):
                        c.source_manifest_binding(root, manifest)

    def test_compatible_json_does_not_relax_frozen_file_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for p in ("configs", "legacy", "artifacts/v6"):
                (root/p).mkdir(parents=True)
            frozen = root/"legacy/frozen.py"; frozen.write_bytes(b"pass\n")
            text = json.dumps({"files": {"legacy/frozen.py": c.digest(frozen)}}, indent=2)+"\n"
            (root/"configs/source_manifest.json").write_bytes(text.encode())
            c.write_json(root/"configs/v6.json", {"submission_sha256":"fixture", "members":[]})
            c.write_json(root/"artifacts/v6/manifest.json", {"source_manifest_sha256":hashlib.sha256(text.replace("\n","\r\n").encode()).hexdigest(),
                "config_sha256":c.digest(root/"configs/v6.json"), "expected_submission_sha256":"fixture", "members":[], "files":{}})
            c.verify_bundle(root)
            frozen.write_bytes(b"changed\n")
            with self.assertRaisesRegex(RuntimeError, "Historical source changed"):
                c.verify_bundle(root)

    def test_concurrent_writers_publish_whole_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"receipt.json"; barrier = threading.Barrier(2); replace = Path.replace
            def coordinated(source, target):
                barrier.wait(timeout=10)
                return replace(source,target)
            values = [{"writer":i,"payload":"x"*(100+i)} for i in range(2)]
            with patch.object(Path,"replace",coordinated), ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lambda v:c.write_json(path,v), values))
            self.assertIn(c.read_json(path), values)
            self.assertEqual(list(Path(tmp).glob(".receipt-*")),[])
            before = path.read_bytes()
            with patch.object(Path,"replace",side_effect=OSError("injected")), self.assertRaises(OSError):
                c.write_json(path,{"new":True})
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(list(Path(tmp).glob(".receipt-*")),[])
