"""Exercise release gates with small Git fixtures, never private competition assets."""

import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import source_readiness as readiness


class SourceReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repo"
        self.root.mkdir()
        self.command("init", "-q", "-b", "main")
        self.command("config", "user.name", "CI fixture")
        self.command("config", "user.email", "ci@example.invalid")
        self.write("legacy/frozen.py", b"# frozen fixture\r\n")
        self.write("legacy/README.md", b"historical notes outside the Python manifest\n")
        self.write("configs/source_manifest.json", json.dumps({"files": {
            "legacy/frozen.py": readiness.sha256((self.root / "legacy/frozen.py").read_bytes()),
        }}).encode())
        self.write("configs/v6.json", json.dumps({"submission_sha256": readiness.V6_SHA256,
            "evaluation_rows": 529390, "evaluation_events": 52939}).encode())
        self.write("README.md", b"source only\n")
        self.commit()
        self.base = self.command("rev-parse", "HEAD").decode().strip()

    def command(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root, stderr=subprocess.STDOUT)

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def commit(self):
        self.command("add", "--all")
        self.command("-c", "commit.gpgsign=false", "commit", "-qm", "fixture")

    def test_clean_source_archive_is_exact_and_receipt_excludes_model_claims(self):
        output = Path(self.temporary.name) / "release"
        report = readiness.package(self.root, output, "success", "success", self.base, model_test_result="success")
        self.assertEqual(report["git_commit"], self.base)
        self.assertEqual(report["frozen_source_files_verified"], 1)
        self.assertFalse(report["model_inference_performed"])
        self.assertFalse(report["private_assets_verified"])
        self.assertFalse(report["competition_submission_created"])
        self.assertEqual(json.loads((output / "source-readiness.json").read_text()), report)
        for line in (output / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split("  ")
            self.assertEqual(expected, readiness.sha256((output / name).read_bytes()))
        readiness.validate_archive(self.root, output / report["archive"]["filename"], readiness.tracked_files(self.root))
        with self.assertRaises(FileExistsError):
            readiness.package(self.root, output, "success", "success", model_test_result="success")

    def test_all_failed_cancelled_or_skipped_prerequisites_block_packaging(self):
        output = Path(self.temporary.name) / "release"
        for result in ("failure", "cancelled", "skipped", ""):
            for tests, integrity in ((result, "success"), ("success", result)):
                with self.subTest(tests=tests, integrity=integrity), self.assertRaises(ValueError):
                    readiness.package(self.root, output, tests, integrity, model_test_result="success")
        self.assertFalse(output.exists())

    def test_model_contract_failure_or_missing_result_blocks_release(self):
        for result in (None,"failure","cancelled","skipped",""):
            with self.subTest(result=result), self.assertRaises(ValueError):
                readiness.package(self.root,Path(self.temporary.name)/"release","success","success",model_test_result=result)

    def test_uncommitted_changes_are_not_certified(self):
        self.write("README.md", b"not committed\n")
        with self.assertRaisesRegex(ValueError, "differs from HEAD"):
            readiness.verify(self.root)

    def test_frozen_manifest_and_config_changes_cannot_self_certify(self):
        self.write("legacy/frozen.py", b"# replacement\n")
        self.write("configs/source_manifest.json", json.dumps({"files": {
            "legacy/frozen.py": readiness.sha256((self.root / "legacy/frozen.py").read_bytes()),
        }}).encode())
        self.commit()
        with self.assertRaisesRegex(ValueError, "Frozen baseline paths changed"):
            readiness.verify(self.root, self.base)

    def test_frozen_hash_mismatch_fails_even_without_a_base_ref(self):
        self.write("legacy/frozen.py", b"# replacement\n")
        self.commit()
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            readiness.verify(self.root)

    def test_invalid_benchmark_fingerprint_is_rejected(self):
        config = json.loads((self.root / "configs/v6.json").read_text())
        config["submission_sha256"] = "0" * 64
        self.write("configs/v6.json", json.dumps(config).encode())
        self.commit()
        with self.assertRaisesRegex(ValueError, "benchmark fingerprint"):
            readiness.verify(self.root)

    def test_manifest_must_cover_all_frozen_sources(self):
        self.write("legacy/extra.py", b"# missing from manifest\n")
        self.commit()
        with self.assertRaisesRegex(ValueError, "exactly match"):
            readiness.verify(self.root)

    def test_tracked_private_formats_directories_and_binary_files_are_rejected(self):
        for name, content in (("docs/model.pt", b"weight"), ("docs/submission.csv", b"table"),
                              ("data/cache.txt", b"private"), ("local/receipt.json", b"{}"),
                              ("artifacts/model.txt", b"private"), ("archive/cache.txt", b"private"),
                              ("docs/binary.txt", b"\0binary"), (".env", b"secret=value")):
            with self.subTest(name=name):
                self.write(name, content)
                with self.assertRaises(ValueError):
                    readiness.validate_files(self.root, [name])

    def test_oversized_files_and_symlinks_are_rejected(self):
        self.write("docs/large.txt", b"x" * (readiness.MAX_FILE_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "exceeds 2 MiB"):
            readiness.validate_files(self.root, ["docs/large.txt"])
        (self.root / "link.md").symlink_to(self.root / "README.md")
        with self.assertRaisesRegex(ValueError, "Symlinks"):
            readiness.validate_files(self.root, ["link.md"])

    def test_committed_model_file_is_rejected_by_the_full_gate(self):
        self.write("docs/model.cbm", b"weights")
        self.commit()
        with self.assertRaisesRegex(ValueError, "Prohibited asset format"):
            readiness.verify(self.root)

    def test_archive_with_missing_or_traversing_files_is_rejected(self):
        output = Path(self.temporary.name) / "bad.tar.gz"
        with tarfile.open(output, "w:gz") as archive:
            archive.add(self.root / "README.md", arcname="techarena-source/README.md")
        with self.assertRaisesRegex(ValueError, "missing tracked"):
            readiness.validate_archive(self.root, output, readiness.tracked_files(self.root))
        with tarfile.open(output, "w:gz") as archive:
            archive.add(self.root / "README.md", arcname="../README.md")
        with self.assertRaisesRegex(ValueError, "Unsafe archive"):
            readiness.validate_archive(self.root, output, readiness.tracked_files(self.root))


if __name__ == "__main__":
    unittest.main()
