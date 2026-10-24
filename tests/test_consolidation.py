"""File-level migration tests using tiny synthetic assets, not real trained models."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import consolidate as c
from predict_v6 import portable_pipeline


def dimensional_fixture(checkpoint):
    dimension = checkpoint["dimension"]
    assert dimension == 8 and checkpoint["epoch"] == 8
    return dimension, checkpoint["epoch"]


class ConsolidationTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / "local/test-runs"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="migration_", dir=temporary_root)
        if not c.inside(self.temp.name, temporary_root):
            raise RuntimeError("Test temporary directory escaped the workspace.")
        self.addCleanup(self.temp.cleanup)
        self.projects = Path(self.temp.name)
        self.root = self.projects / "techarena-team"
        self.root.mkdir()
        self.old = self.projects / "techarena-final"
        self.final = self.projects / "techarena-row-auc-final"
        self.write("tools/predict_v6.py", b"# test predictor\n")
        self.write("tools/consolidate.py", b"# test mover\n")
        self.write("legacy/example.py", b"# historical CRLF\r\n")
        c.write_json(self.root / "configs/source_manifest.json", {"files": {"legacy/example.py": c.digest(self.root / "legacy/example.py")}})
        for name in ["train_events.npz", "evaluation_events.npz", "evaluation_rowids.npy", "preference_training.npz"]:
            self.make(self.old / "data/cache" / name, (name + ": synthetic").encode())
        raw = self.projects / "techarena/data/raw"
        self.make(raw / "example_submission.csv", b"rowid,predicted_probability\r\nr,0.1\r\n")
        self.make(raw / "train.zip", b"synthetic training archive")
        self.make(raw / "evaluation.zip", b"synthetic evaluation archive")
        self.make(self.old / "outputs/techarena_submission_v5.csv", b"synthetic v5")
        self.make(self.final / "outputs/techarena_submission_v6_candidate.csv", b"synthetic v6")
        sha = c.digest(self.final / "outputs/techarena_submission_v6_candidate.csv")
        config = {"submission_sha256": sha, "members": [r[0] for r in c.MEMBERS]}
        c.write_json(self.root / "configs/v6.json", config)
        c.write_json(self.final / "outputs/candidate_submission_checks.json", {
            "submission_ready": True, "v5_files_unchanged": True,
            "validation_pipeline_check_passed": True, "submission_sha256": sha})
        c.write_json(self.old / "outputs/five_model_submission_checks.json", {
            "submission_sha256": c.digest(self.old / "outputs/techarena_submission_v5.csv")})
        for key, folder, checkpoint, _ in c.MEMBERS:
            preference = "preference_best.pt" if checkpoint == "event_best.pt" else "preference_final.pt"
            for name in [checkpoint, preference, "combiner_used.cbm", "evaluation_predictions.npz"]:
                self.make(self.projects / folder / name, (key + name).encode())
        self.study = self.projects / "techarena-failed-study"
        c.write_json(self.study / "outputs/ensemble_trial_results.json", {"supported_for_five_day_check": False, "original_path": str(self.study)})
        self.git = self.projects / "techarena-separate-git"
        (self.git / ".git").mkdir(parents=True)
        self.unrelated = self.projects / "personal-work"
        self.make(self.unrelated / "notes.txt", b"unrelated")

    def make(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def write(self, relative, content):
        self.make(self.root / relative, content)

    def quiet(self):
        return contextlib.redirect_stdout(io.StringIO())

    def bundle(self):
        with self.quiet():
            c.bundle(self.projects, self.root)

    def receipt(self):
        # Test fixture only. Production receipts are written solely by a real
        # five-member replay and byte-identical CSV check in predict_v6.py.
        self.write("artifacts/reproduced_v6.csv", b"synthetic v6")
        required = ["tools/predict_v6.py", "tools/consolidate.py", "configs/v6.json", "configs/source_manifest.json", "legacy/example.py"]
        c.write_json(self.root / "local/v6_relocation_check.json", {
            "passed": True, "bundle_manifest_sha256": c.digest(self.root / "artifacts/v6/manifest.json"),
            "submission_sha256": c.digest(self.root / "artifacts/reproduced_v6.csv"),
            "output_relative": "artifacts/reproduced_v6.csv", "code_sha256": {p: c.digest(self.root / p) for p in required},
            "member_max_abs_differences": {m[0]: 0.0 for m in c.MEMBERS}})

    def test_copy_is_exact_and_never_overwrites_different_bytes(self):
        source, destination = self.projects / "original", self.root / "local/copy"
        source.write_bytes(b"original bytes")
        expected = c.copy_verified(source, destination)
        self.assertEqual(expected, c.digest(destination))
        self.assertEqual(c.copy_verified(source, destination), expected)
        destination.write_bytes(b"different")
        with self.assertRaises(FileExistsError):
            c.copy_verified(source, destination)
        self.assertEqual(source.read_bytes(), b"original bytes")

    def test_dimension_adapter_preserves_original_guard_and_rejects_other_models(self):
        extended = portable_pipeline(dimensional_fixture)
        self.assertEqual(extended({"dimension": 8, "epoch": 8}), (8, 8))
        self.assertEqual(extended({"dimension": 16, "epoch": 8}), (16, 8))
        with self.assertRaises(AssertionError):
            dimensional_fixture({"dimension": 16, "epoch": 8})
        for state in [{"dimension": 4, "epoch": 8}, {"dimension": 8, "epoch": 7}]:
            with self.assertRaises(AssertionError):
                extended(state)

    def test_bundle_resume_fingerprints_and_path_escape(self):
        self.bundle()
        self.bundle()
        self.assertTrue((self.root / "docs/results/techarena-failed-study/ensemble_trial_results.json").exists())
        target = self.root / "artifacts/v6/dim8_seed42/event.pt"
        target.write_bytes(b"tampered")
        with self.assertRaises(RuntimeError):
            c.verify_bundle(self.root)
        with self.assertRaises(ValueError):
            c.asset(self.root, "../outside")

    def test_archive_requires_real_receipt_and_blocks_busy_folders(self):
        self.bundle()
        with patch.object(c, "posix", return_value=True), self.quiet():
            with self.assertRaises(FileNotFoundError):
                c.archive(self.projects, True, self.root)
            self.receipt()
            with patch.object(c, "running_under", return_value=[{"pid": 1, "working_directory": str(self.old)}]):
                with self.assertRaises(RuntimeError):
                    c.archive(self.projects, True, self.root)
        self.assertTrue(self.old.exists())
        self.assertFalse((self.root / "local/archive_journal.json").exists())

    def test_archive_plan_apply_resume_and_restore_preserve_original_bytes(self):
        self.bundle()
        self.receipt()
        original_report = (self.study / "outputs/ensemble_trial_results.json").read_bytes()
        with patch.object(c, "posix", return_value=True), patch.object(c, "running_under", return_value=[]), self.quiet():
            plan = c.archive(self.projects, False, self.root)
            self.assertTrue(self.old.exists())
            self.assertFalse((self.root / "local/archive").exists())
            self.assertIn(str(self.git), plan["excluded"])
            c.archive(self.projects, True, self.root)
            self.assertFalse(self.old.exists())
            self.assertTrue(self.git.exists())
            self.assertTrue(self.unrelated.exists())
            self.assertEqual((self.root / "local/archive/techarena-failed-study/outputs/ensemble_trial_results.json").read_bytes(), original_report)
            c.archive(self.projects, True, self.root)
            c.verify_bundle(self.root)
            c.restore(False, self.root)
            self.assertFalse(self.old.exists())
            c.restore(True, self.root)
            c.restore(True, self.root)
        self.assertTrue(self.old.exists())
        self.assertEqual((self.study / "outputs/ensemble_trial_results.json").read_bytes(), original_report)

    def test_incomplete_receipt_and_changed_code_block_archive(self):
        self.bundle()
        self.receipt()
        path = self.root / "local/v6_relocation_check.json"
        receipt = c.read_json(path)
        receipt["code_sha256"] = {}
        c.write_json(path, receipt)
        with self.quiet(), self.assertRaises(RuntimeError):
            c.require_receipt(self.root)
        self.receipt()
        self.write("tools/predict_v6.py", b"changed after replay")
        with self.quiet(), self.assertRaises(RuntimeError):
            c.require_receipt(self.root)

    def test_restore_refuses_an_occupied_old_location(self):
        self.bundle()
        self.receipt()
        with patch.object(c, "posix", return_value=True), patch.object(c, "running_under", return_value=[]), self.quiet():
            c.archive(self.projects, True, self.root)
            self.old.mkdir()
            with self.assertRaises(RuntimeError):
                c.restore(True, self.root)


if __name__ == "__main__":
    unittest.main()
