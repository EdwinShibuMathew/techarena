"""Synthetic data copies prove the deletion scope and kept bytes before real cleanup."""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import consolidate as c
import deduplicate_archive as archive
import clean_archived_data_copies as clean


class ArchivedDataCopyTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[1] / "local/test-runs"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="data_copies_", dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "techarena-team"
        self.old = self.root / "local/archive/techarena"
        self.old.mkdir(parents=True)
        for relative, contents in [
            ("data/raw/train.zip", b"original competition archive"),
            ("local/archive/techarena/data/raw/train.zip", b"original competition archive"),
            ("local/archive/techarena/data/cache/unknown.npz", b"unique training subset"),
            ("local/archive/techarena/work/v5.pt", b"V5 model"),
            ("local/archive/techarena/work/v6.pt", b"V6 model"),
            ("local/archive/techarena/work/predictions.npz", b"saved predictions"),
            ("local/archive/techarena/outputs/results.json", b'{"auc":0.81}\n'),
            ("local/archive/techarena/outputs/train.zip", b"original competition archive"),
            ("local/archive/techarena/src/train.py", b"# original training source\n"),
            ("tools/clean_archived_data_copies.py", b"synthetic cleanup code")]:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        self.live = self.root / "data/raw/train.zip"
        self.duplicate = self.old / "data/raw/train.zip"
        self.manifest = {"files": {"data/raw/train.zip": {"bytes": self.live.stat().st_size,
                                                          "sha256": c.digest(self.live)}}}
        c.write_json(self.root / "artifacts/v6/manifest.json", self.manifest)
        c.write_json(self.root / "local/archive_journal.json", {
            "repository": str(self.root.resolve()), "projects": str(self.root.parent.resolve()),
            "restored": False, "moves": [{"done": True,
                "source": str(self.root.parent / "techarena"), "destination": str(self.old),
                "identity": [self.old.stat().st_dev, self.old.stat().st_ino]}]})
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        # Live bundle integration is covered by consolidation tests. Fixtures need no real models.
        self.stack.enter_context(patch.object(clean, "checked", return_value=self.manifest))
        self.stack.enter_context(patch.object(clean, "linux_only"))
        self.stack.enter_context(patch.object(c, "running_under", return_value=[]))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def test_only_proven_data_copy_removed_and_restore_recreates_it(self):
        before = archive.snapshot(self.root)
        live_hash = c.digest(self.live)
        record = clean.plan(self.root)
        self.assertEqual([op["archive"] for op in record["operations"]],
                         ["local/archive/techarena/data/raw/train.zip"])
        progress = clean.apply(self.root)
        self.assertTrue(progress["passed"])
        self.assertFalse(self.duplicate.exists())
        self.assertEqual(c.digest(self.live), live_hash)
        for name, expected in before["files"].items():
            if name != record["operations"][0]["archive"]:
                self.assertEqual(c.digest(self.root / name), expected["sha256"])
        self.assertEqual(clean.apply(self.root), progress)
        clean.restore(self.root)
        archive.same_contents(before, archive.snapshot(self.root))

    def test_changed_source_refuses_before_deletion(self):
        clean.plan(self.root)
        self.duplicate.write_bytes(b"changed archived competition data")
        with self.assertRaisesRegex(RuntimeError, "content/permissions changed"):
            clean.apply(self.root)
        self.assertTrue(self.duplicate.exists())

    def test_changed_live_keeper_refuses_before_deletion(self):
        clean.plan(self.root)
        self.live.write_bytes(b"changed canonical bytes")
        with self.assertRaisesRegex(RuntimeError, "Active keeper changed"):
            clean.apply(self.root)
        self.assertTrue(self.duplicate.exists())

    def test_changed_results_refuse_before_deletion(self):
        clean.plan(self.root)
        (self.old / "outputs/results.json").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "content/permissions changed"):
            clean.apply(self.root)
        self.assertTrue(self.duplicate.exists())

    def test_unplanned_missing_file_blocks_cleanup(self):
        clean.plan(self.root)
        (self.old / "work/v5.pt").unlink()
        with self.assertRaisesRegex(RuntimeError, "unplanned archived file"):
            clean.apply(self.root)
        self.assertTrue(self.duplicate.exists())

    def test_busy_archive_blocks_cleanup(self):
        clean.plan(self.root)
        with patch.object(c, "running_under", return_value=[{"pid": 1}]):
            with self.assertRaisesRegex(RuntimeError, "in use"):
                clean.apply(self.root)
        self.assertTrue(self.duplicate.exists())

    def test_resume_after_removal_before_progress_update(self):
        record = clean.plan(self.root)
        progress = self.root / "local/archived-data-cleanup" / record["id"] / "progress.json"
        c.write_json(progress, {"id": record["id"], "passed": False, "removed": []})
        self.duplicate.unlink()
        self.assertTrue(clean.apply(self.root)["passed"])
        clean.restore(self.root)
        self.assertEqual(c.digest(self.duplicate), c.digest(self.live))

    def test_wrong_data_bytes_and_non_data_paths_are_not_selected(self):
        (self.old / "data/raw/train.zip").write_bytes(b"unique different zip")
        (self.old / "work/train.zip").write_bytes(self.live.read_bytes())
        (self.old / "outputs/data").mkdir()
        (self.old / "outputs/data/train.zip").write_bytes(self.live.read_bytes())
        self.assertEqual(clean.plan(self.root)["operations"], [])

    def test_tampered_plan_cannot_delete_a_model(self):
        record = clean.plan(self.root)
        record["operations"][0]["archive"] = "local/archive/techarena/work/v5.pt"
        c.write_json(self.root / "local/archived_data_cleanup_plan.json", record)
        c.write_json(self.root / "local/archived-data-cleanup" / record["id"] / "plan.json", record)
        with self.assertRaisesRegex(RuntimeError, "restrictive data-only rule"):
            clean.apply(self.root)
        self.assertTrue((self.old / "work/v5.pt").exists())


if __name__ == "__main__":
    unittest.main()
