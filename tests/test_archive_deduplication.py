"""Synthetic duplicate/unique files; no competition assets needed."""

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
import deduplicate_archive as d


class ArchiveDeduplicationTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[1] / "local/test-runs"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent, prefix="dedup_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "techarena-team"
        self.archive = self.root / "local/archive"
        self.one = self.archive / "techarena-final"
        self.two = self.archive / "techarena-study"
        self.one.mkdir(parents=True)
        self.two.mkdir()
        for relative in ["tools/deduplicate_archive.py", "artifacts/v6/manifest.json"]:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic fixture", encoding="utf-8")
        c.write_json(self.root / "local/archive_journal.json", {
            "repository": str(self.root.resolve()), "projects": str(self.root.parent.resolve()),
            "restored": False, "moves": [{"source": str(self.root.parent / folder.name),
                "destination": str(folder), "done": True,
                "identity": [folder.stat().st_dev, folder.stat().st_ino]}
                for folder in [self.one, self.two]]})
        self.first = self.one / "cache.npz"
        self.second = self.two / "cache.npz"
        self.first.write_bytes(b"x" * d.MINIMUM_BYTES)
        self.second.write_bytes(self.first.read_bytes())
        (self.one / "v5_unique.pt").write_bytes(b"V5" * d.MINIMUM_BYTES)
        (self.two / "v6_unique.pt").write_bytes(b"V6" * d.MINIMUM_BYTES)
        (self.two / "results.json").write_text('{"auc": 0.819451}\n', encoding="utf-8")
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        # Real receipt verification is covered by test_consolidation. Only synthetic roots bypass it.
        self.stack.enter_context(patch.object(d, "require_receipt"))
        self.stack.enter_context(patch.object(d, "require_self_test"))
        self.stack.enter_context(patch.object(d, "running_under", return_value=[]))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def apply(self):
        # Simulate the Linux-only gate on Windows while retaining native Path/OS calls.
        if os.name == "posix":
            return d.apply_plan(self.root)
        original = d.os

        class PosixGate:
            name = "posix"

            def __getattr__(self, name):
                return getattr(original, name)

        with patch.object(d, "os", PosixGate()):
            return d.apply_plan(self.root)

    def require_hardlinks(self):
        probe = self.two / "hardlink-probe"
        try:
            os.link(self.first, probe)
        except OSError as error:
            self.skipTest(f"Real hardlinks unavailable in this test environment: {error}")
        probe.unlink()

    def test_exact_duplicates_share_storage_and_all_bytes_remain(self):
        self.require_hardlinks()
        before = d.snapshot(self.root)
        plan = d.make_plan(self.root)
        self.assertEqual(len(plan["operations"]), 1)
        self.assertFalse(os.path.samefile(self.first, self.second))
        summary = self.apply()
        self.assertTrue(summary["passed"])
        self.assertTrue(os.path.samefile(self.first, self.second))
        d.same_contents(before, d.snapshot(self.root))
        self.assertNotEqual((self.one / "v5_unique.pt").stat().st_ino,
                            (self.two / "v6_unique.pt").stat().st_ino)
        self.assertEqual(summary, self.apply())

    def test_existing_hardlinks_do_not_count_twice(self):
        self.require_hardlinks()
        self.second.unlink()
        os.link(self.first, self.second)
        plan = d.make_plan(self.root)
        self.assertEqual(plan["operations"], [])
        self.assertEqual(plan["estimated_allocated_bytes_reclaimable"], 0)
        self.apply()

    def test_hardlinks_to_live_assets_are_excluded(self):
        self.require_hardlinks()
        live = self.root / "artifacts/v6/live.npz"
        os.link(self.first, live)
        plan = d.make_plan(self.root)
        self.assertEqual(plan["operations"], [])
        self.apply()
        self.assertFalse(os.path.samefile(live, self.second))

    def test_changed_duplicate_stops_before_any_replacement(self):
        d.make_plan(self.root)
        self.second.write_bytes(b"z" * d.MINIMUM_BYTES)
        with self.assertRaisesRegex(RuntimeError, "content/permissions changed"):
            self.apply()
        self.assertFalse(os.path.samefile(self.first, self.second))

    def test_active_process_stops_cleanup(self):
        d.make_plan(self.root)
        with patch.object(d, "running_under", return_value=[{"pid": 99}]):
            with self.assertRaisesRegex(RuntimeError, "in use"):
                self.apply()
        self.assertFalse(os.path.samefile(self.first, self.second))

    def test_resume_after_atomic_replace_before_progress_write(self):
        self.require_hardlinks()
        plan = d.make_plan(self.root)
        op = plan["operations"][0]
        keeper, target = self.root / op["keeper"], self.root / op["target"]
        temporary = self.root / op["temporary"]
        os.link(keeper, temporary)
        os.replace(temporary, target)
        summary = self.apply()
        self.assertTrue(summary["passed"])
        self.assertTrue(os.path.samefile(keeper, target))

    def test_resume_from_leftover_temporary_link(self):
        self.require_hardlinks()
        plan = d.make_plan(self.root)
        op = plan["operations"][0]
        os.link(self.root / op["keeper"], self.root / op["temporary"])
        self.assertTrue(self.apply()["passed"])
        self.assertFalse((self.root / op["temporary"]).exists())

    def test_live_asset_hardlink_added_after_plan_stops_cleanup(self):
        self.require_hardlinks()
        d.make_plan(self.root)
        os.link(self.first, self.root / "artifacts/v6/live.npz")
        with self.assertRaisesRegex(RuntimeError, "outside the archive"):
            self.apply()
        self.assertFalse(os.path.samefile(self.first, self.second))

    def test_path_escape_and_changed_report_block(self):
        with self.assertRaisesRegex(RuntimeError, "outside archive"):
            d.archived_file(self.root, "local/archive/../../artifacts/v6/manifest.json")
        d.make_plan(self.root)
        (self.two / "results.json").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "content/permissions changed"):
            self.apply()

    def test_duplicate_reports_and_small_files_are_not_candidates(self):
        report = b"experiment evidence" * d.MINIMUM_BYTES
        (self.one / "large-results.json").write_bytes(report)
        (self.two / "large-results.json").write_bytes(report)
        (self.one / "tiny.csv").write_bytes(b"rowid,probability\n")
        (self.two / "tiny.csv").write_bytes(b"rowid,probability\n")
        plan = d.make_plan(self.root)
        self.assertEqual(len(plan["operations"]), 1)
        self.assertEqual(Path(plan["operations"][0]["target"]).name, "cache.npz")

    def test_allocated_space_counts_original_inodes_once(self):
        def entry(inode, count, size=d.MINIMUM_BYTES, sha="a"):
            return {"inode": [1, inode], "links": count, "bytes": size,
                    "allocated_bytes": size, "sha256": sha, "mode": 0o644, "uid": 1000, "gid": 1000}
        state = {"files": {"local/archive/a/one.npz": entry(1, 2),
                           "local/archive/a/two.npz": entry(1, 2),
                           "local/archive/b/one.npz": entry(2, 2),
                           "local/archive/b/two.npz": entry(2, 2)}}
        ops, estimated = d.operations(state, "0" * 32)
        self.assertEqual(len(ops), 2)
        self.assertEqual(estimated, d.MINIMUM_BYTES)
        # The first inode is also used outside the archive, so it cannot participate.
        for key in ["local/archive/a/one.npz", "local/archive/a/two.npz"]:
            state["files"][key]["links"] = 3
        self.assertEqual(d.operations(state, "0" * 32), ([], 0))

    def test_changed_cleanup_script_blocks_apply(self):
        d.make_plan(self.root)
        (self.root / "tools/deduplicate_archive.py").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Plan evidence changed"):
            self.apply()
        self.assertFalse(os.path.samefile(self.first, self.second))

    def test_failed_live_bundle_verification_blocks_plan(self):
        with patch.object(d, "require_receipt", side_effect=RuntimeError("Invalid V6 receipt")):
            with self.assertRaisesRegex(RuntimeError, "Invalid V6"):
                d.make_plan(self.root)
        self.assertFalse((self.root / "local/archive_dedup_plan.json").exists())

    @unittest.skipUnless(os.name == "posix", "Windows symlink privilege is not assumed")
    def test_symlinks_are_preserved_without_following(self):
        (self.two / "original-cache").symlink_to("/missing/original/cache.npz")
        (self.two / "outside").symlink_to(self.root / "artifacts", target_is_directory=True)
        before = d.snapshot(self.root)
        d.make_plan(self.root)
        self.apply()
        d.same_contents(before, d.snapshot(self.root))


if __name__ == "__main__":
    unittest.main()
