"""CPU regressions for verified, non-replacing V6 CSV publication (#7)."""
from concurrent.futures import ThreadPoolExecutor
import csv
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "legacy"))
import predict_v6 as v6
import submission_export


class V6PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.example = self.root / "example.csv"
        self.ids = np.array([f"opaque-{i}" for i in range(20)])
        self.probabilities = np.linspace(.01, .9, 20).reshape(2, 10)
        with self.example.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(submission_export.HEADER)
            writer.writerows((rowid, 0) for rowid in self.ids[::-1])
        expected = self.root / "expected.csv"
        submission_export.export_submission(self.ids, self.probabilities, self.example, expected, 20)
        self.fingerprint = v6.digest(expected)
        self.output = self.root / "outputs" / "reproduced.csv"
        self.verify_inputs = Mock()

    def publish(self, **kwargs):
        options = {"expected_sha256": self.fingerprint, "verify_inputs": self.verify_inputs}
        options.update(kwargs)
        return v6.export_verified_submission(self.ids, self.probabilities, self.example, self.output, 20, **options)

    def assert_clean(self):
        self.assertEqual(list(self.output.parent.glob(".v6-*")), [])
        self.assertFalse(self.output.with_suffix(".csv.tmp").exists())

    def test_canonical_output_and_idempotent_existing_file(self):
        self.publish()
        self.assertEqual(v6.digest(self.output), self.fingerprint)
        before = self.output.stat().st_mtime_ns
        self.publish()
        self.assertEqual(self.output.stat().st_mtime_ns, before)
        self.assertEqual(self.verify_inputs.call_count, 2)
        self.assert_clean()

    def test_existing_different_file_is_preserved(self):
        self.output.parent.mkdir()
        self.output.write_bytes(b"keep existing bytes")
        with self.assertRaises(FileExistsError):
            self.publish()
        self.assertEqual(self.output.read_bytes(), b"keep existing bytes")
        self.verify_inputs.assert_not_called()
        self.assert_clean()

    def test_competing_destination_is_never_replaced(self):
        link = os.link

        def competing_writer(source, destination):
            self.assertEqual(v6.digest(source), self.fingerprint)
            Path(destination).write_bytes(b"another producer owns this")
            return link(source, destination)

        with patch.object(v6.os, "link", side_effect=competing_writer), self.assertRaises(FileExistsError):
            self.publish()
        self.assertEqual(self.output.read_bytes(), b"another producer owns this")
        self.assert_clean()

    def test_simultaneous_producers_use_independent_temporary_files(self):
        barrier = threading.Barrier(2)
        exporter = submission_export.export_submission
        candidates = []

        def coordinated_export(*args):
            candidates.append(Path(args[3]))
            exporter(*args)
            barrier.wait(timeout=10)

        def producer():
            try:
                self.publish()
                return "published"
            except FileExistsError:
                return "occupied"

        with patch.object(submission_export, "export_submission", side_effect=coordinated_export):
            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(lambda _: producer(), range(2)))
        self.assertCountEqual(outcomes, ["published", "occupied"])
        self.assertEqual(len(candidates), 2)
        self.assertNotEqual(candidates[0].parent, candidates[1].parent)
        self.assertEqual(v6.digest(self.output), self.fingerprint)
        self.assert_clean()

    def test_failed_fingerprint_never_publishes(self):
        with self.assertRaisesRegex(RuntimeError, "fingerprint"):
            self.publish(expected_sha256="0" * 64)
        self.assertFalse(self.output.exists())
        self.verify_inputs.assert_not_called()
        self.assert_clean()

    def test_probabilities_are_verified_before_publication(self):
        exporter = submission_export.export_submission

        def wrong_predictions(rowids, probabilities, *args):
            return exporter(rowids, np.asarray(probabilities).ravel()[::-1], *args)

        with patch.object(submission_export, "export_submission", side_effect=wrong_predictions), self.assertRaises(AssertionError):
            self.publish()
        self.assertFalse(self.output.exists())
        self.verify_inputs.assert_not_called()
        self.assert_clean()

    def test_changed_inputs_abort_after_staging_and_before_publication(self):
        def changed_bundle():
            self.assertFalse(self.output.exists())
            staged = list(self.output.parent.glob(".v6-*/candidate.csv"))
            self.assertEqual(len(staged), 1)
            self.assertEqual(v6.digest(staged[0]), self.fingerprint)
            raise RuntimeError("Bundle changed")

        with self.assertRaisesRegex(RuntimeError, "Bundle changed"):
            self.publish(verify_inputs=changed_bundle)
        self.assertFalse(self.output.exists())
        self.assert_clean()

    def test_publication_error_has_no_replacing_fallback(self):
        with patch.object(v6.os, "link", side_effect=OSError("link unavailable")), self.assertRaises(OSError):
            self.publish()
        self.assertFalse(self.output.exists())
        self.assert_clean()


if __name__ == "__main__":
    unittest.main()
