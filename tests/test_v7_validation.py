"""Unit tests validating the V7 candidate model configuration, calibration mathematics, and submission CSV."""

import csv
import json
from pathlib import Path
import unittest
import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from predict_v7 import calibrate_v7_probabilities

CONFIG_PATH = ROOT / "configs/v7.json"
V7_CSV_PATH = ROOT / "artifacts/submissions/v7_candidate.csv"


class V7ValidationTests(unittest.TestCase):
    def test_v7_config_validity(self):
        self.assertTrue(CONFIG_PATH.is_file(), "configs/v7.json must exist")
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.assertEqual(config.get("version"), "v7_calibrated_event_scale_ranking")
        self.assertEqual(config.get("evaluation_events"), 52939)
        self.assertEqual(config.get("evaluation_rows"), 529390)
        self.assertAlmostEqual(config.get("sharpening_gamma", 0.0), 1.12)

        evidence = config.get("validation_evidence", {})
        for window in ["primary_first_user_jan25_26", "secondary_first_user_jan26", "five_day_first_user_jan22_26"]:
            self.assertIn(window, evidence, f"Missing {window} in validation evidence")
            report = evidence[window]
            self.assertTrue(report.get("promotion_gate_passed"), f"Gate not passed for {window}")
            self.assertGreater(report.get("auc_change", 0.0), 0.0)
            ci = report.get("paired_95_ci", [])
            self.assertEqual(len(ci), 2)
            self.assertGreater(ci[0], 0.0, f"Lower bound of 95% CI must be strictly positive for {window}")

    def test_v7_calibration_math(self):
        rng = np.random.default_rng(42)
        n_events = 20
        probs = rng.uniform(0.01, 0.4, size=(n_events, 10))
        event_sums_before = probs.sum(axis=1, keepdims=True)

        v7_probs = calibrate_v7_probabilities(probs, gamma=1.12)
        self.assertEqual(v7_probs.shape, (n_events, 10))

        # Event sums must be preserved to machine precision
        event_sums_after = v7_probs.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(event_sums_after, event_sums_before, rtol=1e-12, atol=1e-14)

        # Probabilities must be in [0, 1]
        self.assertTrue(np.all(v7_probs >= 0.0))
        self.assertTrue(np.all(v7_probs <= 1.0))

        # Check sharpening effect: ratio of top candidate to lowest candidate increases
        ratio_before = probs.max(axis=1) / np.maximum(probs.min(axis=1), 1e-8)
        ratio_after = v7_probs.max(axis=1) / np.maximum(v7_probs.min(axis=1), 1e-8)
        self.assertTrue(np.all(ratio_after > ratio_before))

    def test_v7_submission_integrity_if_present(self):
        if not V7_CSV_PATH.is_file():
            self.skipTest("artifacts/submissions/v7_candidate.csv not yet generated")

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)

        expected_sha = config["submission_sha256"]
        import hashlib
        h = hashlib.sha256(V7_CSV_PATH.read_bytes()).hexdigest()
        self.assertEqual(h, expected_sha, "Submission SHA256 does not match configs/v7.json")

        # Quick CSV check
        with open(V7_CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, ["rowid", "predicted_probability"])
            count = sum(1 for _ in reader)
            self.assertEqual(count, 529390)


if __name__ == "__main__":
    unittest.main()
