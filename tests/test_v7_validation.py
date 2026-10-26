"""Unit tests validating the V7 candidate model configuration, calibration mathematics, and submission CSV."""

import csv
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from predict_v7 import calibrate_v7_probabilities

CONFIG_PATH = ROOT / "configs/v7.json"
V7_CSV_PATH = next((path for path in (
    ROOT / "artifacts/submissions/v7_candidate.csv",
    ROOT / "artifacts/experiments/v7-validation/v7_candidate.csv",
) if path.is_file()), ROOT / "artifacts/submissions/v7_candidate.csv")


class V7ValidationTests(unittest.TestCase):
    def test_v7_config_validity(self):
        self.assertTrue(CONFIG_PATH.is_file(), "configs/v7.json must exist")
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.assertEqual(config.get("version"), "v7_calibrated_event_scale_ranking")
        self.assertEqual(config.get("evaluation_events"), 52939)
        self.assertEqual(config.get("evaluation_rows"), 529390)
        self.assertAlmostEqual(config.get("sharpening_gamma", 0.0), 1.12)

        report_path = ROOT / config["validation_report"]
        self.assertEqual(hashlib.sha256(report_path.read_bytes()).hexdigest(), config["validation_report_sha256"])
        report = json.loads(report_path.read_text())
        self.assertTrue(report["complete"])
        self.assertEqual(report["protocol"]["gamma"], config["sharpening_gamma"])
        self.assertEqual(report["protocol"]["bootstrap"]["resamples"], 1000)
        self.assertEqual(report["protocol"]["bootstrap"]["seed"], 2026)
        results = report["results"]
        decisions = {"primary_ci_lower_positive": results["primary"]["ci_lower"] > 0,
                     "secondary_change_nonnegative": results["secondary"]["auc_diff"] >= 0,
                     "five_day_ci_lower_positive": results["five_day"]["ci_lower"] > 0,
                     "five_day_late_change_nonnegative": results["five_day"]["late_48_120h"]["auc_diff"] >= 0}
        self.assertEqual(report["decision_checks"], decisions)
        self.assertEqual(report["promotion_supported"], all(decisions.values()))
        self.assertEqual(config["promotion_supported"], report["promotion_supported"])
        self.assertFalse(report["model_fitting_performed"])
        self.assertFalse(report["competition_upload_performed"])
        for relative, expected in report["inputs_sha256"].items():
            if relative.endswith(".py") or relative in {"configs/v6.json", "configs/source_manifest.json"}:
                self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected, relative)
        mapping = {"primary_first_user_jan25_26": "primary", "secondary_first_user_jan26": "secondary",
                   "five_day_first_user_jan22_26": "five_day"}
        for window, key in mapping.items():
            result = results[key]
            evidence = config["validation_evidence"][window]
            self.assertEqual(evidence["events"], result["events"])
            self.assertEqual(evidence["candidate_auc"], result["candidate_auc"])
            self.assertEqual(evidence["v6_baseline_auc"], result["baseline_auc"])
            self.assertEqual(evidence["auc_change"], result["auc_diff"])
            self.assertEqual(evidence["paired_95_ci"], [result["ci_lower"], result["ci_upper"]])
        self.assertEqual(config["validation_evidence"]["five_day_late_48_120h"]["auc_change"],
                         results["five_day"]["late_48_120h"]["auc_diff"])
        self.assertTrue(report["submission_verification"]["fingerprint_verified"])
        self.assertEqual(report["submission_verification"]["submission_sha256"], config["submission_sha256"])

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
