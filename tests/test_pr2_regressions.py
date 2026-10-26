"""Regressions for user bootstrap, cold history, and safe V7 generation."""
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from cold_start_enhancer import compute_recency_weighted_profile
from experiment_runner import FastAuc, PairedBootstrapEvaluator, compute_roc_auc
import predict_v7 as v7


def dense_auc(y, p):
    y, p = np.asarray(y).ravel(), np.asarray(p).ravel()
    a, b = p[y == 1], p[y == 0]
    return float(((a[:, None] > b) + .5 * (a[:, None] == b)).mean()) if len(a) and len(b) else np.nan


class BootstrapRegressions(unittest.TestCase):
    def test_repeated_users_match_explicit_duplication(self):
        rng = np.random.default_rng(11)
        users = np.array(["alice"] * 4 + ["bob"] * 2 + ["carol"] * 3)
        codes = np.array([0] * 4 + [1] * 2 + [2] * 3)
        y = np.zeros((9, 10), dtype=np.int8)
        y[:, 0] = 1
        base, cand = [rng.integers(0, 8, y.shape) / 10 for _ in range(2)]
        evaluator = PairedBootstrapEvaluator(n_resamples=100, seed=7)
        actual = evaluator.evaluate(cand, base, y, users)
        expected, draw = [], np.random.default_rng(7)
        for _ in range(100):
            counts = draw.multinomial(3, np.full(3, 1 / 3))
            idx = np.repeat(np.arange(9), counts[codes])
            expected.append(dense_auc(y[idx], cand[idx]) - dense_auc(y[idx], base[idx]))
        np.testing.assert_allclose([actual.ci_lower, actual.ci_upper], np.quantile(expected, [.025, .975]), atol=1e-14)
        self.assertEqual(actual, evaluator.evaluate(cand, base, y, np.repeat(users, 10)))
        one = evaluator.evaluate(cand, base, y, np.zeros(9))
        self.assertAlmostEqual(one.ci_lower, actual.auc_diff)
        self.assertAlmostEqual(one.ci_upper, actual.auc_diff)

    def test_undefined_draws_are_excluded_and_insufficient_valid_draws_fail(self):
        y = np.zeros((10, 10), dtype=np.int8)
        y[:4, 0] = 1
        base = np.arange(100).reshape(10, 10) / 100
        cand = base[:, ::-1]
        values, rng = [], np.random.default_rng(2026)
        for _ in range(1000):
            idx = np.repeat(np.arange(10), rng.multinomial(10, np.full(10, .1)))
            delta = dense_auc(y[idx], cand[idx]) - dense_auc(y[idx], base[idx])
            if np.isfinite(delta):
                values.append(delta)
        self.assertTrue(990 <= len(values) < 1000)
        actual = PairedBootstrapEvaluator().evaluate(cand, base, y, np.arange(10))
        self.assertEqual(actual.resamples, len(values))
        np.testing.assert_allclose([actual.ci_lower, actual.ci_upper], np.quantile(values, [.025, .975]), atol=1e-14)
        y[1:, 0] = 0
        with self.assertRaisesRegex(ValueError, "99%"):
            PairedBootstrapEvaluator().evaluate(cand, base, y, np.arange(10))
        self.assertTrue(np.isnan(FastAuc([0, 0], [.1, .2], [0, 0]).value(np.ones(1))))

    def test_invalid_inputs_and_parameters_fail(self):
        y = np.zeros(20)
        y[0] = 1
        p = np.linspace(.01, .9, 20)
        e = PairedBootstrapEvaluator(n_resamples=10)
        for users in ([0], np.arange(20), [None, 1], [np.nan, 1]):
            with self.assertRaises(ValueError):
                e.evaluate(p, p, y, users)
        for a, b, labels in ((p[:-1], p, y), (p, p, np.full(20, .5)),
                              (p, p, np.zeros(20)), (np.full(20, np.nan), p, y)):
            with self.assertRaises(ValueError):
                e.evaluate(a, b, labels, [0, 1])
        for n, alpha in ((0, .05), (1.5, .05), (True, .05), (10, 0), (10, 1), (10, np.nan)):
            with self.assertRaises(ValueError):
                PairedBootstrapEvaluator(n_resamples=n, alpha=alpha)
        with self.assertRaises(ValueError):
            e.evaluate(p, p, y, [0, 1], candidates_per_event=2.5)
        with self.assertRaises(ValueError):
            compute_roc_auc([0, .5, 1], [0, .5, 1])


class ColdHistoryRegressions(unittest.TestCase):
    def test_old_single_item_and_equal_age_means_do_not_shrink(self):
        embeddings = np.array([[1., 2.], [3., 4.]])
        p, present = compute_recency_weighted_profile(np.array([[0, -1], [0, 1], [-1, -1]]),
            np.array([[720., np.nan], [720., 720.], [np.nan, np.nan]]), embeddings)
        np.testing.assert_allclose(p, [[1, 2], [2, 3], [0, 0]])
        np.testing.assert_array_equal(present, [True, True, False])
        p, _ = compute_recency_weighted_profile(np.array([[0, 1]]), np.array([[719., 720.]]), embeddings, 1e-300)
        np.testing.assert_allclose(p, [[1, 2]])

    def test_relative_weights_preserve_original_decay(self):
        p, _ = compute_recency_weighted_profile(np.array([[0, 1]]), np.array([[48., 72.]]), np.array([[2.], [10.]]))
        expected = (2 * np.exp(-2) + 10 * np.exp(-3)) / (np.exp(-2) + np.exp(-3))
        self.assertAlmostEqual(float(p[0, 0]), expected, places=6)

    def test_invalid_histories_fail(self):
        for items, ages, decay in (([[2]], [[0]], 24), ([[-2]], [[0]], 24), ([[0]], [[np.nan]], 24),
                                    ([[0]], [[-1]], 24), ([[0]], [[0, 1]], 24), ([[0]], [[0]], 0)):
            with self.assertRaises(ValueError):
                compute_recency_weighted_profile(np.array(items), np.array(ages), np.ones((2, 1)), decay)


class V7OutputRegressions(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for directory in ("configs", "data/cache", "data/raw", "artifacts/submissions"):
            (self.root / directory).mkdir(parents=True)
        self.ids, self.probs = np.array([f"opaque-{i}" for i in range(20)]), np.linspace(.01, .2, 20)
        np.save(self.root / "data/cache/evaluation_rowids.npy", self.ids)
        self.baseline = self.root / "artifacts/submissions/v6.csv"
        self.example = self.root / "data/raw/example_submission.csv"
        self.write_csv(self.baseline, self.ids[::-1], self.probs[::-1])
        self.write_csv(self.example, self.ids[np.roll(np.arange(20), 3)], np.zeros(20))
        expected = self.root / "expected.csv"
        v7.export_submission(self.ids, v7.calibrate_v7_probabilities(self.probs), self.example, expected, 20)
        self.config = {"evaluation_rows": 20, "sharpening_gamma": 1.12, "submission_sha256": v7.digest(expected)}
        self.write_config()
        self.output = self.root / "artifacts/submissions/candidate.csv"
        self.original = self.baseline.read_bytes()

    @staticmethod
    def write_csv(path, ids, probs):
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rowid", "predicted_probability"])
            writer.writerows(zip(ids, probs))

    def write_config(self):
        (self.root / "configs/v7.json").write_text(json.dumps(self.config))
        (self.root / "configs/v6.json").write_text(json.dumps({"submission_sha256": v7.digest(self.baseline)}))

    def test_alignment_and_idempotent_generation(self):
        np.testing.assert_array_equal(v7.load_aligned_probabilities(self.baseline, self.ids), self.probs)
        v7.generate(self.output, self.root, gamma=1.12)
        self.assertEqual(v7.digest(self.output), self.config["submission_sha256"])
        before, receipt = self.output.stat().st_mtime_ns, self.output.with_suffix(".csv.receipt.json")
        receipt_before = receipt.read_bytes()
        v7.generate(self.output, self.root)
        self.assertEqual(self.output.stat().st_mtime_ns, before)
        self.assertEqual(receipt.read_bytes(), receipt_before)
        self.assertEqual(self.baseline.read_bytes(), self.original)

    def test_duplicate_missing_unknown_and_nonfinite_rows_fail(self):
        for ids, probs in ((self.ids[:-1], self.probs[:-1]), (np.repeat(self.ids[:1], 20), self.probs),
                           (np.array(["unknown", *self.ids[1:]]), self.probs), (self.ids, np.full(20, np.nan))):
            self.write_csv(self.baseline, ids, probs)
            with self.assertRaises(ValueError):
                v7.load_aligned_probabilities(self.baseline, self.ids)

    def test_existing_files_aliases_and_wrong_gamma_are_preserved(self):
        hardlink, symlink = self.root / "artifacts/hardlink.csv", self.root / "artifacts/symlink.csv"
        os.link(self.baseline, hardlink)
        symlink.symlink_to(self.baseline)
        for output in (self.baseline, hardlink, symlink, self.root / "outside.csv"):
            with self.assertRaises(ValueError):
                v7.generate(output, self.root)
        self.output.write_text("keep these bytes")
        with self.assertRaises(FileExistsError):
            v7.generate(self.output, self.root)
        self.assertEqual(self.output.read_text(), "keep these bytes")
        with self.assertRaises(ValueError):
            v7.generate(self.output, self.root, gamma=1.2)
        self.assertEqual(self.baseline.read_bytes(), self.original)

    def test_failed_verification_and_concurrent_output_do_not_replace_files(self):
        self.config["submission_sha256"] = "0" * 64
        self.write_config()
        with self.assertRaises(RuntimeError):
            v7.generate(self.output, self.root)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob(".v7-*")), [])
        self.assertFalse(self.output.with_suffix(".csv.receipt.json").exists())
        self.config["submission_sha256"] = v7.digest(self.root / "expected.csv")
        self.write_config()
        link = os.link
        def concurrent(source, destination):
            Path(destination).write_text("concurrent producer")
            return link(source, destination)
        with patch.object(v7.os, "link", side_effect=concurrent), self.assertRaises(FileExistsError):
            v7.generate(self.output, self.root)
        self.assertEqual(self.output.read_text(), "concurrent producer")

    def test_zero_underflow_and_invalid_probabilities(self):
        p = np.vstack([np.zeros(10), np.full(10, 1e-300)])
        result = v7.calibrate_v7_probabilities(p)
        self.assertEqual(result.shape, p.shape)
        np.testing.assert_allclose(result[1], p[1], rtol=1e-14, atol=0)
        np.testing.assert_array_equal(result[0], p[0])
        for bad in (np.ones(9), np.ones((10, 2)), np.full(10, np.nan), np.full(10, -1)):
            with self.assertRaises(ValueError):
                v7.calibrate_v7_probabilities(bad)
        with self.assertRaises(ValueError):
            v7.calibrate_v7_probabilities(np.array([[.99] + [.5] * 9]))
