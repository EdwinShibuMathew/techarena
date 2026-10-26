"""Probability caching must preserve ranking and source precision (#6)."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from experiment_runner import compute_roc_auc
from feature_cache import CombinerFeatureCache


class FeatureCachePrecisionTests(unittest.TestCase):
    def test_close_float64_scores_preserve_auc_and_exact_values(self):
        probabilities = np.full((1, 10), .1, dtype=np.float64)
        probabilities[0, 0] = .100000001
        raw = probabilities + 1e-10
        labels = np.array([1] + [0] * 9, dtype=np.int8)
        with tempfile.TemporaryDirectory() as temporary:
            cache = CombinerFeatureCache(Path(temporary))
            cache.save("member", "split", np.zeros((10, 37)), [f"f{i}" for i in range(37)],
                       probabilities, raw, labels)
            loaded = cache.load("member", "split")
        self.assertEqual(loaded["probs"].dtype, np.dtype("float64"))
        self.assertEqual(loaded["raw_probs"].dtype, np.dtype("float64"))
        np.testing.assert_array_equal(loaded["probs"], probabilities)
        np.testing.assert_array_equal(loaded["raw_probs"], raw)
        self.assertEqual(compute_roc_auc(labels, loaded["probs"]), 1.)
        self.assertEqual(compute_roc_auc(labels, loaded["probs"]), compute_roc_auc(labels, probabilities))
        self.assertEqual(loaded["X"].dtype, np.dtype("float32"))
        np.testing.assert_array_equal(loaded["labels"], labels)

    def test_float32_and_mixed_probability_dtypes_round_trip(self):
        for dtype, raw_dtype in ((np.float32, np.float32), (np.float32, np.float64), (np.float64, np.float32)):
            with self.subTest(dtype=dtype, raw_dtype=raw_dtype), tempfile.TemporaryDirectory() as temporary:
                probabilities = np.linspace(.01, .9, 20, dtype=dtype).reshape(2, 10)
                raw = np.linspace(.02, .8, 20, dtype=raw_dtype).reshape(2, 10)
                cache = CombinerFeatureCache(Path(temporary))
                cache.save("member", "split", np.ones((20, 2)), ["a", "b"], probabilities, raw)
                loaded = cache.load("member", "split")
                self.assertEqual(loaded["probs"].dtype, probabilities.dtype)
                self.assertEqual(loaded["raw_probs"].dtype, raw.dtype)
                np.testing.assert_array_equal(loaded["probs"], probabilities)
                np.testing.assert_array_equal(loaded["raw_probs"], raw)
                self.assertNotIn("labels", loaded)
                self.assertEqual(loaded["feature_names"], ["a", "b"])

    def test_legacy_float32_archive_remains_readable_without_invented_precision(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = CombinerFeatureCache(Path(temporary))
            probabilities = np.full((1, 10), .1, dtype=np.float32)
            np.savez_compressed(Path(temporary) / "old_split_features.npz", X=np.zeros((10, 1), dtype=np.float32),
                                feature_names=np.array(["a"]), probs=probabilities, raw_probs=probabilities)
            loaded = cache.load("old", "split")
            self.assertEqual(loaded["probs"].dtype, np.dtype("float32"))
            np.testing.assert_array_equal(loaded["probs"], probabilities)


if __name__ == "__main__":
    unittest.main()
