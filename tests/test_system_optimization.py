"""Unit tests for TechArena System Optimization Tools.

Validates the experiment runner, paired bootstrap evaluation, feature enrichment,
hierarchical event modeling, cold-start recency weighting, and combiner feature caching.
"""

from pathlib import Path
import tempfile
import unittest
import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from experiment_runner import compute_roc_auc, PairedBootstrapEvaluator
from feature_enrichment import (
    compute_relative_event_features,
    enrich_feature_matrix,
    ENRICHED_FEATURE_NAMES
)
from hierarchical_event_model import (
    conditional_item_choice,
    compute_event_features,
    HierarchicalEventCalibrator
)
from cold_start_enhancer import (
    compute_recency_weighted_profile,
    RecencyWeightedRidgeFallback
)
from feature_cache import CombinerFeatureCache


class TestROCAndBootstrap(unittest.TestCase):
    def test_compute_roc_auc_perfect_and_inverse(self):
        y_true = np.array([1, 0, 1, 0], dtype=np.int8)
        y_score_perfect = np.array([0.9, 0.1, 0.8, 0.2])
        self.assertAlmostEqual(compute_roc_auc(y_true, y_score_perfect), 1.0)

        y_score_inverse = np.array([0.1, 0.9, 0.2, 0.8])
        self.assertAlmostEqual(compute_roc_auc(y_true, y_score_inverse), 0.0)

    def test_compute_roc_auc_ties(self):
        y_true = np.array([1, 0, 1, 0], dtype=np.int8)
        y_score_ties = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(compute_roc_auc(y_true, y_score_ties), 0.5)

    def test_paired_bootstrap_evaluator(self):
        # 10 events with 10 candidates each = 100 rows
        n_events = 20
        n_rows = n_events * 10
        rng = np.random.default_rng(42)
        
        labels = np.zeros(n_rows, dtype=np.int8)
        # 1 click per event in first 12 events
        for e in range(12):
            labels[e * 10 + 3] = 1
            
        base_scores = rng.uniform(0.1, 0.9, size=n_rows)
        # Candidate model has higher score on true clicks
        cand_scores = base_scores.copy()
        cand_scores[labels == 1] += 0.3
        
        evaluator = PairedBootstrapEvaluator(n_resamples=50, seed=42)
        metrics = evaluator.evaluate(
            candidate_scores=cand_scores,
            baseline_scores=base_scores,
            labels=labels,
            user_ids=np.arange(n_events)
        )
        self.assertGreater(metrics.auc_diff, 0.0)
        self.assertTrue(metrics.gate_passed)
        self.assertGreater(metrics.ci_lower, 0.0)


class TestFeatureEnrichment(unittest.TestCase):
    def test_enrich_features_shapes_and_values(self):
        n_events = 5
        n_rows = n_events * 10
        probs = np.random.uniform(0.01, 0.8, size=(n_events, 10)).astype(np.float32)
        raw_probs = probs.copy()
        X_37 = np.zeros((n_rows, 37), dtype=np.float32)
        X_37[:, 15] = 2.0  # clicks_6h
        X_37[:, 17] = 5.0  # clicks_72h
        names_37 = [f"feat_{i}" for i in range(37)]

        X_enriched, names_enriched = enrich_feature_matrix(
            X_37, names_37, probs, raw_probs, candidates_per_event=10
        )
        self.assertEqual(X_enriched.shape, (n_rows, 43))
        self.assertEqual(len(names_enriched), 43)
        self.assertEqual(names_enriched[-6:], ENRICHED_FEATURE_NAMES)

        # Check ratio_to_max is <= 1.0
        ratio_to_max = X_enriched[:, 37]
        self.assertTrue(np.all(ratio_to_max <= 1.0001))
        self.assertTrue(np.all(ratio_to_max >= 0.0))

        # Check ranks are in [0, 9]
        ranks = X_enriched[:, 40]
        self.assertTrue(np.all(ranks >= 0.0))
        self.assertTrue(np.all(ranks <= 9.0))


class TestHierarchicalEventModel(unittest.TestCase):
    def test_conditional_item_choice(self):
        probs = np.array([
            [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])
        q = conditional_item_choice(probs, candidates_per_event=10)
        self.assertEqual(q.shape, probs.shape)
        event_sums = q.sum(axis=1)
        np.testing.assert_allclose(event_sums, np.ones(2), rtol=1e-5)

    def test_calibrator_recomposition(self):
        probs = np.random.uniform(0.05, 0.5, size=(4, 10))
        calibrator = HierarchicalEventCalibrator()
        
        # Test identity when propensity equals original event sum
        event_sum = probs.sum(axis=1, keepdims=True)
        recomposed = calibrator.recompose(probs, event_propensity=event_sum, candidates_per_event=10)
        np.testing.assert_allclose(recomposed, probs, rtol=1e-5)


class TestColdStartEnhancer(unittest.TestCase):
    def test_recency_weighted_profile(self):
        n_users = 2
        dim = 8
        embeddings = np.ones((50, dim), dtype=np.float32)
        # Give item 5 different values
        embeddings[5] = np.full(dim, 2.0, dtype=np.float32)

        history_items = np.array([
            [5, 1, -1, -1],  # user 0 has clicks on 5 and 1
            [-1, -1, -1, -1]  # user 1 has empty history
        ])
        # User 0: item 5 is very recent (0h), item 1 is older (72h)
        history_ages = np.array([
            [0.0, 72.0, 1000.0, 1000.0],
            [1000.0, 1000.0, 1000.0, 1000.0]
        ])

        profiles, has_history = compute_recency_weighted_profile(
            history_items, history_ages, embeddings, half_life_hours=24.0
        )
        self.assertEqual(profiles.shape, (2, dim))
        self.assertTrue(has_history[0])
        self.assertFalse(has_history[1])
        # Profile for user 1 should be all zeros
        np.testing.assert_allclose(profiles[1], np.zeros(dim))
        # Profile for user 0 should be weighted towards item 5 (value 2.0)
        self.assertGreater(profiles[0, 0], 1.5)

    def test_ridge_fallback_fit_predict(self):
        rng = np.random.default_rng(42)
        dim = 4
        n_samples = 30
        profiles = rng.normal(size=(n_samples, dim)).astype(np.float32)
        true_w = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 2.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.5]], dtype=np.float32)
        users = np.dot(profiles, true_w) + 0.1

        model = RecencyWeightedRidgeFallback(alpha=0.01)
        model.fit(profiles, users)
        preds = model.predict(profiles)
        self.assertEqual(preds.shape, users.shape)
        np.testing.assert_allclose(preds, users, atol=0.1)


class TestFeatureCache(unittest.TestCase):
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CombinerFeatureCache(cache_dir=Path(tmpdir))
            X = np.ones((10, 37), dtype=np.float32)
            names = [f"f{i}" for i in range(37)]
            probs = np.full((1, 10), 0.1, dtype=np.float32)
            raw = np.full((1, 10), 0.2, dtype=np.float32)
            labels = np.zeros(10, dtype=np.int8)

            cache.save("dim8_seed42", "jan24", X, names, probs, raw, labels)
            self.assertTrue(cache.has_cache("dim8_seed42", "jan24"))

            loaded = cache.load("dim8_seed42", "jan24")
            np.testing.assert_array_equal(loaded["X"], X)
            np.testing.assert_array_equal(loaded["probs"], probs)
            np.testing.assert_array_equal(loaded["labels"], labels)
            self.assertEqual(loaded["feature_names"], names)


if __name__ == "__main__":
    unittest.main()
