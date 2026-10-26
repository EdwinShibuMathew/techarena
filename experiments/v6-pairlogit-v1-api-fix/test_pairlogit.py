"""Synthetic checks. CatBoost tests must also pass in the existing WSL env."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

import pairlogit_core as core
import event_scale_diagnostic as audit


class NumericTests(unittest.TestCase):
    def test_global_pairs_are_bounded_reproducible_and_include_no_click(self):
        y = np.zeros((12, 10), dtype=np.int8)
        y[[0, 2, 5, 8], [0, 1, 2, 3]] = 1
        pairs = core.sample_pairs(y)
        np.testing.assert_array_equal(pairs, core.sample_pairs(y))
        self.assertEqual(pairs.shape, (int(y.sum()) * 16, 2))
        self.assertTrue((y.ravel()[pairs[:, 0]] == 1).all())
        self.assertTrue((y.ravel()[pairs[:, 1]] == 0).all())
        for winner in np.flatnonzero(y.ravel()):
            self.assertEqual(len(np.unique(pairs[pairs[:, 0] == winner, 1])), 16)
        self.assertTrue((y.sum(1)[pairs[:, 1] // 10] == 0).any())
        self.assertTrue((pairs[:, 0] // 10 != pairs[:, 1] // 10).any())

    def test_stable_sigmoid(self):
        p = core.sigmoid(np.array([-10000., 0., 10000.]))
        np.testing.assert_array_equal(p, [0., .5, 1.])
        self.assertTrue(np.isfinite(p).all())

    def test_intercept_matches_training_ctr_and_cancels_constant_offset(self):
        raw = np.linspace(-3, 4, 100)
        y = np.zeros(100, dtype=np.int8)
        y[:13] = 1
        b, proof = core.intercept(raw, y)
        shifted, _ = core.intercept(raw + 100, y)
        self.assertAlmostEqual(float(core.sigmoid(raw + b).mean()), .13, places=12)
        self.assertAlmostEqual(shifted, b - 100, places=12)
        self.assertFalse(proof["target_labels_used"])
        with self.assertRaises(ValueError):
            core.intercept(raw, np.zeros_like(y))

    def test_replacement_preserves_baseline_exactly_for_identity(self):
        rng = np.random.default_rng(71)
        members = rng.uniform(.001, .3, (5, 8, 10))
        baseline = (2 / 3) * members[:3].mean(0) + (1 / 3) * members[3:].mean(0)
        np.testing.assert_array_equal(core.replacement(baseline, members[0], members[0]), baseline)
        new = rng.uniform(.001, .3, (8, 10))
        replaced = core.replacement(baseline, members[0], new)
        expected = baseline - (2 / 9) * members[0] + (2 / 9) * new
        np.testing.assert_allclose(replaced, expected, rtol=0, atol=1e-15)

    def test_parameters_are_fixed_and_do_not_modify_saved_dictionary(self):
        old = {"bootstrap_type": "MVS", "subsample": .8, "boosting_type": "Plain",
               "border_count": 254, "leaf_estimation_iterations": 10, "loss_function": "Logloss"}
        saved = dict(old)
        params = core.ranker_parameters(old)
        self.assertEqual(old, saved)
        self.assertEqual(params["loss_function"], "PairLogit")
        self.assertEqual(params["iterations"], 545)
        self.assertEqual(params["leaf_estimation_iterations"], 10)
        self.assertEqual(params["bootstrap_type"], "MVS")
        self.assertFalse(params["allow_writing_files"])
        self.assertFalse(params["use_best_model"])


@unittest.skipUnless(importlib.util.find_spec("catboost"), "CatBoost unavailable here; run in existing WSL environment")
class CatBoostTests(unittest.TestCase):
    def test_cpu_pair_fit_saved_replay_and_intercept(self):
        from catboost import CatBoost, CatBoostRanker, Pool
        rng = np.random.default_rng(72)
        x = rng.normal(size=(100, 37)).astype(np.float32)
        y = np.zeros(100, dtype=np.int8)
        y[::10] = 1
        names = [f"f{i}" for i in range(37)]
        pairs = core.sample_pairs(y)
        params = core.ranker_parameters({"boosting_type": "Plain", "bootstrap_type": "MVS",
                                         "subsample": .8, "leaf_estimation_iterations": 10})
        params["iterations"] = 2
        model = CatBoostRanker(**params)
        pool = Pool(x, label=y, group_id=np.zeros(100, dtype=np.int64), pairs=pairs,
                    pairs_weight=np.ones(len(pairs)), feature_names=names)
        model.fit(pool, verbose=False)
        raw = model.predict(pool, thread_count=4)
        # Public base API explicitly requests raw values; ranker API must agree.
        explicit_raw = CatBoost.predict(model, pool, prediction_type="RawFormulaVal", thread_count=4)
        np.testing.assert_allclose(raw, explicit_raw, rtol=0, atol=1e-15)
        bias, _ = core.intercept(raw, y)
        p = core.sigmoid(raw + bias)
        self.assertTrue(np.isfinite(p).all())
        self.assertAlmostEqual(float(p.mean()), float(y.mean()), places=12)
        self.assertEqual(model.get_all_params()["loss_function"], "PairLogit")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            path = Path(temporary) / "tiny.cbm"
            model.save_model(str(path))
            loaded = CatBoostRanker()
            loaded.load_model(str(path))
            np.testing.assert_allclose(raw, loaded.predict(pool, thread_count=4),
                                       rtol=0, atol=1e-15)
        self.assertGreaterEqual(audit.auc(y, p), 0)


if __name__ == "__main__":
    unittest.main()
