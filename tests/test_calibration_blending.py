"""Blend-weight, identity and precision regressions for event calibration (#5)."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from hierarchical_event_model import HierarchicalEventCalibrator


class CalibrationBlendingTests(unittest.TestCase):
    def test_documented_weights_interpolate_the_event_scale(self):
        p = np.full((1, 10), .05)
        for weight, expected in ((0., .5), (.5, .65), (1., .8)):
            with self.subTest(weight=weight):
                actual = HierarchicalEventCalibrator(weight).recompose(p, np.array([.8]))
                self.assertEqual(actual.dtype, np.dtype("float64"))
                self.assertAlmostEqual(float(actual.sum()), expected, places=14)

    def test_zero_weight_and_omitted_scale_preserve_exact_values_and_shape(self):
        p = np.linspace(.01, .3, 20, dtype=np.float64)
        p[0] = .010000000000000002
        for original in (p, p.reshape(2, 10), p.astype(np.float32)):
            for weight in (0., .5, 1.):
                with self.subTest(shape=original.shape, weight=weight):
                    model = HierarchicalEventCalibrator(weight)
                    actual = model.recompose(original)
                    np.testing.assert_array_equal(actual, original)
                    self.assertEqual(actual.shape, original.shape)
                    self.assertEqual(actual.dtype, np.dtype("float64"))
                    self.assertFalse(np.shares_memory(actual, original))
                    if weight == 0:
                        np.testing.assert_array_equal(model.recompose(original, np.array([.2, .9])), original)

    def test_positive_rescaling_preserves_candidate_order_and_ties(self):
        p = np.array([[.1, .100000001, .05, .05, .03, .04, .02, .01, .2, .15]])
        original = p.copy()
        for weight in (0., .5, 1.):
            actual = HierarchicalEventCalibrator(weight).recompose(p, np.array([.8]))
            np.testing.assert_array_equal(np.sign(actual[:, :, None] - actual[:, None, :]),
                                          np.sign(p[:, :, None] - p[:, None, :]))
        np.testing.assert_array_equal(p, original)

    def test_zero_and_tiny_events_are_finite_without_a_denominator_floor(self):
        p = np.vstack([np.zeros(10), np.arange(1, 11) * 1e-300])
        actual = HierarchicalEventCalibrator(1.).recompose(p, np.array([.8, .8]))
        np.testing.assert_array_equal(actual[0], np.zeros(10))
        np.testing.assert_allclose(actual[1], np.arange(1, 11) / 55 * .8, rtol=1e-14, atol=0)
        self.assertAlmostEqual(float(actual[1].sum()), .8, places=14)
        self.assertTrue(np.isfinite(actual).all())

    def test_raw_event_sums_above_one_remain_supported(self):
        p = np.full((2, 10), .2)
        for weight in (0., .5, 1.):
            model = HierarchicalEventCalibrator(weight)
            np.testing.assert_allclose(model.recompose(p, p.sum(1, keepdims=True)), p, rtol=1e-14, atol=0)
        actual = HierarchicalEventCalibrator(.5).recompose(p, np.array([.8, .6]))
        np.testing.assert_allclose(actual.sum(1), [1.4, 1.3], rtol=1e-14)

    def test_invalid_weights_are_rejected_at_construction_and_recomposition(self):
        for weight in (-.1, 1.1, np.nan, np.inf, -np.inf, [.5], "bad", None, 1j):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                HierarchicalEventCalibrator(weight)
        model = HierarchicalEventCalibrator()
        model.blend_weight = np.nan
        with self.assertRaises(ValueError):
            model.recompose(np.full((1, 10), .05), np.array([.8]))

    def test_invalid_shapes_and_scales_fail_without_clipping(self):
        model = HierarchicalEventCalibrator(1.)
        for p in (np.array([]), np.ones(9), np.ones((1, 2, 5)), np.ones((10, 2)),
                  np.full(10, np.nan), np.full(10, -.1), np.full(10, 1.1)):
            with self.subTest(shape=p.shape), self.assertRaises(ValueError):
                model.recompose(p)
        p = np.full((2, 10), .05)
        for scale in (np.array([.8]), np.array([[.8, .8]]), np.array([-.1, .8]),
                      np.array([np.nan, .8]), np.array([np.inf, .8])):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                model.recompose(p, scale)
        for candidates in (0, -1, 2.5, True):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                model.recompose(p, candidates_per_event=candidates)
        with self.assertRaisesRegex(ValueError, "outside"):
            model.recompose(np.array([[1.] + [0.] * 9]), np.array([2.]))


if __name__ == "__main__":
    unittest.main()
