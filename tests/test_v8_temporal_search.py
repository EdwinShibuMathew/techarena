"""Fitting-only temporal selection, correction gradients and release regressions."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "experiments/v8-temporal-combiner-search"))
import search_core as core
import predict_v8_selected as predictor
spec = importlib.util.spec_from_file_location("temporal_search_runner", ROOT / "experiments/v8-temporal-combiner-search/runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def matrix(events=80):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(events * 10, 37)).astype(np.float32)
    x[:, 0] = rng.uniform(.01, .9, len(x))
    for name, values in (("history_length", rng.integers(0, 20, events)),
                         ("user_ctr", rng.uniform(.01, .2, events)),
                         ("unseen_user", rng.integers(0, 2, events)),
                         ("time_bucket", rng.integers(0, 4, events)),
                         ("day_type", rng.integers(0, 2, events))):
        x[:, core.BASE_COLUMNS.index(name)] = np.repeat(values, 10)
    x[:, 15], x[:, 17] = 1, 3
    return x


def development(gains=(.01, .02)):
    return {"status": "complete", "folds": [{"fold": number, "reference_auc": .8,
        "candidate_auc": .8 + gain, "auc_diff": gain} for number, gain in enumerate(gains)]}


class TemporalTests(unittest.TestCase):
    def test_forward_folds_never_use_promotion_days(self):
        start = core.FOLDS["jan24"]
        times = start + np.array([0, 7, 8, 15, 16, 23]) * 3600
        first, second = core.development_folds(times)
        np.testing.assert_array_equal(np.flatnonzero(first[0]), [0, 1])
        np.testing.assert_array_equal(np.flatnonzero(first[1]), [2, 3])
        np.testing.assert_array_equal(np.flatnonzero(second[0]), [0, 1, 2, 3])
        np.testing.assert_array_equal(np.flatnonzero(second[1]), [4, 5])
        for extra in (start - 1, start + core.DAY):
            with self.assertRaises(ValueError):
                core.development_folds(np.r_[times, extra])

    def test_timestamp_ties_stay_in_calibration_and_events_stay_whole(self):
        times = np.array([0, 1, 2, 3, 4, 5, 5, 6])
        base, calibration, cutoff = core.chronological_split(times)
        self.assertEqual(cutoff, 5)
        np.testing.assert_array_equal(np.flatnonzero(calibration), [5, 6, 7])
        self.assertLess(times[base].max(), times[calibration].min())
        self.assertEqual(np.repeat(base, 10).sum(), base.sum() * 10)

    def test_selection_is_stable_requires_both_folds_and_ignores_dictionary_order(self):
        result = {"C": development((.02, .02)), "B": development((.03, -.001)), "A": development((.02, .02))}
        self.assertEqual(core.choose(result), "A")
        result["A"] = development((.019, .019))
        self.assertEqual(core.choose(result), "C")
        result["A"] = development((.02 - 5e-13, .02))
        self.assertEqual(core.choose(result), "A")
        self.assertIsNone(core.choose({v: development((0, 0)) for v in "ABC"}))
        for bad in (np.nan, np.inf):
            result["A"]["folds"][0]["auc_diff"] = bad
            with self.assertRaises(ValueError):
                core.choose(result)

    def test_nonconvergence_is_rejection_without_retry_or_fallback(self):
        result = {"A": development((-.01, -.01)), "B": development((-.01, -.01)),
                  "C": {"status": "rejected_nonconvergence", "folds": []}}
        self.assertIsNone(core.choose(result))

    def test_analytic_correction_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(42)
        raw = rng.normal(size=(8, 10))
        features = rng.normal(size=(8, 11))
        labels = np.zeros((8, 10))
        labels[::2, 0] = 1
        theta = rng.normal(size=12) / 10
        value, analytic = core.correction_loss(theta, raw, features, labels)
        numeric = []
        for index in range(12):
            direction = np.zeros(12)
            direction[index] = 1e-6
            numeric.append((core.correction_loss(theta + direction, raw, features, labels)[0]
                            - core.correction_loss(theta - direction, raw, features, labels)[0]) / 2e-6)
        self.assertTrue(np.isfinite(value))
        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)

    def test_event_features_are_permutation_invariant_and_context_checked(self):
        x = matrix(8)
        raw = np.arange(80).reshape(8, 10) / 10 - 4
        expected = core.event_features(raw, x)
        permutation = np.arange(9, -1, -1)
        actual = core.event_features(raw[:, permutation], x.reshape(8, 10, 37)[:, permutation].reshape(-1, 37))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
        x[0, core.BASE_COLUMNS.index("history_length")] += 1
        with self.assertRaises(ValueError):
            core.event_features(raw, x)

    def test_no_winner_blocks_screen_before_budget_or_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core.write_new(root / core.WORK / "inputs.json", {})
            manifest = {"code_sha256": {}}
            core.write_new(root / core.REPORTS / "selection.json", {"complete": True,
                "protocol_sha256": core.canonical(core.PROTOCOL), "source_sha256": {},
                "input_manifest_sha256": core.sha256(root / core.WORK / "inputs.json"),
                "results": {v: development((-.01, -.01)) for v in "ABC"}, "winner": None})
            with patch.object(core, "load_manifest", return_value=manifest), patch.object(runner, "fit_recipe") as fit:
                with self.assertRaisesRegex(ValueError, "No qualifying winner"):
                    runner.screen(root)
                fit.assert_not_called()
            self.assertFalse((root / core.WORK / "budget.json").exists())

    def test_budget_carries_prior_usage_and_refuses_repeated_phases(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            core.write_new(work / "budget.json", {"used_seconds": 432.335, "phases": {}})
            with core.Budget(work, "selection"):
                pass
            self.assertGreater(core.read_json(work / "budget.json")["used_seconds"], 432.335)
            with self.assertRaisesRegex(ValueError, "retries prohibited"):
                with core.Budget(work, "selection"):
                    pass

    def test_failed_screen_cannot_be_bypassed_by_a_passed_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chosen = {"winner": "B"}
            manifest = {"code_sha256": {}}
            core.write_new(root / core.WORK / "inputs.json", {})
            metrics = {"baseline_auc": .8, "candidate_auc": .81, "auc_diff": .01,
                       "ci_lower": -.001, "ci_upper": .02, "n_bootstrap": 1000, "valid_bootstrap_samples": 1000}
            core.write_new(root / core.REPORTS / "screen.json", {"complete": True, "passed": True,
                "protocol_sha256": core.canonical(core.PROTOCOL), "source_sha256": {},
                "input_manifest_sha256": core.sha256(root / core.WORK / "inputs.json"),
                "selected_recipe_sha256": core.canonical(chosen),
                "results": {"primary": metrics, "secondary": metrics}})
            with patch.object(core, "load_manifest", return_value=manifest), patch.object(core, "selection", return_value=chosen), \
                    patch.object(runner, "fit_recipe") as fit:
                for action in (runner.stress, runner.finalize):
                    with self.assertRaisesRegex(ValueError, "screen gate failed"):
                        action(root)
                fit.assert_not_called()
            self.assertFalse((root / core.WORK / "budget.json").exists())

    def test_interrupted_publication_cannot_publish_ready_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(predictor, "validate_release"), patch.object(predictor, "export_verified", side_effect=RuntimeError("interrupted")):
                with self.assertRaises(RuntimeError):
                    predictor.publish_ready(root, {}, [], [], "hash", {}, [])
            self.assertFalse((root / "configs/v8.json").exists())


@unittest.skipUnless(importlib.util.find_spec("scipy"), "Correction fit uses the existing model environment")
class CorrectionTests(unittest.TestCase):
    def test_fit_uses_only_calibration_standardization_and_replays_without_reordering(self):
        x = matrix(80)
        rng = np.random.default_rng(42)
        raw = rng.normal(-2, 1, (80, 10))
        labels = np.zeros((80, 10), dtype=np.int8)
        labels[::2, 0] = 1
        saved = core.fit_correction(raw, x, labels)
        np.testing.assert_allclose(saved["mean"], core.event_features(raw, x).mean(0), rtol=0, atol=1e-14)
        before = copy.deepcopy(saved)
        p = core.apply_correction(raw, x, saved)
        self.assertEqual(saved, before)
        self.assertTrue(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all())
        np.testing.assert_array_equal(np.argsort(raw, axis=1), np.argsort(p, axis=1))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "correction.json"
            core.write_new(path, saved)
            np.testing.assert_array_equal(p, core.apply_correction(raw, x, core.read_json(path)))

    def test_failed_solver_is_not_silently_accepted(self):
        from types import SimpleNamespace
        with patch("scipy.optimize.minimize", return_value=SimpleNamespace(success=False, message="iteration limit")):
            labels = np.zeros((8, 10), dtype=np.int8)
            labels[0, 0] = 1
            with self.assertRaises(core.NonConvergence):
                core.fit_correction(np.zeros((8, 10)), matrix(8), labels)


@unittest.skipUnless(importlib.util.find_spec("catboost"), "Classifier integration uses the existing model environment")
class RecipeTests(unittest.TestCase):
    def test_synthetic_selected_release_and_provenance_fail_closed(self):
        class Budget:
            def check(self):
                pass
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            x = matrix(80)
            y = np.zeros((80, 10), dtype=np.int8)
            y[::2, 0] = 1
            params = {v: core.parameters({"boosting_type": "Plain"}, v) for v in ("A", "B", "C", "reference")}
            manifest = {"parameters": {case: params for case in ("primary", "secondary", "five_day", "final")},
                        "code_sha256": {}, "inputs_sha256": {}, "runtime": {}}
            new, fitted = runner.fit_recipe(root, manifest, "final", "A", x, y, np.arange(80), x, Budget())
            (root / "source.py").write_text("frozen source")
            manifest["code_sha256"] = {"source.py": core.sha256(root / "source.py")}
            core.write_new(root / core.WORK / "inputs.json", {"synthetic": True})
            core.write_new(root / core.REPORTS / "preparation.json", {"complete": True})
            results = {v: development((.01, .02)) for v in "ABC"}
            dummy = root / core.WORK / "development_fixture.txt"
            dummy.write_bytes(b"synthetic development artifact")
            asset = {dummy.relative_to(root).as_posix(): core.sha256(dummy)}
            for variant, result in results.items():
                for fold in result["folds"]:
                    fold.update(fit={"variant": variant, "requested_parameters": params[variant], "files_sha256": asset},
                        reference_fit={"variant": "reference", "requested_parameters": params["reference"], "files_sha256": asset},
                        predictions=dummy.relative_to(root).as_posix(), predictions_sha256=core.sha256(dummy))
            selection_path = root / core.REPORTS / "selection.json"
            core.write_new(selection_path, {"complete": True, "source_sha256": manifest["code_sha256"],
                "protocol_sha256": core.canonical(core.PROTOCOL),
                "input_manifest_sha256": core.sha256(root / core.WORK / "inputs.json"), "results": results, "winner": "A"})
            chosen = {"winner": "A", "protocol_sha256": core.canonical(core.PROTOCOL),
                      "selection_sha256": core.sha256(selection_path), "parameters": {case: params["A"] for case in manifest["parameters"]}}
            core.write_new(root / core.REPORTS / "selected_recipe.json", chosen)
            base_metrics = {"baseline_auc": .8, "candidate_auc": .81, "auc_diff": .01,
                "ci_lower": .001, "ci_upper": .02, "n_bootstrap": 1000, "valid_bootstrap_samples": 1000}
            for stage in ("screen", "stress"):
                values = {}
                for case in (("primary", "secondary") if stage == "screen" else ("five_day",)):
                    values[case] = {**base_metrics, "fit": {**fitted, "case": case},
                                   "predictions": dummy.relative_to(root).as_posix(), "predictions_sha256": core.sha256(dummy)}
                if stage == "stress":
                    values = values["five_day"]
                    values["late_48_120h"] = base_metrics
                core.write_new(root / core.REPORTS / (stage + ".json"), {"complete": True, "passed": True,
                    "protocol_sha256": core.canonical(core.PROTOCOL), "source_sha256": manifest["code_sha256"],
                    "input_manifest_sha256": core.sha256(root / core.WORK / "inputs.json"),
                    "selected_recipe_sha256": core.canonical(chosen), "results": values})
            ids = np.array([f"opaque-{i}" for i in range(len(x))])
            baseline = np.full(y.shape, .2)
            old = np.full(y.shape, .3)
            p = core.replacement(baseline, old, new)
            def export(root, path, rowids, probabilities, expected_sha=None, protected=()):
                content = np.asarray(probabilities, dtype=float).tobytes()
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_bytes(content)
                self.assertEqual(path.read_bytes(), content)
                fingerprint = core.sha256(path)
                self.assertTrue(expected_sha is None or expected_sha == fingerprint)
                return fingerprint
            with patch.object(core, "load_manifest", return_value=manifest), patch.object(core, "CODE_FILES", ("source.py",)), \
                    patch.object(predictor, "replay_v6", return_value=(baseline, old, x, list(core.BASE_COLUMNS), ids, {})), \
                    patch.object(predictor, "export_verified", side_effect=export):
                predictor.release_candidate(root, manifest, fitted, ids, p, Budget())
                config = core.read_json(root / "configs/v8.json")
                self.assertTrue(config["submission_ready"])
                self.assertEqual(predictor.generate(root, root / "artifacts/replayed.csv", "cpu")["sha256"], config["submission_sha256"])
                for field in ("code_sha256", "files_sha256"):
                    bad = copy.deepcopy(config)
                    bad[field].pop(next(iter(bad[field])))
                    with self.assertRaises(ValueError):
                        predictor.validate_release(root, bad)
                chosen["winner"] = "B"
                (root / core.REPORTS / "selected_recipe.json").write_text(__import__("json").dumps(chosen))
                with self.assertRaisesRegex(ValueError, "Selected recipe changed"):
                    predictor.validate_release(root, config)

    def test_all_three_recipes_fit_save_and_reload_with_fixed_parameters(self):
        class Budget:
            def check(self):
                pass
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            x = matrix(80)
            y = np.zeros((80, 10), dtype=np.int8)
            y[::2, 0] = 1
            manifest = {"parameters": {"primary": {v: core.parameters({"boosting_type": "Plain"}, v) for v in "ABC"}}}
            for variant in "ABC":
                predictions, fitted = runner.fit_recipe(root, manifest, "test_" + variant, variant, x, y,
                    np.arange(80), x, Budget(), "primary")
                np.testing.assert_array_equal(predictions, runner.load_predictor(root, fitted, x))
                self.assertEqual(predictions.shape, y.shape)
                self.assertEqual(fitted["resolved_parameters"]["depth"], 6 if variant == "C" else 4)
                self.assertEqual(fitted["calibration_events"], 20 if variant == "C" else 0)
