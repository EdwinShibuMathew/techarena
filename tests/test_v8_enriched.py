"""Successor feature, provenance, gate and successful-publication regressions."""
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
sys.path.insert(0, str(ROOT / "legacy"))
sys.path.insert(0, str(ROOT / "experiments/v8-enriched-logloss"))
import enriched_core as core
import predict_v8_enriched as predictor
spec = importlib.util.spec_from_file_location("enriched_runner", ROOT / "experiments/v8-enriched-logloss/runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def matrix(events=8):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(events * 10, 37)).astype(np.float32)
    x[:, 0] = rng.random(events * 10)
    x[:, 15] = 1
    x[:, 17] = 3
    return x


def metrics():
    return {"baseline_auc": .8, "candidate_auc": .81, "auc_diff": .01,
            "ci_lower": .001, "ci_upper": .02, "n_bootstrap": 1000, "valid_bootstrap_samples": 1000}


class EnrichedTests(unittest.TestCase):
    def test_relative_features_preserve_columns_and_candidate_permutations(self):
        x = matrix(2)
        x[:3, 0] = .75
        result, names = core.enrich(x, core.BASE_COLUMNS)
        np.testing.assert_array_equal(result[:, :37], x)
        self.assertEqual(names, list(core.COLUMNS))
        self.assertEqual(len(np.unique(result[:3, 40])), 1)
        order = np.array([9, 2, 0, 1, 4, 3, 6, 5, 8, 7])
        permuted = x.reshape(2, 10, 37)[:, order].reshape(-1, 37)
        got, _ = core.enrich(permuted, core.BASE_COLUMNS)
        expected = result.reshape(2, 10, 43)[:, order].reshape(-1, 43)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)

    def test_all_equal_and_zero_probability_events_have_tied_average_rank(self):
        for p in (0, .5, 1):
            x = matrix(1)
            x[:, 0] = p
            result, _ = core.enrich(x, core.BASE_COLUMNS)
            np.testing.assert_array_equal(result[:, 40], np.full(10, 4.5))
            np.testing.assert_array_equal(result[:, 39], np.zeros(10))
            self.assertTrue(np.isfinite(result).all())

    def test_feature_schema_shapes_history_and_nonfinite_values_fail(self):
        x = matrix(1)
        bad = x.copy()
        bad[0, 1] = np.nan
        history = x.copy()
        history[0, 15] = 4
        for data, names in ((x, list(reversed(core.BASE_COLUMNS))), (x[:-1], core.BASE_COLUMNS),
                            (x.astype(float), core.BASE_COLUMNS), (bad, core.BASE_COLUMNS), (history, core.BASE_COLUMNS)):
            with self.assertRaises(ValueError):
                core.enrich(data, names)

    def test_gate_requires_finite_complete_consistent_statistics(self):
        self.assertTrue(all(core.screen_gates({"primary": metrics(), "secondary": metrics()}).values()))
        for key, value in (("ci_lower", np.nan), ("valid_bootstrap_samples", 989),
                           ("n_bootstrap", 500), ("auc_diff", .5), ("ci_upper", -.5)):
            bad = metrics()
            bad[key] = value
            with self.assertRaises(ValueError):
                core.screen_gates({"primary": bad, "secondary": metrics()})
        bad = metrics()
        bad["ci_lower"] = 0
        self.assertFalse(all(core.screen_gates({"primary": bad, "secondary": metrics()}).values()))

    def test_failed_gate_blocks_stress_and_final_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {"code_sha256": {}}
            core.write_new(root / core.WORK / "inputs.json", {})
            bad = metrics()
            bad["ci_lower"] = -.001
            core.write_new(root / core.REPORTS / "screen.json", {"complete": True, "passed": True,
                "protocol_sha256": core.canonical(core.PROTOCOL), "source_sha256": {},
                "input_manifest_sha256": core.sha256(root / core.WORK / "inputs.json"),
                "results": {"primary": bad, "secondary": metrics()}})
            with patch.object(core, "load_manifest", return_value=manifest), patch.object(runner, "fit_classifier") as fit:
                for action in (runner.stress, runner.finalize):
                    with self.assertRaisesRegex(ValueError, "gate failed"):
                        action(root)
                fit.assert_not_called()
            self.assertFalse((root / core.WORK / "budget.json").exists())

    def test_reused_cache_rejects_wrong_cutoff_before_fitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.pt").write_bytes(b"checkpoint")
            core.write_new(root / "cache.json", {"identity": {}, "sha256": "wrong"})
            manifest = {"baseline_manifest": {"inputs_sha256": {}, "code_sha256": {}},
                        "reused_caches": {"cached": "cache.npz"}, "models": {"jan24": "model.pt"}}
            # Cache identity validation needs no tensor operations; provide the
            # optional dependency in the CPU-only unit-test environment.
            with patch.dict(sys.modules, {"torch": object()}), self.assertRaisesRegex(ValueError, "Stale cache"):
                runner.features(root, manifest, {}, "jan24", np.array([0]), "cached", None)

    def test_missing_or_unready_release_never_starts_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core.write_new(root / "configs/v8.json", {"submission_ready": False})
            with patch.object(predictor, "compute_candidate") as replay, self.assertRaisesRegex(ValueError, "not submission-ready"):
                predictor.generate(root, root / "artifacts/out.csv")
            replay.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("catboost"), "CatBoost integration uses the existing model environment")
class EnrichedReleaseTests(unittest.TestCase):
    def build_release(self, root):
        class Budget:
            def check(self):
                pass
        x = matrix()
        y = np.zeros((8, 10), dtype=np.int8)
        y[::2, 0] = 1
        parameters = core.classifier_parameters({"boosting_type": "Plain", "bootstrap_type": "MVS", "subsample": .8})
        manifest = {"classifier_parameters": {k: parameters for k in ("primary", "secondary", "five_day", "final")},
                    "code_sha256": {}, "inputs_sha256": {}, "runtime": {}}
        predictions, fit = runner.fit_classifier(root, manifest, "final", x, y, x, Budget())
        self.assertEqual(fit["resolved_parameters"]["loss_function"], "Logloss")
        self.assertEqual(predictions.shape, y.shape)
        (root / "code.py").write_text("frozen code")
        manifest["code_sha256"] = {"code.py": core.sha256(root / "code.py")}
        core.write_new(root / core.WORK / "inputs.json", {"fixture": True})
        core.write_new(root / core.REPORTS / "preparation.json", {"complete": True})
        for stage in ("screen", "stress"):
            results = {}
            for case in (("primary", "secondary") if stage == "screen" else ("five_day",)):
                recorded_fit = {**fit, "case": case}
                path = core.WORK / (case + ".npz")
                runner.save_npz(root / path, candidate=predictions)
                result = {**metrics(), "fit": recorded_fit, "predictions": path.as_posix(),
                          "predictions_sha256": core.sha256(root / path)}
                results[case] = result
            if stage == "stress":
                results = results["five_day"]
                results["late_48_120h"] = metrics()
            core.write_new(root / core.REPORTS / (stage + ".json"), {"complete": True, "passed": True,
                "protocol_sha256": core.canonical(core.PROTOCOL), "source_sha256": manifest["code_sha256"],
                "input_manifest_sha256": core.sha256(root / core.WORK / "inputs.json"), "results": results})
        return manifest, fit, x, predictions, Budget()

    def test_successful_release_reload_and_inventory_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, fit, x, new, budget = self.build_release(root)
            rowids = np.array([f"opaque-{i}" for i in range(len(x))])
            baseline = np.full(new.shape, .2)
            old = np.full(new.shape, .3)
            candidate = core.replacement(baseline, old, new)
            # Tiny publication adapter exercises orchestration; the separate full-size
            # CSV test below exercises production serialization and validation.
            def export(root, path, ids, p, expected_sha=None, protected=()):
                core.require(np.array_equal(ids, rowids), "Fixture row IDs differ")
                content = np.asarray(p, dtype=np.float64).tobytes()
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
                core.require(path.read_bytes() == content, "Different fixture output")
                fingerprint = core.sha256(path)
                core.require(expected_sha is None or expected_sha == fingerprint, "Fixture fingerprint differs")
                return fingerprint
            with patch.object(core, "CODE_FILES", ("code.py",)), patch.object(core, "load_manifest", return_value=manifest), \
                    patch.object(predictor, "replay_v6", return_value=(baseline, old, x, list(core.BASE_COLUMNS), rowids, {})), \
                    patch.object(predictor, "export_verified", side_effect=export):
                predictor.release_candidate(root, manifest, fit, rowids, candidate, budget)
                config = core.read_json(root / "configs/v8.json")
                self.assertTrue(config["submission_ready"])
                replay = predictor.generate(root, root / "artifacts/replayed.csv", "cpu")
                self.assertEqual(replay["sha256"], config["submission_sha256"])
                for field in ("code_sha256", "files_sha256"):
                    bad = copy.deepcopy(config)
                    bad[field].pop(next(iter(bad[field])))
                    with self.assertRaises(ValueError):
                        predictor.validate_release(root, bad)
                bad_report = core.read_json(root / core.REPORTS / "screen.json")
                bad_report["source_sha256"] = {}
                (root / core.REPORTS / "screen.json").write_text(__import__("json").dumps(bad_report))
                with self.assertRaises(ValueError):
                    predictor.validate_release(root, config)

    def test_interrupted_publication_has_no_ready_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(predictor, "validate_release"), \
                    patch.object(predictor, "export_verified", side_effect=RuntimeError("interrupted")):
                with self.assertRaises(RuntimeError):
                    predictor.publish_ready(root, {}, [], [], "fingerprint", {}, [])
            self.assertFalse((root / "configs/v8.json").exists())


class FullSizeCSVTests(unittest.TestCase):
    def test_real_export_validates_rows_fingerprint_aliases_and_preserves_existing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            example = root / "data/raw/example_submission.csv"
            example.parent.mkdir(parents=True)
            n = 529390
            ids = np.array([f"id-{i}" for i in range(n)])
            with example.open("w") as stream:
                stream.write("rowid,predicted_probability\n")
                stream.writelines(f"{i},0\n" for i in ids)
            probabilities = np.full(n, .125)
            output = root / "artifacts/out.csv"
            fingerprint = predictor.export_verified(root, output, ids, probabilities, protected=[example])
            self.assertEqual(predictor.export_verified(root, output, ids, probabilities, fingerprint), fingerprint)
            with self.assertRaises(ValueError):
                predictor.export_verified(root, output, ids, probabilities, "0" * 64)
            self.assertEqual(core.sha256(output), fingerprint)
            alias = root / "artifacts/source-alias.csv"
            alias.symlink_to(example)
            with self.assertRaises(ValueError):
                predictor.export_verified(root, alias, ids, probabilities, protected=[example])


if __name__ == "__main__":
    unittest.main()
