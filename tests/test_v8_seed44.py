"""Seed44 objective extension: cutoffs, process cleanup, neural replay and release guards."""
import copy
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "experiments/v8-seed44-row-auc"))
import seed44_core as core
import train_seed44
import predict_v8_seed44 as predictor
spec = importlib.util.spec_from_file_location("seed44_test_runner", ROOT / "experiments/v8-seed44-row-auc/runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def metrics():
    return {"baseline_auc": .8, "candidate_auc": .81, "auc_diff": .01, "ci_lower": .001,
            "ci_upper": .02, "n_bootstrap": 1000, "valid_bootstrap_samples": 1000}


class GuardTests(unittest.TestCase):
    def test_protocol_changes_only_seed44_and_preserves_final_combiner_policy(self):
        self.assertEqual(core.PROTOCOL["member"], "dim8_seed44")
        self.assertEqual(core.PROTOCOL["pair_sampling_seed"], 3044)
        self.assertIn("reuse validated Jan25", core.PROTOCOL["cohorts"]["final"][0])
        rng = np.random.default_rng(44)
        members = rng.random((5, 8, 10))
        weights = np.array([2/9, 2/9, 2/9, 1/6, 1/6])
        baseline = np.sum(members * weights[:, None, None], axis=0)
        new = rng.random((8, 10))
        expected = sum(weights[i] * (new if i == 2 else members[i]) for i in range(5))
        np.testing.assert_allclose(core.replacement(baseline, members[2], new), expected, rtol=0, atol=1e-15)

    def test_failed_gate_blocks_training_even_with_a_passed_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core.write_new(root / core.WORK / "inputs.json", {})
            bad = metrics()
            bad["ci_lower"] = -.001
            core.write_new(root / core.REPORTS / "screen.json", {"complete": True, "passed": True,
                "protocol_sha256": core.canonical(core.PROTOCOL), "source_sha256": {},
                "input_manifest_sha256": core.sha256(root / core.WORK / "inputs.json"),
                "results": {"primary": bad, "secondary": metrics()}})
            with patch.object(core, "load_manifest", return_value={"code_sha256": {}}), patch.object(runner, "train_fold") as train:
                for action in (runner.stress, runner.finalize):
                    with self.assertRaisesRegex(ValueError, "screen gate failed"):
                        action(root)
                train.assert_not_called()
            self.assertFalse((root / core.WORK / "budget.json").exists())

    def test_timeout_kills_training_process_group_and_records_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            core.write_new(work / "budget.json", {"used_seconds": 543.611, "phases": {}})
            children = []
            original = subprocess.Popen
            def launch(*args, **kwargs):
                child = original(*args, **kwargs)
                children.append(child)
                return child
            with patch.dict(core.LIMITS, {"screen": .2}), patch.object(core.Budget, "expire"), \
                    patch.object(core.subprocess, "Popen", side_effect=launch), patch.object(core.os, "killpg", wraps=os.killpg) as kill:
                with self.assertRaises(subprocess.TimeoutExpired):
                    with core.Budget(work, "screen") as budget:
                        budget.run([sys.executable, "-c", "import time; time.sleep(10)"], work)
                kill.assert_any_call(children[0].pid, signal.SIGKILL)
            self.assertEqual(children[0].poll(), -signal.SIGKILL)
            saved = core.read_json(work / "budget.json")
            self.assertEqual(saved["phases"]["screen"]["status"], "failed")
            self.assertGreater(saved["used_seconds"], 543.611)
            with self.assertRaisesRegex(ValueError, "retries prohibited"):
                with core.Budget(work, "screen"):
                    pass

    def test_interrupted_publication_never_marks_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(predictor, "validate_release"), patch.object(predictor, "export_verified", side_effect=RuntimeError("interrupted")):
                with self.assertRaises(RuntimeError):
                    predictor.publish_ready(root, {}, [], [], "sha", {}, [])
            self.assertFalse((root / "configs/v8.json").exists())


@unittest.skipUnless(importlib.util.find_spec("torch"), "Neural tests use the existing model environment")
class NeuralTests(unittest.TestCase):
    def preference(self, root):
        import torch
        from train_final_models import PreferenceModel
        torch.manual_seed(44)
        model = PreferenceModel(12, 20, 8, torch.ones(12, dtype=torch.bool))
        path = root / "preference.pt"
        torch.save({"model_state": model.state_dict(), "n_users": 12, "n_items": 20, "dimension": 8,
                    "epoch": 260, "fine_tuning_sampling_seed": 1044}, path)
        return path

    def arrays(self):
        labels = np.zeros((24, 10), dtype=np.int8)
        labels[::2, 0] = 1
        return {"u": np.arange(24) % 12, "i": np.tile(np.arange(10), (24, 1)),
                "t": np.arange(24) + core.FOLDS["jan21"] - 100, "tb": np.arange(24) % 4,
                "dt": np.arange(24) % 2, "y": labels, "indices_sha256": "synthetic-indices"}

    def test_mixed_loss_oracle_gradients_extremes_and_shuffle_rng(self):
        from row_auc_objective import self_test
        self_test()

    def test_training_rejects_future_rows_before_creating_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preference = self.preference(root)
            arrays = self.arrays()
            arrays["t"][0] = core.FOLDS["jan21"]
            with self.assertRaisesRegex(ValueError, "timeline"):
                train_seed44.train(preference, arrays, core.FOLDS["jan21"], root / "bad.pt", "cpu")
            self.assertFalse((root / "bad.pt").exists())

    def test_eight_epochs_save_reload_and_strict_provenance(self):
        import torch
        from train_final_models import PreferenceModel, EventModel
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preference = self.preference(root)
            output = root / core.neural_path("jan21")
            state = train_seed44.train(preference, self.arrays(), core.FOLDS["jan21"], output, "cpu")
            self.assertEqual(state["epoch"], 8)
            self.assertEqual(len(core.read_json(output.with_suffix(".json"))["history"]), 8)
            manifest = {"preferences": {"jan21": "preference.pt"}, "inputs_sha256": {"preference.pt": core.sha256(preference)}}
            restored = core.validate_neural(root, manifest, "jan21")
            model = EventModel(PreferenceModel(12, 20, 8, torch.ones(12, dtype=torch.bool)), 12, 20)
            model.load_state_dict(restored["model_state"], strict=True)
            for key in state["model_state"]:
                torch.testing.assert_close(state["model_state"][key], restored["model_state"][key], rtol=0, atol=0)
            self.assertLess(state["maximum_fitting_timestamp"], core.FOLDS["jan21"])
            changed = dict(restored, seed=42)
            torch.save(changed, output)
            with self.assertRaisesRegex(ValueError, "recipe differs"):
                core.validate_neural(root, manifest, "jan21")
            with self.assertRaisesRegex(ValueError, "Preserve"):
                train_seed44.train(preference, self.arrays(), core.FOLDS["jan21"], output, "cpu")


@unittest.skipUnless(importlib.util.find_spec("catboost") and importlib.util.find_spec("torch"), "Release integration uses existing model dependencies")
class ReleaseTests(unittest.TestCase):
    preference = NeuralTests.preference
    arrays = NeuralTests.arrays

    def test_synthetic_neural_release_reuses_secondary_combiner_and_rejects_tampering(self):
        import torch
        from catboost import CatBoostClassifier, Pool
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preference = self.preference(root)
            state = train_seed44.train(preference, self.arrays(), core.FOLDS["jan21"], root / "template.pt", "cpu")
            # Exercise the unmodified production feature builder on a saved
            # synthetic neural model, in addition to release orchestration.
            from build_submission import neural_pipeline, FIT_FIELDS
            positive = root / "positive.npz"
            np.savez(positive, users=np.arange(12), items=np.arange(12))
            target = {"u": np.arange(4), "i": np.tile(np.arange(10), (4, 1)),
                "t": np.full(4, core.FOLDS["jan21"] + 1), "tb": np.arange(4), "dt": np.zeros(4, dtype=int),
                "h": np.tile(np.r_[np.arange(3), np.full(17, -1)], (4, 1)),
                "ht": np.tile(np.r_[np.full(3, core.FOLDS["jan21"] - 200), np.zeros(17, dtype=int)], (4, 1))}
            built, names, *_ = neural_pipeline(root / "template.pt", {k: self.arrays()[k] for k in FIT_FIELDS},
                                               target, positive, torch.device("cpu"))
            self.assertEqual(built.shape, (40, 37))
            self.assertEqual(tuple(names), core.BASE_COLUMNS)
            self.assertTrue(np.isfinite(built).all())
            manifest = {"preferences": {fold: "preference.pt" for fold in core.FOLDS},
                "inputs_sha256": {"preference.pt": core.sha256(preference)}, "code_sha256": {}, "runtime": {}}
            for fold in core.FOLDS:
                path = root / core.neural_path(fold)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({**state, "cutoff": core.FOLDS[fold]}, path)
                core.write_new(path.with_suffix(".json"), {"synthetic": True})
            (root / "source.py").write_text("frozen source")
            manifest["code_sha256"] = {"source.py": core.sha256(root / "source.py")}
            core.write_new(root / core.WORK / "inputs.json", {"synthetic": True})
            core.write_new(root / core.REPORTS / "preparation.json", {"complete": True})
            rng = np.random.default_rng(44)
            x = rng.normal(size=(240, 37)).astype(np.float32)
            params = core.previous.prior.classifier_parameters({"boosting_type": "Plain"})
            manifest["parameters"] = {case: params for case in core.CASE_FOLDS}
            stage_results = {}
            for case, folds in core.CASE_FOLDS.items():
                model = CatBoostClassifier(**params)
                model.fit(Pool(x, label=self.arrays()["y"].ravel(), feature_names=list(core.BASE_COLUMNS)), verbose=False)
                path = root / core.WORK / "combiners" / (case + ".cbm")
                path.parent.mkdir(parents=True, exist_ok=True)
                model.save_model(str(path))
                predictions = root / core.WORK / (case + ".npz")
                runner.save_npz(predictions, synthetic=model.predict_proba(x))
                stage_results[case] = {**metrics(), "fit": {"case": case, "model": path.relative_to(root).as_posix(),
                    "model_sha256": core.sha256(path), "parameters": params,
                    "neural_sha256": {core.neural_path(f).as_posix(): core.sha256(root / core.neural_path(f)) for f in folds}},
                    "predictions": predictions.relative_to(root).as_posix(), "predictions_sha256": core.sha256(predictions)}
            for stage in ("screen", "stress"):
                results = ({case: stage_results[case] for case in ("primary", "secondary")} if stage == "screen" else
                           {**stage_results["five_day"], "late_48_120h": metrics()})
                core.write_new(root / core.REPORTS / (stage + ".json"), {"complete": True, "passed": True,
                    "protocol_sha256": core.canonical(core.PROTOCOL), "source_sha256": manifest["code_sha256"],
                    "input_manifest_sha256": core.sha256(root / core.WORK / "inputs.json"), "results": results})
            ids, p = np.array([f"opaque-{i}" for i in range(240)]), np.full((24, 10), .1)
            def export(root, path, rowids, probabilities, expected_sha=None, protected=()):
                content = np.asarray(probabilities, dtype=float).tobytes()
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_bytes(content)
                self.assertEqual(path.read_bytes(), content)
                value = core.sha256(path)
                self.assertTrue(expected_sha is None or value == expected_sha)
                return value
            class Budget:
                def check(self):
                    pass
            with patch.object(core, "load_manifest", return_value=manifest), patch.object(core, "CODE_FILES", ("source.py",)), \
                    patch.object(predictor, "compute_candidate", return_value=(ids, p, {})), patch.object(predictor, "export_verified", side_effect=export):
                predictor.release_candidate(root, manifest, Budget())
                config = core.read_json(root / "configs/v8.json")
                predictor.validate_release(root, config)
                self.assertTrue(config["submission_ready"])
                self.assertEqual(config["combiner"], stage_results["secondary"]["fit"]["model"])
                self.assertEqual(predictor.generate(root, root / "artifacts/replay.csv", "cpu")["sha256"], config["submission_sha256"])
                for field in ("code_sha256", "files_sha256"):
                    bad = copy.deepcopy(config)
                    bad[field].pop(next(iter(bad[field])))
                    with self.assertRaises(ValueError):
                        predictor.validate_release(root, bad)
                bad = {**config, "member": "dim8_seed42"}
                with self.assertRaises(ValueError):
                    predictor.validate_release(root, bad)
