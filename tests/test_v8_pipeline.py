"""V8 temporal, provenance, statistical-gate and publication regressions."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/v8-pairlogit"))
sys.path.insert(0, str(ROOT / "tools"))
import v8_core as core
import predict_v8 as predictor
spec = importlib.util.spec_from_file_location("v8_runner", ROOT / "experiments/v8-pairlogit/runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class V8ProtocolTests(unittest.TestCase):
    def test_manifest_line_ending_adapter_does_not_accept_content_changes(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            content = b'{\n  "file": "checksum"\n}\n'
            path.write_bytes(content)
            bound = hashlib.sha256(content.replace(b"\n", b"\r\n")).hexdigest()
            self.assertTrue(core.manifest_line_endings_match(path, bound))
            path.write_bytes(content.replace(b"checksum", b"changed"))
            self.assertFalse(core.manifest_line_endings_match(path, bound))

    def test_first_user_selection_and_temporal_separation(self):
        day = core.FOLDS["jan24"]
        raw = {"u": np.array([1, 1, 2, 3, 1, 2, 4]),
               "t": day + np.array([10, 1, 2, 2, core.DAY + 1, core.DAY + 2, core.DAY + 3])}
        fit = core.first_indices(raw["u"], raw["t"], day, day + core.DAY)
        np.testing.assert_array_equal(fit, [1, 2, 3])
        core.temporal_check(raw, fit, [4, 5, 6], day, day + core.DAY)
        for bad_fit, bad_target in (([0, 2, 3], [4, 5, 6]), ([4, 5, 6], [1, 2, 3]), (fit, [1, 5, 6])):
            with self.assertRaises(ValueError):
                core.temporal_check(raw, bad_fit, bad_target, day, day + core.DAY)

    def test_cache_identity_binds_order_cutoff_model_data_and_columns(self):
        original = core.cache_identity("model", 10, [1, 2], "inputs", ["a", "b"])
        alternatives = [core.cache_identity("changed", 10, [1, 2], "inputs", ["a", "b"]),
                        core.cache_identity("model", 11, [1, 2], "inputs", ["a", "b"]),
                        core.cache_identity("model", 10, [2, 1], "inputs", ["a", "b"]),
                        core.cache_identity("model", 10, [1, 2], "changed", ["a", "b"]),
                        core.cache_identity("model", 10, [1, 2], "inputs", ["b", "a"])]
        for alternative in alternatives:
            self.assertNotEqual(core.canonical(original), core.canonical(alternative))

    def test_replacement_is_the_five_member_convex_ensemble(self):
        rng = np.random.default_rng(42)
        members = rng.random((5, 20, 10))
        baseline = (2 / 3) * members[:3].mean(0) + (1 / 3) * (.5 * members[3] + .5 * members[4])
        np.testing.assert_array_equal(core.replacement(baseline, members[0], members[0]), baseline)
        new = rng.random((20, 10))
        expected = (2 / 3) * ((new + members[1] + members[2]) / 3) + (1 / 3) * (.5 * members[3] + .5 * members[4])
        np.testing.assert_allclose(core.replacement(baseline, members[0], new), expected, atol=1e-15, rtol=0)

    def test_pairs_include_no_click_cross_event_negatives_and_link_uses_only_fit_ctr(self):
        labels = np.zeros((10, 10), dtype=np.int8)
        labels[:3, 0] = 1
        pairs = core.sample_pairs(labels)
        self.assertEqual(pairs.shape, (48, 2))
        self.assertTrue(np.all(labels.ravel()[pairs[:, 0]] == 1))
        self.assertTrue(np.all(labels.ravel()[pairs[:, 1]] == 0))
        self.assertTrue(np.any(labels.sum(1)[pairs[:, 1] // 10] == 0))
        for positive in np.flatnonzero(labels.ravel()):
            self.assertEqual(len(np.unique(pairs[pairs[:, 0] == positive, 1])), 16)
        raw = np.linspace(-5, 5, labels.size)
        bias, link = core.intercept(raw, labels)
        self.assertAlmostEqual(core.sigmoid(raw + bias).mean(), labels.mean(), places=12)
        self.assertFalse(link["target_labels_used"])

    def test_no_gate_can_be_bypassed_by_a_passed_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = {"complete": True, "passed": True, "protocol_sha256": core.canonical(core.PROTOCOL),
                      "results": {"primary": {"ci_lower": -.0001}, "secondary": {"auc_diff": .01}}}
            core.write_new(root / runner.REPORTS / "screen.json", report)
            with self.assertRaisesRegex(ValueError, "gate failed"):
                runner.stress(root)
            with self.assertRaisesRegex(ValueError, "gate failed"):
                runner.finalize(root)
            self.assertFalse((root / runner.WORK).exists())
        self.assertFalse(all(core.stress_gates({"ci_lower": .001, "late_48_120h": {"auc_diff": -.001}}).values()))

    def test_frozen_file_and_manifest_mutations_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "code.py").write_text("original")
            (root / "input.txt").write_text("input")
            (root / "cache.txt").write_text("cache")
            manifest = {"protocol": core.PROTOCOL, "protocol_sha256": core.canonical(core.PROTOCOL),
                        "code_sha256": {"code.py": core.sha256(root / "code.py")},
                        "inputs_sha256": {"input.txt": core.sha256(root / "input.txt")},
                        "prepared_sha256": {"cache.txt": core.sha256(root / "cache.txt")}}
            core.write_new(root / runner.WORK / "inputs.json", manifest)
            core.write_new(root / runner.REPORTS / "preparation.json", {
                "complete": True, "manifest_sha256": core.sha256(root / runner.WORK / "inputs.json")})
            with patch.object(core, "CODE_FILES", ("code.py",)):
                core.load_manifest(root)
                (root / "cache.txt").write_text("changed")
                with self.assertRaisesRegex(ValueError, "Frozen input changed"):
                    core.load_manifest(root)
            with self.assertRaises(ValueError):
                core.inside(root, "../outside")

    def test_budget_records_failures_and_refuses_retries_or_exhaustion(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            with self.assertRaises(RuntimeError):
                with core.Budget(work, "prepare"):
                    raise RuntimeError("preserve failure")
            state = core.read_json(work / "budget.json")
            self.assertEqual(state["phases"]["prepare"]["status"], "failed")
            self.assertGreater(state["used_seconds"], 0)
            with self.assertRaises(ValueError):
                with core.Budget(work, "prepare"):
                    pass
            state["used_seconds"] = 14400
            (work / "budget.json").write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "exhausted"):
                with core.Budget(work, "screen"):
                    pass


class V8PublicationTests(unittest.TestCase):
    def test_output_aliases_and_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "artifacts/model.cbm"
            source.parent.mkdir()
            source.write_bytes(b"model")
            hard, symbolic = source.parent / "alias.csv", source.parent / "symbolic.csv"
            os.link(source, hard)
            symbolic.symlink_to(source)
            for output in (source, hard, symbolic, root / "outside.csv"):
                with self.assertRaises(ValueError):
                    predictor.output_path(root, output, [source])
            self.assertEqual(source.read_bytes(), b"model")

    def test_unready_candidate_cannot_run_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core.write_new(root / "configs/v8.json", {"submission_ready": False})
            with patch.object(predictor, "compute_candidate") as compute, self.assertRaisesRegex(ValueError, "not submission-ready"):
                predictor.generate(root, root / "artifacts/out.csv")
            compute.assert_not_called()

    def test_failed_validation_does_not_publish_or_replace_output(self):
        sys.path.insert(0, str(ROOT / "legacy"))
        import submission_export
        import row_auc_final_utils
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifacts/out.csv"
            def exporter(ids, probabilities, example, path, rows):
                path.write_bytes(b"candidate")
            with patch.object(submission_export, "export_submission", side_effect=exporter), \
                    patch.object(submission_export, "validate_csv", return_value={}), \
                    patch.object(row_auc_final_utils, "verify_csv_probabilities", return_value=None):
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    predictor.export_verified(root, output, np.array(["id"]), np.array([.5]), "0" * 64)
                self.assertFalse(output.exists())
                self.assertEqual(list(output.parent.glob(".v8-*")), [])
                output.write_bytes(b"existing")
                with self.assertRaises(ValueError):
                    predictor.export_verified(root, output, np.array(["id"]), np.array([.5]))
                self.assertEqual(output.read_bytes(), b"existing")


@unittest.skipUnless(importlib.util.find_spec("catboost"), "Run CatBoost integration in the existing model environment")
class V8CatBoostTests(unittest.TestCase):
    def test_fixed_ranker_fit_save_replay_and_training_link(self):
        class Budget:
            def check(self):
                pass
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(42)
            x = rng.normal(size=(80, 37)).astype(np.float32)
            y = np.zeros((8, 10), dtype=np.int8)
            y[::2, 0] = 1
            manifest = {"columns": [f"feature_{i}" for i in range(37)],
                        "ranker_parameters": {"primary": core.ranker_parameters({"boosting_type": "Plain", "bootstrap_type": "MVS", "subsample": .8})}}
            p, fit = runner.fit_ranker(root, manifest, "primary", x, y, x, Budget())
            self.assertEqual(p.shape, y.shape)
            self.assertTrue(np.isfinite(p).all())
            self.assertAlmostEqual(p.mean(), y.mean(), places=12)
            self.assertEqual(fit["resolved_parameters"]["iterations"], 545)
            self.assertEqual(fit["model_sha256"], core.sha256(root / fit["model"]))
            self.assertGreater(fit["pairs_with_no_click_negative"], 0)
