"""Run PairLogit numeric tests in CI and exercise staged-input preparation."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments/v6-pairlogit-v1-api-fix"
sys.path.insert(0, str(EXPERIMENT))
import pairlogit_trial as trial
spec = importlib.util.spec_from_file_location("pairlogit_numeric_tests", EXPERIMENT / "test_pairlogit.py")
numeric = importlib.util.module_from_spec(spec)
spec.loader.exec_module(numeric)
NumericTests, CatBoostTests = numeric.NumericTests, numeric.CatBoostTests


class PairLogitPreparationRegressions(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self.source = self.repo / "artifacts/experiments/v6-pairlogit-v1-inputs/source"
        self.source.mkdir(parents=True)

    def test_separate_inputs_and_preservation_of_existing_outputs(self):
        source, work = trial.input_layout(self.repo, self.source)
        self.assertEqual(source, self.source)
        self.assertFalse(work.exists())
        work.mkdir()
        marker = work / "existing.txt"
        marker.write_text("preserve")
        with self.assertRaisesRegex(ValueError, "Verification already exists"):
            trial.input_layout(self.repo, self.source)
        self.assertEqual(marker.read_text(), "preserve")
        with self.assertRaises(ValueError):
            trial.input_layout(self.repo, self.repo)

    def test_split_lookup_is_bounded_and_hash_verified(self):
        path = self.source / "data/cache/split_jan24.npz"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"synthetic split")
        self.assertEqual(trial.locate_split(self.source, trial.audit.sha256(path)), path)
        with self.assertRaises(FileNotFoundError):
            trial.locate_split(self.source, "0" * 64)
        path.unlink()
        outside = self.repo / "outside.npz"
        outside.write_bytes(b"synthetic split")
        path.symlink_to(outside)
        with self.assertRaises(FileNotFoundError):
            trial.locate_split(self.source, trial.audit.sha256(outside))

    def test_wrapper_uses_tracked_sources_and_preserves_logs(self):
        experiment = self.repo / "experiments/v6-pairlogit-v1-api-fix"
        experiment.mkdir(parents=True)
        script = experiment / "prepare_pairlogit_api_fix.sh"
        shutil.copy2(EXPERIMENT / script.name, script)
        fakebin = self.repo / "fakebin"
        fakebin.mkdir()
        log = self.repo / "calls.txt"
        for command in ("python", "git", "sha256sum"):
            executable = fakebin / command
            executable.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$PAIRLOGIT_TEST_LOG"\n')
            executable.chmod(0o755)
        env = {**os.environ, "PATH": str(fakebin) + os.pathsep + os.environ["PATH"],
               "CONDA_DEFAULT_ENV": "techarena", "PAIRLOGIT_TEST_LOG": str(log)}
        result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("pairlogit_trial.py verify --repo " + str(self.repo), log.read_text())
        self.assertIn("--source-dir " + str(self.source), log.read_text())
        self.assertIn("pairlogit_trial.py prepare --repo " + str(self.repo), log.read_text())
        prep_log = self.repo / "artifacts/experiments/v6-pairlogit-v1-prepare-api-fix.log"
        prep_log.write_text("existing log")
        result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(prep_log.read_text(), "existing log")
