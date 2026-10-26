"""Supplementary checks stay descriptive and within the shared compute budget."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/v8-pairlogit"))
spec = importlib.util.spec_from_file_location("v8_diagnostics", ROOT / "experiments/v8-pairlogit/diagnostics.py")
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class V8DiagnosticTests(unittest.TestCase):
    def test_verification_allowance_is_shared_not_reset(self):
        state = {"used_seconds": 1000, "phases": {"verify": {"seconds": 700}}, "supplementary_verification_seconds": 100}
        self.assertEqual(diagnostics.remaining_verification_seconds(state), 1000)
        state["used_seconds"] = 14350
        self.assertEqual(diagnostics.remaining_verification_seconds(state), 50)

    def test_single_class_strata_are_reported_as_undefined(self):
        y = np.zeros((2, 10), dtype=int)
        p = np.full((2, 10), .1)
        report = diagnostics.metrics(y, p, p, np.array([True, False]))
        self.assertFalse(report["auc_defined"])
        self.assertEqual(report["events"], 1)
        self.assertNotIn("auc_diff", report)
