"""The fixed validation protocol rejects mutated inputs and unsafe reports."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v7_protocol", ROOT / "experiments/v7-validation/validate_v7.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


class V7ProtocolTests(unittest.TestCase):
    def test_frozen_input_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.txt"
            source.write_text("original")
            inputs = {"input.txt": protocol.audit.sha256(source)}
            protocol.verify_inputs(root, inputs)
            source.write_text("replacement")
            with self.assertRaisesRegex(ValueError, "Frozen input changed"):
                protocol.verify_inputs(root, inputs)

    def test_reports_cannot_escape_results_directory_or_replace_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside.json"
            with self.assertRaises(ValueError):
                protocol.run(root, outside)
            report = root / "docs/results/v7-validation/validation.json"
            report.parent.mkdir(parents=True)
            report.write_text("preserve existing evidence")
            with self.assertRaises(FileExistsError):
                protocol.run(root, report)
            self.assertEqual(report.read_text(), "preserve existing evidence")

    def test_recipe_hash_is_order_independent_and_rejects_nonfinite_values(self):
        self.assertEqual(protocol.canonical_hash({"gamma": 1.12, "seed": 2026}),
                         protocol.canonical_hash({"seed": 2026, "gamma": 1.12}))
        with self.assertRaises(ValueError):
            protocol.canonical_hash({"gamma": float("nan")})
