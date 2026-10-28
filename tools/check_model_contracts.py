"""CI entry point: missing dependencies, zero tests or any skip are failures."""
import importlib.metadata
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]


def main():
    for name in ("numpy","torch","catboost","scikit-learn"):
        print(name+"="+importlib.metadata.version(name),flush=True)
    suite=unittest.TestSuite()
    for pattern in ("test_model_contracts.py","test_workflow_integration.py","test_workflow_contracts.py","test_shared_feature_cache.py","test_postprocessing.py","test_joint_calibration.py","test_v7_trial.py","test_history_graph.py"):
        found=unittest.defaultTestLoader.discover(str(ROOT/"tests"),pattern=pattern)
        if not found.countTestCases():raise RuntimeError("No tests found: "+pattern)
        suite.addTests(found)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or result.testsRun < 13:
        return 1
    # Reconstruction contracts use pytest fixtures; require them without skips too.
    import pytest
    class RequiredRebuild:
        skipped = False
        passed = 0
        def pytest_runtest_logreport(self, report):
            if report.skipped: self.skipped = True
            if report.when == 'call' and report.passed: self.passed += 1
    required = RequiredRebuild()
    status = pytest.main([str(ROOT / 'tests/test_rebuild.py'), '-q'], plugins=[required])
    return 0 if status == 0 and required.passed >= 12 and not required.skipped else 1


if __name__=="__main__":raise SystemExit(main())
