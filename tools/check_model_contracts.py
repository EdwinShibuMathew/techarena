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
    for pattern in ("test_model_contracts.py","test_workflow_integration.py","test_workflow_contracts.py","test_shared_feature_cache.py"):
        found=unittest.defaultTestLoader.discover(str(ROOT/"tests"),pattern=pattern)
        if not found.countTestCases():raise RuntimeError("No tests found: "+pattern)
        suite.addTests(found)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() and not result.skipped and result.testsRun>=13 else 1


if __name__=="__main__":raise SystemExit(main())
