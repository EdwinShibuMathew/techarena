# Rule 01: Repository Architecture & Operational Entrypoints

## 1. System Components

- **`tools/predict_v6.py`**:
  The primary runner for replicating the submitted V6 benchmark.
  - Verifies asset digests against `configs/v6.json` before importing heavy ML dependencies.
  - Runs all 5 ensemble members through portable neural pipeline and CatBoost combiners.
  - Validates exact SHA256 digest (`c386d545b3746e05dd515020c6185afda06cefdbcdf7d48df022454762fb378c`).
- **`tools/consolidate.py`**:
  Consolidation and asset management tool.
  - `python tools/consolidate.py verify`: Checks integrity of all bundled assets against the manifest.
  - `python tools/consolidate.py inventory`: Reports project tree and sizes.
- **`tests/`**:
  Regression test suite for archive deduplication, data copy management, and consolidation safety.
  - Run with: `pytest tests/`

## 2. Environment & Execution
- Runtime: Python 3.12+ (or packed Linux x86_64 environment) in WSL2.
- GPU: CUDA-enabled PyTorch (NVIDIA RTX GPU with 8GB+ VRAM). Fallback CPU execution supported via `--device cpu`.
- Threading: Pinned to `torch.set_num_threads(4)` for stability.
