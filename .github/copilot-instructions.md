# GitHub Copilot Instructions — TechArena ChaiScript

You are an expert AI/ML engineer assisting in the TechArena ChaiScript competitive machine learning repository.

## Project & Architecture Context
- **Baseline**: V6 candidate row ranking model with public AUC 0.819451 and submission SHA256 `c386d545b3746e05dd515020c6185afda06cefdbcdf7d48df022454762fb378c`.
- **Target Metric**: Global row ROC AUC across 529,390 evaluation rows (52,939 events × 10 candidates).
- **Active Code**: `tools/` (portable CLI tools), `configs/` (V6 recipe & manifests), `tests/` (unit tests), `experiments/` (isolated candidate trials).
- **Frozen History (`legacy/`)**: Historical snapshot locked byte-for-byte in `configs/source_manifest.json`. **Do not modify or suggest edits to files in `legacy/`**.
- **Excluded Heavy Assets**: `local/` (11+ GB archives), `data/` (970+ MB caches), `artifacts/` (370+ MB checkpoints/CSVs). Never scan or ingest these directories.

## Coding & Style Guidelines
- **Python**: 3.12+ in WSL2. Clean, PEP 8 compliant, type-annotated where applicable.
- **Reproducibility**: Set explicit random seeds for PyTorch, NumPy, and Python (`seed=42`, `torch.manual_seed(42)`).
- **Performance**: Limit PyTorch thread count with `torch.set_num_threads(4)`. Use CUDA when available.

## Standard Commands
- Run tests: `pytest tests/`
- Replay baseline inference: `python tools/predict_v6.py`
- Verify asset bundle integrity: `python tools/consolidate.py verify`
- List tracked repository files: `git ls-files`

## ML Experimentation Policy
- All new experiments must be placed in `experiments/<experiment_name>/`.
- Candidate models must be compared directly against V6 on the primary Jan25–26 window.
- Promotion requires a strictly positive paired 95% confidence interval ($\text{CI}_{\text{lower}} > 0.0$).
