# Codex CLI Instructions — TechArena ChaiScript

These instructions guide the OpenAI Codex CLI when generating shell commands, modifying code, or performing ML tasks in this repository.

## Operational Environment
- Operating System: Linux (WSL2 Ubuntu on x86_64).
- Working Directory: `/home/edwin/projects/techarena-team`.
- Python Environment: `conda activate techarena` (or system Python 3.12+).
- Hardware: NVIDIA RTX GPU with CUDA enabled.

## Critical Guardrails
1. **Never Scan Excluded Directories**:
   - `local/` (11 GB), `data/` (970 MB), `artifacts/` (370 MB) must never be traversed.
   - Use `git ls-files` to list repository files.
2. **Read-Only `legacy/`**:
   - Files in `legacy/` are frozen and verified by SHA256 in `configs/source_manifest.json`.
   - Never edit files in `legacy/`.
3. **Execution Conventions**:
   - Run tests: `pytest tests/`
   - Replay baseline: `python tools/predict_v6.py`
   - Verify assets: `python tools/consolidate.py verify`
   - Place all new experiments under `experiments/<experiment_name>/`.
