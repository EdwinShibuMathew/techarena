# Antigravity CLI Workspace Rules — TechArena ChaiScript

Welcome to the TechArena ChaiScript repository. You are operating as an expert AI/ML Engineer in WSL2 Linux (`/home/edwin/projects/techarena-team`).

## Core Directives

1. **Active vs Frozen Directories**:
   - **Active Code**: `tools/`, `configs/`, `tests/`, `experiments/`.
   - **Frozen History (`legacy/`)**: 62 historical Python scripts. SHA256-verified by `configs/source_manifest.json`. **DO NOT EDIT OR RUN DIRECTLY**.
   - **Untracked Heavy Assets (`local/`, `data/`, `artifacts/`)**: Over 12 GB of archives, binary data, and checkpoints. **NEVER search, index, or cat these folders**.
2. **Deterministic Fast Execution**:
   - Replay baseline: `python tools/predict_v6.py`
   - Run tests: `pytest tests/`
   - Verify assets: `python tools/consolidate.py verify`
   - Discover files: `git ls-files` (avoids 11GB traversal)
3. **ML Protocol & Promotion Gate**:
   - Target metric: Global row ROC AUC.
   - Baseline: V6 candidate row ranking (0.819451 public AUC).
   - Gate: Candidate must show strictly positive paired 95% confidence interval on Jan25–26 primary evaluation before promotion to five-day validation.
4. **Context & Token Economy**:
   - Use progressive disclosure skills located in `.agents/skills/`.
   - Do not print entire CSVs or arrays to stdout.
   - For detailed rules, see:
     - [Token Efficiency](.agents/rules/00-token-efficiency.md)
     - [Architecture & Entrypoints](.agents/rules/01-architecture-entrypoints.md)
     - [ML Protocol & Validation](.agents/rules/02-ml-protocol.md)
