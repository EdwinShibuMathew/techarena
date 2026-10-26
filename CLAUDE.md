# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## Environment

- Python 3.12. This machine has `python3.12` but no `python` alias or global `pytest`. Code-only test env: `python3.12 -m venv .venv && source .venv/bin/activate && python -m pip install -r requirements-ci.txt` (numpy + pytest only).
- Inference (`tools/predict_v6.py`, `tools/predict_v7.py`, `tools/consolidate.py verify`) needs the private asset tarball and the packed inference env (see `docs/REPRODUCE_V6_WSL.md`). Without them `tests/test_v7_validation.py` skips (no `artifacts/submissions/v7_candidate.csv`) — a skip is not a pass. `predict_v7.py` also needs `artifacts/submissions/v6.csv` or `artifacts/reproduced_v6.csv`.
- Commands in AGENTS.md use `python`/`pytest`; here run them inside the venv (or as `python3.12 -m pytest`). Single test: `pytest tests/test_x.py::Class::test_name` (unittest classes).
- Paths like `/home/edwin/...` and `conda activate techarena` in GEMINI.md, `.codex/instructions.md` and `docs/HANDOFF.md` are the owner's machine, not this one.

## CI (`.github/workflows/source-readiness.yml`) fails if

- Anything under `legacy/`, `configs/source_manifest.json` or `configs/v6.json` changes (including line endings). New model configs go in new files (e.g. `configs/v7.json`).
- A tracked file is under `data/`, `artifacts/`, `local/`, `archive/`; has a binary/data suffix (`.csv .npz .npy .pt .cbm .log .pkl .parquet .gz` …); is a `.env`; is a symlink; or exceeds 2 MiB.
- Agent config files checked by `tests/test_agent_configurations.py` are removed, or lose pinned content (`.codex/config.json` `baseline=v6`; copilot file must mention `predict_v6.py`, `legacy/`, "Global row ROC AUC"; rule files ≤ 24 KB; skills need a `name:` frontmatter line) (AGENTS.md, GEMINI.md, `.agentignore`, `.ignore`, `.codex/*`, `.github/copilot-instructions.md`, `.agents/rules/*`, `.agents/skills/*/SKILL.md`).

Local CI equivalent (requires no uncommitted changes to tracked files; untracked files are ignored): `python tools/source_readiness.py verify --base-ref origin/main`. Without `--base-ref` the frozen-path check is skipped.

## Where AGENTS.md is out of date

- The promotion gate has **four** checks (`experiments/v7-validation/validate_v7.py`): primary paired-CI lower bound > 0, Jan26 change ≥ 0, five-day CI lower bound > 0, five-day late (48–120h) change ≥ 0. If any fails, stop; don't sweep parameters (e.g. V7 `--gamma` is fixed at 1.12).
- V7 (`tools/predict_v7.py`, `configs/v7.json`) exists but is **not promoted**: it fails the late-window check. V6 remains the submitted baseline.
- The legacy hash lock is enforced by `tools/source_readiness.py`, not `tests/test_consolidation.py`.
- `validate_v7.py prepare` reads three hash-locked `local/archive/*.npz` files: the one sanctioned exception to "never touch `local/`" (see `experiments/v7-validation/RUN_V7.md`).

## Conventions

- `legacy/` is imported as a library by `predict_v6.py` (via `sys.path`); never edit or run it directly.
- Experiment tests are collected by `tests/` (e.g. `tests/test_pairlogit_regressions.py` loads `experiments/v6-pairlogit-v1-api-fix/test_pairlogit.py`).
- Experiments: seed 42, `torch.set_num_threads(4)`, start from the epoch-260 checkpoints; ask before any multi-hour retraining. Log decisions in `docs/RESULTS.md` as well as `docs/results/<name>/`.
- Branches `feat/`, `fix/`, `ci/` off `main`; Conventional Commits with scope (`fix(calibration): ...`); PRs into `main`. `tools/publish_private.py` is the repo owner's helper only.
- No linter/formatter is configured; follow PEP 8 with type hints.
