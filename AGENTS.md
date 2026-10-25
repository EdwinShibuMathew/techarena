# AGENTS.md — Agentic Development & Token-Efficiency Guide

This repository hosts the **TechArena — ChaiScript** competition project.
This document provides high-density context and operational guardrails for autonomous coding agents (**Codex CLI**, **GitHub Copilot CLI**, **Antigravity CLI**). Following these instructions minimizes token consumption and prevents context pollution.

---

## 1. Executive ML Baseline & Metric

- **Domain**: Candidate recommendation for e-commerce / user-event sessions.
- **Active Baseline**: **V6** ensemble (3x dim8 preference models + 2x dim16 models + CatBoost combiners).
- **Official Metric**: Global row ROC AUC across 529,390 evaluation rows (52,939 events × 10 candidates).
- **Benchmark Submission SHA256**:
  `c386d545b3746e05dd515020c6185afda06cefdbcdf7d48df022454762fb378c`
- **Historical Public AUC**: `0.819451` (V6) vs `0.819402` (V5).

---

## 2. Directory Layout & Boundaries

| Directory | Scope / Purpose | Agent Policy |
|---|---|---|
| `tools/` | Active portable CLI utilities (`predict_v6.py`, `consolidate.py`, etc.) | **Active**: Inspect and execute. |
| `configs/` | Benchmark configurations (`v6.json`) and SHA256 manifests (`source_manifest.json`) | **Active**: Read-only reference unless explicitly modifying configs. |
| `tests/` | Path safety, deduplication, and migration regression tests (`test_*.py`) | **Active**: Run with `pytest tests/`. |
| `experiments/` | Workspaces for active/new experimental trials (`experiments/<trial_id>/`) | **Active**: Create new candidate trials here. |
| `docs/` | Model architecture, WSL reproduction, handoff, and results (`docs/results/`) | **Active**: Consult for domain context and record trial logs. |
| `legacy/` | 62 historical Python scripts | **FROZEN / READ-ONLY**: SHA256-locked by `source_manifest.json`. **NEVER EDIT**. Do not run directly (they rely on stale absolute paths). |
| `local/` | 11+ GB local archives, deduplication journals, run logs | **IGNORE**: Never search, grep, or read. |
| `data/` | 970+ MB raw caches and `.npz` arrays | **IGNORE**: Binary assets. Read only via Python scripts. |
| `artifacts/` | 370+ MB checkpoints (`.pt`), combiners (`.cbm`), CSVs | **IGNORE**: Local-only weights and predictions. |

---

## 3. Golden Commands (Instant CLI Entrypoints)

Do **NOT** guess commands or write exploratory scripts to find CLI syntax. Use these exact commands:

```bash
# 1. Run full test suite (regression & migration checks)
pytest tests/

# 2. Verify collected asset bundle and manifest integrity
python tools/consolidate.py verify

# 3. Replay V6 baseline inference (CUDA required; verifies exact SHA256)
python tools/predict_v6.py

# 4. Replay V6 baseline on CPU (or specify alternate output path)
python tools/predict_v6.py --device cpu --output artifacts/reproduced_v6.csv

# 5. List all tracked source files efficiently without traversing local/
git ls-files
```

---

## 4. Token Conservation Guardrails

To prevent burning tens of thousands of tokens per session:

1. **Avoid Unbounded Search**:
   - **NEVER** run `find .`, `grep -r`, or `ls -R` from the root without excluding `local/`, `data/`, `artifacts/`, and `legacy/`.
   - Prefer: `git ls-files | grep <pattern>` or `find tools configs tests -type f`.
2. **Never Ingest Binary or Tabular Data**:
   - Never print or read raw `.csv`, `.npz`, `.npy`, `.pt`, or `.cbm` files directly into context.
   - Inspect tabular shapes or schemas using short Python one-liners via `python -c`.
3. **Preserve `legacy/` Immutability**:
   - `tests/test_consolidation.py` strictly checks that every file in `legacy/` matches `configs/source_manifest.json`.
   - Modifying any byte in `legacy/` will fail CI/tests.
4. **Targeted Reading**:
   - Read line ranges (e.g. lines 1–100) instead of dumping entire files.
   - Use the pre-computed configurations in `configs/v6.json` rather than reverse-engineering weights from code.

---

## 5. ML Experimentation & Promotion Gate

When developing new modeling ideas:
1. **Location**: Place all code in `experiments/<experiment_name>/`.
2. **Hypothesis & Gate**:
   - Declare one hypothesis, one comparator, and stopping rule.
   - Evaluate against V6 on the primary first-user Jan25–26 window.
   - **Promotion Gate**: Strictly positive primary paired 95% confidence interval (`[lower, upper]` with `lower > 0`) AND non-negative secondary Jan26 performance.
   - Follow with a fixed five-day comparison.
3. **Reporting**: Store compact result metrics in `docs/results/<experiment_name>/`.
4. **No Overfitting**: Never tune weights against the public leaderboard, and do not label reused dates as independent holdout data.
