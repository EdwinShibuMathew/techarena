---
name: token-saver
description: >-
  Use this skill when seeking fast, token-conserving patterns for browsing, searching, or inspecting the TechArena codebase.
---

# Token-Saver Cheat Sheet

Use these low-overhead commands and patterns instead of expensive broad context operations.

## 1. Quick File & Code Lookups

| Goal | Low-Token Pattern | High-Token Pattern (AVOID) |
|---|---|---|
| List repo files | `git ls-files` | `find .` / `dir /s` |
| Search code text | `git grep -n "keyword" tools/ configs/` | `grep -r "keyword" .` |
| Inspect file head | `head -n 50 tools/predict_v6.py` | `cat tools/predict_v6.py` |
| Inspect array shape | `python -c "import numpy as np; d = np.load('data/cache/train_events.npz'); print(d.files)"` | Dumping array into stdout |
| Check git changes | `git status -s` | `git diff` without file filter |

## 2. Protected & Excluded Paths

Never read, index, or grep inside:
- `local/` (11+ GB archives)
- `data/` (970+ MB binary data)
- `artifacts/` (370+ MB checkpoints and predictions)
- `legacy/` (historical frozen code)

## 3. Fast Config Reference

Before reading multiple python scripts for model weights or submission hashes, check:
- `configs/v6.json` (exact weights, member list, and submission SHA256).
- `configs/source_manifest.json` (tracked historical files and SHA256 digests).
