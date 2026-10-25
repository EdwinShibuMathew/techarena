# Rule 00: Token Efficiency & Context Boundaries

To maximize response speed and minimize LLM context token usage, observe the following constraints:

## 1. Directory Traversal Bounds
- **Strictly Avoid**:
  - `local/` (11+ GB): Contains 34 archived experiment runs, deduplication journals, logs.
  - `data/` (970+ MB): Contains large binary `.npz` arrays and raw zip archives.
  - `artifacts/` (370+ MB): Contains PyTorch model weights (`.pt`), CatBoost trees (`.cbm`), predictions, and CSVs.
  - `legacy/`: Historical scripts; do not index unless checking specific historical implementation details.
- **Allowed Discovery Commands**:
  - `git ls-files`: Instant listing of tracked files (<1 KB output).
  - Targeted commands: `find tools configs tests -type f` or `ls -l tools/`.

## 2. File Reading Guidelines
- Always read specific line slices using line ranges instead of loading full files.
- Never dump CSV files, model dictionaries, or numpy arrays into chat/prompt contexts.
- Use Python CLI one-liners with `--help` or small sample inspects (`head -n 20`) when examining outputs.

## 3. Preserving Budget
- Antigravity CLI enforces a 24 KB per-rule cap and a 20,000-token aggregate rules budget.
- Keep workspace modifications minimal, focused, and test-driven.
