---
name: verify-pipeline
description: >-
  Use this skill when running tests, verifying code integrity, checking source manifest hashes, or testing archive deduplication.
---

# Verify Pipeline and Code Integrity

This skill covers the fast execution of automated test suites and hash verification checks.

## 1. Run Regression Tests

Execute pytest across all test files:
```bash
pytest tests/
```

Expected result: All tests pass without warnings or errors.
- `tests/test_consolidation.py`: Verifies byte integrity of `legacy/` against `configs/source_manifest.json`.
- `tests/test_archive_deduplication.py`: Verifies deduplication logic on synthetic directories.
- `tests/test_archived_data_copies.py`: Verifies cleanup and restoration safety.

## 2. Verify Asset Integrity

Verify that local bundled assets match the declared SHA256 checksums:
```bash
python tools/consolidate.py verify
```

## 3. Git Status & Safety Check

Confirm no untracked binaries or large files were accidentally generated:
```bash
git status --short
```
Ensure no modified files appear in `legacy/` or `configs/source_manifest.json`.
