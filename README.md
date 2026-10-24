# TechArena — ChaiScript

One repository for the competition code, model explanation and experiment history.
The current submitted benchmark is **V6**, public AUC **0.819451** in Edwin's last
reported screenshot. It is not a verified live ranking or a private score.

## Start here

1. [Model and validation](docs/MODEL.md): what V6 does and how it was evaluated.
2. [Experiment results](docs/RESULTS.md): successful and rejected changes.
3. [Full handoff](docs/HANDOFF.md): personal environment, paths and research history.
4. [Consolidation](docs/CONSOLIDATION.md): collecting assets and archiving old folders.
5. [Competition instructions](docs/competition/INSTRUCTIONS.md).

## Layout

```text
techarena-team/
  README.md
  docs/                 Team documentation and compact experiment reports
  configs/              V6 recipe and known submission fingerprint
  tools/                Portable prediction, consolidation and private publication
  tests/                Migration/path-safety regression tests
  legacy/               Original Python sources, preserved byte-for-byte
  data/                 Local only: original archives, example CSV and caches
  artifacts/            Local only: V5/V6 submissions, V6 models and prediction caches
  local/                Local only: inventory, receipts and archived experiment folders
```

Only code, documentation and small metadata enter Git. The ignore rules exclude
training data, model weights, prediction arrays and submission CSVs. Team members
need the separately supplied local assets to reproduce inference. Do not assume
that a Git clone contains trained models or a reproducible training environment.

`legacy/` is a source snapshot for review, not a collection of independently
supported entry points. Historical runners depend on old absolute paths and
study folders. Use the portable entry point below for the submitted V6 model.
Future experiments should use configurations within this repository instead of
creating another sibling project folder.

## Edwin: consolidate the existing WSL project

Use the already working `techarena` conda environment; no new packages are needed.
The destination must not already exist for this initial copy:

```bash
conda activate techarena
test ! -e "$HOME/projects/techarena-team" &&
cp -a /mnt/c/Users/Edwin/Documents/Codex/techarena-team "$HOME/projects/techarena-team"
cd "$HOME/projects/techarena-team"
python tools/consolidate.py inventory
python tools/consolidate.py bundle
python tools/predict_v6.py
python tools/consolidate.py archive
```

`inventory` reads the existing project tree. `bundle` copies and hashes the
required V6 assets and V5/V6 submission backups; original folders stay intact.
`predict_v6.py` runs the existing model from the new paths, checks each member's
predictions and the exact V6 CSV SHA256, then writes a relocation receipt.
The final command prints a proposed archive plan and **does not move anything**.

Review the plan. Close training processes and move other terminals out of the old
project folders. Then perform the reversible archive move:

```bash
python tools/consolidate.py archive --apply
```

It requires the successful V6 receipt and verifies the collected asset hashes.
Old plain `techarena*` experiment folders move into `local/archive/`, leaving one
active working directory. Existing Git repositories, symlink roots and busy
folders are excluded or blocked for separate review. No experiment is deleted.
**Archiving reduces directory clutter; it does not reclaim disk space.**

Publish the code to the requested new private repository from the WSL terminal
where `gh auth login` was completed:

```bash
python tools/publish_private.py
```

The publisher checks the authenticated account and file list, creates
`EdwinShibuMathew/techarena` privately if absent, verifies privacy and pushes the
single `main` branch. It does not invite collaborators or send messages. Add your
team through GitHub's repository access settings.

## Teammates: reproduce inference

Use Python 3.12 and a compatible CUDA PyTorch installation. The unpinned
requirements list is a starting point, not the original environment lock.
Edwin's bundle records observed package versions in `local/environment.json`
and the shareable `docs/runtime_versions.json` (created when bundling).
Transfer the ignored `data/` and `artifacts/` assets separately and run:

```bash
python tools/consolidate.py verify
python tools/predict_v6.py
```

The default prediction output is `artifacts/reproduced_v6.csv`. Existing output
is validated rather than overwritten. There is no competition upload command.

## Preservation and limits

- V5 and V6 stay the benchmarks; recent failed experiments are not promoted.
- Full-data retraining has not been refactored into a portable one-command job.
  The original training code and frozen evidence are retained for inspection.
- Archived frozen reports retain their original paths as historical evidence.
  They are not silently rewritten to claim that old runners are portable.
- `python tools/consolidate.py restore --apply` reverses completed folder moves,
  provided none of the original paths have been reused.
- No unrelated Windows folders or Downloads files are automatically removed.
- Repeated validation dates mean reported intervals are development evidence,
  not independent guarantees of private leaderboard gains.
