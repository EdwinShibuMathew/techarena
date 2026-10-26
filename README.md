# TechArena — ChaiScript

I'm Edwin, and this is our ChaiScript repository for the TechArena competition.
I keep our code, model notes and experiment history here so the team can review
the work and reproduce our current baseline.

Our current submitted benchmark is **V6**, with a public AUC of **0.819451** in
my latest saved leaderboard screenshot. That is a historical public result;
our private score and final ranking are still unknown.

## Start here

If you're joining the team, start with [Reproduce V6 in WSL](docs/REPRODUCE_V6_WSL.md).
I've also documented [how I manage access and prepare our private downloads](docs/TEAM_ACCESS.md).
See [CI, branch protection and source releases](docs/CI_RELEASE.md) for merge checks
and the distinction between source readiness and exact V6 inference replay.

1. [Model and validation](docs/MODEL.md): what V6 does and how it was evaluated.
2. [Experiment results](docs/RESULTS.md): successful and rejected changes.
3. [Full handoff](docs/HANDOFF.md): personal environment, paths and research history.
4. [Consolidation](docs/CONSOLIDATION.md): collecting assets and archiving old folders.
5. [Competition instructions supplied by the organizers](docs/competition/INSTRUCTIONS.md).

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

I keep code, documentation and small metadata in Git. Our ignore rules exclude
training data, model weights, prediction arrays and submission CSVs. I share the
verified assets and a packed copy of my working environment privately with the
team. Cloning the repository gets you the code; you also need those downloads.

`legacy/` is a source snapshot for review, not a collection of independently
supported entry points. Historical runners depend on old absolute paths and
study folders. Use the portable entry point below for the submitted V6 model.
Future experiments should use configurations within this repository instead of
creating another sibling project folder.

## How I consolidated the project

I moved 34 old experiment folders into `local/archive/`, verified the active
V6 assets, and reproduced the original submission exactly. My working repository
is now `/home/edwin/projects/techarena-team`.

The commands below record the original migration. I keep them for reference;
they are not setup steps for a teammate or something to rerun on my completed
workspace. The initial copy required an unused destination and my existing
`techarena` environment:

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
**Archiving reduced directory clutter without reclaiming disk space.** I later
removed seven verified duplicate data copies, reclaiming about 0.947 GiB while
keeping the active data, models and results. See [the cleanup record](docs/results/archived_data_cleanup.json)
and [data-copy cleanup notes](docs/ARCHIVED_DATA_COPIES.md).

I use the following helper to publish source and documentation from the WSL
terminal where I signed into GitHub:

```bash
python tools/publish_private.py
```

The helper requires a feature branch, checks my authenticated account and the
file list, verifies that `EdwinShibuMathew/techarena` is private, and pushes that
branch with a link to open a pull request. It refuses `main` and detached HEAD
before staging or committing. I add teammates through
GitHub's repository access settings; the helper does not send invitations.

## Teammates: reproduce inference

Follow the [WSL reproduction guide](docs/REPRODUCE_V6_WSL.md) for repository access,
downloads, a packed copy of my working environment, and verification.
I prepared and checked the five private transfer files described in the
[team access guide](docs/TEAM_ACCESS.md). A clone alone does not contain the models.
After installing the supplied environment and assets, run:

```bash
python tools/consolidate.py verify
python tools/predict_v6.py --output artifacts/teammate_reproduced_v6.csv
```

The default output without `--output` remains `artifacts/reproduced_v6.csv`.
Existing output is validated rather than overwritten. The unpinned requirements
file is not a reproduction lock; cross-hardware numerical identity must pass
the replay checks. There is no competition upload command.

## Preservation and limits

- V5 and V6 stay the benchmarks; recent failed experiments are not promoted.
- Full-data retraining has not been refactored into a portable one-command job.
  The original training code and frozen evidence are retained for inspection.
- Archived frozen reports retain their original paths as historical evidence.
  They are not silently rewritten to claim that old runners are portable.
- I can reverse the folder moves with `python tools/consolidate.py restore --apply`
  if the original paths are free. After data-copy cleanup, I must first recreate
  the removed archive data copies with `python tools/clean_archived_data_copies.py restore`.
- No unrelated Windows folders or Downloads files are automatically removed.
- Repeated validation dates mean reported intervals are development evidence,
  not independent guarantees of private leaderboard gains.
