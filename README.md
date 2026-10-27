# TechArena — ChaiScript

Candidate click prediction for the TechArena competition. Each event contains
ten candidate items and at most one click. The official metric is **global row
ROC AUC** across all candidate rows.

The repository contains the submitted V6 inference pipeline, experiment tools,
regression tests, and recorded model evidence. Competition data, trained models,
and submission CSVs are distributed separately through private team access.

[Code-only setup](#code-only-setup) · [Reproduce V6](#reproduce-v6) ·
[Contributing and CI](#contributing-and-ci) · [Documentation](#documentation)

## Model status

| Model | Status | Evidence |
|---|---|---|
| **V6** | Submitted benchmark; supported inference entry point | Historical public ROC AUC: **0.819451**. [Configuration](configs/v6.json) and [model details](docs/MODEL.md). |
| **V7** | Research candidate; **not promoted** | Overall gains reproduce, but AUC declines by **0.0001676** on the five-day late-window subset. [Validation report](docs/results/v7-validation/README.md). |
| **V7.1 entropy postprocessing** | Three gates passed; **not promoted**, no submission exported | Late 48–120h AUC declines by **0.0004835**. [Frozen experiment and decision](docs/results/v7.1-entropy-sharpening-v1/README.md). |
| **V8 research archive** | Six closed campaigns; **none promoted**, no submission-ready V8 | [Preserved evidence and review dispositions](docs/results/v8-archive/README.md). |

V6 combines three dimension-8 preference models and two dimension-16 models,
with event training, exposure adjustment, cold-user history fallback, and
CatBoost combiners. Evaluation covers **52,939 events / 529,390 candidate rows**.

V7 applies fixed power sharpening at `gamma=1.12`. It preserves event sums and
within-event ordering; its AUC changes arise from comparisons across events.
Recorded experiment dates have been reused for development, so their intervals
are not independent holdout guarantees or adjusted for prior model selection.

## Code-only setup

Use Git and Python 3.12 with virtual-environment support in a Linux or WSL shell.
Accept repository access before cloning this private repository:

```bash
git clone https://github.com/EdwinShibuMathew/techarena.git
cd techarena
```

Check out the intended branch or commit before continuing. When reviewing a PR,
use its branch: this README describes the checked-out revision. Run subsequent
commands from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-ci.txt
pytest tests/
```

These dependencies support CPU source checks without private assets. Tests that
need CatBoost or a generated submission may be skipped when those are absent.
This environment is separate from the V6 inference environment.

On a clean, committed checkout, verify the source and frozen baseline against
the fetched `origin/main` revision:

```bash
python tools/source_readiness.py verify --base-ref origin/main
```

## Reproduce V6

Follow the [WSL reproduction guide](docs/REPRODUCE_V6_WSL.md) for the verified
reference setup, environment activation, and asset installation. Request one
matching private export containing:

- `techarena-v6-assets.tar.gz` — data, checkpoints, combiners, and reference predictions.
- `techarena-v6-env-linux-x86_64.tar.gz` — the packed inference environment.
- `SHA256SUMS`, `CODE_COMMIT.txt`, and `runtime_versions.json` — integrity and revision records.

Use the supplied inference environment for the following commands; the
code-only `.venv` above contains only test dependencies. A Git clone alone
does not include the assets. See [team access](docs/TEAM_ACCESS.md) for delivery
and sharing instructions.

```bash
python tools/consolidate.py verify
python tools/predict_v6.py
```

The predictor defaults to CUDA and writes `artifacts/reproduced_v6.csv`. It
replays all five saved members against reference predictions, then checks the
exported CSV against the canonical submission fingerprint:

```text
c386d545b3746e05dd515020c6185afda06cefdbcdf7d48df022454762fb378c
```

CPU execution and alternate output paths are available:

```bash
python tools/predict_v6.py --device cpu --output artifacts/reproduced_v6_cpu.csv
```

Both modes must pass the same strict reproduction checks; matching asset hashes
alone does not establish numerical identity on different hardware. Existing
output is validated and a different file is never overwritten. New V6 output is
validated in a private staging directory before atomic publication; a destination
created by another run is preserved and publication fails. The unpinned
`requirements.txt` is a package inventory, not an inference environment lock.

## Repository layout

| Path | Purpose |
|---|---|
| `tools/` | Supported inference, source verification, asset management, and reusable research utilities. |
| `configs/` | Model recipes, frozen-source manifest, and desired GitHub protection policy. |
| `tests/` | Source, path-safety, migration, evaluation, and prediction regression tests. |
| `experiments/` | Isolated research trials with their own protocols and execution instructions. |
| `docs/` | Setup guides, model documentation, and compact results under `docs/results/`. |
| `.github/` | Required CI workflow and contributor-tool instructions. |
| `legacy/` | Historical Python sources preserved byte-for-byte; use supported wrappers instead of running them directly. |
| `data/`, `artifacts/`, `local/` | Ignored private data, models, predictions, receipts, and archives. |

Git tracks source, documentation, and small reports. Keep datasets, checkpoints,
environment archives, submission CSVs, and credentials out of commits. Archive
and cleanup operations are documented separately; they are not onboarding steps.

## Contributing and CI

Create a feature branch from current `main`, commit the intended changes, and
open a PR. Protected `main` requires an up-to-date branch, resolved conversations,
and all four GitHub Actions checks, including for administrators:

| Required check | What it establishes |
|---|---|
| `cpu-tests` | Regression tests on Ubuntu 24.04 with Python 3.12 and the CPU test dependencies. |
| `model-contracts` | Actual CPU model fitting, numerical contracts and the full synthetic experiment workflow; skips fail this check. |
| `source-integrity` | Frozen baseline/source integrity and checks against tracked private assets, binary files, and oversized files. |
| `release-readiness` | Successful prerequisite jobs and a verified source archive from the tested commit. |

The current policy requires zero external approvals; force pushes and deletion
of `main` are disabled. Every PR runs CI, including documentation-only changes.
Successful `main` runs retain a private source package and checksum receipt for
14 days. **Source readiness does not establish model reproduction or promotion.**
See [CI and source releases](docs/CI_RELEASE.md) for artifacts and policy details.

Preserve `legacy/`, `configs/source_manifest.json`, and `configs/v6.json`.
Place new model variants under `experiments/` with separate configurations.
The owner-specific `tools/publish_private.py` helper publishes feature branches;
other contributors should use ordinary Git and PR workflows.

## Research workflow

Declare a hypothesis, the unchanged V6 comparator, and a stopping rule before
scoring a new recipe. Evaluate the primary first-user Jan25–26 window with a
paired 95% interval strictly above zero and nonnegative secondary Jan26 change,
then perform the fixed five-day comparison and its declared stress checks.
Record results under `docs/results/`. Do not tune against the public leaderboard
or relabel reused dates as independent holdout data.

Use the [standalone ML workflow](docs/ML_WORKFLOW.md) for new work: committed source snapshots, explicit input fingerprints, shared verified features, chronological development and promotion, persistent compute limits, and two saved-model replays. The public synthetic adapter validates this machinery; production model adapters and measured V8 improvement remain separate research work.

The feature cache, enrichment, event-scale, recency-history, and evaluation tools
are research building blocks. They do not change the frozen V6 inference recipe.
For specific protocols, use the [PairLogit trial guide](experiments/v6-pairlogit-v1-api-fix/RUN_PAIRLOGIT.md)
and [fixed V7 validation guide](experiments/v7-validation/RUN_V7.md). PairLogit's
five-day runner and full-data fitting/export stage are not implemented in that
trial. Full-data retraining is not available as a supported one-command workflow.
Competition uploads remain manual.

## Documentation

- [V6 model and validation](docs/MODEL.md)
- [Experiment results and decisions](docs/RESULTS.md)
- [V6 reproduction in WSL](docs/REPRODUCE_V6_WSL.md)
- [Team access and private assets](docs/TEAM_ACCESS.md)
- [CI, protection, and source releases](docs/CI_RELEASE.md)
- [Repository instructions and experiment rules](AGENTS.md)
- [Competition instructions](docs/competition/INSTRUCTIONS.md)
- Historical references: [handoff](docs/HANDOFF.md), [consolidation](docs/CONSOLIDATION.md),
  [archive cleanup](docs/ARCHIVE_CLEANUP.md), and [data-copy cleanup](docs/ARCHIVED_DATA_COPIES.md).
