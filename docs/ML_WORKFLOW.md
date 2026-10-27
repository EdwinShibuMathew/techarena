# Standalone ML workflow

`tools/run_experiment.py` owns chronology, metrics, gates, budgeting, source
snapshots and publication. `tools/ml_workflow/` is the shared implementation.
An adapter under `experiments/<name>/` owns model training, feature extraction
and CatBoost fitting/prediction. Existing experiments and V6 remain intact.

## Run the public validation campaign

Use Linux/WSL, Python 3.12 and a clean committed checkout. These commands create
only ignored synthetic inputs and validation outputs, with actual Torch and
CatBoost fitting. They require no competition assets or prior campaign folders.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-model-ci.txt
python -m pip check
python tools/check_model_contracts.py
python experiments/workflow-smoke/prepare_fixture.py --fixture-id smoke-01
python tools/run_experiment.py prepare --run-id smoke-01 --spec artifacts/validation-inputs/smoke-01/experiment.json
python tools/run_experiment.py develop --run-id smoke-01
python tools/run_experiment.py screen --run-id smoke-01
python tools/run_experiment.py stress --run-id smoke-01
python tools/run_experiment.py finalize --run-id smoke-01
python tools/run_experiment.py verify --run-id smoke-01
```

Validation writes `artifacts/validation/smoke-01/`: source archive, manifest,
stage reports and seals, checkpoints, combiners, predictions, cache timing
receipts and two identical private CSVs. It reports `validation_complete: true`
and **`submission_ready: false`**. Its comparator is explicitly synthetic; its
AUC is orchestration evidence only. It cannot publish a production config or
submission. Use a new fixture/run ID for another validation. Never overwrite
old run evidence or retry a scientifically failed recipe.

## Source and asset contract

The generator above is the executable specification example. Its JSON declares
`schema_version`, `purpose`, `hypothesis`, V6 identity, `adapter`, `seed`,
`candidates`, explicit `candidate_order`, `development`, fixed `promotion`,
`inputs`, `bootstrap`, `selection` and `budget`. Each input has a relative path
and SHA256. Research also requires `release_name`, `device` and a fingerprinted
`carry` decision with `compute_limit_seconds` and `total_compute_seconds`.
Commit research specifications under their experiment directory before prepare.
All source inputs must be committed; assets stay ignored and private.

Prepare requires clean tracked source and verifies frozen legacy hashes. It
captures a complete `git archive` plus per-file hashes, safely extracts only
regular files/directories, and binds every stage to the archive, spec and input
fingerprints. The child executes the archived CLI/adapter. A later runtime edit
requires invoking the old snapshot explicitly, for example:

```bash
python artifacts/validation/smoke-01/source/tools/run_experiment.py verify --root "$PWD" --run-id smoke-01
```

Stage reports never import other historical campaign scripts. Hash-verified
assets can live outside a campaign directory through explicitly declared paths
inside this checkout. Root escapes and symlinks are rejected. Source archives
and frozen code are evidence; do not edit them to resume work.

## Chronology and promotion

Event NPZ inputs have `u,i,t,tb,dt,h,ht,rowids`, plus `y` for training only.
Candidate and row-ID arrays have shape `(events,10)`; history arrays have shape
`(events,20)`. Every present history timestamp precedes its event. No labels
reach the extraction target. Preprocessing any official cache into this schema
must preserve row order and be committed, reviewed experiment source.

Each case has a neural prefix before `meta_cutoff`, a one-day meta-label cohort
starting there, then a separate neural prefix before `target_cutoff`. The same
extractor/schema transfers to first-user targets through `target_end`.
Development ends before Jan25. Candidates must have nonnegative gains on every
declared origin and positive mean gain. Choose the largest minimum origin gain;
ties within 1e-12 follow the committed candidate order. Freeze that winner.

| Case | Meta cutoff | Target cutoff | Target end (exclusive) |
|---|---|---|---|
| Primary | Jan24 | Jan25 | Jan27 |
| Secondary | Jan25 | Jan26 | Jan27 |
| Five-day | Jan21 | Jan22 | Jan27 |

All dates are UTC, January 2025. Primary requires paired 95% CI lower bound >0;
secondary requires AUC difference >=0. Five-day requires lower bound >0 and
nonnegative difference on its 48–120h late subset. Bootstrap is fixed at 1,000
user-cluster draws with seed 2026 and at least 990 valid draws. Warm/cold/history
subsets are diagnostics only. Reused dates are retrospective evidence, never
independent or selection-adjusted holdouts. Failure closes that recipe; no
weight tuning against the leaderboard.

## Adapter contract and research boundaries

The synthetic reference adapter exercises both dimension 8 and 16. A research
adapter must declare `PURPOSES={"research"}` and implement:

- `train(fitting, metadata, checkpoint_path, candidate_params)`: fit only the
  supplied prefix; save `metadata` unchanged and a finite `model_state` using
  Torch. Metadata binds dimension, seed, cutoff, fitting count/index hash and
  prescribed epoch. Training receives defensive deep copies of metadata and
  candidate parameters; saved metadata is checked against independent expected
  values before any success receipt is issued. Initialize preferences from verified **same-cutoff** dim8
  epoch-260 or dim16 epoch-196 assets; never reuse later-prefix weights or train
  preference embeddings from scratch. Declare initialization assets in inputs.
- `extract(fitting, unlabeled_target, checkpoint_path)`: return float32 `X` and
  ordered Unicode `feature_names`. Declare `FEATURE_NAMES`, `EXTRACTOR_FILES`
  including the actual extraction function and every helper, and actual device
  and precision through `EXTRACTION_DEVICE` / `EXTRACTION_PRECISION` (CPU/float32
  defaults). Keep training/report code separate from extraction helpers.
- `fit_combiner(features, labels, output_path, seed)` and
  `predict(features, saved_combiner_path, rowids)`: save/reload the combiner;
  prediction returns `rowids` unchanged and `(events,10)` probabilities.

Each baseline input declares `version` (`V6` for research), all five ordered
`member_names`, `rowids`, member predictions and reconstructed ensemble `scores`.
Generate these from the unchanged V6 recipe at the same chronological prefixes;
review/cache their provenance before committing the specification. A label or
shape alone does not establish scientific validity. The runtime verifies input
hashes, case row order and the historical ensemble formula. Final research
validation additionally compares every evaluation member to canonical V6
references and performs two fresh exact V6 CLI replays.

Research carries the latest closed V8 decision's **1,974.359347462001 seconds**
into the original **14,400-second** total. All research run IDs share
`artifacts/experiments/compute-ledger.json`; changing run ID does not reset it.
Use the original verified carry decision, never a fabricated replacement.
An exclusive lock allows one active phase. Lock contention does not claim or
charge the waiting phase; retry that same phase after the lock is released.
After budget ownership and reservation succeed, the immutable phase claim is
written before worker launch, and existing retry protection applies. The entire
phase allowance is reserved before launch, wall time is charged including extraction, hashing,
evaluation and export, and unused time is refunded only after a normal
controller exit. A crashed controller retains the reservation. Timeout kills
the worker process group; a worker watchdog kills that group if its controller
dies. No automatic retries, silent refits or phase overwrite are supported.

After successful research gates, finalize fits only the prescribed final
models and reuses the secondary combiner. Two uncached saved-model replays,
matching CSV hashes, canonical V6 replays, input/source rechecks and successful
phase accounting precede publication. `configs/<release_name>.json` is the last
readiness marker; existing releases are never replaced. Competition upload is
manual. A new model is not promoted by green synthetic CI.

## Shared feature cache

`artifacts/cache/ml-workflow/v1/<sha256>/` stores complete NPZ features and a
metadata commit record. The key binds extractor/helper/runtime source hashes,
checkpoint bytes, exact fitting prefix including labels, unlabeled target
content/order, feature names, dtype and numerical runtime/device/precision.
Training-only and reporting-only source files are excluded. Cache hits verify
bytes, array schema, shape and dtype; corruption fails rather than silently
recomputing. Per-key locks serialize concurrent producers. Unique staging files
preserve completed entries on failure; an orphan NPZ without metadata is never
a hit. Final independent extraction bypasses this cache.

Each use records hit/miss and elapsed time in the run's `cache-receipts/`. Keep
cache receipts with compact results under `docs/results/<experiment>/`; large
features, models and CSVs remain ignored.

## Minimal next-trial handoff

For a reviewed research adapter, provide the verified 29-file V6 bundle, the
schema-normalized event inputs and matching per-case V6 baseline matrices,
same-cutoff preference initialization assets with their true dimension/epoch,
and the latest closed compute decision. Record each reused asset's producer
revision, cutoff and lineage in the committed experiment protocol alongside its
hash. None of these inputs requires a complete predecessor campaign workspace.
For example, after creating and committing `experiments/next-trial/experiment.json`
and its adapter, prepare exactly with:

```bash
python tools/consolidate.py verify
python tools/run_experiment.py prepare --run-id next-trial-01 --spec experiments/next-trial/experiment.json
```

Use `develop`, `screen`, `stress`, `finalize`, then `verify` with that run ID.
Preparation itself never authorizes restarting a closed recipe or resetting the
research budget. The supplied adapter is synthetic only; implementing and
reviewing a scientific hypothesis remains a separate research change.

## Schema-v2 ensemble postprocessing

Schema-v1 retains the neural/member-replacement interface above. Schema-v2 with
`mode: ensemble_postprocessor` supports a calibration adapter without fake neural
checkpoints. Candidates declare `kind: ensemble_postprocessor` and a frozen
`config` matching the adapter's `CONFIG`. The adapter exposes `user_folds`,
`fit_calibrator`, `load_calibrator` and `predict_calibrator`; predictions receive
no target labels. The controller owns chronology, case metrics, gates and export.

Preparation is budgeted, snapshots committed source, and normalizes event inputs
while preserving official evaluation row IDs. Generated arrays and their hashes
are sealed into the immutable run manifest only after preparation accounting
succeeds. Worker execution requires the active budget-owning controller. Research
continues the shared cumulative ledger and cannot reset it through a new run ID.

User-grouped cross-fitting creates label-safe calibration scores from frozen
neural features and temporary copies of the original CatBoost recipe. Its
content-addressed cache binds features, fitting labels/users/fold assignments,
cutoff, parameters, implementation and runtime. Complete entries retain hashes
for every fitted fold model and score array; corrupted entries fail verification.
Original model/combiner files are never changed. At most two independent CPU
folds run concurrently, preserving each model's fixed parameters and seed.

See the [V7.1 frozen protocol](../experiments/v7.1-entropy-sharpening-v1/RUN_V7_1.md)
for the first such experiment and its explicit asset handoff. It is a distinct
successor to the existing V7 recipe. A failed development or promotion gate closes
the recipe and produces no ready configuration. Publication and independent
release replay retain the shared accounting, provenance and no-replacement rules.


## Schema-v3 joint calibration

Schema-v3 adds candidate-specific saved state for bounded event-wide and
conditional-score calibration. Its explicit contract permits learned event mass
while preserving within-event order and ties. Schema-v1/v2 keep their interfaces
and checks, including schema-v2 mass preservation.

The adapter declares the exact `CANDIDATES` family. `fit_calibrator` receives
independent candidate parameters; `predict_calibrator` receives the saved state;
`load_calibrator` checks metadata and parameters. Saved state includes fitting-only
normalization. The controller independently recomputes that normalization and
binds the candidate identity and parameters to its fitting metadata. Candidate
scoring shares hash-verified temporary development reference combiners.

Selection additionally requires nonnegative aged-development change. One winner
is frozen before the unchanged four promotion gates; there is no runner-up
promotion after a failure. See the [V7.2 protocol](../experiments/v7.2-joint-calibration-v1/RUN_V7_2.md).
General replay uses `python tools/predict_postprocessor.py --release v7_2`;
the existing V7.1 CLI remains compatible. Synthetic final predictions use only
explicitly synthetic baselines and never claim private V6 reproduction or publish
a ready release.
