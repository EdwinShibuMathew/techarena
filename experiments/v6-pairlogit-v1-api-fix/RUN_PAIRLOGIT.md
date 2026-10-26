# Fixed PairLogit Combiner Trial

This trial replaces only the dim8 seed42 combiner at its existing 2/9 ensemble
weight. It keeps the five neural models, four other combiners, 37 features,
exposure adjustment and fallback fixed. The recipe asks whether a global
pairwise combiner improves row ordering; it does not promise a leaderboard gain.

## Recipe and gates

- CPU CatBoost PairLogit: 545 trees, depth 6, learning rate 0.05, L2 20, seed
  42, four threads. Use the existing explicit sampler: 16 uniform global
  negatives per positive, seed 3042, including no-click events.
- Use the fixed unit-slope sigmoid and fitting-day-only intercept. No target
  labels fit the link. No feature, pair-count, tree, weight or calibration
  sweeps.
- Primary: Jan24 meta-training and first-user Jan25–26 targets. Secondary:
  Jan25 meta-training and first-user Jan26 targets. Require primary paired 95%
  interval lower bound > 0 and secondary AUC change >= 0.
- Use 1,000 paired whole-user bootstrap draws with seed 2026. The inspected
  cohorts contain one event per user; the runner must preserve that check.
- This fixed run has a 45-minute compute limit. A pass only allows a separate
  earlier-cutoff five-day review; it does not authorize submission.

The five-day runner and full-data fitting/export stage are not implemented in
this folder. A passing short-window screen must remain a research result until
those stages are added, their earlier-cutoff inputs are hash-verified, and the
same fixed recipe passes the five-day gates.

Dates are reused development data, so intervals are descriptive and not
selection-adjusted.

## Explicit input staging

The runner reads no `local/` paths and performs no repository walk. Before
verification, place the original, already hash-verified study inputs under:

```text
artifacts/experiments/v6-pairlogit-v1-inputs/source/
  outputs/frozen_protocol.json
  outputs/training_results.json
  outputs/ensemble_trial_results.json
  work/dim8_seed42_jan24.cbm
  work/dim8_seed42_jan25.cbm
  work/two_day_ensemble_predictions.npz
  work/jan26_ensemble_predictions.npz
  dim8_seed42/jan24/work/event_best.pt
  dim8_seed42/jan25/work/event_best.pt
  dim8_seed42/jan26/work/event_best.pt
  dim8_seed42/jan24/src/train_events.py
  dim8_seed42/jan24/src/row_auc_objective.py
  dim8_seed42/jan25/src/train_events.py
  dim8_seed42/jan25/src/row_auc_objective.py
  dim8_seed42/jan26/src/train_events.py
  dim8_seed42/jan26/src/row_auc_objective.py
  data/cache/split_jan24.npz
```

The frozen protocol, checkpoints, combiners, prediction arrays, split and
source files are checked against their recorded hashes. Missing or mismatched
files stop the run; do not substitute regenerated data. The prepared verification
also requires CUDA to replay the original feature pipeline.

## WSL commands

Run from the repository in the existing `techarena` environment with the RTX
4070 available. This agent did not run training or modify model artifacts.

```bash
cd /home/edwin/projects/techarena-team
source /home/edwin/miniconda3/etc/profile.d/conda.sh
conda activate techarena
repo=/home/edwin/projects/techarena-team
experiment="$repo/experiments/v6-pairlogit-v1-api-fix"
source_dir="$repo/artifacts/experiments/v6-pairlogit-v1-inputs/source"

python -u "$experiment/pairlogit_trial.py" verify \
  --repo "$repo" --source-dir "$source_dir"
python -u "$experiment/pairlogit_trial.py" prepare --repo "$repo"
python -u "$experiment/pairlogit_trial.py" run \
  --manifest "$repo/artifacts/experiments/v6-pairlogit-v1/inputs.json"
```

`verify` and `prepare` create new files under ignored `artifacts/`; `run` writes
its result only if the output path does not already exist. Review the report and
input hashes before any five-day follow-up. No upload or full-data fit is part
of these commands.
