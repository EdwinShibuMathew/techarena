# My fixed seed42 PairLogit test against V6: API repair

This repair removes an unsupported prediction_type argument from five ranker
prediction calls. The installed ranker returns raw scores directly. My smoke
test now compares them with an explicit raw prediction through the public base
API. My failed code folder and log stay intact; the modeling recipe is unchanged.

I keep V6 as the comparator. Event normalization, attention allocation and the
conditional-choice expert failed their fixed gates. Those results do not prove
that ranking loss will help the final combiner. This test asks whether changing
that combiner's objective, using exactly its original inputs, adds useful global
row ordering to the ensemble.

I replace only dim8 seed42's final Logloss combiner at its existing 2/9 weight.
All neural checkpoints, 37 feature formulas, fallback/exposure rules and four
other ensemble members stay fixed. This is a bounded screen of one member, not
evidence about replacing every combiner.

## Fixed recipe and gate

- CPU CatBoostRanker PairLogit, 545 trees, depth6, learning rate0.05, L2=20,
  seed42, four threads, no early stopping. I copy applicable saved tree/sampling
  settings and test their compatibility on synthetic data before freezing them.
  There is no automatic parameter retry or change on incompatibility.
- Sixteen uniformly sampled global negative rows per positive, without
  replacement per positive, RNG3042. Explicit pairs use one group across the
  fitting day, including negatives from no-click events. I never request an
  automatic all-pairs construction.
- PairLogit scores have no identified absolute offset. I use a fixed unit-slope
  sigmoid and a fitting-day-only intercept, solved by 80 bisections so the mean
  fitting probability equals that day's row CTR. This does not establish target
  calibration; no target label fits the link and no slope is selected.
- Candidate = cached V6 + (2/9) * (new seed42 probability - cached seed42
  probability). No clipping, event normalization or additional blend is used.
  This replacement is allowed to change event sums.
- Primary meta-training: Jan24 first-user rows with pre-Jan24 features. Targets:
  original Jan25-26 first-user cohort with pre-Jan25 features. Secondary
  meta-training uses exactly the Jan25 subset of those primary features; targets
  use the original Jan26 cohort and pre-Jan26 features.
- Global row AUC includes all ten candidates and no-click events. I use 1,000
  paired whole-user resamples, seed2026. Advance only if the primary 95%
  percentile interval has lower bound strictly above zero and secondary point
  change is nonnegative. A pass requires a separate same-recipe five-day review
  before any full-data promotion. A failure closes this recipe without a sweep.
- Verification/preparation plus fitting/comparison share a 45-minute compute
  budget. Time waiting for report review is excluded. There is no neural
  training, evaluation-data access, submission export or automatic promotion.

These development dates have been reused adaptively. Bootstrap intervals are
descriptive and are not adjusted for that selection. They do not establish
qualification or predict a leaderboard gain. Portal timezone and official
selection rules remain unconfirmed.

## What verification must establish

I read the original row-AUC trial's protocol and training/result evidence in
`local/archive/techarena-row-auc-trial/outputs`. The two saved models are:

```
local/archive/techarena-row-auc-trial/work/dim8_seed42_jan24.cbm
SHA256 813544ab4ae8a7a702ed4ef20180f34d231946ef31a890435a6303e8d20341a4
local/archive/techarena-row-auc-trial/work/dim8_seed42_jan25.cbm
SHA256 0382bc46dc70ce3e817e10736291a3d1ab5af88a068994fce1c7d489ca5cdebc
```

I verify those bytes and model settings, the original protocol hash, historical
V6 prediction hashes, relevant legacy source hashes, and three seed42 event
checkpoints against their original training report. I require the original
Jan24 split matching the original protocol's hash; the verifier searches only
that filename inside this repository if its usual location is unavailable.
Missing originals stop preparation. I do not restore deleted archive data or
substitute a newly selected cohort.

The active full click cache supplies derived prefix caches inside this new
experiment only. Each prefix multiset must match the fitting users' initial
histories plus observed prefix clicks, retaining repeats and excluding
future-only users. Original fitting indices, Jan25 meta-subset alignment,
disjoint meta/target events and 37-feature order must agree.

Verification runs GPU inference with the original event checkpoints and fits
the original prefix-only Ridge fallback to reconstruct the features. It does
not update any neural checkpoint or competition combiner. Both original saved
combiners must reproduce their cached seed42 target predictions at rtol0,
atol1e-15. Identity replacement must preserve V6 exactly, with primary AUC
0.8184592533262289 and secondary AUC0.8271277122502162. A mismatch stops the test;
I do not relax tolerance. Verification does not repeat the full five-member V6
replay or repository consolidation.

Preparation freezes feature caches, explicit pairs, original/resolved synthetic
parameters, source/input hashes and the recipe. It scores no new competition
candidate. The run checks those hashes again before fitting and after scoring.

## Exact WSL verification and preparation

The Windows workspace contains all seven source/document files below plus the
checksum manifest. No packages are installed or changed. I activate my existing
environment. The assistant ran five NumPy tests locally; its runtime lacks
CatBoost. All six tests, including the tiny CPU CatBoost fit/save/replay test,
must run here before I accept preparation. The actual saved settings also pass
a synthetic compatibility fit during verification.

```bash
cd /home/edwin/projects/techarena-team
source /home/edwin/miniconda3/etc/profile.d/conda.sh
conda activate techarena

(
  set -euo pipefail
  test "${CONDA_DEFAULT_ENV:-}" = techarena
  export PYTHONDONTWRITEBYTECODE=1
  repo=/home/edwin/projects/techarena-team
  bundle='/mnt/c/Users/Edwin/Documents/ChatGPT/TechArena — ChaiScript/pairlogit-api-fix'
  experiment="$repo/experiments/v6-pairlogit-v1-api-fix"
  prep_log="$repo/artifacts/experiments/v6-pairlogit-v1-prepare-api-fix.log"
  cd "$repo"
  git check-ignore -q artifacts/experiments/v6-pairlogit-v1/probe.npz
  test ! -e "$repo/artifacts/experiments/v6-pairlogit-v1"
  test ! -e "$prep_log"
  mkdir -p "$repo/experiments" "$repo/artifacts/experiments"
  mkdir "$experiment"
  for name in pairlogit_trial.py pairlogit_core.py test_pairlogit.py \
              event_scale_diagnostic.py attention_allocation_test.py \
              conditional_choice_trial.py RUN_PAIRLOGIT.md prepare_pairlogit_api_fix.sh SHA256SUMS_PAIRLOGIT.txt; do
    cp -- "$bundle/$name" "$experiment/"
  done
  cd "$experiment"
  sha256sum -c SHA256SUMS_PAIRLOGIT.txt
  (
    set -euo pipefail
    python -c 'import sys,numpy,torch,sklearn,catboost; print(sys.version); print("NumPy",numpy.__version__,"torch",torch.__version__,"sklearn",sklearn.__version__,"CatBoost",catboost.__version__)'
    python -m unittest discover -s "$experiment" -p test_pairlogit.py -v
    timeout --signal=TERM --kill-after=15s 45m bash -c '
      set -euo pipefail
      python -u "$1/pairlogit_trial.py" verify --repo "$2"
      python -u "$1/pairlogit_trial.py" prepare --repo "$2"
    ' _ "$experiment" "$repo"
  ) 2>&1 | tee "$prep_log"
)
```

All new paths must be absent; existing results are never overwritten. A stop
leaves its partial outputs intact for inspection. I do not automatically rerun
or remove them. If the ignore check fails, I stop and inspect the repository's
ignore rules rather than copy large caches into tracked paths.

## Copy preparation reports into Windows Downloads

I send only these small reports, not model/data/feature files:

```bash
(
  set -euo pipefail
  repo=/home/edwin/projects/techarena-team
  work="$repo/artifacts/experiments/v6-pairlogit-v1"
  files=("$work/verification.json" "$work/inputs.json"
         "$repo/artifacts/experiments/v6-pairlogit-v1-prepare-api-fix.log")
  for file in "${files[@]}"; do test -f "$file"; done
  destination=$(mktemp -d /mnt/c/Users/Edwin/Downloads/pairlogit-preparation.XXXXXX)
  cp -- "${files[@]}" "$destination/"
  wslpath -w "$destination"
)
```

If preparation stops, I copy its log with this command and send that log first:

```bash
(
  set -euo pipefail
  log=/home/edwin/projects/techarena-team/artifacts/experiments/v6-pairlogit-v1-prepare-api-fix.log
  test -f "$log"
  destination=$(mktemp -d /mnt/c/Users/Edwin/Downloads/pairlogit-preparation-error.XXXXXX)
  cp -- "$log" "$destination/"
  wslpath -w "$destination"
)
```

An error before logging starts is copied from the terminal as text. I request
only the specific missing evidence identified by that error.

## Separate fixed run, after preparation is checked

This step performs two CPU competition ranker fits. The internal budget deducts
the recorded verification/preparation compute from 45 minutes. I run this only
after the frozen preparation is accepted, without changing sources or inputs.

```bash
(
  set -euo pipefail
  test "${CONDA_DEFAULT_ENV:-}" = techarena
  export PYTHONDONTWRITEBYTECODE=1
  repo=/home/edwin/projects/techarena-team
  work="$repo/artifacts/experiments/v6-pairlogit-v1"
  test -f "$work/inputs.json"
  test ! -e "$work/run"
  test ! -e "$work/run.log"
  test ! -e "$repo/docs/results/v6-pairlogit-v1/pairlogit.json"
  remaining=$(python -c 'import json,math,sys; r=2700-json.load(open(sys.argv[1]))["preparation_compute_seconds"]; assert r>=1,"Compute budget exhausted"; print(math.floor(r))' "$work/inputs.json")
  timeout --signal=TERM --kill-after=15s "${remaining}s" \
    python -u "$repo/experiments/v6-pairlogit-v1-api-fix/pairlogit_trial.py" run \
      --manifest "$work/inputs.json" 2>&1 | tee "$work/run.log"
)
```

After success I copy the reports to Windows for review:

```bash
(
  set -euo pipefail
  repo=/home/edwin/projects/techarena-team
  work="$repo/artifacts/experiments/v6-pairlogit-v1"
  files=("$work/verification.json" "$work/inputs.json" "$work/run.log"
         "$repo/docs/results/v6-pairlogit-v1/pairlogit.json")
  for file in "${files[@]}"; do test -f "$file"; done
  destination=$(mktemp -d /mnt/c/Users/Edwin/Downloads/pairlogit-results.XXXXXX)
  cp -- "${files[@]}" "$destination/"
  wslpath -w "$destination"
)
```

If the run stops, I use the preparation-error copy block with `log` changed to
`/home/edwin/projects/techarena-team/artifacts/experiments/v6-pairlogit-v1/run.log`.

CatBoost's official references describe [explicit pair fitting](https://catboost.ai/docs/en/concepts/python-reference_catboostranker_fit),
[ranking objectives](https://catboost.ai/docs/en/concepts/loss-functions-ranking)
and [resolved saved parameters](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_get_all_params).
