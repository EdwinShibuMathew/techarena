# V8 robust ensemble selection

One fixed comparison of two compositions, selected using Jan21-only forward
full-ensemble validation. This addresses weak early-window transfer, no-click
negative regressions, and prior selection based on component averages. It does
not guarantee improvement or relax any promotion gate.

## Candidate and reference

- A updates both dim16 members using the original mixed event/global row loss.
- B makes the same updates and reuses the mixed-loss seed44 component.
- Retain V6's dim8 seeds42/43, original dimensions, inference formulas, 37
  features, combiner settings and weights [2/9,2/9,2/9,1/6,1/6].

Dim16 preferences are the reviewed original epoch 196 checkpoints; seed44 uses
epoch 260. Do not retrain preferences or weaken the old validators. New event
models use eight epochs, fresh Adam lr 0.0015, batch 2048, anchor 0.1/effect 0.05,
neural seeds42/43 and separate pair RNGs 3042/3043. The existing mixed loss,
16-negative sampler and no-click-batch fallback are unchanged. Seed44 prefix
models and outputs are reused only after hash/cohort/provenance checks; its
standalone failed decision remains intact.

## Jan21 development

Use first-user Jan21 events and pre-Jan21 neural models only. Fold0 fits
combiners before 08:00 UTC and validates 08:00–16:00; fold1 fits before 16:00 and
validates 16:00–24:00. All ten candidates stay together. Freshly fit all five
reference combiners on each fold's fitting block, never on the full Jan21 day.
The reference is a rolling reconstruction of the V6 recipe, not the cached
Jan22–26 benchmark. Both candidates are scored as complete ensembles.

Qualify only compositions with both fold ensemble gains>=0, positive mean gain,
and mean weighted no-click-negative contribution>=0. Select the largest minimum
fold gain; ties within 1e-12 prefer A, changing fewer members. A larger average
cannot compensate for a weaker minimum. Record clicked/no-click decomposition,
warm/cold/history/day diagnostics and exact cross-group pair contributions.
Conditional group AUCs are descriptive, not an additive global AUC estimate or
a routing rule. Pair contributions must reconstruct global change within 1e-12.

Freeze the winner and development evidence before promotion. Jan22–26 labels
do not select the composition. These dates have nevertheless been reused in
earlier research; no interval is presented as independent or selection-adjusted.

## Commands and stages

From the repository root in the existing CUDA environment:

```bash
python experiments/v8-robust-ensemble/runner.py prepare --source-root /home/edwin/projects/techarena-team
python experiments/v8-robust-ensemble/runner.py development
python experiments/v8-robust-ensemble/runner.py screen
python experiments/v8-robust-ensemble/runner.py stress
python experiments/v8-robust-ensemble/runner.py finalize
```

Run later commands only after success. No eligible composition or failed gate
exits 2 and closes the recipe. Integrity errors/timeouts also preserve the
attempt; no automatic retries, fallback compositions, weight tuning, extra
epochs or deletion of journals/results. No promotion outcomes select new members.

Development trains only the two pre-Jan21 dim16 models. Screen trains the
pre-Jan24/25/26 models and fits Jan24/Jan25 combiners on the existing cohorts.
Stress reuses pre-Jan21 and trains pre-Jan22, fits Jan21 combiners, and evaluates
the original five-day/late cohorts. Reused seed44 predictions retain their
original event order and labels. Final training runs only after all four gates
pass and trains selected pre-Jan27 members, reusing their Jan25 combiners.

Require primary paired 95% lower > 0, secondary Jan26 change>=0, five-day lower > 0,
and late 48–120h change>=0. Use 1000 whole-user paired draws, seed 2026 and at
least 990 valid draws. Preserve all frozen source and baseline checks.

Carry 748.9314011110018 prior seconds against 14400 total. Caps are prepare 600,
development 1800, screen 3000, stress 2400, final 2700, verify 1500 seconds, further
bounded by remaining total. Existing hard timers terminate training process
groups and record failures. Engineering tests/human waits are excluded.

## Verified delivery

```bash
python tools/predict_v8_candidate.py --device cuda --output artifacts/reproduced_v8.csv
```

The CLI never trains or bypasses selection/gates. A release requires complete
source/input/model/selection/report inventories, a canonical five-member V6
replay, independently loaded candidate replay, and identical independent CSV
hashes. Validate exactly 529,390 unique template-ordered row IDs, the required
schema and finite [0,1] probabilities. Use protected-input checks and atomic
no-replace exports; publish configs/v8.json readiness last. Keep models and CSVs
private/ignored. PR #4 becomes review-ready only after real submission replay and
all required source CI pass; no automatic merge or competition upload.

Historical archives, previous trials, V6 configuration, legacy sources, branch
protection and the original dirty checkout remain unchanged.
