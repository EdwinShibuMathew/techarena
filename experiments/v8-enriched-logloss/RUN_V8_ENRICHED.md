# Fixed V8 enriched Logloss successor

This is one separately frozen successor to the rejected V8 PairLogit trial.
The hypothesis is that six relative event features improve the seed42 Logloss
combiner's contribution to global row AUC. V6 is the sole comparator. Dates are
reused development windows, not an independent holdout or selection-adjusted
confidence guarantee. Never tune the recipe against these outcomes or the public
leaderboard.

## Recipe

Keep all five neural models, four other combiners, original inference formulas,
and ensemble weights. Replace dim8 seed42 at weight 2/9. Append six float32
features to the original ordered 37 columns: pipeline probability/event max
(floor 1e-8), difference from event mean, event z-score (standard deviation floor
1e-6), zero-based descending average rank with ties, log-odds clipped to
[1e-7, 1-1e-7], and clicks_last_6h/(clicks_last_72h+1). Candidate blocks contain
exactly ten rows. No labels enter feature construction. Original columns remain
byte-identical; invalid names, values, or history counts fail before fitting.

CatBoostClassifier uses Logloss, 545 trees, depth 6, learning rate .05, L2 20,
seed 42, four CPU threads, unit row weights, and all fitting rows. Compatible
sampling, quantization, and leaf settings come from each verified original
combiner and are frozen in the input manifest. There is no evaluation-set early
stopping, probability calibration, blend search, clipping of ensemble scores,
event normalization, or neural retraining.

| Stage | Fitting day / prefix | Target / prefix |
|---|---|---|
| Primary | Jan24 / pre-Jan24 | Jan25–26 first-user / pre-Jan25 |
| Secondary | Jan25 / pre-Jan25 | Jan26 first-user / pre-Jan26 |
| Five-day | Jan21 / pre-Jan21 | Jan22–26 first-user / pre-Jan22 |
| Final | Jan26 / pre-Jan26 | Unlabeled evaluation / pre-Jan27 |

The final replacement is deliberately fitted on Jan26; the protected V6 member
it replaces retains its original saved combiner. This shift is part of the
predeclared recipe and does not alter the benchmark.

## Commands and stopping rule

Use the existing private-asset model environment with CUDA access for original
neural feature generation. Run from the repository root:

```bash
python experiments/v8-enriched-logloss/runner.py prepare
python experiments/v8-enriched-logloss/runner.py screen
python experiments/v8-enriched-logloss/runner.py stress
python experiments/v8-enriched-logloss/runner.py finalize
```

Run each later command only after success of its prerequisite. A failed metric
gate exits 2 and closes this recipe. Exceptions/timeouts also preserve the
attempt and prohibit automatic retries. Do not delete reports or reset budgets.

Preparation verifies the closed PairLogit source and inputs and binds the exact
three reusable caches, event order, cutoff, checkpoint, source, and feature
schema. Newly generated caches and models belong only to this successor's
ignored artifact directory. Reports live in `docs/results/v8-enriched-logloss/`.

Carry all 370.28605625100136 prior seconds against the original 14,400-second
compute budget. Phase caps are prepare 1,800, screen 3,600, stress 3,600, final
fit 3,600, and verify 1,800 seconds, further limited by remaining total time.
Engineering work and human waits are excluded. Hard timers and a persistent
journal include failed attempts.

Promotion requires primary paired whole-user bootstrap 95% CI lower > 0,
secondary Jan26 AUC change >= 0, five-day CI lower > 0, and five-day late
48–120-hour AUC change >= 0. Bootstrap uses 1,000 draws, seed 2026, and at least
990 valid draws. Recompute gates from complete finite evidence, never a passed
flag alone. Failure stops before later competition fits or export.

## Release contract

Only after all four gates pass, fit/save the final classifier and independently
reload it while reconstructing all five V6 members and enriched target features.
Require the canonical V6 CSV SHA256, prediction agreement at rtol=0/atol=1e-15,
and identical candidate CSV fingerprints. Validate 529,390 unique row IDs,
template order, schema, finite probabilities, and [0,1] bounds.

The guarded successor replay command is:

```bash
python tools/predict_v8_enriched.py --device cuda --output artifacts/reproduced_v8.csv
```

It requires a complete `configs/v8.json` ready marker and bound release report;
it never fits a classifier. The old `tools/predict_v8.py` remains the closed
PairLogit recipe's entrypoint and does not support this successor configuration.
Publication uses validated temporary CSVs and no-replace hardlinks; the ready
configuration is published last. An interrupted release without that marker
cannot run inference. Source, input, model, and validation-report inventories
must all match. Model assets and submissions remain ignored/private.

Keep PR #4 in draft unless every gate, independent replay, and required CI check
passes. Do not merge, upload to the competition, or claim a submission exists
when the recipe fails.
