# V8 temporal combiner search

One bounded development search, one frozen winner, unchanged promotion gates.
The hypothesis is that temporal selection of regularized or separately
calibrated seed42 combiners improves global row AUC versus V6. All neural
weights, other four combiners, and ensemble weights remain frozen. The selected
model replaces dim8 seed42 at weight 2/9. Both previous V8 recipes remain closed.

## Declare and select

Use only the verified Jan24 first-user feature cache from the pre-Jan24 model.
Two UTC forward folds train before 08:00/16:00 and validate 08:00–16:00/
16:00–24:00 respectively. All ten candidates stay together. Jan24 contains
35,040 unique-user events; the three blocks have 12,063, 13,135, and 9,842 events.

Each fold freshly fits the original 37-feature depth6/L2=20 Logloss reference.
Compare exactly three fixed variants:

| ID | Features | Depth | L2 | Additional stage |
|---|---:|---:|---:|---|
| A | Original 37 | 4 | 50 | None |
| B | Checked enriched 43 | 4 | 50 | None |
| C | Original 37 | 6 | 20 | Separate event-shared logit correction |

Every classifier uses 545 trees, learning rate .05, seed 42, four CPU threads,
unit row weights, and compatible hash-bound original sampling/quantization/leaf
settings. No early stopping, target-based tuning, or neural retraining.

For C, fit the base on the earliest 75% of the fitting block and calibrate on
the rest. At the boundary, all tied timestamps enter calibration. Use eleven
event features: raw-margin mean/std/log-sum-exp, softmax entropy, log history
length, user CTR, unseen-user flag, three time-bucket indicators, and day type.
Fit standardization on calibration events only (std below 1e-6 becomes 1).
Minimize mean row Logloss plus .001*sum(weights^2)/2 with an unpenalized
intercept. L-BFGS-B uses analytic gradients, maxiter=200, gtol=1e-8, ftol=1e-12;
nonconvergence rejects C with no retry. Prediction is sigmoid(base raw margin +
shared event offset), without interpreting event sums as probabilities.

Qualify only candidates with both fold AUC gains >=0 and positive mean gain.
Choose the highest mean; ties within 1e-12 prefer A, then B, then C. Save all
development metrics, model/prediction hashes, and an immutable selected-recipe
receipt before promotion. If none qualifies, close the search. Do not fall back
to another candidate after a failed promotion screen.

## Run in the existing private model environment

From the repository root:

```bash
python experiments/v8-temporal-combiner-search/runner.py prepare
python experiments/v8-temporal-combiner-search/runner.py selection
python experiments/v8-temporal-combiner-search/runner.py screen
python experiments/v8-temporal-combiner-search/runner.py stress
python experiments/v8-temporal-combiner-search/runner.py finalize
```

Proceed only when the preceding command succeeds. A failed metric gate or no
qualifying winner exits 2. Exceptions/timeouts preserve the attempt and prohibit
automatic retries. Never delete results or reset budgets. Preparation binds
existing caches and previous manifests, checks all original assets, and freshly
replays all five V6 members and the canonical baseline CSV. New five-day caches
and all model outputs belong only to this experiment's ignored directory.

Carry all 432.33501409900055 previous seconds against the original 14,400-second
limit. Caps: prepare 600, selection 1,800, screen 1,800, stress 2,700, final fit
2,700, verification 1,800 seconds, further limited by remaining total time. Hard
timers include failed attempts. Engineering work and human waits are excluded.

## Promotion and release

The unchanged primary/secondary/five-day/final fitting and prefix cohorts are
recorded in the protocol. A/B use the full fitting day; C always retains its
chronological base/calibration split. Require primary paired 95% CI lower >0,
secondary Jan26 change >=0, five-day CI lower >0, and late 48–120h change >=0.
Bootstrap uses whole users, 1,000 draws, seed 2026, and at least 990 valid draws.
All dates remain reused development evidence, not an independent holdout or
selection-adjusted guarantee.

After all gates pass, independently reload the selected saved recipe and
reconstruct all five V6 members and target features. Require the canonical V6
CSV SHA256, prediction agreement at rtol=0/atol=1e-15, identical V8 CSV hashes,
529,390 unique template-ordered row IDs, and finite [0,1] probabilities.

```bash
python tools/predict_v8_selected.py --device cuda --output artifacts/reproduced_v8.csv
```

This replay requires a fully bound ready `configs/v8.json`; it never fits a
classifier. Complete inventories bind selection, recipe, sources, inputs,
models, correction coefficients when applicable, and validation reports. CSV
publication uses validated temporary files and no-replace hardlinks; the ready
configuration is published last. No gate bypass, automatic merge, or competition
upload is provided. Keep PR #4 draft until all gates, real release replay and
required CI pass. Preserve the original dirty checkout and frozen baseline.
