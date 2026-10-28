# V7.5 history and prefix-age combiner

This generation is **closed at development**. A recovered the aged regression
but failed the Jan24 comparison against rebuilt V7; B/C also failed eligibility.
The verified rebuilt-V6 competition CSV was delivered with matching independent
exports and supported replay. See [measured results and delivery](../../docs/results/v7.5-history-age-v1/README.md).
The scientific commands below preserve the failed decision; they do not reopen it.

Hypothesis: graph evidence and real stale-prefix training improve aged transfer
without losing near-window gains. Comparators are rebuilt V6 and fixed-gamma V7.
Historical fingerprints and closed outcomes are preserved. All dates are reused
retrospective evidence, not independent holdout evidence.

The user approved the existing 43,200-second cumulative cap, carrying all
5,048.286446463993 already charged seconds, and requested a verified baseline CSV
even if V7.5 fails. First train five full-prefix neural members, fit the original
Jan25 combiners, independently replay twice, and publish the rebuilt baseline.
Then train additional six-hour, twelve-hour and Jan23 prefix ensembles. Prefix
models only use clicks and first-observed histories available before their cutoff.

The fixed 47-column schema extends the original 37 with two graph affinities,
two centered affinities, log item support, history and edge coverage, log prefix
age capped at 168h, and log prefix event/user counts. Graph parameters are V7.4's
frozen parameters. No hard age/date routing or public-leaderboard tuning.

The three declared variants are A (current-prefix fitting, gamma1.12), B (pooled
prefix fitting, gamma1) and C (B's same fitted models, gamma1.12). Pool fitting
uses each metadata day's first-user cohort with all prefixes no later than that
day and within 96h. Each event's K replicas receive weight1/K per row. All model
features precede their fitting labels. CatBoost uses the fixed original recipe.

Development: Jan22–24 and Jan24, including the aged subset. Eligibility requires
nonnegative gains against both comparators in each window and aged subset, and
positive mean overall gain against each. Select the maximum minimum overall
gain, ties A/B/C. One frozen winner must pass all eight original promotion checks
against rebuilt V6/V7, using1000 paired user bootstraps seed2026/min990. Stop on
failure; no alternate candidate or retuning. Protect3600 seconds for final replay.

Final inference uses the validated Jan25 combiners and full Jan27 neural models
and graph. Publish V7.5 only after independent replay and settled accounting.
CSV529390 rows must match the example IDs/header/order, with finite scores in
[0,1]. Baseline fallback and promoted-candidate receipts remain distinct.

```bash
python experiments/v7.5-history-age-v1/campaign.py baseline
python experiments/v7.5-history-age-v1/campaign.py replay-baseline
python experiments/v7.5-history-age-v1/campaign.py prepare
python experiments/v7.5-history-age-v1/campaign.py develop
python experiments/v7.5-history-age-v1/campaign.py screen
python experiments/v7.5-history-age-v1/campaign.py stress
python experiments/v7.5-history-age-v1/campaign.py finalize
python experiments/v7.5-history-age-v1/campaign.py verify
```
