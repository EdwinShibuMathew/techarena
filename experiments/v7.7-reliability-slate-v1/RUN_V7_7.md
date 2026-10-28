# V7.7: user reliability and slate evidence

This is an authorized controlled campaign, not a guaranteed empirical result.
The frozen executable specification is `v77_protocol.SPEC`. Historical recipes
remain closed. Raw data, checkpoints, original combiners, V6/V7 configurations,
the original 43,200-second ledger and every prior charge remain intact.

Hypothesis: bounded user reliability and candidate slate corrections improve
global row ROC AUC on first-return users, including old prefixes and cold users.
Comparators: the original rebuilt V6 ensemble and its fixed gamma-1.12 V7 anchor.

Four candidates share identical teacher predictions, fitting populations,
weights, normalization inputs and fixed global negatives: A control event head;
B event head plus reliability; C control plus relative slate head; D both.
No neural retraining, graph coverage mask, direct model-age feature, probability
mass preservation, leaderboard tuning, or post-gate recipe changes are allowed.

Five stable user folds exclude held users from all metadata-day teacher fits.
Use mature neural prefixes at metadata age 0, 24 and 48 hours when available.
Retain only each user's first event since that prefix within the metadata day.
Balance lag-day/six-hour strata and cap each original event's replicated weight
at one. Report unique users/events, effective sample sizes and weighted coverage.
All heads use float64 CPU, zero initialization, bounded tanh corrections and
fitting-only normalization. The specification fixes the optimizer and loss.

Select once on Jan22–24 and Jan24, including the 48–72-hour development subset.
Require nonnegative gains against both anchors everywhere and positive mean
overall gain for each; maximize the worst overall gain, resolving ties A–D.
Then refit only that algorithm at Jan24/25. Stop at the first failed stage.

Promotion requires all eight original paired/point gates plus strict historical
point floors: primary .8187833364842867; secondary .8275321389476477; five-day
.7822476123338514; late .7395732277865564. Their immutable source hashes and
cohort definitions are recorded at preflight. These reused dates and point
thresholds are not independent holdouts or paired historical comparisons.

Release needs three separately owned fresh inference runs, identical CSV bytes,
member agreement at absolute 1e-15, source and data identity, a fixed verification
descriptor with settled accounting, paid verification, and final accounting
extending that seal. Workers only write attempt directories. The controller
registers logical success after settlement. Failed exports retry into fresh
directories; interrupted registration can adopt an already completed attempt.
Resume reuses the original paid verification descriptor/result. CSV and sidecars
precede config/readiness. Different existing outputs are preserved. Upload is manual.

Prepare/develop/promotion ceilings are 5,000/8,000/10,000 seconds within the
original ledger, with a 3,600-second finalization reserve and per-job hard limits
900 seconds for fits, 1,200 otherwise. No new research may begin until the
committed source passes CI and has a matching engineering package.

Commands (production runtime and CUDA required):

```bash
python experiments/v7.7-reliability-slate-v1/v77_runner.py preflight
python experiments/v7.7-reliability-slate-v1/v77_runner.py run
python experiments/v7.7-reliability-slate-v1/v77_runner.py status
python experiments/v7.7-reliability-slate-v1/v77_runner.py verify
```

The campaign writes compact terminal results under
`docs/results/v7.7-reliability-slate-v1/`. Only an artifact-verified successful
campaign publishes `artifacts/submissions/v7_7.csv` and `configs/v7_7.json`.
