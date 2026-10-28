# V7.4 from raw data

The user authorized reconstructing missing assets from the original raw archives
and approved a new 43,200-second local compute budget. This generation uses
`artifacts/rebuild/compute-ledger.json`; the lost original ledger is not recreated.
The last reported historical usage is unverified and is not a restored balance.

Hypothesis: the predeclared 24-hour or 72-hour causal co-click graph correction
improves the fixed V7 calibration without degrading late-window performance.
Comparators are newly fingerprinted reconstructions of the V6 recipe and its
gamma-1.12 V7 calibration. Results cannot establish superiority to the lost
historical vectors. Historical configs and results remain intact.

Raw preparation streams both archives, checks row/event identities, reconstructs
opaque-ID dictionaries and writes hashed memory-mapped arrays. Each prefix uses
the first observed history of users appearing before its cutoff plus labeled
prefix clicks, preserving repeats. Five members are rebuilt at Jan21, Jan22,
Jan24, Jan25 and Jan26. Fixed schedules and combiner parameters are in recipe.py.
Checkpoints retain Adam state and all RNG state for exact epoch-boundary recovery.
Training source is frozen once for this generation; each phase executes an
immutable committed source snapshot under the cumulative budget controller.

Development compares the two declared variants on Jan22–24 and Jan24 windows,
including the aged subset. Stop if neither meets selection rules. The winner
must pass primary, secondary, five-day and late checks against **both** rebuilt
comparators. Primary and five-day require strictly positive paired 95% confidence
lower bounds; secondary and late point differences must be non-negative.
All reused dates are retrospective checks, not independent holdout evidence.
Gate failure is terminal; no retuning or alternate candidate after failure.

Only a passing candidate permits full-prefix training and independent final
replays. Publication follows successful durable accounting. Until then there is
no submission-ready V7.4 and no modification to official V7.

This generation has completed and was **rejected at development**. All 25 prefix
models were trained, but neither variant passed the aged comparison against
rebuilt V6. Screening and finalization refuse this recorded failure. See the
[terminal report](../../docs/results/v7.4-rebuild-v1/README.md). The commands below
describe the frozen workflow; they do not authorize another scientific attempt.

```bash
python experiments/v7.4-rebuild-v1/run.py prepare
python experiments/v7.4-rebuild-v1/run.py baselines
python experiments/v7.4-rebuild-v1/run.py develop
python experiments/v7.4-rebuild-v1/run.py screen
python experiments/v7.4-rebuild-v1/run.py stress
python experiments/v7.4-rebuild-v1/run.py finalize
python experiments/v7.4-rebuild-v1/run.py verify
```
