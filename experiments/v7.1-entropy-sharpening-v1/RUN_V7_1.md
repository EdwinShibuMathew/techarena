# V7.1 entropy-conditioned allocation

This is a single new recipe. Existing V7, its gamma=1.12 transform, configuration
and evidence remain unchanged. V6 is the scientific comparator; existing V7 is
reported descriptively only. No competition upload is automated.

Hypothesis: fitting a two-parameter entropy-conditioned allocation from
user-grouped cross-fitted V6 scores transfers more robustly than fixed sharpening.
The event mass S is an ensemble score sum, not asserted to be a true event click
probability. Set q=p/S, H=-sum(q log q)/log(10), gamma=exp(a+b*(H-.5)); constrain a,b
to [-.25,.25]. Preserve S, within-event ordering, zeros and ties. A fixed 40-step
bisection caps gamma only where necessary to keep output probabilities <=1.
Exported scores are never clipped. Zero-mass events remain zero; identity returns
original values exactly. There are no date, age, history or warm/cold routing rules.

Fit with mean row BCE/mean sampled global pair-logistic loss at weights .5/.5,
plus .01*(a*a+b*b). Sample 16 global negative rows per positive with dedicated
seed 2026, including no-click events. Five deterministic user-grouped folds keep
all ten candidates together. Temporary CatBoost copies use the exact declared
original parameters (545 trees, no early stopping); original neural and combiner
files remain unchanged. Float64 L-BFGS-B starts at identity, maxiter200,
ftol1e-12, gtol1e-8, maxls50. Non-convergence is terminal. No parameter sweep,
promotion-based adjustments or fallback recipe is allowed.

| Case | Calibration day | Target model cutoff | Target end (exclusive) |
|---|---|---|---|
| jan22_long | Jan21 | Jan22 | Jan25 |
| jan24_forward | Jan22 | Jan24 | Jan25 |
| primary | Jan24 | Jan25 | Jan27 |
| secondary | Jan25 | Jan26 | Jan27 |
| five_day | Jan21 | Jan22 | Jan27 |

All dates are January 2025 UTC. Development requires each declared origin gain
>=0 and mean gain>0. Promotion requires primary paired95% lower>0, Jan26 point
change>=0, five-day paired95% lower>0 and late48–120h point change>=0. Use1000
paired whole-user draws, seed2026, >=990 valid draws. These dates are reused
retrospective evidence, not independent holdout or selection-adjusted intervals.

The committed specification freezes the recipe and explicit reused asset hashes.
The imported inventory's producer manifest must match the archived preparation
receipt. No old campaign code is imported. Budgeted preparation checks all inputs,
creates deterministic training row identifiers and preserves official evaluation
row IDs. Generated inputs are hashed into the immutable run manifest. Source
snapshots, controller leases, exclusive accounting and publication-last behavior
are shared with schema-v1; v1 keeps its prior API and behavior.

Use a clean feature checkout and the working private inference environment:

```bash
python experiments/v7.1-entropy-sharpening-v1/prepare_spec.py --asset-root /path/to/preserved-assets --baseline-root /path/to/verified-v6-bundle
# Commit the generated experiment.json and all experiment/runtime source before prepare.
python tools/run_experiment.py prepare --run-id v7.1-entropy-sharpening-v1 --spec experiments/v7.1-entropy-sharpening-v1/experiment.json
python tools/run_experiment.py develop --run-id v7.1-entropy-sharpening-v1
python tools/run_experiment.py screen --run-id v7.1-entropy-sharpening-v1
python tools/run_experiment.py stress --run-id v7.1-entropy-sharpening-v1
python tools/run_experiment.py finalize --run-id v7.1-entropy-sharpening-v1
python tools/run_experiment.py verify --run-id v7.1-entropy-sharpening-v1
```

Stop at the first failed phase. Each phase, including preparation, has a hard
1200-second cap and consumes the shared14400-second cumulative research ledger.
Carry the verified1974.359347462001 seconds; do not reset existing usage. Import
refuses a source with a newer authoritative shared ledger instead of silently
seeding an older carry. Source/runtime hashes are frozen; changing current
runtime requires the recorded snapshot for old evidence, not patched receipts.

Only after all scientific gates pass, finalization reuses the Jan25-fitted
secondary calibrator and performs two fresh exact V6 replays and independent
calibrator/CSV computations. Accounting completes before readiness is published.
The ready release is configs/v7_1.json and artifacts/submissions/v7_1.csv, never
replacing existing files. Replay is separately budgeted and controller-guarded:

```bash
python tools/predict_v7_1.py --output artifacts/submissions/v7_1_replayed.csv
```

Absent or failed gates yield no ready configuration. A verified old V7 CSV or
an improved Huawei rank does not authorize bypassing any gate.
