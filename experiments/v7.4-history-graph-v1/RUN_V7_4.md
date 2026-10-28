# V7.4 causal history-neighbor correction

Status: implemented research protocol; no scientific promotion result yet.
This is a new hypothesis, not a retry of V7.1, V7.2, V7.3 or archived V8 recipes.

Hypothesis: direct item-neighbor evidence in observed click histories adds
information beyond the fixed V7 low-dimensional embedding/combiner ensemble,
improving global row AUC without the historical late-window regression.
The competition starter motivates the graph signal; performance is unproven.

## Frozen recipe

All five V6 neural members, preferences, released combiners, weights and fixed
V7 gamma=1.12 remain unchanged. The V7 anchor is bound to SHA256
`9aebf08df53b9b1d0661479151c37118859831b3b4c9d371d533c3bf203b141a`.
V6 remains an independent required comparator. No public score selects parameters.

For each required prefix, form binary user-item interactions from clicked
candidates and observed history clicks in events strictly before the cutoff.
Include warm-up clicks, deduplicate repeated user-item observations and exclude
non-clicked impressions. Prefix histories must themselves precede their events.
Do not pool target histories into global graph construction. Stable encoded IDs
are opaque keys, never numerical features; deterministic neighbor ties use their
serialized encoding order. Each graph binds the exact encoded prefix arrays.

With shared-user count C and item-user counts Ni,Nj, similarity is
`C/sqrt(Ni*Nj) * C/(C+20)`. Keep C>=3, omit self-links, and retain 64 strongest
neighbors per source item. Construct in sparse blocks of 256 active items;
never allocate a dense 50,000 by 50,000 matrix. Candidate scoring is batched in
2,048 events with vectorized sparse-key lookups.

Two candidates only, in tie order: history_24h, history_72h. Average similarities
against the event's supplied history with weights `2**(-age_hours/half_life)`.
Target history repeats contribute as observed. Missing histories or unsupported
items contribute zero. Center affinities across the ten candidates, divide by
the fitting-only centered RMS (floor 1e-6), and bound features to [-5,5].

For fixed V7 probabilities p, output
`sigmoid(logit(p)+alpha*z+event_offset)`. Learn alpha in [0,.25]. Solve the common
offset by 64 float64 bisection steps, preserving each event's original sum to
absolute tolerance 1e-14, including sums above one. Keep exact zero/one endpoints;
alpha=0 returns fixed V7 bit-for-bit. Candidate ordering may change. Probability
clipping is forbidden; loss-only clipping uses 1e-15 for stable logarithms.

Fit half mean row BCE plus half mean global row-pair logistic loss plus
`.01*alpha**2`, including no-click-event negatives. Sample 16 negatives per
positive with RNG seed 2026. Use float64 L-BFGS-B, maxiter200, ftol1e-12,
gtol1e-8, maxls50, at alpha=0 with an implicit derivative of the mass constraint.
Non-convergence is terminal. Save normalization and independent metadata exactly.
Five user-grouped folds with seed2026 supply cross-fitted V6 combiner scores;
convert them to fixed V7 before fitting. Earlier user histories in the graph are
permitted; metadata-day labels never enter graph construction. Original CatBoost
parameters and tree counts remain fixed.

## Chronology, selection and promotion

| Case | Metadata labels | Target prefix | End exclusive |
| --- | --- | --- | --- |
| jan22_long | Jan21 | Jan22 | Jan25 |
| jan24_forward | Jan22 | Jan24 | Jan25 |
| primary | Jan24 | Jan25 | Jan27 |
| secondary | Jan25 | Jan26 | Jan27 |
| five_day | Jan21 | Jan22 | Jan27 |

All dates are UTC January 2025. All targets use first users with whole events.
Build fitting graphs before the metadata-label day and target graphs before the
target cutoff. Reuse the Jan21 calibrator unchanged for the five-day comparison.

Eligibility requires nonnegative overall changes against both V6 and V7 on both
development origins and nonnegative changes against both on the available aged
48–72h development subset. Mean overall change must be positive for each anchor.
Choose the largest minimum overall gain across anchors/origins; ties within
1e-12 use 24h then 72h. Freeze one winner. No fallback follows a failure.

The explicit `v6_v7_four_gate` policy requires all eight checks: for EACH anchor,
primary and five-day paired 95% lower bounds >0, Jan26 point change>=0, and
five-day late48–120h point change>=0. Use 1,000 whole-user paired draws, seed2026,
at least990 valid draws. All other cohort diagnostics are descriptive. Historical
dates are reused retrospective evidence, not independent or selection-adjusted
holdouts. A competition improvement is not guaranteed by offline acceptance.

Stop at the first eligibility, scientific, integrity, timeout or budget failure.
Do not retune, screen the runner-up or revive a closed recipe.

## Asset prerequisite and execution

The template contains the exact existing asset fingerprints, not fabricated
placeholders. It is not an executable research claim. Restore the verified V6
bundle, chronological prefix inputs and authoritative shared ledger first.
The raw Competition Folder archives cannot replace trained checkpoints.
Asset transport must preserve one authoritative ledger; do not fork live compute
ownership by copying a writable ledger between simultaneously active workspaces.

The latest recorded usage is 4,459.699446460001 seconds. Continue the original
carry identity and 14,400-second cumulative allowance; every phase has at most
1,200 seconds. Reconcile any newer completed attempts. Missing, failed or running
accounting must be resolved before research, never silently reset or refunded.
Preflight and failed missing-asset checks make no claims or ledger writes.

```bash
python experiments/v7.4-history-graph-v1/prepare_spec.py
# Only after restoration and successful read-only checks:
python experiments/v7.4-history-graph-v1/prepare_spec.py --freeze
# Commit the new experiment.json and all source before prepare.
python tools/run_experiment.py prepare --run-id v7.4-history-graph-v1 --spec experiments/v7.4-history-graph-v1/experiment.json
python tools/run_experiment.py develop --run-id v7.4-history-graph-v1
python tools/run_experiment.py screen --run-id v7.4-history-graph-v1
python tools/run_experiment.py stress --run-id v7.4-history-graph-v1
python tools/run_experiment.py finalize --run-id v7.4-history-graph-v1
python tools/run_experiment.py verify --run-id v7.4-history-graph-v1
python tools/predict_postprocessor.py --release v7_4
```

Preparation performs fresh exact V6/V7 anchor verification under its compute
lease before development. Graph caches bind source, prefix arrays/labels/encoded
IDs, cutoff, parameters and numerical runtime. Finalization reuses the validated
Jan25 coefficient/normalization, constructs the pre-Jan27 graph twice without
cache reuse, and requires identical graph arrays, candidate predictions and CSV
hashes. Final private inference also verifies all five V6 members and both anchor
fingerprints independently. Supported replay builds another fresh graph, binding
its files into the replay validation evidence. Accounting and durable replay
ledgers precede public CSV publication; readiness is last.

Only successful research publishes configs/v7_4.json and a separate
artifacts/submissions/v7_4.csv with all eight gate outcomes. Preserve configs/v7.json
and all historical receipts. Update the recommended-model documentation only
after actual promotion. Competition upload remains manual.

## Public implementation validation

The public fixture uses synthetic baselines, actual graph building, scalar
optimization/reload, bootstrap, gates and independent CSV export. It cannot
publish or claim private model reproduction:

```bash
python experiments/v7.4-history-graph-v1/prepare_fixture.py --fixture-id history-smoke
python tools/run_experiment.py prepare --run-id history-smoke --spec artifacts/validation-inputs/history-smoke/experiment.json
python tools/run_experiment.py develop --run-id history-smoke
python tools/run_experiment.py screen --run-id history-smoke
python tools/run_experiment.py stress --run-id history-smoke
python tools/run_experiment.py finalize --run-id history-smoke
python tools/run_experiment.py verify --run-id history-smoke
```
