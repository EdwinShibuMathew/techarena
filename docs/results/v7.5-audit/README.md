# V7.5 rejection audit and frozen V7.6 design

**V7.5 was correctly rejected at development.** Training and export engineering
succeeded, but no candidate met the frozen development rule against both rebuilt
comparators. The eight subsequent promotion checks were **not run**. There is no
promoted V7.5 model or V7.5 submission. The existing verified rebuilt-V6 CSV remains
available.

The [reproducible audit](audit.json) checks 766 protected inputs, reproduces all
six development comparisons and saved combiner predictions, and records diagnostic
metrics without fitting a new model. Its frozen numerical source is
`2554e7d6367a3d7501b4de4e7d08671320761452`. Later verifier hardening additionally
pins raw/data identities and checks published report copies; it does not change
the frozen analysis or its results. The [controller](../../../experiments/v7.5-audit/audit.py)
publishes after final accounting, and successful reruns reuse the completed audit.

| Evidence category | Outcome |
|---|---|
| Development selection | Rejected; no eligible A/B/C winner |
| Primary, secondary, five-day and late checks versus V6 and V7 | Not run after terminal development rejection |
| Scientific source engineering | Original CI succeeded; this is not promotion evidence |
| Competition CSV | Verified rebuilt-V6 fallback; distinct from historical V6 and from V7.5 |
| V7.6 | Design only; no training, promotion or new CSV |

All dates are reused retrospective data. Evaluation Jan27–Feb2 has no labels.
Neither this audit nor the proposed successor can turn existing dates into an
independent holdout or guarantee that a future candidate passes the gates.

## Measured reason for rejection

| Candidate | Jan22–24 Δ versus rebuilt V6 / V7 | Jan24 Δ versus rebuilt V6 / V7 |
|---|---:|---:|
| A: current-prefix 47 features, gamma 1.12 | +0.000306011 / +0.000037990 | +0.000036400 / **−0.000146431** |
| B: pooled-prefix 47 features, raw | −0.000154184 / −0.000422206 | −0.000334133 / −0.000516965 |
| C: B with gamma 1.12 | +0.000158569 / −0.000109453 | −0.000175803 / −0.000358635 |

A improved the aged cohort against both anchors, recovering V7.4's particular
regression, but its Jan24 comparison against rebuilt V7 was negative. The frozen
rule required nonnegative development and aged differences against both anchors,
with a positive mean against each. B/C also failed. No runner-up screening,
comparator waiver, or public-leaderboard tuning followed rejection.

The fixed post-hoc whole-user bootstrap for A versus V7 gives Jan24 a 95% interval
of **[−0.000339992, +0.000055689]** (1,000 draws, seed 2026). Jan22–24 is
[−0.000142156, +0.000206416]. Both cross zero. The point estimate fails the
predeclared development rule; these diagnostics do not establish a statistically
certain degradation or demonstrate that graph information is inherently harmful.

## What the diagnostics suggest

| Jan24 metric | Rebuilt V7 | A |
|---|---:|---:|
| Global row AUC, official metric | 0.809803498 | 0.809657067 |
| Within-event pair AUC | 0.800323526 | 0.800358912 |
| Click/no-click discrimination using event score sum | 0.664886717 | 0.664409354 |
| Row logloss | 0.193800311 | 0.193865299 |
| Row Brier score | 0.052408543 | 0.052425060 |

Within-event ranking improved slightly while global AUC and both row losses
worsened. Global AUC compares positive rows against negatives in other events
too; improved ordering inside an event cannot certify improvement in the official
metric. The audit decomposes global AUC into clicked-event and no-click-event
negative comparisons, checks reconstruction including ties, and reports warm/cold,
history, repeat-candidate, and prefix-age subgroups. Small subgroups remain noisy.
The sum of row scores can exceed one and is not treated as an event probability.

Preserving V7's event score sum does **not** repair A: the score-only swap of A's
allocation onto V7's sum changes Jan24 AUC by −0.000206445 versus V7. Keeping
V7's allocation with A's sum gives −0.000015201. These nonlinear component swaps
are diagnostics, not causal attribution or exportable candidates. They argue
against imposing event-mass preservation as the default successor fix.

PredictionValuesChange importance on Jan24 current-prefix models assigns about
96.98% to the original 37 columns, 2.49% to graph columns, 0.52% to age, and zero
to prefix-size columns. Pooled models assign about 7.84% to prefix size. This is
descriptive importance, not proof of feature harm. Single-day current-prefix age
also equals time since midnight; it does not independently identify staleness.
Pooling varies prefix volume, neural quality, graph coverage and age together.
Target prefix volume exceeds the fitting prefixes' range. Replicas each carry
1/K weight, so repeated event views contribute total weight one and do not create
extra independent labels.

The audit records all five fixed-weight member replacements/restorations rather
than selecting a favorable member. The existing experiment is an incomplete
factorial. Any later causal feature study needs separately frozen controls:

| Proposed control | Question it would resolve | Status |
|---|---|---|
| 37-feature current versus pooled training | Does stale-prefix pooling help without extra columns? | Requires a new declared experiment |
| Current 47 raw versus sharpened scores | How much is attributable to the shared sharpening transform? | Post-hoc score comparison only |
| Add graph, then age, then size under matched training views | Which feature group changes transfer? | Requires separate training ablations |
| Matched event labels and total weights across views | Is a gain attributable to label duplication or training exposure? | Required control for any future study |

None of these controls was trained by this audit. They cannot reopen V7.5 or be
used to select a new candidate on its already inspected outer dates.

## Frozen next-model proposal

The [V7.6 specification](successor_spec.json) proposes a small, zero-initialized
residual on rebuilt V7, bounded to ±0.25 in row log-odds. It retains the existing
neural models/combiners, uses seven graph and six anchor/context features, and
excludes raw prefix age, calendar date and global prefix counts. Empty-history
or uncovered events return the anchor. There is no event-sum constraint.

The proposed objective combines row BCE, globally sampled row-ranking pairs and
an anchor divergence penalty. Six-hour historical forecasts respect causal
neural, graph and combiner cutoffs. Inner selection excludes validation users
from prior residual fitting records and finishes before the earliest Jan22
outer target. Three fixed divergence penalties are compared only there; one is
frozen before any outer evaluation, or the experiment stops. Normalization uses
fitting records only. No outer hyperparameter updates or runner-up after failure
are allowed. This is a testable hypothesis, not evidence that V7.6 will improve.

The [future promotion verifier](../../../tools/ml_workflow/promotion.py) recomputes
all eight gates from pinned cohort predictions and checks source, model,
accounting and independent export evidence. Its [contract](../../../experiments/v7.5-audit/PROMOTION_CONTRACT.md)
defines the trust boundary: controllers must freeze genuine inputs/receipts;
the verifier does not authenticate arbitrary caller-created manifests. Synthetic
test successes and diagnostic scores cannot establish scientific eligibility.

## Accounting and delivery

The audit charged **64.80082819300151 seconds** to the existing 43,200-second
ledger. Cumulative usage is **8,622.898038783052 seconds**, leaving
**34,577.10196121695 seconds**. All 196 prior attempts are preserved, followed by
one completed audit attempt. [Sealed accounting](accounting.json) and its
[receipt](receipt.json) preserve the exact owner, prefix hash and report hash.
The original V7.4/V7.5 ledgers, reports, models and CSV hashes remain unchanged.

Existing competition file: `artifacts/submissions/rebuilt_v6_submission.csv`,
529,390 official rows, SHA256
`9320ca6465c5ce6576746c5fc8f00d13109e507d7a20d5d506a9e03f63fa2b85`.
Its original independent inference/replay evidence remains valid; no new CSV
was exported and no competition upload occurred.
