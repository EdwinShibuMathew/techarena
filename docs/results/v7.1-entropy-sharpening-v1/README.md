# V7.1 entropy-conditioned allocation: not submission-ready

**Decision: rejected at the late-window gate.** Three scientific gates passed;
the late 48–120h AUC change was negative. Finalization refused before any phase
claim, final inference or submission export. No configs/v7_1.json or V7.1 CSV
was produced. Existing V6/V7 files and decisions remain unchanged.

| Cohort | AUC change vs V6 | Paired 95% interval | Gate |
|---|---:|---|---|
| Primary Jan25–26, 43,964 events | +0.000253392017 | [+0.000135593223, +0.000375645545] | Passed |
| Secondary Jan26, 35,138 events | +0.000299762461 | [+0.000178194607, +0.000414059705] | Passed: point >=0 |
| Five-day Jan22–26, 51,488 events | +0.000208877797 | [+0.000023714297, +0.000373496981] | Passed |
| Late48–120h, 7,596 events | **−0.000483522408** | Point comparison | **Failed** |

Every paired interval used 1,000 valid whole-user draws with seed 2026. Existing
V7's recorded changes were +0.000242447044 primary, +0.000276540602 Jan26,
+0.000267002885 five-day and −0.000167595880 late. V7.1 improves the first two point
estimates slightly but worsens five-day and late results. V7 was descriptive
only; no selection decision used those comparisons or the public leaderboard.

## Frozen method and development

[Protocol and exact commands](../../../experiments/v7.1-entropy-sharpening-v1/RUN_V7_1.md)
fix a two-parameter gamma=exp(a+b*(entropy-.5)) map, bounds [-.25,.25], global
row BCE/pairwise loss, identity regularization and numerical feasibility caps.
The five original neural scorers and released combiners stayed unchanged.
Temporary user-grouped CatBoost copies supplied out-of-fold fitting predictions;
all ten candidates remained together and each user was excluded from its fold's
combiner training. Original V6 parameters, 545 trees and no early stopping were
retained. No new preference or neural model was trained.

Both predeclared development cases passed: Jan22–24 +0.000192078063, Jan24 forward
+0.000111157155. Selection froze the sole eligible recipe before promotion.
The Jan21 calibrator coefficients were [0.1723938825024883, 0.06584918284146862];
that same saved calibrator was reused unchanged for five-day testing.
No coefficients, bounds, loss weights, folds or routing rules changed after
promotion. Development dates and historical benchmark dates have been reused;
these intervals are retrospective, not independent or selection-adjusted proof.

## Integrity, accounting and evidence

- [Source/input preparation](preparation.json)
- [Development and selection](develop.json), [screen](screen.json), [stress](stress.json)
- [Terminal decision and integrity checks](decision.json)

Recorded source is `c64eab4b1e7a42a664900edfabbca1804137b359`, captured in an immutable
source archive. All declared assets and frozen V6 source fingerprints verified.
The original V6 reference vectors reproduced with **zero** maximum error on
primary, Jan26, five-day and the development subset with saved references.
Across every scored case, maximum event-mass error was 4.44e-16, no within-event
order/tie changed, and every probability was finite and in [0,1].

One hundred temporary fold combiners and five development reference combiners
were fitted. Sixty feature/OOF cache requests yielded 16 hits; stress reused the
Jan21 fitting features, cross-fitted scores and saved calibrator. No retired
campaign code was imported; reuse came from the explicitly hash-bound asset
inventory. Evaluation inputs were normalized during preparation, but no final
candidate probabilities were computed and no evaluation labels exist or were
used for fitting.

New scientific compute was **915.106481503 seconds**. Cumulative usage is
**2,889.465828965005 seconds** of the original 14,400-second cap, continuing the
verified 1,974.359347462001-second carry. Each phase had a 1,200-second limit.
Remaining compute does not authorize reopening this rejected recipe.

Locked CPU validation passed 142 tests with one optional private-artifact skip;
source-only validation passed 128 with 15 optional skips. New contracts exercise
numerical edge cases, real fitting/reload, grouped folds, cache corruption,
budgeted preparation, input tampering and unleased/unready refusals. The real
successful schema-v2 release path remains unrun because the late gate failed.
The source/CPU checks do not establish a submission-ready candidate. No new
private full-data CUDA replay or competition upload was performed.

The learned confidence-dependent sharpening improved early aggregate ranking
but did not solve late transfer. This result does not establish a causal reason
for the late loss or justify post-outcome cohort routing. A different scientific
hypothesis requires a separate plan and protocol; this recipe is closed.
