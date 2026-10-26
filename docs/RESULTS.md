# Experiment register

I keep our experiment results here so we can compare changes without repeating
work. The original compact JSON reports are in `docs/results/` where available.
I treat a rejected recipe as evidence about that configuration, not proof that
its entire model family is ineffective.

## Submitted benchmark

V5 public: 0.819402. V6 public: 0.819451. Last reported V6 rank: 27; tenth-place
score in that snapshot: 0.826331. These are historical, not live standings.

## Research candidate: V7 (fixed power sharpening; not promoted)

| Evaluation | Metric / Change vs V6 | Paired 95% Bootstrap Interval | Gate Status |
|---|---:|---|---|
| **Primary (Jan25–26, 43,964 events)** | **+0.0002424470** | **[+0.0001509391, +0.0003354438]** | **PASSED** (Lower > 0) |
| **Secondary (Jan26, 35,138 events)** | **+0.0002765406** | **[+0.0001768461, +0.0003696035]** | **PASSED** (Nonnegative) |
| **Five-Day Stress (Jan22–26, 51,488 events)** | **+0.0002670029** | **[+0.0001555146, +0.0003699263]** | **PASSED** (Lower > 0) |
| **Five-Day Late (48–120h, 7,596 events)** | **−0.0001675959** | Point comparison only | **FAILED** (Change < 0) |

- **Method**: Within-event power sharpening ($\gamma=1.12$) preserving exact event click sums.
- **Submission Artifact**: `artifacts/submissions/v7_candidate.csv` (529,390 rows).
- **Candidate SHA256**: `9aebf08df53b9b1d0661479151c37118859831b3b4c9d371d533c3bf203b141a`.
- **Status**: Not promoted. Overall gains reproduce, but the late-window check fails.
- **Reproduced evidence**: [fixed protocol, source/input hashes, metrics and fingerprint verification](results/v7-validation/validation.json).
- Sharpening preserves within-event ordering; gains concern global row comparisons across events. These are reused development dates, not independent holdout evidence. No model fitting or competition upload was performed.

Our successful steps included preference learning, event training, exposure
adjustment, cold-history fallback, CatBoost combiners, longer preference
training, lower-rate fine-tuning, dimension/seed diversity and the small V6
global row-ranking update.

## V8 PairLogit screen (not promoted)

| Evaluation | AUC change vs V6 | Paired 95% interval | Decision |
|---|---:|---|---|
| Primary Jan25–26 | +0.0000802855 | [−0.0000116083, +0.0001711445] | Failed: interval lower is not positive |
| Secondary Jan26 | +0.0000623605 | [−0.0000407558, +0.0001594924] | Passed: point change is nonnegative |

This fixed trial replaced only the dim8 seed42 combiner at weight 2/9, retaining
the five neural models, 37 features, and other four combiners. The primary gate
failed, so the recipe was closed without five-day testing, final fitting, or
submission export. V6 remains the benchmark. See the [protocol and evidence](results/v8-pairlogit/README.md).

## V8 enriched Logloss successor (not promoted)

| Evaluation | AUC change vs V6 | Paired 95% interval | Decision |
|---|---:|---|---|
| Primary Jan25–26 | +0.0000463279 | [−0.0000192994, +0.0001089003] | Failed: interval lower is not positive |
| Secondary Jan26 | +0.0000496322 | [−0.0000108388, +0.0001105305] | Passed: point change is nonnegative |

One predeclared successor replaced the seed42 combiner with a 43-feature Logloss
classifier, retaining the neural models, other combiners, and 2/9 member weight.
Both original screen members replayed with zero maximum error; all 1,000 paired
bootstrap draws were valid. The primary gate failed, closing this recipe before
five-day testing, final fitting, or submission export. No V8 configuration or CSV
was created. V6 remains the benchmark; PR #4 stays draft as requested.
See the [successor protocol and evidence](results/v8-enriched-logloss/README.md).

## V8 temporal combiner search (selected B; not promoted)

Three predeclared candidates were compared against freshly fitted original
seed42 combiners on Jan24 forward folds. A gained +0.0010613850 then lost
0.0002884511 and was ineligible. B gained +0.0011859932 and +0.0000019671;
C gained +0.0004956737 and +0.0005838487. B had the highest mean gain among
candidates nonnegative on both folds and was frozen before promotion. These
development gains concern the seed42 component, not the V6 ensemble.

| Promotion window | AUC change vs V6 | Paired 95% interval | Decision |
|---|---:|---|---|
| Primary Jan25–26 | −0.0000104611 | [−0.0000734081, +0.0000509818] | Failed |
| Secondary Jan26 | +0.0000109097 | [−0.0000638571, +0.0000860025] | Passed: point change is nonnegative |

B uses 43 features, 545 Logloss trees, depth 4, L2 50, and the unchanged 2/9
replacement weight. Both original comparison members replayed with zero maximum
error and all 1,000 paired draws were valid in each window. No fallback candidate
was screened after failure; five-day testing, final fitting, and export did not
run. V6 remains the benchmark and PR #4 remains draft. See the
[selection protocol and evidence](results/v8-temporal-combiner-search/README.md).

## V8 seed44 row-ranking extension (screen passed; not promoted)

One fixed trial applies V6's existing half-event-CE/half-global-row-pair loss to
seed44, starting from its epoch260 preference checkpoints. All five required
prefix models completed eight event epochs; preferences were not retrained.
The original 37 features, combiner settings, other members and 2/9 replacement
weight remain unchanged. Final inference was specified to reuse the validated
Jan25 combiner, matching V6, without a Jan26 refit.

| Window | AUC change vs V6 | Paired 95% interval | Gate |
|---|---:|---|---|
| Primary Jan25–26 | +0.0000714876 | [+0.0000218718, +0.0001248046] | Passed |
| Secondary Jan26 | +0.0000714093 | [+0.0000151797, +0.0001285389] | Passed |
| Five-day Jan22–26 | +0.0000123114 | [−0.0000492575, +0.0000774763] | Failed |
| Late 48–120h | +0.0001018467 | Point comparison | Passed |

All 1,000 paired draws were valid in each bootstrap window. The five-day lower
bound is not positive, closing the recipe. Actual finalization refused before
training, no full-data seed44 event model or V8 CSV was produced, and PR #4
remains draft. Cumulative V8 compute is 748.931 seconds, including 205.320 for
this trial and refusal verification. V6 remains the submitted benchmark.
See the [protocol, integrity checks and full evidence](results/v8-seed44-row-auc/README.md).

## V8 robust ensemble selection (A selected; not promoted)

This revision addresses earlier weak early-window transfer and selection based
on component averages. Both compositions update the two dim16 members; B also
updates seed44. Selection uses Jan21-only forward folds with pre-Jan21 neural
features and freshly fitted reference combiners for all five members. It uses
complete ensemble AUC, requires a nonnegative mean no-click-negative contribution,
and maximizes the weaker fold rather than the average.

| Composition | First fold ensemble gain | Second fold ensemble gain | Decision |
|---|---:|---:|---|
| A: update dim16 seeds42/43 only | +0.000102899108 | +0.000123615320 | Selected: stronger weaker fold |
| B: same plus seed44 | +0.000094229285 | +0.000159499262 | Eligible, weaker minimum |

Both compositions had positive no-click-negative contributions in both folds.
The winner was frozen before evaluating new promotion outcomes.

| Promotion window | AUC change vs V6 | Paired 95% interval | Gate |
|---|---:|---|---|
| Primary Jan25–26 | +0.000010808166 | [−0.000051946950, +0.000075899505] | Failed |
| Secondary Jan26 | +0.000070042895 | [−0.000004834098, +0.000145198289] | Passed: point change nonnegative |

All 1,000 paired draws were valid. Eight new dim16 prefix models completed eight
epochs from verified epoch196 preferences without preference retraining.
Original dim16 vectors reproduced with zero error in all three historical
cohorts, and all full-data V6 members matched their canonical reference.
The primary gain is insufficient to pass its interval gate. No fallback was
screened, no new stress/full-data fit or CSV export ran, and PR #4 stays draft.
See the [full development, diagnostics and stopping evidence](results/v8-robust-ensemble/README.md).

## V8 residual event (B selected; not promoted)

A bounded campaign compared event-wide corrections on dim8 seed42 (A), event-wide
plus candidate-choice corrections on that member (B), and both heads on all five
members (C). Original event weights and embeddings stayed frozen. Three temporal
origins, Jan21/22/24, included available predictions beyond48h, with fresh reference
combiners fit in each origin's first 8h. All development targets precede Jan25.

| Composition | Jan21 gain | Jan22 gain | Jan24 gain | Decision |
|---|---:|---:|---:|---|
| A | −0.000006838704 | −0.000065255922 | +0.000007491925 | Origin regressions |
| B | +0.000012948099 | +0.000059153058 | +0.000039583501 | Selected |
| C | +0.000018307792 | +0.000259494304 | −0.000094088960 | Jan24 regression |

B alone qualified, including nonnegative mean late-development gain and mean
weighted no-click-negative contribution. Selection was frozen before promotion.

| Window | AUC change | Paired 95% interval | Gate |
|---|---:|---|---|
| Primary Jan25–26 | −0.000012945717 | [−0.000071759803, +0.000039411469] | Failed |
| Secondary Jan26 | +0.000028583175 | [−0.000032561827, +0.000093630479] | Point gate passed |

Both 1,000-draw bootstraps were complete. The primary failure closed this recipe;
actual stress and finalization commands refused before further training or export.
Twenty adapters completed eight epochs each and retained their original base
weights exactly. Zero adapters reproduced all five members with zero probability
error on samples from all three historical cohorts. Full V6 replay remained exact.

No alternative was screened after failure. No V8 configuration or CSV was
produced; PR #4 remains draft. Cumulative compute is 1,974.359347 seconds. Jan22–24
development overlaps the reused five-day dates: this is retrospective evidence,
not independent confirmation. See the [full protocol and evidence](results/v8-residual-event/README.md).

## Recent rejected candidates versus V6

| Fixed experiment | Primary AUC change | Primary paired 95% interval | Jan26 change | Decision |
|---|---:|---|---:|---|
| exposure-matched preference, 85/15 expert blend | -0.0001047702 | [-0.0002941105, +0.0000950952] | +0.0001325543 | reject; no five-day run |
| time-bilinear seed42 replacement | -0.0000960311 | [-0.0001498893, -0.0000362835] | -0.0000791266 | reject; no five-day run |

Both passed reproduction and preservation checks. Neither produced a submission.

Earlier rejected/unsupported recipes: seen-history blending; joint combiner;
24 event epochs; dimension4; pooled-history correction; dim8-only history
integration; standalone attention history expert; all-event equal-user combiner
training; global implicit ALS blend; ALS cold-user routing; past-exposure
aggregate features. See the full handoff and historical plan for details.

## How I want us to choose the next experiment

I want us to audit the implementation and existing evidence before another trial.
We should rank at most three directions, then declare one hypothesis,
configuration, comparator, promotion gate and stopping rule. We keep new
experiments inside this repository and use the positive primary paired-interval /
nonnegative secondary gate followed by the fixed five-day comparison, while
acknowledging adaptive date reuse. We don't select weights against the public
leaderboard or relabel reused dates as an independent holdout.
