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
