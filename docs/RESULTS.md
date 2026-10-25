# Experiment register

I keep our experiment results here so we can compare changes without repeating
work. The original compact JSON reports are in `docs/results/` where available.
I treat a rejected recipe as evidence about that configuration, not proof that
its entire model family is ineffective.

## Submitted benchmark

V5 public: 0.819402. V6 public: 0.819451. Last reported V6 rank: 27; tenth-place
score in that snapshot: 0.826331. These are historical, not live standings.

## Promoted candidate: V7 (Calibrated Event-Scale Ranking)

| Evaluation | Metric / Change vs V6 | Paired 95% Bootstrap Interval | Gate Status |
|---|---:|---|---|
| **Primary (Jan25–26, 43,964 events)** | **+0.0002424470** | **[+0.0001509391, +0.0003354438]** | **PASSED** (Lower > 0) |
| **Secondary (Jan26, 35,138 events)** | **+0.0002765406** | **[+0.0001768461, +0.0003696035]** | **PASSED** (Nonnegative) |
| **Five-Day Stress (Jan22–26, 51,488 events)** | **+0.0002670029** | **[+0.0001555146, +0.0003699263]** | **PASSED** (Lower > 0) |

- **Method**: Within-event power sharpening ($\gamma=1.12$) preserving exact event click sums.
- **Submission Artifact**: `artifacts/submissions/v7_candidate.csv` (529,390 rows).
- **Candidate SHA256**: `9aebf08df53b9b1d0661479151c37118859831b3b4c9d371d533c3bf203b141a`.
- **Status**: Passes all primary, secondary, and five-day stress gates; promoted for competition upload.

Our successful steps included preference learning, event training, exposure
adjustment, cold-history fallback, CatBoost combiners, longer preference
training, lower-rate fine-tuning, dimension/seed diversity and the small V6
global row-ranking update.

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
