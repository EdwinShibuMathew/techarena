# Experiment register

This register summarizes Edwin's reported runs. Original compact JSON reports
are collected into `docs/results/` by the WSL bundler when available. Do not
interpret a rejected recipe as proof that its entire model family is ineffective.

## Submitted benchmark

V5 public: 0.819402. V6 public: 0.819451. Last reported V6 rank: 27; tenth-place
score in that snapshot: 0.826331. These are historical, not live standings.

Successful steps included preference learning, event training, exposure
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

## Future experiment policy

Audit the implementation and existing evidence before another trial. Rank at
most three directions, then declare one hypothesis, configuration, comparator,
promotion gate and stopping rule. Keep new experiments inside this repository.
Use the positive primary paired-interval / nonnegative secondary gate followed
by the fixed five-day comparison, while acknowledging adaptive date reuse.
Do not select weights against the public leaderboard or relabel reused dates
as an independent holdout.
