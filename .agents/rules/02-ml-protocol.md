# Rule 02: ML Protocol, Validation, and Promotion Gates

## 1. Problem Formulation & Metric
- **Dataset**: E-commerce impression logs where each event presents 10 item candidates with at most 1 click.
- **Evaluation Metric**: Global row ROC AUC across all 529,390 rows.
- **Benchmark Baseline**: V6 model ensemble with 5 members:
  - 3x dim8 preference models (seeds 42, 43, 44, weight 2/9 each)
  - 2x dim16 models (seeds 42, 43, weight 1/6 each)
  - 37-feature, 545-tree CatBoost combiners

## 2. Experiment Hygiene
- **Folder Isolation**: All new trials must reside under `experiments/<experiment_name>/`.
- **Result Logging**: Record metrics and parameters under `docs/results/<experiment_name>/`.
- **Promotion Gate**:
  1. Primary Gate: Evaluated on first-user Jan25–26 window. Must achieve a strictly positive paired 95% confidence interval over V6 (`lower_bound > 0.0`).
  2. Secondary Gate: Non-negative change on Jan26 window.
  3. Final Gate: Replicate advantage on the 5-day evaluation window.
- **Prohibitions**:
  - Never optimize or tune weights against public leaderboard submissions.
  - Do not treat repeatedly evaluated historical dates as independent holdout sets.
  - Never execute unbounded multi-hour full retraining without explicit user confirmation.
