---
name: run-experiment
description: >-
  Use this skill when developing, testing, or running a new machine learning experiment, model variant, or comparative study against V6.
---

# Run and Gate New ML Experiments

This skill outlines the strict ML engineering workflow for proposing, running, and evaluating new candidate models against the V6 baseline.

## 1. Directory Structure

Create a dedicated folder for the experiment under `experiments/`:
```text
experiments/<experiment-name>/
├── RUN_<EXPERIMENT_NAME>.md       # Plan, hypothesis, and parameters
├── <model_definition>.py          # Architecture and loss functions
├── <experiment_trial>.py          # Execution script
└── test_<experiment_name>.py     # Unit tests for the custom components
```

## 2. Experiment Execution Rules

1. **Baseline Comparator**:
   All evaluations must directly compare against the saved V6 baseline predictions.
2. **Deterministic Seeds**:
   Always set explicit random generators (`torch.manual_seed`, `np.random.default_rng`).
3. **No Scratch Full Retraining**:
   Initialize from verified epoch-260 checkpoints in `local/archive/` or collected assets. Never train full preference embeddings from scratch without approval.
4. **Target Evaluation**:
   Evaluate on the first-user Jan25–26 window (43,964 events).

## 3. Promotion Gate Verification

Before advancing or proposing any submission:
1. **Primary Gate**:
   Calculate paired difference: $\Delta \text{AUC} = \text{AUC}_{\text{candidate}} - \text{AUC}_{V6}$.
   The paired 95% confidence interval must be strictly positive: $\text{CI}_{\text{lower}} > 0.0$.
2. **Secondary Gate**:
   Verify Jan26 single-day improvement is non-negative ($\Delta \text{AUC}_{\text{Jan26}} \ge 0$).
3. **Log Results**:
   Save output JSON metrics under `docs/results/<experiment-name>/`.
