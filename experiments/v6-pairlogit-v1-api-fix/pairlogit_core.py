"""Bounded global pair sampling and fixed training-only probability link."""

import numpy as np


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    p = np.empty_like(x)
    nonnegative = x >= 0
    p[nonnegative] = 1 / (1 + np.exp(-x[nonnegative]))
    e = np.exp(x[~nonnegative])
    p[~nonnegative] = e / (1 + e)
    return p


def intercept(raw, labels):
    raw, labels = np.asarray(raw, dtype=np.float64).ravel(), np.asarray(labels).ravel()
    if raw.shape != labels.shape or not np.isfinite(raw).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid intercept-fitting inputs")
    ctr = float(labels.mean())
    if not 0 < ctr < 1:
        raise ValueError("Intercept fitting requires both label classes")
    low, high = -float(raw.max()) - 60, -float(raw.min()) + 60
    for _ in range(80):
        mid = (low + high) / 2
        if float(sigmoid(raw + mid).mean()) > ctr:
            high = mid
        else:
            low = mid
    value = (low + high) / 2
    fitted_mean = float(sigmoid(raw + value).mean())
    if abs(fitted_mean - ctr) > 1e-12:
        raise ValueError("Training-only intercept did not converge")
    return value, {"training_row_ctr": ctr, "training_mean_probability": fitted_mean,
                   "bias": value, "slope": 1., "target_labels_used": False,
                   "rule": "mean(sigmoid(raw_train+b)) = fitting_day_CTR; 80 bisections"}


def sample_pairs(labels):
    y = np.asarray(labels).ravel()
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Pair labels must be binary")
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(positive) or len(negative) < 16:
        raise ValueError("Need positives and at least sixteen negatives")
    rng = np.random.default_rng(3042)
    pairs = np.empty((len(positive) * 16, 2), dtype=np.int64)
    for j, winner in enumerate(positive):
        pairs[j * 16:(j + 1) * 16, 0] = winner
        pairs[j * 16:(j + 1) * 16, 1] = rng.choice(negative, 16, replace=False)
    if not ((y[pairs[:, 0]] == 1) & (y[pairs[:, 1]] == 0)).all():
        raise ValueError("Pair sampler produced an invalid pair")
    return pairs


def ranker_parameters(original):
    """Reuse saved compatible settings; never retry a different configuration.

    A tiny synthetic fit in verification catches unsupported installed-version
    combinations before fitting competition labels. Full resolved parameters
    of the original and tiny model are recorded, including objective defaults.
    """
    copied = ("border_count", "feature_border_type", "boosting_type", "grow_policy",
              "bootstrap_type", "bagging_temperature", "subsample", "sampling_frequency",
              "random_strength", "rsm", "leaf_estimation_iterations", "leaf_estimation_method",
              "leaf_estimation_backtracking", "min_data_in_leaf", "score_function",
              "random_score_type", "sampling_unit", "nan_mode", "model_size_reg", "mvs_reg")
    params = {key: original[key] for key in copied if key in original}
    params.update(loss_function="PairLogit", eval_metric="AUC", iterations=545, depth=6,
                  learning_rate=.05, l2_leaf_reg=20, random_seed=42, thread_count=4,
                  task_type="CPU", allow_writing_files=False, use_best_model=False)
    if params.get("boosting_type", "Plain") != "Plain":
        raise ValueError("Saved boosting type is not Plain; inspect before changing the recipe")
    return params


def replacement(v6, old_member, new_member):
    v6, old_member, new_member = [np.asarray(x, dtype=np.float64) for x in (v6, old_member, new_member)]
    if v6.shape != old_member.shape or v6.shape != new_member.shape:
        raise ValueError("Replacement arrays do not align")
    result = v6 + (2 / 9) * (new_member - old_member)
    if not np.isfinite(result).all() or not ((result >= 0) & (result <= 1)).all():
        raise ValueError("Invalid ensemble probabilities; no clipping is permitted")
    return result
