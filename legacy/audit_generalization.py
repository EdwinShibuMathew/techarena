"""Compare first-event validation over one versus two days, without retraining.

Run from techarena-final: python src/audit_generalization.py
Uses prefix models trained before Jan 25 and frozen combiners trained on Jan 24.
Jan 25 and Jan 26 have both been used in development; this is a retrospective
diagnostic, not an untouched test or a prediction of private leaderboard AUC.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from build_submission import TARGET_FIELDS, FIT_FIELDS, neural_pipeline, load_combiner, predict_combiner, check_close


def auc_or_none(labels, predictions):
    return float(roc_auc_score(labels, predictions)) if len(np.unique(labels)) == 2 else None


def group_metrics(target, labels, predictions, fit_users, n_users):
    counts = np.bincount(fit_users, minlength=n_users)
    activity = counts[target["u"]]
    history = (target["h"] >= 0).sum(axis=1)
    first_day = target["t"] < 1737849600
    groups = {
        "all_first_events_jan25_to_26": np.ones(len(history), dtype=bool),
        "first_event_on_jan25": first_day,
        "first_event_on_jan26_no_jan25_event": ~first_day,
        "seen_users": activity > 0,
        "unseen_users": activity == 0,
        "history_empty": history == 0,
        "history_1_to_5": (history >= 1) & (history <= 5),
        "history_6_to_19": (history >= 6) & (history <= 19),
        "history_20": history == 20,
        "training_activity_1_to_5": (activity >= 1) & (activity <= 5),
        "training_activity_6_to_15": (activity >= 6) & (activity <= 15),
        "training_activity_16_plus": activity >= 16,
    }
    result = {}
    for name, mask in groups.items():
        if not mask.any():
            continue
        y = labels[mask].ravel()
        p = predictions[mask].ravel()
        result[name] = {
            "events": int(mask.sum()),
            "rows": len(y),
            "clicks": int(y.sum()),
            "auc": auc_or_none(y, p),
            "actual_click_rate": float(y.mean()),
            "mean_prediction": float(p.mean()),
            "mean_history_length": float(history[mask].mean()),
        }
    return result


def population_profile(target, fit_users, n_users, predictions, labels=None):
    counts = np.bincount(fit_users, minlength=n_users)
    activity = counts[target["u"]]
    history = (target["h"] >= 0).sum(axis=1)
    result = {
        "events": len(history),
        "unseen_users": int((activity == 0).sum()),
        "unseen_user_fraction": float((activity == 0).mean()),
        "empty_history_fraction": float((history == 0).mean()),
        "mean_history_length": float(history.mean()),
        "training_activity_0_to_5_fraction": float((activity <= 5).mean()),
        "median_training_events_per_target_user": float(np.median(activity)),
        "mean_prediction": float(predictions.mean()),
    }
    if labels is not None:
        result["actual_click_rate"] = float(labels.mean())
        result["auc"] = auc_or_none(labels.ravel(), predictions.ravel())
    else:
        result["evaluation_labels_available"] = False
    return result


def main():
    start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    projects = root.parent
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    with np.load(root / "data/cache/train_events.npz") as saved:
        raw = {name: saved[name] for name in set(TARGET_FIELDS + FIT_FIELDS)}

    cutoff = 1737763200  # Jan 25 00:00 UTC
    fit_idx = np.flatnonzero(raw["t"] < cutoff)
    eligible = np.flatnonzero((raw["t"] >= cutoff) & (raw["t"] < 1737936000))
    ordered = eligible[np.argsort(raw["t"][eligible], kind="stable")]
    _, first_positions = np.unique(raw["u"][ordered], return_index=True)
    target_idx = np.sort(ordered[first_positions])
    assert len(np.unique(raw["u"][target_idx])) == len(target_idx)
    first_day = raw["t"][target_idx] < 1737849600
    training = {name: raw[name][fit_idx] for name in FIT_FIELDS}
    target = {name: raw[name][target_idx] for name in TARGET_FIELDS}
    labels = raw["y"][target_idx]

    folders = {
        42: projects / "techarena-long320-jan25",
        43: projects / "techarena-seed43-jan25",
    }
    combiner_paths = {
        42: projects / "techarena-confirm-combiner/work/frozen_combiner.cbm",
        43: projects / "techarena-confirm-seed43-combiner/work/frozen_combiner.cbm",
    }
    reference_paths = {
        42: projects / "techarena-confirm-combiner/work/confirmation_predictions.npz",
        43: projects / "techarena-confirm-seed43-combiner/work/confirmation_predictions.npz",
    }
    reference_keys = {42: "combiner_predictions", 43: "seed43_predictions"}
    predictions = []
    seed_metrics = {}

    for seed in [42, 43]:
        folder = folders[seed]
        with np.load(folder / "data/cache/split_jan25.npz") as split:
            assert np.array_equal(fit_idx, split["train_idx"])
            assert np.array_equal(target_idx[first_day], split["valid_idx"])
        print(f"Scoring the two-day validation window with seed {seed}...", flush=True)
        X, names, _, _, _, _ = neural_pipeline(
            folder / "work/event_best.pt", training, target,
            folder / "data/cache/preference_training.npz", device,
        )
        model = load_combiner(combiner_paths[seed])
        p = predict_combiner(model, X, names).reshape(-1, 10)
        with np.load(reference_paths[seed]) as saved:
            assert np.array_equal(saved["valid_idx"], target_idx[first_day])
            check_close(p[first_day].ravel(), saved[reference_keys[seed]].ravel(), "Original Jan 25 confirmation")
        predictions.append(p)
        seed_metrics[str(seed)] = {
            "january25_reproduction_auc": auc_or_none(labels[first_day].ravel(), p[first_day].ravel()),
            "two_day_auc": auc_or_none(labels.ravel(), p.ravel()),
        }
        del X, model

    ensemble = 0.5 * predictions[0] + 0.5 * predictions[1]
    groups = group_metrics(target, labels, ensemble, training["u"], 54456)

    # Compare validation and evaluation populations. Evaluation has no labels.
    with np.load(projects / "techarena-long320/data/cache/split_jan26.npz") as split:
        old_fit_idx, old_valid_idx = split["train_idx"], split["valid_idx"]
    with np.load(projects / "techarena-seed43/work/seed_ensemble_validation.npz") as saved:
        assert np.array_equal(saved["valid_idx"], old_valid_idx)
        original_predictions = saved["predictions"].reshape(-1, 10)
    with np.load(root / "data/cache/evaluation_events.npz") as saved:
        evaluation = {name: saved[name] for name in TARGET_FIELDS}
    with np.load(root / "work/ensemble_evaluation.npz") as saved:
        evaluation_predictions = saved["predictions"].reshape(-1, 10)

    populations = {
        "original_jan26_validation": population_profile(
            {name: raw[name][old_valid_idx] for name in TARGET_FIELDS},
            raw["u"][old_fit_idx], 54456, original_predictions, raw["y"][old_valid_idx],
        ),
        "two_day_validation": population_profile(target, training["u"], 54456, ensemble, labels),
        "official_evaluation": population_profile(evaluation, raw["u"], 54456, evaluation_predictions),
    }
    submission_checks = json.loads((root / "outputs/submission_checks.json").read_text(encoding="utf-8"))
    report = {
        "diagnostic_type": "retrospective_first_event_window_comparison",
        "model_training_before": "2025-01-25",
        "combiner_training_date": "2025-01-24",
        "validation_window": ["2025-01-25", "2025-01-26"],
        "observed_public_auc": 0.814522,
        "observed_public_rank": 24,
        "seed_metrics": seed_metrics,
        "ensemble_groups": groups,
        "population_comparison": populations,
        "submission_checks_passed": bool(submission_checks["submission_ready"]),
        "runtime_seconds": round(time.perf_counter() - start, 2),
    }
    (root / "outputs/generalization_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(root / "work/two_day_diagnostic.npz", valid_idx=target_idx, labels=labels, predictions=ensemble)
    print("\nGENERALIZATION AUDIT")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
