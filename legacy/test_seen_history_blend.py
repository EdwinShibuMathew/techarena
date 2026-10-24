"""One fixed diagnostic: use current history for seen, low-activity users.

Run in WSL from techarena-final: python src/test_seen_history_blend.py
No saved model is changed and no submission is written.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from build_submission import (
    TARGET_FIELDS, FIT_FIELDS, neural_pipeline, predict_events,
    adjust_exposure, load_combiner, predict_combiner, check_close,
)
from submission_features import build_features
from train_final_models import PreferenceModel, EventModel


def auc(labels, predictions):
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels.ravel(), predictions.ravel()))


def score_trial(checkpoint_path, positive_path, training, target, combiner_path, device):
    base_X, names, raw_p, pipeline_p, encoder, _ = neural_pipeline(
        checkpoint_path, training, target, positive_path, device
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    n_users = int(checkpoint["n_users"])
    n_items = int(checkpoint["n_items"])
    dimension = int(checkpoint["dimension"])
    state = checkpoint["model_state"]

    activity = np.bincount(training["u"], minlength=n_users)[target["u"]]
    present = target["h"] >= 0
    length = present.sum(axis=1)
    eligible = (activity >= 1) & (activity <= 5) & (length > 0)
    assert eligible.any()
    assert len(np.unique(target["u"])) == len(target["u"])

    # Fit/predict the encoder with the original weights first, as in the baseline.
    # Only the selected seen users receive this fixed, equal profile mixture.
    item_vectors = state["preference.item.weight"].numpy()
    h = target["h"][eligible]
    mask = h >= 0
    history_profiles = (
        item_vectors[h.clip(min=0)] * mask[:, :, None]
    ).sum(axis=1) / mask.sum(axis=1, keepdims=True)
    inferred = encoder.predict(history_profiles)
    selected_users = target["u"][eligible]
    original = state["preference.user.weight"][selected_users].numpy().copy()
    original *= state["preference.active_users"][selected_users].numpy()[:, None]
    blended = 0.5 * original + 0.5 * inferred
    state["preference.user.weight"] = state["preference.user.weight"].clone()
    state["preference.active_users"] = state["preference.active_users"].clone()
    state["preference.user.weight"][selected_users] = torch.tensor(blended, dtype=torch.float32)
    state["preference.active_users"][selected_users] = True

    preference = PreferenceModel(n_users, n_items, dimension, torch.ones(n_users, dtype=torch.bool))
    model = EventModel(preference, n_users, n_items).to(device)
    model.load_state_dict(state)
    model.eval()
    subset = {name: value[eligible] for name, value in target.items()}
    selected_p = predict_events(model, subset, device)
    exposure = np.bincount(training["i"].ravel(), minlength=n_items)
    updated_pipeline = pipeline_p.copy()
    updated_pipeline[eligible] = adjust_exposure(selected_p, target["i"][eligible], exposure)

    # Raw probability remains the original event-model feature. The pipeline
    # probability and user-vector features reflect the proposed profile mixture.
    candidate_X, candidate_names = build_features(
        training, target, state, encoder.coef_, encoder.intercept_,
        n_users, n_items, updated_pipeline, raw_p,
    )
    assert candidate_names == names
    untouched_rows = np.repeat(~eligible, 10)
    assert np.array_equal(base_X[untouched_rows], candidate_X[untouched_rows])
    combiner = load_combiner(combiner_path)
    base = predict_combiner(combiner, base_X, names).reshape(-1, 10)
    candidate = predict_combiner(combiner, candidate_X, candidate_names).reshape(-1, 10)
    assert np.array_equal(base[~eligible], candidate[~eligible])
    del model, preference, checkpoint, base_X, candidate_X
    torch.cuda.empty_cache()
    return base, candidate, eligible


def run_case(name, fit_idx, target_idx, raw, folders, combiner_paths, device, reference_path):
    assert raw["t"][fit_idx].max() < raw["t"][target_idx].min()
    training = {key: raw[key][fit_idx] for key in FIT_FIELDS}
    target = {key: raw[key][target_idx] for key in TARGET_FIELDS}
    labels = raw["y"][target_idx]
    bases, candidates = [], []
    eligible_reference = None
    for seed in [42, 43]:
        print(f"{name}: scoring seed {seed}", flush=True)
        base, candidate, eligible = score_trial(
            folders[seed] / "work/event_best.pt",
            folders[seed] / "data/cache/preference_training.npz",
            training, target, combiner_paths[seed], device,
        )
        if eligible_reference is None:
            eligible_reference = eligible
        assert np.array_equal(eligible_reference, eligible)
        bases.append(base)
        candidates.append(candidate)

    base = 0.5 * bases[0] + 0.5 * bases[1]
    candidate = 0.5 * candidates[0] + 0.5 * candidates[1]
    with np.load(reference_path) as reference:
        assert np.array_equal(reference["valid_idx"], target_idx)
        check_close(base.ravel(), reference["predictions"].ravel(), f"{name} baseline")
    baseline_auc, candidate_auc = auc(labels, base), auc(labels, candidate)
    eligible = eligible_reference
    metrics = {
        "events": len(target_idx),
        "eligible_seen_low_activity_events": int(eligible.sum()),
        "baseline_auc": baseline_auc,
        "candidate_auc": candidate_auc,
        "auc_change": candidate_auc - baseline_auc,
        "eligible_group_baseline_auc": auc(labels[eligible], base[eligible]),
        "eligible_group_candidate_auc": auc(labels[eligible], candidate[eligible]),
        "eligible_group_actual_click_rate": float(labels[eligible].mean()),
        "eligible_group_baseline_mean_prediction": float(base[eligible].mean()),
        "eligible_group_candidate_mean_prediction": float(candidate[eligible].mean()),
        "unselected_predictions_unchanged": bool(np.array_equal(base[~eligible], candidate[~eligible])),
    }
    print(json.dumps({name: metrics}, indent=2), flush=True)
    return metrics, labels, base, candidate, eligible


def main():
    start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    projects = root.parent
    output = root / "outputs/seen_history_blend_trial.json"
    if output.exists():
        raise FileExistsError(f"Trial results already exist: {output}")
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    protocol = {
        "eligible_users": "seen users with 1-5 fitting events and nonempty target history",
        "learned_profile_weight": 0.5,
        "history_profile_weight": 0.5,
        "ensemble_seed_weights": [0.5, 0.5],
        "primary_check": "first event per user across Jan 25-26",
        "secondary_check": "original Jan 26 validation",
        "bootstrap_resamples": 300,
        "bootstrap_seed": 2026,
        "support_rule": "primary paired gain interval entirely above zero and secondary AUC change nonnegative",
        "scope": "retrospective diagnostic; no model files changed and no submission created",
    }
    (root / "outputs/seen_history_blend_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    with np.load(root / "data/cache/train_events.npz") as saved:
        raw = {key: saved[key] for key in set(TARGET_FIELDS + FIT_FIELDS)}
    with np.load(root / "work/two_day_diagnostic.npz") as saved:
        primary_idx = saved["valid_idx"]
    primary_fit = np.flatnonzero(raw["t"] < 1737763200)
    primary = run_case(
        "two_day_primary", primary_fit, primary_idx, raw,
        {42: projects / "techarena-long320-jan25", 43: projects / "techarena-seed43-jan25"},
        {42: projects / "techarena-confirm-combiner/work/frozen_combiner.cbm", 43: projects / "techarena-confirm-seed43-combiner/work/frozen_combiner.cbm"},
        device, root / "work/two_day_diagnostic.npz",
    )
    with np.load(projects / "techarena-long320/data/cache/split_jan26.npz") as saved:
        secondary_fit, secondary_idx = saved["train_idx"], saved["valid_idx"]
    secondary = run_case(
        "jan26_secondary", secondary_fit, secondary_idx, raw,
        {42: projects / "techarena-long320", 43: projects / "techarena-seed43"},
        {42: projects / "techarena-long320/work/combiner_best.cbm", 43: projects / "techarena-seed43/work/combiner_best.cbm"},
        device, projects / "techarena-seed43/work/seed_ensemble_validation.npz",
    )
    primary_metrics, labels, base, candidate, eligible = primary
    secondary_metrics = secondary[0]
    rng = np.random.default_rng(2026)
    gains = []
    for repetition in range(300):
        q = rng.integers(0, len(labels), size=len(labels))
        gains.append(auc(labels[q], candidate[q]) - auc(labels[q], base[q]))
        if (repetition + 1) % 50 == 0:
            print(f"Paired event bootstrap: {repetition + 1}/300", flush=True)
    interval = np.quantile(gains, [0.025, 0.975]).tolist()
    report = {
        "protocol": protocol,
        "two_day_primary": primary_metrics,
        "jan26_secondary": secondary_metrics,
        "primary_paired_gain_95_percentile_interval": interval,
        "sampling_unit": "whole first event per user, preserving all 10 candidates",
        "supported_for_further_confirmation": bool(interval[0] > 0 and secondary_metrics["auc_change"] >= 0),
        "runtime_seconds": round(time.perf_counter() - start, 2),
    }
    work = root / "work/seen_history_blend_trial"
    work.mkdir(exist_ok=True)
    np.savez_compressed(work / "two_day_predictions.npz", valid_idx=primary_idx, labels=labels, baseline=base, candidate=candidate, eligible=eligible)
    np.savez_compressed(work / "jan26_predictions.npz", valid_idx=secondary_idx, labels=secondary[1], baseline=secondary[2], candidate=secondary[3], eligible=secondary[4])
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSEEN HISTORY BLEND TRIAL")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
