"""Evaluate a fixed 80% V3 + 20% dimension-4 blend; creates no submission."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from evaluate_dim16_trial import fit_combiner
from prepare_dim16_trial import digest


def auc(labels, predictions):
    return float(roc_auc_score(labels.ravel(), predictions.ravel()))


def metrics(labels, baseline, dim4):
    candidate = 0.8 * baseline + 0.2 * dim4
    return {
        "events": len(labels),
        "baseline_four_model_auc": auc(labels, baseline),
        "dim4_single_model_auc": auc(labels, dim4),
        "candidate_five_model_auc": auc(labels, candidate),
        "auc_change": auc(labels, candidate) - auc(labels, baseline),
    }, candidate


def main():
    start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    trial = root.parent / "techarena-dim4-trial"
    output = trial / "outputs/ensemble_trial_results.json"
    if output.exists():
        raise FileExistsError(f"Trial result already exists: {output}")
    protocol = json.loads((trial / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    assert protocol["dimension"] == 4 and protocol["neural_seed"] == 42
    assert protocol["preference_epochs"] == 196 and protocol["event_epochs"] == 8
    assert protocol["combiner_trees"] == 545 and not protocol["combiner_early_stopping"]
    assert protocol["candidate_baseline_weight"] == 0.8 and protocol["candidate_dim4_weight"] == 0.2
    for name, expected in protocol["source_sha256"].items():
        assert digest(trial / name) == expected, f"Source changed: {name}"
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    sys.path.insert(0, str(trial / "src"))
    from dim4_pipeline import TARGET_FIELDS, FIT_FIELDS, neural_pipeline, predict_combiner

    with np.load(root / "data/cache/train_events.npz") as saved:
        raw = {name: saved[name] for name in set(TARGET_FIELDS + FIT_FIELDS)}

    def fold_features(fold, indices, cutoff):
        folder = trial / fold
        with np.load(folder / "data/cache" / f"split_{fold}.npz") as saved:
            fit_idx = saved["train_idx"]
        assert (raw["t"][fit_idx] < cutoff).all()
        assert (raw["t"][indices] >= cutoff).all()
        assert (raw["t"][indices] < 1737936000).all()
        checkpoint = torch.load(folder / "work/preference_best.pt", map_location="cpu", weights_only=True)
        assert checkpoint["dimension"] == 4 and checkpoint["epoch"] == 196
        del checkpoint
        print(f"Building dimension-4 features for {fold}...", flush=True)
        X, names, _, _, _, _ = neural_pipeline(
            folder / "work/event_best.pt",
            {name: raw[name][fit_idx] for name in FIT_FIELDS},
            {name: raw[name][indices] for name in TARGET_FIELDS},
            folder / "data/cache/preference_training.npz", device,
        )
        return X, names

    baseline_folder = root.parent / "techarena-dim16-seed43-trial/work"

    def reference(filename):
        with np.load(baseline_folder / filename) as saved:
            indices = saved["valid_idx"]
            labels = saved["labels"]
            baseline = saved["candidate"].reshape(-1, 10)
        assert np.array_equal(labels, raw["y"][indices])
        assert baseline.shape == labels.shape
        assert np.isfinite(baseline).all() and ((baseline >= 0) & (baseline <= 1)).all()
        return indices, labels, baseline

    primary_idx, primary_labels, primary_baseline = reference("two_day_ensemble_predictions.npz")
    secondary_idx, secondary_labels, secondary_baseline = reference("jan26_ensemble_predictions.npz")
    # Confirm that the cached reference is the previously validated public-V3 recipe.
    assert abs(auc(primary_labels, primary_baseline) - 0.8162401445033456) < 1e-10
    assert abs(auc(secondary_labels, secondary_baseline) - 0.8256163176107802) < 1e-10
    assert len(np.unique(raw["u"][primary_idx])) == len(primary_idx)
    with np.load(trial / "jan24/data/cache/split_jan24.npz") as saved:
        jan24_idx = saved["valid_idx"]
    with np.load(trial / "jan26/data/cache/split_jan26.npz") as saved:
        assert np.array_equal(secondary_idx, saved["valid_idx"])
    assert (raw["t"][jan24_idx] >= 1737676800).all()
    assert (raw["t"][jan24_idx] < 1737763200).all()
    assert np.intersect1d(jan24_idx, primary_idx).size == 0

    X24, names = fold_features("jan24", jan24_idx, 1737676800)
    X25, names25 = fold_features("jan25", primary_idx, 1737763200)
    assert names == names25 and len(names) == 37
    print("Fitting the Jan24 combiner for the two-day primary check...", flush=True)
    model = fit_combiner(X24, raw["y"][jan24_idx], names)
    primary_dim4 = predict_combiner(model, X25, names).reshape(-1, 10)
    model.save_model(str(trial / "work/combiner_jan24.cbm"))
    del X24, model
    primary_metrics, primary_candidate = metrics(primary_labels, primary_baseline, primary_dim4)

    jan25_mask = raw["t"][primary_idx] < 1737849600
    jan25_idx = primary_idx[jan25_mask]
    with np.load(trial / "jan25/data/cache/split_jan25.npz") as saved:
        assert np.array_equal(jan25_idx, saved["valid_idx"])
    assert np.intersect1d(jan25_idx, secondary_idx).size == 0
    print("Fitting the Jan25 combiner for the Jan26 secondary check...", flush=True)
    model = fit_combiner(X25[np.repeat(jan25_mask, 10)], primary_labels[jan25_mask], names)
    model.save_model(str(trial / "work/combiner_jan25.cbm"))
    del X25
    X26, names26 = fold_features("jan26", secondary_idx, 1737849600)
    assert names26 == names
    secondary_dim4 = predict_combiner(model, X26, names).reshape(-1, 10)
    del X26, model
    secondary_metrics, secondary_candidate = metrics(secondary_labels, secondary_baseline, secondary_dim4)
    print(json.dumps({"two_day_primary": primary_metrics, "jan26_secondary": secondary_metrics}, indent=2), flush=True)

    rng = np.random.default_rng(protocol["bootstrap_seed"])
    gains = []
    for repetition in range(protocol["bootstrap_resamples"]):
        q = rng.integers(0, len(primary_labels), size=len(primary_labels))
        gains.append(auc(primary_labels[q], primary_candidate[q]) - auc(primary_labels[q], primary_baseline[q]))
        if (repetition + 1) % 50 == 0:
            print(f"Paired event bootstrap: {repetition + 1}/{protocol['bootstrap_resamples']}", flush=True)
    interval = np.quantile(gains, [0.025, 0.975]).tolist()
    report = {
        "protocol": {k: v for k, v in protocol.items() if k != "source_sha256"},
        "two_day_primary": primary_metrics,
        "jan26_secondary": secondary_metrics,
        "primary_paired_gain_95_percentile_interval": interval,
        "sampling_unit": "whole first event per user, preserving all 10 candidates",
        "supported_for_further_confirmation": bool(interval[0] > 0 and secondary_metrics["auc_change"] >= 0),
        "runtime_seconds": round(time.perf_counter() - start, 2),
    }
    for case, idx, labels, baseline, dim4, candidate in [
        ("two_day", primary_idx, primary_labels, primary_baseline, primary_dim4, primary_candidate),
        ("jan26", secondary_idx, secondary_labels, secondary_baseline, secondary_dim4, secondary_candidate),
    ]:
        np.savez_compressed(trial / "work" / f"{case}_ensemble_predictions.npz",
                            valid_idx=idx, labels=labels, baseline=baseline, dim4=dim4, candidate=candidate)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nDIMENSION 4 ENSEMBLE TRIAL", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
