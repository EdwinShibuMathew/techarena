"""Evaluate the predeclared equal three-model ensemble after dim16 training.

Run in techarena-final: python src/evaluate_dim16_trial.py
Trains two fixed 545-tree combiners using past labels; creates no submission.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score


def auc(labels, predictions):
    return float(roc_auc_score(labels.ravel(), predictions.ravel()))


def fit_combiner(X, labels, names):
    model = CatBoostClassifier(
        iterations=545, depth=6, learning_rate=0.05,
        l2_leaf_reg=20, loss_function="Logloss", eval_metric="AUC",
        random_seed=42, thread_count=4, verbose=100,
        allow_writing_files=False,
    )
    model.fit(Pool(X, label=labels.ravel(), feature_names=names))
    assert model.tree_count_ == 545
    return model


def metrics(labels, baseline, dim16):
    candidate = (2 / 3) * baseline + (1 / 3) * dim16
    return {
        "events": len(labels),
        "baseline_two_model_auc": auc(labels, baseline),
        "dim16_single_model_auc": auc(labels, dim16),
        "candidate_three_model_auc": auc(labels, candidate),
        "auc_change": auc(labels, candidate) - auc(labels, baseline),
    }, candidate


def main():
    start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    trial = root.parent / "techarena-dim16-trial"
    output = trial / "outputs/ensemble_trial_results.json"
    if output.exists():
        raise FileExistsError(f"Trial result already exists: {output}")
    protocol = json.loads((trial / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    assert protocol["dimension"] == 16
    assert protocol["neural_seed"] == 42
    assert protocol["combiner_trees"] == 545
    assert protocol["planned_candidate_weights"] == [1 / 3, 1 / 3, 1 / 3]
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    sys.path.insert(0, str(trial / "src"))
    from dim16_pipeline import TARGET_FIELDS, FIT_FIELDS, neural_pipeline, predict_combiner

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
        assert checkpoint["dimension"] == 16 and checkpoint["epoch"] == 196
        del checkpoint
        print(f"Building dimension-16 features for {fold}...", flush=True)
        X, names, _, _, _, _ = neural_pipeline(
            folder / "work/event_best.pt",
            {name: raw[name][fit_idx] for name in FIT_FIELDS},
            {name: raw[name][indices] for name in TARGET_FIELDS},
            folder / "data/cache/preference_training.npz", device,
        )
        return X, names

    with np.load(trial / "jan24/data/cache/split_jan24.npz") as saved:
        jan24_idx = saved["valid_idx"]
    with np.load(root / "work/two_day_diagnostic.npz") as saved:
        primary_idx = saved["valid_idx"]
        primary_labels = saved["labels"]
        primary_baseline = saved["predictions"].reshape(-1, 10)
    with np.load(trial / "jan26/data/cache/split_jan26.npz") as saved:
        secondary_idx = saved["valid_idx"]
    with np.load(root.parent / "techarena-seed43/work/seed_ensemble_validation.npz") as saved:
        assert np.array_equal(saved["valid_idx"], secondary_idx)
        secondary_baseline = saved["predictions"].reshape(-1, 10)
    secondary_labels = raw["y"][secondary_idx]
    assert np.array_equal(primary_labels, raw["y"][primary_idx])
    assert (raw["t"][jan24_idx] >= 1737676800).all()
    assert (raw["t"][jan24_idx] < 1737763200).all()
    assert np.intersect1d(jan24_idx, primary_idx).size == 0

    X24, names = fold_features("jan24", jan24_idx, 1737676800)
    X25, names25 = fold_features("jan25", primary_idx, 1737763200)
    assert names == names25 and len(names) == 37
    print("Fitting the Jan24 combiner for the two-day primary check...", flush=True)
    primary_combiner = fit_combiner(X24, raw["y"][jan24_idx], names)
    primary_dim16 = predict_combiner(primary_combiner, X25, names).reshape(-1, 10)
    primary_combiner.save_model(str(trial / "work/combiner_jan24.cbm"))
    del X24, primary_combiner
    primary_metrics, primary_candidate = metrics(primary_labels, primary_baseline, primary_dim16)
    print(json.dumps({"two_day_primary": primary_metrics}, indent=2), flush=True)

    # Secondary fitting uses only Jan25 labels, excluding all Jan26 rows.
    jan25_mask = raw["t"][primary_idx] < 1737849600
    jan25_idx = primary_idx[jan25_mask]
    with np.load(trial / "jan25/data/cache/split_jan25.npz") as saved:
        assert np.array_equal(jan25_idx, saved["valid_idx"])
    assert np.intersect1d(jan25_idx, secondary_idx).size == 0
    print("Fitting the Jan25 combiner for the Jan26 secondary check...", flush=True)
    secondary_combiner = fit_combiner(
        X25[np.repeat(jan25_mask, 10)], primary_labels[jan25_mask], names
    )
    secondary_combiner.save_model(str(trial / "work/combiner_jan25.cbm"))
    del X25
    X26, names26 = fold_features("jan26", secondary_idx, 1737849600)
    assert names26 == names
    secondary_dim16 = predict_combiner(secondary_combiner, X26, names).reshape(-1, 10)
    del X26, secondary_combiner
    secondary_metrics, secondary_candidate = metrics(secondary_labels, secondary_baseline, secondary_dim16)
    print(json.dumps({"jan26_secondary": secondary_metrics}, indent=2), flush=True)

    rng = np.random.default_rng(2026)
    gains = []
    for repetition in range(300):
        q = rng.integers(0, len(primary_labels), size=len(primary_labels))
        gains.append(auc(primary_labels[q], primary_candidate[q]) - auc(primary_labels[q], primary_baseline[q]))
        if (repetition + 1) % 50 == 0:
            print(f"Paired event bootstrap: {repetition + 1}/300", flush=True)
    interval = np.quantile(gains, [0.025, 0.975]).tolist()
    report = {
        "protocol": {k: v for k, v in protocol.items() if k != "source_sha256"},
        "two_day_primary": primary_metrics,
        "jan26_secondary": secondary_metrics,
        "primary_paired_gain_95_percentile_interval": interval,
        "supported_for_further_confirmation": bool(interval[0] > 0 and secondary_metrics["auc_change"] >= 0),
        "runtime_seconds": round(time.perf_counter() - start, 2),
    }
    np.savez_compressed(trial / "work/two_day_ensemble_predictions.npz", valid_idx=primary_idx, labels=primary_labels, baseline=primary_baseline, dim16=primary_dim16, candidate=primary_candidate)
    np.savez_compressed(trial / "work/jan26_ensemble_predictions.npz", valid_idx=secondary_idx, labels=secondary_labels, baseline=secondary_baseline, dim16=secondary_dim16, candidate=secondary_candidate)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nDIMENSION 16 ENSEMBLE TRIAL", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
