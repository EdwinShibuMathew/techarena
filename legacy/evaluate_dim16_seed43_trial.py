"""Test a second dim16 seed, preserving the public v2 dimension weights.

Run in techarena-final: python src/evaluate_dim16_seed43_trial.py
Weights are [1/3, 1/3, 1/6, 1/6] for dim8_42, dim8_43, dim16_42, dim16_43.
Creates retrospective diagnostic results and combiners, not a submission.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from evaluate_dim16_trial import fit_combiner


def auc(labels, predictions):
    return float(roc_auc_score(labels.ravel(), predictions.ravel()))


def compare(labels, base8, old16, baseline, new16):
    # Keep total dim8 weight at 2/3 and total dim16 weight at 1/3.
    candidate = (2 / 3) * base8 + (1 / 3) * (0.5 * old16 + 0.5 * new16)
    expected_baseline = (2 / 3) * base8 + (1 / 3) * old16
    assert np.allclose(expected_baseline, baseline, rtol=0, atol=1e-15)
    return {
        "events": len(labels),
        "baseline_three_model_auc": auc(labels, baseline),
        "dim16_seed43_single_auc": auc(labels, new16),
        "candidate_four_model_auc": auc(labels, candidate),
        "auc_change": auc(labels, candidate) - auc(labels, baseline),
    }, candidate


def main():
    start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    trial = root.parent / "techarena-dim16-seed43-trial"
    previous = root.parent / "techarena-dim16-trial"
    output = trial / "outputs/ensemble_trial_results.json"
    if output.exists():
        raise FileExistsError(f"Trial result already exists: {output}")
    protocol = json.loads((trial / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    assert protocol["dimension"] == 16 and protocol["neural_seed"] == 43
    assert protocol["planned_candidate_weights"] == [1 / 3, 1 / 3, 1 / 6, 1 / 6]
    assert protocol["combiner_trees"] == 545
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    sys.path.insert(0, str(trial / "src"))
    from dim16_pipeline import TARGET_FIELDS, FIT_FIELDS, neural_pipeline, predict_combiner

    with np.load(root / "data/cache/train_events.npz") as saved:
        raw = {name: saved[name] for name in set(TARGET_FIELDS + FIT_FIELDS)}
    with np.load(trial / "jan24/data/cache/split_jan24.npz") as saved:
        jan24_idx = saved["valid_idx"]
    with np.load(previous / "work/two_day_ensemble_predictions.npz") as saved:
        primary_idx = saved["valid_idx"]
        primary_labels = saved["labels"]
        primary_base8 = saved["baseline"]
        primary_old16 = saved["dim16"]
        primary_baseline = saved["candidate"]
    with np.load(previous / "work/jan26_ensemble_predictions.npz") as saved:
        secondary_idx = saved["valid_idx"]
        secondary_labels = saved["labels"]
        secondary_base8 = saved["baseline"]
        secondary_old16 = saved["dim16"]
        secondary_baseline = saved["candidate"]
    assert np.array_equal(primary_labels, raw["y"][primary_idx])
    assert np.array_equal(secondary_labels, raw["y"][secondary_idx])

    def features(fold, target_idx, cutoff):
        folder = trial / fold
        with np.load(folder / "data/cache" / f"split_{fold}.npz") as saved:
            fit_idx = saved["train_idx"]
        assert (raw["t"][fit_idx] < cutoff).all()
        assert (raw["t"][target_idx] >= cutoff).all()
        assert (raw["t"][target_idx] < 1737936000).all()
        checkpoint = torch.load(folder / "work/preference_best.pt", map_location="cpu", weights_only=True)
        assert checkpoint["dimension"] == 16 and checkpoint["epoch"] == 196
        del checkpoint
        print(f"Building seed43 dimension-16 features for {fold}...", flush=True)
        X, names, _, _, _, _ = neural_pipeline(
            folder / "work/event_best.pt",
            {name: raw[name][fit_idx] for name in FIT_FIELDS},
            {name: raw[name][target_idx] for name in TARGET_FIELDS},
            folder / "data/cache/preference_training.npz", device,
        )
        return X, names

    assert np.intersect1d(jan24_idx, primary_idx).size == 0
    assert (raw["t"][jan24_idx] < 1737763200).all()
    X24, names = features("jan24", jan24_idx, 1737676800)
    X25, names25 = features("jan25", primary_idx, 1737763200)
    assert names == names25 and len(names) == 37
    print("Fitting the Jan24 combiner...", flush=True)
    model = fit_combiner(X24, raw["y"][jan24_idx], names)
    primary_new16 = predict_combiner(model, X25, names).reshape(-1, 10)
    model.save_model(str(trial / "work/combiner_jan24.cbm"))
    del X24, model
    primary_metrics, primary_candidate = compare(primary_labels, primary_base8, primary_old16, primary_baseline, primary_new16)
    print(json.dumps({"two_day_primary": primary_metrics}, indent=2), flush=True)

    jan25_mask = raw["t"][primary_idx] < 1737849600
    with np.load(trial / "jan25/data/cache/split_jan25.npz") as saved:
        assert np.array_equal(primary_idx[jan25_mask], saved["valid_idx"])
    with np.load(trial / "jan26/data/cache/split_jan26.npz") as saved:
        assert np.array_equal(secondary_idx, saved["valid_idx"])
    assert np.intersect1d(primary_idx[jan25_mask], secondary_idx).size == 0
    print("Fitting the Jan25 combiner...", flush=True)
    model = fit_combiner(X25[np.repeat(jan25_mask, 10)], primary_labels[jan25_mask], names)
    model.save_model(str(trial / "work/combiner_jan25.cbm"))
    del X25
    X26, names26 = features("jan26", secondary_idx, 1737849600)
    assert names26 == names
    secondary_new16 = predict_combiner(model, X26, names).reshape(-1, 10)
    del X26, model
    secondary_metrics, secondary_candidate = compare(secondary_labels, secondary_base8, secondary_old16, secondary_baseline, secondary_new16)
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
    np.savez_compressed(trial / "work/two_day_ensemble_predictions.npz", valid_idx=primary_idx, labels=primary_labels, baseline=primary_baseline, new_dim16=primary_new16, candidate=primary_candidate)
    np.savez_compressed(trial / "work/jan26_ensemble_predictions.npz", valid_idx=secondary_idx, labels=secondary_labels, baseline=secondary_baseline, new_dim16=secondary_new16, candidate=secondary_candidate)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSECOND DIMENSION-16 SEED TRIAL", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
