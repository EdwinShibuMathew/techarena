"""Evaluate longer event training with the original ensemble weights.

Run in techarena-final: python src/evaluate_event24_trial.py
Uses cached dim16 predictions and newly trained dim8 event24 models.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from evaluate_dim16_trial import fit_combiner
from prepare_event24_trial import KEYS


def auc(labels, predictions):
    return float(roc_auc_score(labels.ravel(), predictions.ravel()))


def main():
    start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    projects = root.parent
    trial = projects / "techarena-event24-trial"
    previous = projects / "techarena-dim16-seed43-trial"
    old16 = projects / "techarena-dim16-trial"
    output = trial / "outputs/ensemble_trial_results.json"
    if output.exists():
        raise FileExistsError(f"Trial result already exists: {output}")
    protocol = json.loads((trial / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    assert protocol["event_epochs"] == 24 and protocol["dimension"] == 8
    assert protocol["weights"] == [1 / 3, 1 / 3, 1 / 6, 1 / 6]
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    sys.path.insert(0, str(trial / "src"))
    from event24_pipeline import TARGET_FIELDS, FIT_FIELDS, neural_pipeline, predict_combiner

    with np.load(root / "data/cache/train_events.npz") as saved:
        raw = {name: saved[name] for name in set(TARGET_FIELDS + FIT_FIELDS)}
    with np.load(trial / KEYS[0] / "jan24/data/cache/split_jan24.npz") as saved:
        jan24_idx = saved["valid_idx"]

    references = {}
    for case, new_name, old_name in [
        ("primary", "two_day_ensemble_predictions.npz", "two_day_ensemble_predictions.npz"),
        ("secondary", "jan26_ensemble_predictions.npz", "jan26_ensemble_predictions.npz"),
    ]:
        with np.load(previous / "work" / new_name) as saved:
            indices = saved["valid_idx"]
            labels = saved["labels"]
            baseline = saved["candidate"]
            second16 = saved["new_dim16"]
        with np.load(old16 / "work" / old_name) as saved:
            assert np.array_equal(saved["valid_idx"], indices)
            first16 = saved["dim16"]
            old8_mean = saved["baseline"]
        mean16 = 0.5 * first16 + 0.5 * second16
        assert np.allclose((2 / 3) * old8_mean + (1 / 3) * mean16, baseline, rtol=0, atol=1e-15)
        assert np.array_equal(labels, raw["y"][indices])
        references[case] = dict(indices=indices, labels=labels, baseline=baseline, mean16=mean16)
    primary = references["primary"]
    secondary = references["secondary"]
    assert (raw["t"][jan24_idx] >= 1737676800).all()
    assert (raw["t"][jan24_idx] < 1737763200).all()
    assert np.intersect1d(jan24_idx, primary["indices"]).size == 0
    new_primary, new_secondary = [], []

    def features(key, fold, target_idx, cutoff):
        folder = trial / key / fold
        with np.load(folder / "data/cache" / f"split_{fold}.npz") as saved:
            fit_idx = saved["train_idx"]
        assert (raw["t"][fit_idx] < cutoff).all()
        assert (raw["t"][target_idx] >= cutoff).all()
        assert (raw["t"][target_idx] < 1737936000).all()
        print(f"Building event24 features: {key}, {fold}", flush=True)
        X, names, _, _, _, _ = neural_pipeline(
            folder / "work/event_best.pt",
            {name: raw[name][fit_idx] for name in FIT_FIELDS},
            {name: raw[name][target_idx] for name in TARGET_FIELDS},
            folder / "data/cache/preference_training.npz", device,
        )
        return X, names

    for key in KEYS:
        X24, names = features(key, "jan24", jan24_idx, 1737676800)
        X25, names25 = features(key, "jan25", primary["indices"], 1737763200)
        assert names == names25 and len(names) == 37
        print(f"Fitting Jan24 event24 combiner: {key}", flush=True)
        model = fit_combiner(X24, raw["y"][jan24_idx], names)
        new_primary.append(predict_combiner(model, X25, names).reshape(-1, 10))
        model.save_model(str(trial / "work" / f"{key}_combiner_jan24.cbm"))
        del X24, model
        jan25_mask = raw["t"][primary["indices"]] < 1737849600
        with np.load(trial / key / "jan25/data/cache/split_jan25.npz") as saved:
            assert np.array_equal(primary["indices"][jan25_mask], saved["valid_idx"])
        with np.load(trial / key / "jan26/data/cache/split_jan26.npz") as saved:
            assert np.array_equal(secondary["indices"], saved["valid_idx"])
        assert np.intersect1d(primary["indices"][jan25_mask], secondary["indices"]).size == 0
        print(f"Fitting Jan25 event24 combiner: {key}", flush=True)
        model = fit_combiner(X25[np.repeat(jan25_mask, 10)], primary["labels"][jan25_mask], names)
        model.save_model(str(trial / "work" / f"{key}_combiner_jan25.cbm"))
        del X25
        X26, names26 = features(key, "jan26", secondary["indices"], 1737849600)
        assert names26 == names
        new_secondary.append(predict_combiner(model, X26, names).reshape(-1, 10))
        del X26, model

    def compare(reference, updated):
        mean8 = 0.5 * updated[0] + 0.5 * updated[1]
        candidate = (2 / 3) * mean8 + (1 / 3) * reference["mean16"]
        metrics = {
            "events": len(reference["labels"]),
            "baseline_event8_auc": auc(reference["labels"], reference["baseline"]),
            "candidate_event24_auc": auc(reference["labels"], candidate),
            "auc_change": auc(reference["labels"], candidate) - auc(reference["labels"], reference["baseline"]),
            "event24_dim8_seed42_auc": auc(reference["labels"], updated[0]),
            "event24_dim8_seed43_auc": auc(reference["labels"], updated[1]),
        }
        return metrics, candidate

    primary_metrics, primary_candidate = compare(primary, new_primary)
    secondary_metrics, secondary_candidate = compare(secondary, new_secondary)
    print(json.dumps({"two_day_primary": primary_metrics, "jan26_secondary": secondary_metrics}, indent=2), flush=True)
    rng = np.random.default_rng(2026)
    gains = []
    for repetition in range(300):
        q = rng.integers(0, len(primary["labels"]), size=len(primary["labels"]))
        gains.append(auc(primary["labels"][q], primary_candidate[q]) - auc(primary["labels"][q], primary["baseline"][q]))
        if (repetition + 1) % 50 == 0:
            print(f"Paired event bootstrap: {repetition + 1}/300", flush=True)
    interval = np.quantile(gains, [0.025, 0.975]).tolist()
    report = {
        "protocol": {key: value for key, value in protocol.items() if not key.endswith("sha256")},
        "two_day_primary": primary_metrics,
        "jan26_secondary": secondary_metrics,
        "primary_paired_gain_95_percentile_interval": interval,
        "supported_for_further_confirmation": bool(interval[0] > 0 and secondary_metrics["auc_change"] >= 0),
        "runtime_seconds": round(time.perf_counter() - start, 2),
    }
    for case, reference, candidate, updated in [
        ("two_day", primary, primary_candidate, new_primary),
        ("jan26", secondary, secondary_candidate, new_secondary),
    ]:
        np.savez_compressed(trial / "work" / f"{case}_predictions.npz", valid_idx=reference["indices"], labels=reference["labels"], baseline=reference["baseline"], candidate=candidate, new_dim8_seed42=updated[0], new_dim8_seed43=updated[1])
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nEVENT24 ENSEMBLE TRIAL", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
