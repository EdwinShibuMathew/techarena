"""Evaluate the fixed three-dim8-seed ensemble against public V4.

Run in techarena-final: python src/evaluate_seed44_trial.py
Fits two fixed 545-tree combiners; creates no competition submission.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from build_submission import TARGET_FIELDS, FIT_FIELDS, neural_pipeline, predict_combiner
from evaluate_dim16_trial import fit_combiner
from prepare_seed44_trial import verify_inputs


def auc(labels, predictions):
    return float(roc_auc_score(labels.ravel(), predictions.ravel()))


def candidate_probabilities(first8, second8, new8, mean16):
    members = [np.asarray(p, dtype=np.float64) for p in [first8, second8, new8, mean16]]
    assert all(p.shape == members[0].shape for p in members)
    for p in members:
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return (2 / 3) * ((members[0] + members[1] + members[2]) / 3) + (1 / 3) * members[3]


def compare(reference, new8):
    candidate = candidate_probabilities(reference["first8"], reference["second8"], new8, reference["mean16"])
    labels, baseline = reference["labels"], reference["baseline"]
    return {
        "events": len(labels), "baseline_v4_auc": auc(labels, baseline),
        "seed44_single_model_auc": auc(labels, new8),
        "candidate_five_model_auc": auc(labels, candidate),
        "auc_change": auc(labels, candidate) - auc(labels, baseline),
    }, candidate


def main():
    start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    projects = root.parent
    trial = projects / "techarena-seed44-trial"
    previous = projects / "techarena-preference-finetune-trial"
    output = trial / "outputs/ensemble_trial_results.json"
    if output.exists():
        raise FileExistsError(f"Trial result already exists: {output}")
    protocol = json.loads((trial / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    assert protocol["dimension"] == 8 and protocol["neural_seed"] == 44
    assert protocol["weights"] == [2 / 9, 2 / 9, 2 / 9, 1 / 6, 1 / 6]
    assert protocol["preference_total_epochs"] == 260 and protocol["event_epochs"] == 8
    assert protocol["combiner_trees"] == 545 and not protocol["combiner_early_stopping"]
    verify_inputs(trial, protocol)
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    with np.load(root / "data/cache/train_events.npz") as saved:
        raw = {name: saved[name] for name in set(TARGET_FIELDS + FIT_FIELDS)}
    references = {}
    for case, filename, expected_auc in [
        ("primary", "two_day", 0.8168650717272858),
        ("secondary", "jan26", 0.8260509325478992),
    ]:
        with np.load(previous / "work" / f"{filename}_predictions.npz") as saved:
            indices, labels = saved["valid_idx"], saved["labels"]
            baseline, first8, second8 = saved["candidate"], saved["new_dim8_seed42"], saved["new_dim8_seed43"]
        with np.load(projects / "techarena-dim16-trial/work" / f"{filename}_ensemble_predictions.npz") as saved:
            assert np.array_equal(saved["valid_idx"], indices)
            first16 = saved["dim16"]
        with np.load(projects / "techarena-dim16-seed43-trial/work" / f"{filename}_ensemble_predictions.npz") as saved:
            assert np.array_equal(saved["valid_idx"], indices)
            second16 = saved["new_dim16"]
        mean16 = 0.5 * first16 + 0.5 * second16
        assert np.allclose((2 / 3) * (0.5 * first8 + 0.5 * second8) + (1 / 3) * mean16, baseline, rtol=0, atol=1e-15)
        assert np.array_equal(labels, raw["y"][indices])
        assert len(np.unique(raw["u"][indices])) == len(indices)
        assert abs(auc(labels, baseline) - expected_auc) < 1e-10
        references[case] = dict(indices=indices, labels=labels, baseline=baseline,
                                first8=first8, second8=second8, mean16=mean16)
    primary, secondary = references["primary"], references["secondary"]
    with np.load(trial / "jan24/data/cache/split_jan24.npz") as saved:
        jan24_idx = saved["valid_idx"]
    assert (raw["t"][jan24_idx] >= 1737676800).all() and (raw["t"][jan24_idx] < 1737763200).all()
    assert np.intersect1d(jan24_idx, primary["indices"]).size == 0

    def features(fold, indices, cutoff):
        folder = trial / fold
        with np.load(folder / "data/cache" / f"split_{fold}.npz") as saved:
            fit_idx = saved["train_idx"]
        assert (raw["t"][fit_idx] < cutoff).all()
        assert (raw["t"][indices] >= cutoff).all() and (raw["t"][indices] < 1737936000).all()
        checkpoint = torch.load(folder / "work/preference_best.pt", map_location="cpu", weights_only=True)
        assert checkpoint["dimension"] == 8 and checkpoint["epoch"] == 260
        assert checkpoint["fine_tuning_sampling_seed"] == 1044
        del checkpoint
        print(f"Building seed44 features for {fold}...", flush=True)
        X, names, _, _, _, _ = neural_pipeline(
            folder / "work/event_best.pt",
            {name: raw[name][fit_idx] for name in FIT_FIELDS},
            {name: raw[name][indices] for name in TARGET_FIELDS},
            folder / "data/cache/preference_training.npz", device,
        )
        return X, names

    X24, names = features("jan24", jan24_idx, 1737676800)
    X25, names25 = features("jan25", primary["indices"], 1737763200)
    assert names == names25 and len(names) == 37
    print("Fitting the Jan24 seed44 combiner...", flush=True)
    model = fit_combiner(X24, raw["y"][jan24_idx], names)
    primary_new = predict_combiner(model, X25, names).reshape(-1, 10)
    model.save_model(str(trial / "work/combiner_jan24.cbm"))
    del X24, model
    primary_metrics, primary_candidate = compare(primary, primary_new)
    jan25_mask = raw["t"][primary["indices"]] < 1737849600
    with np.load(trial / "jan25/data/cache/split_jan25.npz") as saved:
        assert np.array_equal(primary["indices"][jan25_mask], saved["valid_idx"])
    with np.load(trial / "jan26/data/cache/split_jan26.npz") as saved:
        assert np.array_equal(secondary["indices"], saved["valid_idx"])
    assert np.intersect1d(primary["indices"][jan25_mask], secondary["indices"]).size == 0
    print("Fitting the Jan25 seed44 combiner...", flush=True)
    model = fit_combiner(X25[np.repeat(jan25_mask, 10)], primary["labels"][jan25_mask], names)
    model.save_model(str(trial / "work/combiner_jan25.cbm"))
    del X25
    X26, names26 = features("jan26", secondary["indices"], 1737849600)
    assert names26 == names
    secondary_new = predict_combiner(model, X26, names).reshape(-1, 10)
    del X26, model
    secondary_metrics, secondary_candidate = compare(secondary, secondary_new)
    print(json.dumps({"two_day_primary": primary_metrics, "jan26_secondary": secondary_metrics}, indent=2), flush=True)
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    gains = []
    for repetition in range(protocol["bootstrap_resamples"]):
        q = rng.integers(0, len(primary["labels"]), size=len(primary["labels"]))
        gains.append(auc(primary["labels"][q], primary_candidate[q]) - auc(primary["labels"][q], primary["baseline"][q]))
        if (repetition + 1) % 50 == 0:
            print(f"Paired event bootstrap: {repetition + 1}/{protocol['bootstrap_resamples']}", flush=True)
    interval = np.quantile(gains, [0.025, 0.975]).tolist()
    report = {
        "protocol": {k: v for k, v in protocol.items() if not k.endswith("sha256")},
        "two_day_primary": primary_metrics, "jan26_secondary": secondary_metrics,
        "primary_paired_gain_95_percentile_interval": interval,
        "sampling_unit": "whole first event per user, preserving all 10 candidates",
        "supported_for_further_confirmation": bool(interval[0] > 0 and secondary_metrics["auc_change"] >= 0),
        "runtime_seconds": round(time.perf_counter() - start, 2),
    }
    for name, reference, new8, candidate in [
        ("two_day", primary, primary_new, primary_candidate),
        ("jan26", secondary, secondary_new, secondary_candidate),
    ]:
        np.savez_compressed(trial / "work" / f"{name}_ensemble_predictions.npz",
                            valid_idx=reference["indices"], labels=reference["labels"],
                            baseline=reference["baseline"], new_dim8_seed44=new8, candidate=candidate)
    verify_inputs(trial, protocol)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nTHIRD DIMENSION-8 SEED TRIAL", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
