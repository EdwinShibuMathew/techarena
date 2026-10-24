"""Test a joint neural-feature combiner, using the existing four models.

Run in techarena-final: python src/test_joint_combiner.py
No neural training or submission is performed. Both combiners fit only past days.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from build_submission import TARGET_FIELDS, FIT_FIELDS, neural_pipeline as pipeline8
from evaluate_dim16_trial import fit_combiner
from joint_features import MODEL_KEYS, build_joint_features
from submission_export import sha256


def auc(labels, predictions):
    return float(roc_auc_score(labels.ravel(), predictions.ravel()))


def compare(labels, baseline, joint):
    candidate = 0.5 * baseline + 0.5 * joint
    return {
        "events": len(labels),
        "baseline_four_model_auc": auc(labels, baseline),
        "joint_combiner_auc": auc(labels, joint),
        "candidate_blend_auc": auc(labels, candidate),
        "auc_change": auc(labels, candidate) - auc(labels, baseline),
    }, candidate


def main():
    start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    projects = root.parent
    previous = projects / "techarena-dim16-seed43-trial"
    trial = projects / "techarena-joint-combiner-trial"
    output = trial / "outputs/ensemble_trial_results.json"
    if output.exists():
        raise FileExistsError(f"Joint-combiner trial result already exists: {output}")
    for name in ["work", "outputs"]:
        (trial / name).mkdir(parents=True, exist_ok=True)
    protocol = {
        "model_inputs": list(MODEL_KEYS),
        "common_features_included_once": 16,
        "model_specific_features_per_model": 21,
        "joint_feature_count": 100,
        "combiner_trees": 545,
        "combiner_depth": 6,
        "combiner_learning_rate": 0.05,
        "combiner_l2_leaf_reg": 20,
        "combiner_seed": 42,
        "combiner_early_stopping": False,
        "candidate_current_ensemble_weight": 0.5,
        "candidate_joint_combiner_weight": 0.5,
        "fitting_features": "neural features from models trained before fitting day; no fitted-combiner predictions",
        "primary_check": "first event per user across Jan25-26; combiner trained on Jan24",
        "secondary_check": "first event per user on Jan26; combiner trained on Jan25",
        "support_rule": "primary paired-gain interval entirely above zero and secondary AUC change nonnegative",
        "bootstrap_resamples": 300,
        "bootstrap_seed": 2026,
        "status": "retrospective development experiment",
        "source_sha256": {name: sha256(root / "src" / name) for name in ["joint_features.py", "test_joint_combiner.py", "build_submission.py", "submission_features.py", "evaluate_dim16_trial.py"]},
    }
    protocol_path = trial / "outputs/frozen_protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding="utf-8")) == protocol
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    sys.path.insert(0, str(projects / "techarena-dim16-trial/src"))
    from dim16_pipeline import neural_pipeline as pipeline16, predict_combiner

    folders = {
        "dim8_seed42": {
            "jan24": projects / "techarena-confirm-jan24",
            "jan25": projects / "techarena-long320-jan25",
            "jan26": projects / "techarena-long320",
        },
        "dim8_seed43": {
            "jan24": projects / "techarena-confirm-seed43-jan24",
            "jan25": projects / "techarena-seed43-jan25",
            "jan26": projects / "techarena-seed43",
        },
        "dim16_seed42": {fold: projects / "techarena-dim16-trial" / fold for fold in ["jan24", "jan25", "jan26"]},
        "dim16_seed43": {fold: previous / fold for fold in ["jan24", "jan25", "jan26"]},
    }
    with np.load(root / "data/cache/train_events.npz") as saved:
        raw = {name: saved[name] for name in set(TARGET_FIELDS + FIT_FIELDS)}
    with np.load(folders["dim8_seed42"]["jan24"] / "data/cache/split_jan24.npz") as saved:
        jan24_idx = saved["valid_idx"]
    with np.load(previous / "work/two_day_ensemble_predictions.npz") as saved:
        primary_idx = saved["valid_idx"]
        primary_labels = saved["labels"]
        primary_baseline = saved["candidate"]
    with np.load(previous / "work/jan26_ensemble_predictions.npz") as saved:
        secondary_idx = saved["valid_idx"]
        secondary_labels = saved["labels"]
        secondary_baseline = saved["candidate"]
    assert np.array_equal(primary_labels, raw["y"][primary_idx])
    assert np.array_equal(secondary_labels, raw["y"][secondary_idx])
    assert np.intersect1d(jan24_idx, primary_idx).size == 0
    assert (raw["t"][jan24_idx] >= 1737676800).all()
    assert (raw["t"][jan24_idx] < 1737763200).all()

    def features(fold, target_idx, cutoff):
        with np.load(folders["dim8_seed42"][fold] / "data/cache" / f"split_{fold}.npz") as saved:
            fit_idx = saved["train_idx"]
        assert (raw["t"][fit_idx] < cutoff).all()
        assert (raw["t"][target_idx] >= cutoff).all()
        assert (raw["t"][target_idx] < 1737936000).all()
        assert len(np.unique(raw["u"][target_idx])) == len(target_idx)
        sources = {}
        for key in MODEL_KEYS:
            folder = folders[key][fold]
            with np.load(folder / "data/cache" / f"split_{fold}.npz") as saved:
                assert np.array_equal(fit_idx, saved["train_idx"])
            checkpoint = torch.load(folder / "work/preference_best.pt", map_location="cpu", weights_only=True)
            expected_dimension = 8 if key.startswith("dim8_") else 16
            assert checkpoint["dimension"] == expected_dimension and checkpoint["epoch"] == 196
            del checkpoint
            sources[key] = {"checkpoint": folder / "work/event_best.pt", "positive_path": folder / "data/cache/preference_training.npz"}
        print(f"Building 100 joint features for {fold}...", flush=True)
        return build_joint_features(
            {name: raw[name][fit_idx] for name in FIT_FIELDS},
            {name: raw[name][target_idx] for name in TARGET_FIELDS},
            sources, pipeline8, pipeline16, device,
        )

    X24, names = features("jan24", jan24_idx, 1737676800)
    X25, names25 = features("jan25", primary_idx, 1737763200)
    assert names == names25 and len(names) == 100
    print("Fitting the Jan24 joint combiner...", flush=True)
    model = fit_combiner(X24, raw["y"][jan24_idx], names)
    primary_joint = predict_combiner(model, X25, names).reshape(-1, 10)
    model.save_model(str(trial / "work/joint_combiner_jan24.cbm"))
    del X24, model
    primary_metrics, primary_candidate = compare(primary_labels, primary_baseline, primary_joint)
    print(json.dumps({"two_day_primary": primary_metrics}, indent=2), flush=True)

    jan25_mask = raw["t"][primary_idx] < 1737849600
    with np.load(folders["dim8_seed42"]["jan25"] / "data/cache/split_jan25.npz") as saved:
        assert np.array_equal(primary_idx[jan25_mask], saved["valid_idx"])
    assert np.intersect1d(primary_idx[jan25_mask], secondary_idx).size == 0
    print("Fitting the Jan25 joint combiner...", flush=True)
    model = fit_combiner(X25[np.repeat(jan25_mask, 10)], primary_labels[jan25_mask], names)
    model.save_model(str(trial / "work/joint_combiner_jan25.cbm"))
    del X25
    X26, names26 = features("jan26", secondary_idx, 1737849600)
    assert names26 == names
    secondary_joint = predict_combiner(model, X26, names).reshape(-1, 10)
    del X26, model
    secondary_metrics, secondary_candidate = compare(secondary_labels, secondary_baseline, secondary_joint)
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
        "protocol": {key: value for key, value in protocol.items() if key != "source_sha256"},
        "two_day_primary": primary_metrics,
        "jan26_secondary": secondary_metrics,
        "primary_paired_gain_95_percentile_interval": interval,
        "supported_for_further_confirmation": bool(interval[0] > 0 and secondary_metrics["auc_change"] >= 0),
        "runtime_seconds": round(time.perf_counter() - start, 2),
    }
    np.savez_compressed(trial / "work/two_day_predictions.npz", valid_idx=primary_idx, labels=primary_labels, baseline=primary_baseline, joint=primary_joint, candidate=primary_candidate)
    np.savez_compressed(trial / "work/jan26_predictions.npz", valid_idx=secondary_idx, labels=secondary_labels, baseline=secondary_baseline, joint=secondary_joint, candidate=secondary_candidate)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nJOINT COMBINER TRIAL", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
