"""Prepare, train and screen one pooled-history correction experiment.

In techarena-final:
    python src/run_history_trial.py setup
    python src/run_history_trial.py train
    python src/run_history_trial.py evaluate

Creates techarena-history-trial; changes no V5 model or submission.
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

from pooled_history_model import CONFIG, train_model, predict_model
from build_submission import TARGET_FIELDS, FIT_FIELDS, neural_pipeline, load_combiner, predict_combiner, check_close
from evaluate_dim16_trial import fit_combiner
from audit_v5 import AucOrder, auc
from submission_export import sha256

FEATURE = "history_correction_probability"
WEIGHTS = [2 / 9, 2 / 9, 2 / 9, 1 / 6, 1 / 6]
CUTOFFS = {"jan24": 1737676800, "jan25": 1737763200, "jan26": 1737849600}


def member_specs(projects):
    fine = projects / "techarena-preference-finetune-trial"
    return [
        ("dim8_seed42", 8, fine / "dim8_seed42", fine / "work", "dim8_seed42_combiner_"),
        ("dim8_seed43", 8, fine / "dim8_seed43", fine / "work", "dim8_seed43_combiner_"),
        ("dim8_seed44", 8, projects / "techarena-seed44-trial", projects / "techarena-seed44-trial/work", "combiner_"),
        ("dim16_seed42", 16, projects / "techarena-dim16-trial", projects / "techarena-dim16-trial/work", "combiner_"),
        ("dim16_seed43", 16, projects / "techarena-dim16-seed43-trial", projects / "techarena-dim16-seed43-trial/work", "combiner_"),
    ]


def verify(protocol):
    assert protocol["history_model"] == CONFIG
    assert protocol["weights"] == WEIGHTS
    for name, expected in protocol["input_sha256"].items():
        assert sha256(Path(name)) == expected, f"Changed frozen input: {name}"


def read_raw(root):
    with np.load(root / "data/cache/train_events.npz") as saved:
        return {name: saved[name] for name in set(TARGET_FIELDS + FIT_FIELDS)}


def ensemble(members):
    assert len(members) == 5 and all(p.shape == members[0].shape for p in members)
    return (2 / 3) * ((members[0] + members[1] + members[2]) / 3) + (1 / 3) * (0.5 * members[3] + 0.5 * members[4])


def setup(root, trial):
    if trial.exists():
        raise FileExistsError(f"Already prepared: {trial}. Use train or evaluate after setup.")
    projects = root.parent
    specs = member_specs(projects)
    paths = [root / "src" / name for name in [
        "pooled_history_model.py", "run_history_trial.py", "train_final_models.py",
        "build_submission.py", "submission_features.py", "evaluate_dim16_trial.py", "audit_v5.py",
    ]]
    audit_path = root / "outputs/v5_audit/audit_summary.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["validation_source_checks_passed"] is True
    paths.append(audit_path)
    checks = root / "outputs/five_model_submission_checks.json"
    checked = json.loads(checks.read_text(encoding="utf-8"))
    assert checked["submission_ready"] is True and checked["version"] == "v5"
    assert checked["weights"] == WEIGHTS
    paths.append(checks)
    for filename in ["two_day_ensemble_predictions.npz", "jan26_ensemble_predictions.npz"]:
        paths.append(projects / "techarena-seed44-trial/work" / filename)
    for _, _, folder, combiners, prefix in specs:
        for fold in CUTOFFS:
            paths.extend([folder / fold / "work/event_best.pt",
                          folder / fold / "data/cache/preference_training.npz",
                          folder / fold / "data/cache" / f"split_{fold}.npz"])
        for day in ["jan24", "jan25"]:
            paths.append(combiners / f"{prefix}{day}.cbm")
    dim16_helper = projects / "techarena-dim16-trial/src/dim16_pipeline.py"
    paths.append(dim16_helper)
    canonical = specs[0][2]
    folds = {}
    for fold, cutoff in CUTOFFS.items():
        base_path = canonical / fold / "work/event_best.pt"
        pref = torch.load(canonical / fold / "work/preference_best.pt", map_location="cpu", weights_only=True)
        assert pref["dimension"] == 8 and pref["epoch"] == 260
        base = torch.load(base_path, map_location="cpu", weights_only=True)
        assert base["dimension"] == 8 and base["epoch"] == 8
        folds[fold] = {"cutoff": cutoff, "base_checkpoint": str(base_path),
                       "split_path": str(canonical / fold / "data/cache" / f"split_{fold}.npz")}
    protocol = {
        "experiment": "pooled history correction as one extra feature in all five V5 combiners",
        "history_model": CONFIG.copy(), "base_neural_member": "fine-tuned dim8 seed42",
        "combiner_trees": 545, "combiner_seed": 42, "combiner_early_stopping": False,
        "extra_feature": FEATURE, "feature_count_per_combiner": 38,
        "members": [spec[0] for spec in specs], "weights": WEIGHTS,
        "primary_check": "first event per user across Jan25-26; combiners trained on Jan24",
        "secondary_check": "first event per user on Jan26; combiners trained on Jan25",
        "bootstrap_resamples": 300, "bootstrap_seed": 2026,
        "support_rule": "primary paired gain interval entirely positive and secondary AUC change nonnegative",
        "next_gate": "pass a longer-window paired stress check before full-data promotion or submission",
        "status": "retrospective development experiment; reused validation dates",
        "folds": folds, "dim16_helper": str(dim16_helper),
        "input_sha256": {str(path.resolve(strict=True)): sha256(path) for path in paths},
    }
    for folder in ["work", "outputs"]:
        (trial / folder).mkdir(parents=True)
    for fold in CUTOFFS:
        (trial / "work" / fold).mkdir()
    (trial / "outputs/frozen_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({"setup": "passed", "trial_workspace": str(trial),
                      "history_model": CONFIG, "combiner_features": 38, "weights": WEIGHTS}, indent=2), flush=True)


def train(root, trial, protocol):
    raw = read_raw(root)
    device = torch.device("cuda")
    reports = {}
    for fold, info in protocol["folds"].items():
        checkpoint = trial / "work" / fold / "history_model.pt"
        with np.load(info["split_path"]) as saved:
            fit_idx = saved["train_idx"]
        assert np.array_equal(fit_idx, np.flatnonzero(raw["t"] < info["cutoff"]))
        print(f"\nTraining history correction: {fold}, {len(fit_idx):,} events", flush=True)
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert saved["config"] == CONFIG and saved["epoch"] == CONFIG["epochs"]
            assert saved["cutoff"] == info["cutoff"] and saved["base_sha256"] == sha256(Path(info["base_checkpoint"]))
            reports[fold] = {"checkpoint": str(checkpoint), "using_completed_checkpoint": True}
            del saved
        else:
            reports[fold] = train_model(Path(info["base_checkpoint"]),
                                        {name: raw[name][fit_idx] for name in TARGET_FIELDS},
                                        raw["y"][fit_idx], info["cutoff"], checkpoint, device)
        torch.cuda.empty_cache()
    verify(protocol)
    (trial / "outputs/training_results.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print("\nPOOLED HISTORY TRAINING RESULTS", flush=True)
    print(json.dumps({fold: {k: v for k, v in data.items() if k != "history"} for fold, data in reports.items()}, indent=2), flush=True)


def evaluate(root, trial, protocol):
    start = time.perf_counter()
    output = trial / "outputs/ensemble_trial_results.json"
    if output.exists():
        raise FileExistsError(f"Results already exist: {output}")
    raw = read_raw(root)
    device = torch.device("cuda")
    spec = importlib.util.spec_from_file_location("history_trial_dim16_pipeline", protocol["dim16_helper"])
    pipeline16 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline16)
    with np.load(protocol["folds"]["jan24"]["split_path"]) as saved:
        jan24_idx = saved["valid_idx"]
    references = {}
    for name, filename in [("primary", "two_day"), ("secondary", "jan26")]:
        with np.load(root.parent / "techarena-seed44-trial/work" / f"{filename}_ensemble_predictions.npz") as saved:
            references[name] = {"indices": saved["valid_idx"], "labels": saved["labels"], "baseline": saved["candidate"]}
        assert np.array_equal(references[name]["labels"], raw["y"][references[name]["indices"]])
        assert len(np.unique(raw["u"][references[name]["indices"]])) == len(references[name]["indices"])
    primary, secondary = references["primary"], references["secondary"]
    assert (raw["t"][jan24_idx] >= CUTOFFS["jan24"]).all() and (raw["t"][jan24_idx] < CUTOFFS["jan25"]).all()
    assert np.intersect1d(jan24_idx, primary["indices"]).size == 0
    targets = {"jan24": jan24_idx, "jan25": primary["indices"], "jan26": secondary["indices"]}
    history_predictions = {}
    for fold, indices in targets.items():
        print(f"Predicting pooled-history feature for {fold}...", flush=True)
        history_predictions[fold] = predict_model(trial / "work" / fold / "history_model.pt",
                                                  {name: raw[name][indices] for name in TARGET_FIELDS},
                                                  device, CUTOFFS[fold])
    member_primary, member_secondary, old_primary, old_secondary = [], [], [], []
    member_results = {}

    def features(folder, fold, indices, pipeline):
        with np.load(folder / fold / "data/cache" / f"split_{fold}.npz") as saved:
            fit_idx = saved["train_idx"]
            if fold == "jan24":
                assert np.array_equal(indices, saved["valid_idx"])
        assert np.array_equal(fit_idx, np.flatnonzero(raw["t"] < CUTOFFS[fold]))
        assert (raw["t"][indices] >= CUTOFFS[fold]).all() and (raw["t"][indices] < 1737936000).all()
        X, names, _, _, _, _ = pipeline(folder / fold / "work/event_best.pt",
                                        {name: raw[name][fit_idx] for name in FIT_FIELDS},
                                        {name: raw[name][indices] for name in TARGET_FIELDS},
                                        folder / fold / "data/cache/preference_training.npz", device)
        assert len(names) == 37 and FEATURE not in names
        return X, names

    day25 = raw["t"][primary["indices"]] < CUTOFFS["jan26"]
    for key, dimension, folder, combiners, prefix in member_specs(root.parent):
        print(f"\nBuilding 38-feature combiners: {key}", flush=True)
        pipeline = neural_pipeline if dimension == 8 else pipeline16.neural_pipeline
        X24, names = features(folder, "jan24", jan24_idx, pipeline)
        X25, names25 = features(folder, "jan25", primary["indices"], pipeline)
        assert names == names25
        with np.load(folder / "jan25/data/cache/split_jan25.npz") as saved:
            assert np.array_equal(primary["indices"][day25], saved["valid_idx"])
        with np.load(folder / "jan26/data/cache/split_jan26.npz") as saved:
            assert np.array_equal(secondary["indices"], saved["valid_idx"])
        assert np.intersect1d(primary["indices"][day25], secondary["indices"]).size == 0
        old_primary.append(predict_combiner(load_combiner(combiners / f"{prefix}jan24.cbm"), X25, names).reshape(-1, 10))
        augmented_names = list(names) + [FEATURE]
        A24 = np.column_stack([X24, history_predictions["jan24"].ravel()])
        A25 = np.column_stack([X25, history_predictions["jan25"].ravel()])
        del X24, X25
        model = fit_combiner(A24, raw["y"][jan24_idx], augmented_names)
        member_primary.append(predict_combiner(model, A25, augmented_names).reshape(-1, 10))
        model.save_model(str(trial / "work" / f"{key}_combiner_jan24.cbm"))
        del A24, model
        model = fit_combiner(A25[np.repeat(day25, 10)], primary["labels"][day25], augmented_names)
        model.save_model(str(trial / "work" / f"{key}_combiner_jan25.cbm"))
        del A25
        X26, names26 = features(folder, "jan26", secondary["indices"], pipeline)
        assert names26 == names
        old_secondary.append(predict_combiner(load_combiner(combiners / f"{prefix}jan25.cbm"), X26, names).reshape(-1, 10))
        A26 = np.column_stack([X26, history_predictions["jan26"].ravel()])
        member_secondary.append(predict_combiner(model, A26, augmented_names).reshape(-1, 10))
        del X26, A26, model
        member_results[key] = {"primary_auc": auc(primary["labels"], member_primary[-1]),
                               "secondary_auc": auc(secondary["labels"], member_secondary[-1])}
    check_close(ensemble(old_primary), primary["baseline"], "Reproduced V5 primary", rtol=0, atol=1e-15)
    check_close(ensemble(old_secondary), secondary["baseline"], "Reproduced V5 secondary", rtol=0, atol=1e-15)
    candidate_primary, candidate_secondary = ensemble(member_primary), ensemble(member_secondary)

    def metrics(reference, predictions, history_probability):
        return {"events": len(reference["indices"]), "baseline_v5_auc": auc(reference["labels"], reference["baseline"]),
                "candidate_history_auc": auc(reference["labels"], predictions),
                "auc_change": auc(reference["labels"], predictions) - auc(reference["labels"], reference["baseline"]),
                "history_network_alone_auc": auc(reference["labels"], history_probability)}

    primary_metrics = metrics(primary, candidate_primary, history_predictions["jan25"])
    secondary_metrics = metrics(secondary, candidate_secondary, history_predictions["jan26"])
    print(json.dumps({"two_day_primary": primary_metrics, "jan26_secondary": secondary_metrics}, indent=2), flush=True)
    events = np.repeat(np.arange(len(primary["indices"])), 10)
    old_order = AucOrder(primary["labels"].ravel(), primary["baseline"].ravel(), events)
    new_order = AucOrder(primary["labels"].ravel(), candidate_primary.ravel(), events)
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    gains = []
    for iteration in range(protocol["bootstrap_resamples"]):
        counts = np.bincount(rng.integers(0, len(primary["indices"]), size=len(primary["indices"])), minlength=len(primary["indices"]))
        gains.append(new_order.value(counts) - old_order.value(counts))
        if (iteration + 1) % 50 == 0:
            print(f"Whole-event paired bootstrap: {iteration + 1}/300", flush=True)
    interval = np.quantile(gains, [0.025, 0.975]).tolist()
    with np.load(protocol["folds"]["jan25"]["split_path"]) as saved:
        user_counts = np.bincount(raw["u"][saved["train_idx"]], minlength=int(raw["u"].max()) + 1)
    activity = user_counts[raw["u"][primary["indices"]]]
    length = (raw["h"][primary["indices"]] >= 0).sum(axis=1)
    groups = {}
    for name, mask in {
        "activity_1_5": (activity >= 1) & (activity <= 5), "history_1_5": (length >= 1) & (length <= 5),
        "unseen_with_history": (activity == 0) & (length > 0), "history_empty": length == 0,
        "activity_16_plus": activity >= 16, "horizon_24_48h": ~day25,
    }.items():
        old_auc, new_auc = auc(primary["labels"][mask], primary["baseline"][mask]), auc(primary["labels"][mask], candidate_primary[mask])
        groups[name] = {"events": int(mask.sum()), "v5_auc": old_auc, "candidate_auc": new_auc,
                        "auc_change": new_auc - old_auc if old_auc is not None and new_auc is not None else None}
    report = {
        "protocol": {k: v for k, v in protocol.items() if k not in ["input_sha256", "folds", "dim16_helper"]},
        "two_day_primary": primary_metrics, "jan26_secondary": secondary_metrics,
        "primary_paired_gain_95_percentile_interval": interval, "primary_groups": groups,
        "member_auc": member_results, "v5_reproduction_passed": True,
        "supported_for_long_window_check": bool(interval[0] > 0 and secondary_metrics["auc_change"] >= 0),
        "submission_created": False, "runtime_seconds": round(time.perf_counter() - start, 2),
    }
    for name, reference, candidate, hp in [
        ("two_day", primary, candidate_primary, history_predictions["jan25"]),
        ("jan26", secondary, candidate_secondary, history_predictions["jan26"]),
    ]:
        np.savez_compressed(trial / "work" / f"{name}_predictions.npz", valid_idx=reference["indices"],
                            labels=reference["labels"], baseline=reference["baseline"], candidate=candidate,
                            history_model_probability=hp)
    verify(protocol)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nPOOLED HISTORY CORRECTION TRIAL", flush=True)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "train", "evaluate"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    trial = root.parent / "techarena-history-trial"
    if args.action == "setup":
        setup(root, trial)
        return
    protocol = json.loads((trial / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    verify(protocol)
    assert torch.cuda.is_available(), "Activate techarena with working CUDA."
    torch.set_num_threads(4)
    if args.action == "train":
        train(root, trial, protocol)
    else:
        evaluate(root, trial, protocol)


if __name__ == "__main__":
    main()
