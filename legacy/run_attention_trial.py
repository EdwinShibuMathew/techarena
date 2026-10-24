"""Fixed independent attention-history expert trial against unchanged V5.

In techarena-final, run self-test, setup, train, evaluate in that order.
Trial artifacts go to techarena-attention-trial; the self-test uses temporary
files inside techarena-final. Never reads evaluation data or exports CSV.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

CUTOFFS = {"jan24": 1737676800, "jan25": 1737763200, "jan26": 1737849600}
FIELDS = ["u", "i", "t", "tb", "dt", "h", "ht", "y"]
EXPERT_WEIGHT = 0.15


def blend(baseline, expert, history):
    assert baseline.shape == expert.shape == (len(history), 10)
    assert np.isfinite(baseline).all() and np.isfinite(expert).all()
    assert ((baseline >= 0) & (baseline <= 1)).all() and ((expert >= 0) & (expert <= 1)).all()
    # Predeclared: a history-only expert cannot help events with no history.
    return np.where((history >= 0).any(1)[:, None],
                    (1 - EXPERT_WEIGHT) * baseline + EXPERT_WEIGHT * expert, baseline)


def calibration_features(fitting, target, probability, n_items):
    assert probability.shape == target["i"].shape
    displays = np.bincount(fitting["i"].ravel(), minlength=n_items)
    clicks = np.bincount(fitting["i"].ravel(), weights=fitting["y"].ravel(), minlength=n_items)
    item_ctr = (clicks + 50 * fitting["y"].mean()) / (displays + 50)
    user_events = np.bincount(fitting["u"], minlength=max(int(fitting["u"].max()), int(target["u"].max())) + 1)
    h, items = target["h"], target["i"]
    present = h >= 0
    age = np.where(present, (target["t"][:, None] - target["ht"]) / 3600, 720).clip(0, 720)
    broadcast = lambda x: np.broadcast_to(x[:, None], items.shape)
    features = {
        "attention_probability": probability,
        "log_attention_probability": np.log(np.maximum(probability, 1e-8)),
        "relative_attention_probability": probability - probability.mean(1, keepdims=True),
        "attention_event_probability": broadcast(probability.sum(1)),
        "attention_max_probability": broadcast(probability.max(1)),
        "item_ctr": item_ctr[items], "log_item_displays": np.log1p(displays[items]),
        "log_user_events": broadcast(np.log1p(user_events[target["u"]])),
        "time_bucket": broadcast(target["tb"]), "day_type": broadcast(target["dt"]),
        "history_length": broadcast(present.sum(1)),
        "log_last_click_age": broadcast(np.log1p(age.min(1))),
        "history_clicks_last24h": broadcast((present & (age < 24)).sum(1)),
        "repeat_count": ((items[:, :, None] == h[:, None, :]) & present[:, None, :]).sum(2),
    }
    result = np.stack(list(features.values()), -1).reshape(-1, len(features)).astype(np.float32)
    assert np.isfinite(result).all()
    return result, list(features)


def reference_paths(projects, name):
    return [
        projects / "techarena-preference-finetune-trial/work" / f"{name}_predictions.npz",
        projects / "techarena-seed44-trial/work" / f"{name}_ensemble_predictions.npz",
        projects / "techarena-dim16-trial/work" / f"{name}_ensemble_predictions.npz",
        projects / "techarena-dim16-seed43-trial/work" / f"{name}_ensemble_predictions.npz",
    ]


def references(root, raw):
    from audit_v5 import auc
    result = {}
    for name, expected in [("two_day", 0.8183722315263706), ("jan26", 0.8269940935263441)]:
        paths = reference_paths(root.parent, name)
        with np.load(paths[1]) as saved:
            idx, y, v5, third8 = [saved[k] for k in ["valid_idx", "labels", "candidate", "new_dim8_seed44"]]
        members = []
        for path, keys in [(paths[0], ["new_dim8_seed42", "new_dim8_seed43"]),
                           (paths[2], ["dim16"]), (paths[3], ["new_dim16"])]:
            with np.load(path) as saved:
                assert np.array_equal(saved["valid_idx"], idx)
                assert np.array_equal(saved["labels"], y)
                members.extend([saved[k] for k in keys])
        p42, p43, p16a, p16b = members
        reconstructed = (2 / 3) * ((p42 + p43 + third8) / 3) + (1 / 3) * (0.5 * p16a + 0.5 * p16b)
        np.testing.assert_allclose(reconstructed, v5, rtol=0, atol=1e-15)
        assert abs(auc(y, v5) - expected) < 1e-12
        assert np.array_equal(raw["y"][idx], y)
        assert len(np.unique(raw["u"][idx])) == len(idx)
        assert (raw["t"][idx] >= CUTOFFS["jan25" if name == "two_day" else "jan26"]).all()
        assert (raw["t"][idx] < 1737936000).all()
        result[name] = {"indices": idx, "labels": y, "baseline": v5}
    return result


def setup(root, trial):
    from attention_history_model import CONFIG
    from submission_export import sha256
    if trial.exists():
        raise FileExistsError(f"Already prepared: {trial}; use train or evaluate.")
    dim16 = root.parent / "techarena-dim16-trial"
    inputs = [root / "data/cache/train_events.npz"]
    inputs += [root / "src" / f for f in ["attention_history_model.py", "run_attention_trial.py",
               "audit_v5.py", "submission_export.py", "train_final_models.py", "evaluate_dim16_trial.py"]]
    folds = {}
    original_dim16_protocol = dim16 / "outputs/frozen_protocol.json"
    dim16_protocol = json.loads(original_dim16_protocol.read_text(encoding="utf-8"))
    assert dim16_protocol["dimension"] == 16 and dim16_protocol["neural_seed"] == 42
    assert dim16_protocol["preference_epochs"] == 196
    inputs.append(original_dim16_protocol)
    for filename, expected in dim16_protocol["source_sha256"].items():
        path = dim16 / filename
        assert sha256(path) == expected, f"Changed dim16 initialization training source: {path}"
        inputs.append(path)
    for fold, cutoff in CUTOFFS.items():
        checkpoint = dim16 / fold / "work/preference_best.pt"
        split = dim16 / fold / "data/cache" / f"split_{fold}.npz"
        positive_cache = dim16 / fold / "data/cache/preference_training.npz"
        with np.load(positive_cache) as saved:
            assert int(saved["cutoff"]) == cutoff and (saved["timestamps"] < cutoff).all()
        inputs += [checkpoint, split, positive_cache]
        folds[fold] = {"cutoff": cutoff, "initialization_checkpoint": str(checkpoint), "split_path": str(split)}
    for name in ["two_day", "jan26"]:
        inputs += reference_paths(root.parent, name)
    checks = root / "outputs/five_model_submission_checks.json"
    checked = json.loads(checks.read_text(encoding="utf-8"))
    assert checked["submission_ready"] is True and checked["version"] == "v5"
    assert checked["weights"] == [2/9, 2/9, 2/9, 1/6, 1/6]
    inputs.append(checks)
    protocol = {
        "experiment": "standalone candidate-conditioned attention history expert",
        "neural_model": CONFIG.copy(), "v5_weight": 1 - EXPERT_WEIGHT,
        "attention_expert_weight": EXPERT_WEIGHT,
        "empty_history_rule": "retain V5 unchanged",
        "v5_members_and_combiners_unchanged": True,
        "new_expert_calibration": {"trees": 545, "depth": 6, "learning_rate": 0.05,
                                  "l2_leaf_reg": 20, "seed": 42, "early_stopping": False},
        "calibration_feature_count": 14,
        "primary_check": "first event per user across Jan25-26; calibration trained on Jan24",
        "secondary_check": "first event per user on Jan26; calibration trained on Jan25",
        "bootstrap_resamples": 300, "bootstrap_seed": 2026,
        "support_rule": "primary paired gain interval entirely positive and secondary AUC change nonnegative",
        "next_gate": "longer-window paired stress check before full-data promotion or submission",
        "status": "retrospective development experiment; reused validation dates; interval not selection-adjusted",
        "folds": folds,
        "input_sha256": {str(p.resolve(strict=True)): sha256(p) for p in inputs},
    }
    (trial / "outputs").mkdir(parents=True)
    (trial / "work").mkdir()
    for fold in folds:
        (trial / "work" / fold).mkdir()
    (trial / "outputs/frozen_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("ATTENTION TRIAL SETUP", flush=True)
    print(json.dumps({k: v for k, v in protocol.items() if k not in ["folds", "input_sha256"]}, indent=2), flush=True)


def verify(protocol):
    from attention_history_model import CONFIG
    from submission_export import sha256
    assert protocol["neural_model"] == CONFIG
    assert protocol["attention_expert_weight"] == EXPERT_WEIGHT
    for filename, expected in protocol["input_sha256"].items():
        assert sha256(Path(filename)) == expected, f"Changed frozen input: {filename}"


def load_raw(root):
    with np.load(root / "data/cache/train_events.npz") as saved:
        return {k: saved[k] for k in FIELDS}


def subset(raw, indices):
    return {k: raw[k][indices] for k in FIELDS if k != "y"}


def fit_indices(raw, info):
    with np.load(info["split_path"]) as saved:
        idx = saved["train_idx"]
    assert np.array_equal(idx, np.flatnonzero(raw["t"] < info["cutoff"]))
    return idx


def train_all(root, trial, protocol, device):
    import torch
    from attention_history_model import train
    from submission_export import sha256
    raw = load_raw(root)
    references(root, raw)  # Verify the unchanged V5 comparison before training.
    reports = {}
    for fold, info in protocol["folds"].items():
        path = trial / "work" / fold / "attention.pt"
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            assert saved["config"] == protocol["neural_model"] and saved["cutoff"] == info["cutoff"]
            assert saved["initialization_sha256"] == sha256(Path(info["initialization_checkpoint"]))
            print(f"Reusing completed {fold} checkpoint", flush=True)
        else:
            idx = fit_indices(raw, info)
            print(f"Training independent attention model: {fold}, {len(idx):,} events", flush=True)
            train(Path(info["initialization_checkpoint"]), subset(raw, idx), raw["y"][idx],
                  info["cutoff"], path, device)
            saved = torch.load(path, map_location="cpu", weights_only=True)
        reports[fold] = {"training_events": saved["training_events"],
                         "pretraining_click_events": saved["pretraining_click_events"],
                         "last_training_record": saved["records"][-1], "checkpoint": str(path)}
        del saved
        torch.cuda.empty_cache()
    verify(protocol)
    (trial / "outputs/training_results.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print("ATTENTION TRAINING RESULTS", flush=True)
    print(json.dumps(reports, indent=2), flush=True)


def evaluate(root, trial, protocol, device):
    from audit_v5 import AucOrder, auc
    from attention_history_model import predict
    from evaluate_dim16_trial import fit_combiner
    from build_submission import predict_combiner
    from submission_export import sha256
    start = time.perf_counter()
    output = trial / "outputs/ensemble_trial_results.json"
    if output.exists() or (trial / "work/calibration").exists():
        raise FileExistsError("Evaluation already started; preserve its saved results and models.")
    checkpoints = [trial / "work" / f / "attention.pt" for f in CUTOFFS]
    model_hashes = {str(p): sha256(p) for p in checkpoints}
    (trial / "outputs/evaluation_inputs.json").write_text(json.dumps(model_hashes, indent=2), encoding="utf-8")
    (trial / "work/calibration").mkdir()
    raw = load_raw(root)
    refs = references(root, raw)
    with np.load(protocol["folds"]["jan24"]["split_path"]) as saved:
        meta_idx = saved["valid_idx"]
    assert (raw["t"][meta_idx] >= CUTOFFS["jan24"]).all() and (raw["t"][meta_idx] < CUTOFFS["jan25"]).all()
    assert np.intersect1d(meta_idx, refs["two_day"]["indices"]).size == 0
    cases = {"jan24": meta_idx, "jan25": refs["two_day"]["indices"], "jan26": refs["jan26"]["indices"]}
    features, targets, probabilities = {}, {}, {}
    names = None
    for fold, idx in cases.items():
        print(f"Predicting independent attention expert: {fold}", flush=True)
        info = protocol["folds"][fold]
        fitting = fit_indices(raw, info)
        target = subset(raw, idx)
        p = predict(trial / "work" / fold / "attention.pt", target, info["cutoff"], device)
        import torch
        saved = torch.load(trial / "work" / fold / "attention.pt", map_location="cpu", weights_only=True)
        X, columns = calibration_features({k: raw[k][fitting] for k in ["u", "i", "y"]}, target, p, saved["n_items"])
        del saved
        assert len(columns) == protocol["calibration_feature_count"]
        if names is None:
            names = columns
        assert columns == names
        features[fold], targets[fold], probabilities[fold] = X, target, p
    day25 = raw["t"][refs["two_day"]["indices"]] < CUTOFFS["jan26"]
    with np.load(protocol["folds"]["jan25"]["split_path"]) as saved:
        assert np.array_equal(refs["two_day"]["indices"][day25], saved["valid_idx"])
    with np.load(protocol["folds"]["jan26"]["split_path"]) as saved:
        assert np.array_equal(refs["jan26"]["indices"], saved["valid_idx"])
    assert np.intersect1d(refs["two_day"]["indices"][day25], refs["jan26"]["indices"]).size == 0
    model = fit_combiner(features["jan24"], raw["y"][meta_idx], names)
    model.save_model(str(trial / "work/calibration/attention_jan24.cbm"))
    primary_expert = predict_combiner(model, features["jan25"], names).reshape(-1, 10)
    model = fit_combiner(features["jan25"][np.repeat(day25, 10)], refs["two_day"]["labels"][day25], names)
    model.save_model(str(trial / "work/calibration/attention_jan25.cbm"))
    secondary_expert = predict_combiner(model, features["jan26"], names).reshape(-1, 10)
    metrics, candidates = {}, {}
    for name, fold, expert in [("two_day", "jan25", primary_expert), ("jan26", "jan26", secondary_expert)]:
        ref = refs[name]
        candidate = blend(ref["baseline"], expert, targets[fold]["h"])
        candidates[name] = candidate
        old_auc, new_auc = auc(ref["labels"], ref["baseline"]), auc(ref["labels"], candidate)
        empty = (targets[fold]["h"] < 0).all(1)
        assert np.array_equal(candidate[empty], ref["baseline"][empty])
        metrics[name] = {"events": len(ref["indices"]), "baseline_v5_auc": old_auc,
                         "attention_raw_auc": auc(ref["labels"], probabilities[fold]),
                         "attention_calibrated_auc": auc(ref["labels"], expert),
                         "candidate_auc": new_auc, "auc_change": new_auc - old_auc,
                         "empty_history_predictions_unchanged": True}
        np.savez_compressed(trial / "work" / f"{name}_predictions.npz", valid_idx=ref["indices"], labels=ref["labels"],
                            baseline=ref["baseline"], candidate=candidate, attention_expert=expert,
                            attention_raw=probabilities[fold])
    print(json.dumps(metrics, indent=2), flush=True)
    ref = refs["two_day"]
    n = len(ref["indices"])
    event_ids = np.repeat(np.arange(n), 10)
    old_order = AucOrder(ref["labels"].ravel(), ref["baseline"].ravel(), event_ids)
    new_order = AucOrder(ref["labels"].ravel(), candidates["two_day"].ravel(), event_ids)
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    gains = []
    for iteration in range(protocol["bootstrap_resamples"]):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        gains.append(new_order.value(counts) - old_order.value(counts))
        if (iteration + 1) % 50 == 0:
            print(f"Whole-event paired bootstrap: {iteration + 1}/300", flush=True)
    interval = np.quantile(gains, [0.025, 0.975]).tolist()
    fitting = fit_indices(raw, protocol["folds"]["jan25"])
    activity = np.bincount(raw["u"][fitting], minlength=int(raw["u"].max()) + 1)[raw["u"][ref["indices"]]]
    length = (raw["h"][ref["indices"]] >= 0).sum(1)
    groups = {}
    for key, mask in {"activity_1_5": (activity >= 1) & (activity <= 5),
                      "unseen_with_history": (activity == 0) & (length > 0),
                      "history_1_5": (length >= 1) & (length <= 5),
                      "activity_16_plus": activity >= 16, "horizon_24_48h": ~day25}.items():
        before, after = auc(ref["labels"][mask], ref["baseline"][mask]), auc(ref["labels"][mask], candidates["two_day"][mask])
        groups[key] = {"events": int(mask.sum()), "baseline_auc": before, "candidate_auc": after,
                       "auc_change": after - before if before is not None and after is not None else None}
    verify(protocol)
    for path, expected in model_hashes.items():
        assert sha256(Path(path)) == expected
    report = {
        "protocol": {k: v for k, v in protocol.items() if k not in ["input_sha256", "folds"]},
        "two_day_primary": metrics["two_day"], "jan26_secondary": metrics["jan26"],
        "primary_paired_gain_95_percentile_interval": interval, "primary_groups": groups,
        "v5_reproduction_passed": True,
        "supported_for_long_window_check": bool(interval[0] > 0 and metrics["jan26"]["auc_change"] >= 0),
        "sampling_unit": "whole first event per user, preserving all 10 candidates",
        "submission_created": False, "runtime_seconds": round(time.perf_counter() - start, 2),
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nCANDIDATE ATTENTION HISTORY TRIAL", flush=True)
    print(json.dumps(report, indent=2), flush=True)


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["self-test", "setup", "train", "evaluate"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    trial = root.parent / "techarena-attention-trial"
    if args.action == "self-test":
        from attention_history_model import self_test
        self_test()
        return
    if args.action == "setup":
        setup(root, trial)
        return
    protocol = json.loads((trial / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    verify(protocol)
    assert torch.cuda.is_available(), "Activate techarena with working CUDA."
    torch.set_num_threads(4)
    device = torch.device("cuda")
    if args.action == "train":
        train_all(root, trial, protocol, device)
    else:
        evaluate(root, trial, protocol, device)


if __name__ == "__main__":
    main()
