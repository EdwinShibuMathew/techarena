"""All meta-training-day events, equal total weight per user, unchanged V5 inputs.

In techarena-final: self-test, setup, evaluate. Fits ten new combiners only.
Primary: fit Jan24, evaluate first-user Jan25-26. Secondary: fit Jan25, evaluate
first-user Jan26. Full-data fitting/submission and five-day promotion are separate.
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from expanded_combiner_features import make_pipeline, user_equal_weights
from audit_v5 import AucOrder, auc
from submission_export import sha256

CUTOFFS = {"jan24": 1737676800, "jan25": 1737763200, "jan26": 1737849600}
WEIGHTS = [2 / 9, 2 / 9, 2 / 9, 1 / 6, 1 / 6]


def member_specs(projects):
    fine = projects / "techarena-preference-finetune-trial"
    return [
        ("dim8_seed42", 8, fine / "dim8_seed42", fine / "work", "dim8_seed42_combiner_"),
        ("dim8_seed43", 8, fine / "dim8_seed43", fine / "work", "dim8_seed43_combiner_"),
        ("dim8_seed44", 8, projects / "techarena-seed44-trial", projects / "techarena-seed44-trial/work", "combiner_"),
        ("dim16_seed42", 16, projects / "techarena-dim16-trial", projects / "techarena-dim16-trial/work", "combiner_"),
        ("dim16_seed43", 16, projects / "techarena-dim16-seed43-trial", projects / "techarena-dim16-seed43-trial/work", "combiner_"),
    ]


def ensemble(members):
    assert len(members) == 5 and all(p.shape == members[0].shape for p in members)
    return (2 / 3) * ((members[0] + members[1] + members[2]) / 3) + (1 / 3) * (0.5 * members[3] + 0.5 * members[4])


def reference_paths(projects, name):
    return [projects / "techarena-preference-finetune-trial/work" / f"{name}_predictions.npz",
            projects / "techarena-seed44-trial/work" / f"{name}_ensemble_predictions.npz",
            projects / "techarena-dim16-trial/work" / f"{name}_ensemble_predictions.npz",
            projects / "techarena-dim16-seed43-trial/work" / f"{name}_ensemble_predictions.npz"]


def load_references(root, raw):
    cases = {}
    for name, expected in [("two_day", 0.8183722315263706), ("jan26", 0.8269940935263441)]:
        paths = reference_paths(root.parent, name)
        with np.load(paths[1]) as saved:
            idx, y, baseline, third8 = [saved[k] for k in ["valid_idx", "labels", "candidate", "new_dim8_seed44"]]
        members = []
        for path, keys in [(paths[0], ["new_dim8_seed42", "new_dim8_seed43"]),
                           (paths[2], ["dim16"]), (paths[3], ["new_dim16"])]:
            with np.load(path) as saved:
                assert np.array_equal(saved["valid_idx"], idx) and np.array_equal(saved["labels"], y)
                members.extend([saved[k] for k in keys])
        ordered = [members[0], members[1], third8, members[2], members[3]]
        np.testing.assert_allclose(ensemble(ordered), baseline, rtol=0, atol=1e-15)
        assert abs(auc(y, baseline) - expected) < 1e-12
        assert np.array_equal(raw["y"][idx], y)
        assert len(np.unique(raw["u"][idx])) == len(idx)
        assert (raw["t"][idx] >= CUTOFFS["jan25" if name == "two_day" else "jan26"]).all()
        assert (raw["t"][idx] < 1737936000).all()
        cases[name] = {"indices": idx, "labels": y, "baseline": baseline, "members": ordered}
    return cases


def read_raw(root):
    with np.load(root / "data/cache/train_events.npz") as saved:
        return {k: saved[k] for k in ["u", "i", "t", "tb", "dt", "h", "ht", "y"]}


def verify(protocol):
    assert protocol["weights"] == WEIGHTS
    for name, expected in protocol["input_sha256"].items():
        assert sha256(Path(name)) == expected, f"Changed frozen input: {name}"


def setup(root, trial):
    import torch
    if trial.exists():
        protocol = json.loads((trial / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
        verify(protocol)
        print("Expanded-combiner setup already complete; continue with evaluate.", flush=True)
        return
    paths = [Path(__file__)] + [root / "src" / name for name in ["expanded_combiner_features.py", "build_submission.py",
             "submission_features.py", "train_final_models.py", "submission_export.py", "audit_v5.py"]]
    paths.append(root / "data/cache/train_events.npz")
    dim16_helper = root.parent / "techarena-dim16-trial/src/dim16_pipeline.py"
    paths.append(dim16_helper)
    for name in ["two_day", "jan26"]:
        paths.extend(reference_paths(root.parent, name))
    checks_path = root / "outputs/five_model_submission_checks.json"
    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    assert checks["submission_ready"] is True and checks["version"] == "v5" and checks["weights"] == WEIGHTS
    paths.append(checks_path)
    for key, dimension, folder, combiners, prefix in member_specs(root.parent):
        for fold, cutoff in CUTOFFS.items():
            pref_path, event_path = folder / fold / "work/preference_best.pt", folder / fold / "work/event_best.pt"
            pref = torch.load(pref_path, map_location="cpu", weights_only=True)
            assert pref["dimension"] == dimension and pref["epoch"] == (260 if dimension == 8 else 196)
            if dimension == 8:
                assert pref["fine_tuning_sampling_seed"] == int(key.rsplit("seed", 1)[1]) + 1000
            event = torch.load(event_path, map_location="cpu", weights_only=True)
            assert event["dimension"] == dimension and event["epoch"] == 8
            positive_path = folder / fold / "data/cache/preference_training.npz"
            with np.load(positive_path) as saved:
                assert int(saved["cutoff"]) == cutoff and (saved["timestamps"] < cutoff).all()
            paths += [pref_path, event_path, positive_path, folder / fold / "data/cache" / f"split_{fold}.npz"]
            del pref, event
        paths += [combiners / f"{prefix}{fold}.cbm" for fold in ["jan24", "jan25"]]
    raw = read_raw(root)
    load_references(root, raw)
    weighting = {}
    for fold in ["jan24", "jan25"]:
        idx = np.flatnonzero((raw["t"] >= CUTOFFS[fold]) & (raw["t"] < CUTOFFS[fold] + 86400))
        _, weighting[fold] = user_equal_weights(raw["u"][idx])
    protocol = {
        "experiment": "all fitting-day events with equal total user weight in each V5 combiner",
        "members": [s[0] for s in member_specs(root.parent)], "weights": WEIGHTS,
        "neural_models_unchanged": True, "feature_count": 37,
        "weight_rule": "event weight = (events / users) / that user's fitting-day event count; repeat for all 10 rows",
        "weight_normalization": "mean row weight one; equal summed row weight per user",
        "weighting": weighting,
        "combiner": {"trees": 545, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 20,
                     "loss": "Logloss", "seed": 42, "thread_count": 4, "early_stopping": False},
        "primary_check": "all Jan24 fitting events, first-user Jan25-26 evaluation",
        "secondary_check": "all Jan25 fitting events, first-user Jan26 evaluation",
        "feature_fitting_rule": "neural models, ridge history encoder and aggregates strictly before the event's fitting day",
        "cold_user_rule": "one inferred profile per event; no mutation of shared user embeddings",
        "bootstrap_resamples": 300, "bootstrap_seed": 2026,
        "support_rule": "primary paired gain interval entirely positive and secondary AUC change nonnegative",
        "next_gate": "five-day paired comparison using Jan21 expanded combiners before promotion or submission",
        "status": "retrospective development experiment; reused validation dates; interval not selection-adjusted",
        "dim16_helper": str(dim16_helper),
        "input_sha256": {str(p.resolve(strict=True)): sha256(p) for p in paths},
    }
    (trial / "work").mkdir(parents=True)
    (trial / "outputs").mkdir()
    (trial / "outputs/frozen_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    verify(protocol)
    print("EXPANDED COMBINER SETUP", flush=True)
    print(json.dumps({k: v for k, v in protocol.items() if k not in ["input_sha256", "dim16_helper"]}, indent=2), flush=True)


def fit_weighted(X, labels, names, row_weights):
    from catboost import CatBoostClassifier, Pool
    assert len(names) == 37 and X.shape == (labels.size, 37)
    assert row_weights.shape == (labels.size,) and (row_weights > 0).all()
    model = CatBoostClassifier(iterations=545, depth=6, learning_rate=0.05, l2_leaf_reg=20,
                               loss_function="Logloss", eval_metric="AUC", random_seed=42,
                               thread_count=4, verbose=100, allow_writing_files=False)
    model.fit(Pool(X, label=labels.ravel(), weight=row_weights, feature_names=names))
    assert model.tree_count_ == 545 and len(model.feature_names_) == 37
    return model


def evaluate(root, trial, protocol):
    import torch
    from build_submission import TARGET_FIELDS, FIT_FIELDS, neural_pipeline, predict_combiner, load_combiner, check_close
    output = trial / "outputs/ensemble_trial_results.json"
    if output.exists():
        print("Saved expanded-combiner result already complete; displaying it.", flush=True)
        print("\nEXPANDED COMBINER TRAINING TRIAL", flush=True)
        print(output.read_text(encoding="utf-8"), flush=True)
        return
    start = time.perf_counter()
    assert torch.cuda.is_available(), "Activate techarena with working CUDA."
    torch.set_num_threads(4)
    device = torch.device("cuda")
    raw = read_raw(root)
    refs = load_references(root, raw)
    primary, secondary = refs["two_day"], refs["jan26"]
    day25 = raw["t"][primary["indices"]] < CUTOFFS["jan26"]
    spec = importlib.util.spec_from_file_location("expanded_combiner_dim16_original", protocol["dim16_helper"])
    original16 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original16)
    expanded_pipeline = make_pipeline()
    meta_indices = {fold: np.flatnonzero((raw["t"] >= CUTOFFS[fold]) & (raw["t"] < CUTOFFS[fold] + 86400)) for fold in ["jan24", "jan25"]}
    row_weights = {fold: user_equal_weights(raw["u"][idx])[0] for fold, idx in meta_indices.items()}
    protocol_hash = sha256(trial / "outputs/frozen_protocol.json")
    member_primary, member_secondary, member_checks = [], [], {}
    for member_index, (key, dimension, folder, combiners, prefix) in enumerate(member_specs(root.parent)):
        cache = trial / "work" / f"{key}_predictions.npz"
        if cache.exists():
            with np.load(cache) as saved:
                assert str(saved["protocol_sha256"].item()) == protocol_hash
                assert np.array_equal(saved["primary_idx"], primary["indices"]) and np.array_equal(saved["secondary_idx"], secondary["indices"])
                for fold in ["jan24", "jan25"]:
                    assert str(saved[f"{fold}_combiner_sha256"].item()) == sha256(trial / "work" / f"{key}_{fold}.cbm")
                member_primary.append(saved["primary_prediction"])
                member_secondary.append(saved["secondary_prediction"])
                member_checks[key] = json.loads(str(saved["checks_json"].item()))
            print(f"Using completed combiners and predictions: {key}", flush=True)
            continue
        print(f"\nExpanded training data: {key}", flush=True)
        original_pipeline = neural_pipeline if dimension == 8 else original16.neural_pipeline

        def features(fold, indices, expanded=False):
            with np.load(folder / fold / "data/cache" / f"split_{fold}.npz") as saved:
                fit_idx, first_idx = saved["train_idx"], saved["valid_idx"]
            assert np.array_equal(fit_idx, np.flatnonzero(raw["t"] < CUTOFFS[fold]))
            assert (raw["t"][indices] >= CUTOFFS[fold]).all() and (raw["t"][indices] < 1737936000).all()
            pipeline = expanded_pipeline if expanded else original_pipeline
            X, names, *_ = pipeline(folder / fold / "work/event_best.pt",
                                    {name: raw[name][fit_idx] for name in FIT_FIELDS},
                                    {name: raw[name][indices] for name in TARGET_FIELDS},
                                    folder / fold / "data/cache/preference_training.npz", device)
            assert len(names) == 37
            return X, names, first_idx

        X24, names, first24 = features("jan24", meta_indices["jan24"], expanded=True)
        original24, original_names, _ = features("jan24", first24)
        assert names == original_names
        positions24 = np.searchsorted(meta_indices["jan24"], first24)
        assert np.array_equal(meta_indices["jan24"][positions24], first24)
        diff24 = check_close(X24.reshape(-1, 10, 37)[positions24].reshape(-1, 37), original24,
                             f"{key} Jan24 original first-event features")
        del original24
        print(f"Fitting Jan24 weighted combiner: {key}", flush=True)
        model24 = fit_weighted(X24, raw["y"][meta_indices["jan24"]], names, row_weights["jan24"])
        model24.save_model(str(trial / "work" / f"{key}_jan24.cbm"))
        del X24
        Xprimary, primary_names, first25 = features("jan25", primary["indices"])
        assert primary_names == names
        old_primary = predict_combiner(load_combiner(combiners / f"{prefix}jan24.cbm"), Xprimary, names).reshape(-1, 10)
        primary_diff = check_close(old_primary, primary["members"][member_index], f"{key} V5 primary", rtol=0, atol=1e-15)
        new_primary = predict_combiner(model24, Xprimary, names).reshape(-1, 10)
        reloaded = predict_combiner(load_combiner(trial / "work" / f"{key}_jan24.cbm"), Xprimary, names).reshape(-1, 10)
        check_close(new_primary, reloaded, f"{key} saved primary combiner", rtol=0, atol=1e-15)
        del model24, reloaded
        assert np.array_equal(primary["indices"][day25], first25)
        X25, names25, _ = features("jan25", meta_indices["jan25"], expanded=True)
        assert names25 == names
        positions25 = np.searchsorted(meta_indices["jan25"], first25)
        assert np.array_equal(meta_indices["jan25"][positions25], first25)
        diff25 = check_close(X25.reshape(-1, 10, 37)[positions25].reshape(-1, 37),
                             Xprimary.reshape(-1, 10, 37)[day25].reshape(-1, 37),
                             f"{key} Jan25 original first-event features")
        del Xprimary
        print(f"Fitting Jan25 weighted combiner: {key}", flush=True)
        model25 = fit_weighted(X25, raw["y"][meta_indices["jan25"]], names, row_weights["jan25"])
        model25.save_model(str(trial / "work" / f"{key}_jan25.cbm"))
        del X25
        Xsecondary, secondary_names, first26 = features("jan26", secondary["indices"])
        assert secondary_names == names and np.array_equal(first26, secondary["indices"])
        old_secondary = predict_combiner(load_combiner(combiners / f"{prefix}jan25.cbm"), Xsecondary, names).reshape(-1, 10)
        secondary_diff = check_close(old_secondary, secondary["members"][member_index], f"{key} V5 secondary", rtol=0, atol=1e-15)
        new_secondary = predict_combiner(model25, Xsecondary, names).reshape(-1, 10)
        reloaded = predict_combiner(load_combiner(trial / "work" / f"{key}_jan25.cbm"), Xsecondary, names).reshape(-1, 10)
        check_close(new_secondary, reloaded, f"{key} saved secondary combiner", rtol=0, atol=1e-15)
        checks = {"jan24_first_event_feature_max_abs_difference": diff24, "jan25_first_event_feature_max_abs_difference": diff25,
                  "primary_baseline_max_abs_difference": primary_diff, "secondary_baseline_max_abs_difference": secondary_diff,
                  "saved_combiners_reproduced": True}
        np.savez_compressed(cache, primary_idx=primary["indices"], secondary_idx=secondary["indices"],
                            primary_prediction=new_primary, secondary_prediction=new_secondary,
                            protocol_sha256=protocol_hash, checks_json=json.dumps(checks),
                            jan24_combiner_sha256=sha256(trial / "work" / f"{key}_jan24.cbm"),
                            jan25_combiner_sha256=sha256(trial / "work" / f"{key}_jan25.cbm"))
        member_primary.append(new_primary)
        member_secondary.append(new_secondary)
        member_checks[key] = checks
        del Xsecondary, model25, reloaded
        torch.cuda.empty_cache()
    candidates = {"two_day": ensemble(member_primary), "jan26": ensemble(member_secondary)}
    metrics = {}
    for case, ref in refs.items():
        candidate = candidates[case]
        assert candidate.shape == ref["labels"].shape and np.isfinite(candidate).all() and ((candidate >= 0) & (candidate <= 1)).all()
        before, after = auc(ref["labels"], ref["baseline"]), auc(ref["labels"], candidate)
        metrics[case] = {"events": len(ref["indices"]), "baseline_v5_auc": before,
                         "expanded_combiner_auc": after, "auc_change": after - before,
                         "actual_click_rate": float(ref["labels"].mean()),
                         "baseline_mean_prediction": float(ref["baseline"].mean()), "candidate_mean_prediction": float(candidate.mean())}
        np.savez_compressed(trial / "work" / f"{case}_ensemble_predictions.npz", valid_idx=ref["indices"],
                            labels=ref["labels"], baseline=ref["baseline"], candidate=candidate,
                            members=np.stack(member_primary if case == "two_day" else member_secondary), weights=np.asarray(WEIGHTS))
    print(json.dumps(metrics, indent=2), flush=True)
    n = len(primary["indices"])
    event_ids = np.repeat(np.arange(n), 10)
    order_old = AucOrder(primary["labels"].ravel(), primary["baseline"].ravel(), event_ids)
    order_new = AucOrder(primary["labels"].ravel(), candidates["two_day"].ravel(), event_ids)
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    gains = []
    for iteration in range(protocol["bootstrap_resamples"]):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        gains.append(order_new.value(counts) - order_old.value(counts))
        if (iteration + 1) % 50 == 0:
            print(f"Paired whole-user bootstrap: {iteration + 1}/300", flush=True)
    interval = np.quantile(gains, [0.025, 0.975]).tolist()
    idx = primary["indices"]
    activity = np.bincount(raw["u"][raw["t"] < CUTOFFS["jan25"]], minlength=int(raw["u"].max()) + 1)[raw["u"][idx]]
    length = (raw["h"][idx] >= 0).sum(1)
    groups = {}
    for name, mask in {"activity_1_5": (activity >= 1) & (activity <= 5), "activity_16_plus": activity >= 16,
                       "unseen_with_history": (activity == 0) & (length > 0), "history_empty": length == 0,
                       "history_1_5": (length >= 1) & (length <= 5), "horizon_24_48h": ~day25}.items():
        a, b = auc(primary["labels"][mask], primary["baseline"][mask]), auc(primary["labels"][mask], candidates["two_day"][mask])
        groups[name] = {"events": int(mask.sum()), "baseline_auc": a, "candidate_auc": b,
                        "auc_change": b - a if a is not None and b is not None else None}
    verify(protocol)
    report = {
        "protocol": {k: v for k, v in protocol.items() if k not in ["input_sha256", "dim16_helper"]},
        "two_day_primary": metrics["two_day"], "jan26_secondary": metrics["jan26"],
        "primary_paired_gain_95_percentile_interval": interval, "primary_groups": groups,
        "member_checks": member_checks, "v5_reproduction_passed": True,
        "equal_user_weights_verified": True, "neural_training_performed": False,
        "supported_for_long_window_check": bool(interval[0] > 0 and metrics["jan26"]["auc_change"] >= 0),
        "sampling_unit": "whole first event per user, preserving all 10 candidate rows",
        "submission_created": False, "runtime_seconds": round(time.perf_counter() - start, 2),
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nEXPANDED COMBINER TRAINING TRIAL", flush=True)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["self-test", "setup", "evaluate"])
    args = parser.parse_args()
    if args.action == "self-test":
        from expanded_combiner_features import self_test
        self_test()
        return
    root = Path(__file__).resolve().parents[1]
    trial = root.parent / "techarena-expanded-combiner-trial"
    if args.action == "setup":
        setup(root, trial)
    else:
        protocol = json.loads((trial / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
        verify(protocol)
        evaluate(root, trial, protocol)


if __name__ == "__main__":
    main()
