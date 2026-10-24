"""One fixed V5 trial: add past-exposure and activity summaries to combiners.

In techarena-final: self-test, setup, evaluate. Reuses every neural checkpoint,
fits ten 53-feature combiners, then tests 50% V5 + 50% augmented ensemble.
Everything new lives in techarena-exposure-trial. No submission is created.
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from exposure_features import ExposureFeatures, NAMES, SMOOTHING, RECENT_HOURS, MAX_AGE_HOURS
from run_expanded_combiner_trial import member_specs, ensemble, reference_paths, load_references, read_raw, CUTOFFS, WEIGHTS
from audit_v5 import AucOrder, auc
from submission_export import sha256


CONFIG = {
    "extra_features": NAMES, "base_feature_count": 37, "total_feature_count": 53,
    "recent_window_hours": RECENT_HOURS, "maximum_age_hours": MAX_AGE_HOURS,
    "ctr_smoothing_candidate_rows": SMOOTHING,
    "time_context": "four six-hour time buckets, pooled over both day types",
    "combiner_trees": 545, "combiner_depth": 6, "combiner_learning_rate": .05,
    "combiner_l2_leaf_reg": 20, "combiner_seed": 42, "combiner_threads": 4,
    "combiner_early_stopping": False,
    "candidate_v5_weight": .5, "candidate_augmented_weight": .5,
}


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(protocol):
    assert protocol["configuration"] == CONFIG and protocol["member_weights"] == WEIGHTS
    for path, expected in protocol["input_sha256"].items():
        assert sha256(Path(path)) == expected, f"Changed frozen input: {path}"


def candidate_blend(baseline, augmented):
    assert baseline.shape == augmented.shape and baseline.ndim == 2 and baseline.shape[1] == 10
    for p in [baseline, augmented]:
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return CONFIG["candidate_v5_weight"] * baseline + CONFIG["candidate_augmented_weight"] * augmented


def setup(root, trial):
    if trial.exists():
        protocol = read_json(trial / "outputs/frozen_protocol.json")
        verify(protocol)
        print("Exposure trial already prepared; continue with evaluate.", flush=True)
        return
    import torch
    checks = root / "outputs/five_model_submission_checks.json"
    saved_checks = read_json(checks)
    assert saved_checks["version"] == "v5" and saved_checks["submission_ready"] is True
    assert saved_checks["weights"] == WEIGHTS
    paths = [Path(__file__), checks, root / "data/cache/train_events.npz"]
    paths += [root / "src" / name for name in ["exposure_features.py", "run_expanded_combiner_trial.py",
              "expanded_combiner_features.py", "audit_v5.py", "submission_export.py", "build_submission.py",
              "submission_features.py", "train_final_models.py"]]
    helper = root.parent / "techarena-dim16-trial/src/dim16_pipeline.py"
    paths.append(helper)
    for case in ["two_day", "jan26"]:
        paths += reference_paths(root.parent, case)
    for key, dimension, folder, combiners, prefix in member_specs(root.parent):
        for fold, cutoff in CUTOFFS.items():
            pref_path = folder / fold / "work/preference_best.pt"
            event_path = folder / fold / "work/event_best.pt"
            pref = torch.load(pref_path, map_location="cpu", weights_only=True)
            assert pref["dimension"] == dimension and pref["epoch"] == (260 if dimension == 8 else 196)
            if dimension == 8:
                assert pref["fine_tuning_sampling_seed"] == int(key.rsplit("seed", 1)[1]) + 1000
            event = torch.load(event_path, map_location="cpu", weights_only=True)
            assert event["dimension"] == dimension and event["epoch"] == 8
            positive = folder / fold / "data/cache/preference_training.npz"
            with np.load(positive, allow_pickle=False) as saved:
                assert int(saved["cutoff"]) == cutoff and (saved["timestamps"] < cutoff).all()
            paths += [pref_path, event_path, positive, folder / fold / "data/cache" / f"split_{fold}.npz"]
            del pref, event
        paths += [combiners / f"{prefix}{day}.cbm" for day in ["jan24", "jan25"]]
    raw = read_raw(root)
    load_references(root, raw)
    protocol = {
        "experiment": "past exposure, recent activity and time-bucket CTR summaries added to V5 combiners",
        "configuration": CONFIG, "members": [s[0] for s in member_specs(root.parent)],
        "member_weights": WEIGHTS, "neural_models_unchanged": True,
        "fitting_selection": "first event per user on the same meta-training days as V5; no weighting change",
        "feature_fitting_rule": "all impression counts, outcomes, smoothing priors and timestamps strictly before each prefix cutoff",
        "target_rule": "target user, item, time and bucket are queries only; no aggregate updates from targets or histories",
        "nonclick_interpretation": "observed exposure without a click; not explicit dislike",
        "primary_check": "Jan24 prefix/meta day for combiner fitting; Jan25 prefix, first-user Jan25-26 evaluation",
        "secondary_check": "Jan25 prefix/meta day for combiner fitting; Jan26 prefix, first-user Jan26 evaluation",
        "bootstrap_resamples": 300, "bootstrap_seed": 2026,
        "support_rule": "global primary paired gain interval entirely positive and global secondary AUC change nonnegative",
        "next_gate": "same fixed feature/blend recipe versus V5 in earlier-cutoff five-day paired check before full-data promotion",
        "weight_search_performed": False, "submission_created": False,
        "status": "retrospective development experiment; reused validation dates; interval not selection-adjusted",
        "dim16_helper": str(helper), "input_sha256": {str(p.resolve(strict=True)): sha256(p) for p in paths},
    }
    (trial / "outputs").mkdir(parents=True)
    (trial / "work").mkdir()
    write_json(trial / "outputs/frozen_protocol.json", protocol)
    verify(protocol)
    print("EXPOSURE TRIAL SETUP", flush=True)
    print(json.dumps({k: v for k, v in protocol.items() if k not in ["input_sha256", "dim16_helper"]}, indent=2), flush=True)


def fit_augmented(X, labels, names):
    from catboost import CatBoostClassifier, Pool
    assert X.shape == (labels.size, 53) and len(names) == len(set(names)) == 53
    model = CatBoostClassifier(iterations=545, depth=6, learning_rate=.05, l2_leaf_reg=20,
                               loss_function="Logloss", eval_metric="AUC", random_seed=42,
                               thread_count=4, verbose=100, allow_writing_files=False)
    model.fit(Pool(X, label=labels.ravel(), feature_names=names))
    assert model.tree_count_ == 545 and model.feature_names_ == names
    return model


def load_augmented(path, names):
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(str(path))
    assert model.tree_count_ == 545 and model.feature_names_ == names and len(names) == 53
    return model


def augment(X, names, extra):
    assert X.shape[1] == len(names) == 37 and extra.shape == (len(X), 16)
    assert not set(names).intersection(NAMES)
    combined = np.column_stack([X, extra])
    assert combined.dtype == np.float32 and np.isfinite(combined).all()
    return combined, list(names) + list(NAMES)


def bootstrap(labels, baseline, candidate, protocol):
    n = len(labels)
    ids = np.repeat(np.arange(n), 10)
    before = AucOrder(labels.ravel(), baseline.ravel(), ids)
    after = AucOrder(labels.ravel(), candidate.ravel(), ids)
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    gains = []
    for iteration in range(protocol["bootstrap_resamples"]):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        old, new = before.value(counts), after.value(counts)
        assert old is not None and new is not None
        gains.append(new - old)
        if (iteration + 1) % 50 == 0:
            print(f"Global paired user bootstrap: {iteration + 1}/{protocol['bootstrap_resamples']}", flush=True)
    return np.quantile(gains, [.025, .975]).tolist()


def evaluate(root, trial, protocol):
    output = trial / "outputs/ensemble_trial_results.json"
    if output.exists():
        report = read_json(output)
        for path, expected in report["artifact_sha256"].items():
            assert sha256(trial / path) == expected, f"Changed trial output: {path}"
        print("Saved exposure trial complete; displaying it.", flush=True)
        print("\nPAST EXPOSURE FEATURE TRIAL", flush=True)
        print(output.read_text(encoding="utf-8"), flush=True)
        return
    import torch
    from build_submission import TARGET_FIELDS, FIT_FIELDS, neural_pipeline, predict_combiner, load_combiner, check_close
    assert torch.cuda.is_available(), "Activate techarena with working CUDA."
    torch.set_num_threads(4)
    device = torch.device("cuda")
    start = time.perf_counter()
    raw = read_raw(root)
    refs = load_references(root, raw)
    primary, secondary = refs["two_day"], refs["jan26"]
    day25 = raw["t"][primary["indices"]] < CUTOFFS["jan26"]
    assert np.intersect1d(primary["indices"][day25], secondary["indices"]).size == 0
    specs = member_specs(root.parent)
    with np.load(specs[0][2] / "jan24/data/cache/split_jan24.npz", allow_pickle=False) as saved:
        meta24 = saved["valid_idx"]
    assert (raw["t"][meta24] >= CUTOFFS["jan24"]).all() and (raw["t"][meta24] < CUTOFFS["jan25"]).all()
    assert len(np.unique(raw["u"][meta24])) == len(meta24)
    assert np.intersect1d(meta24, primary["indices"]).size == 0
    targets = {"jan24": meta24, "jan25": primary["indices"], "jan26": secondary["indices"]}
    extra, coverage = {}, {}
    for fold, idx in targets.items():
        print(f"Building strictly-before-cutoff exposure summaries: {fold}", flush=True)
        stats = ExposureFeatures(raw, CUTOFFS[fold])
        extra[fold], coverage[fold] = stats.transform(raw, idx)
        del stats
    spec = importlib.util.spec_from_file_location("exposure_trial_original_dim16", protocol["dim16_helper"])
    helper16 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper16)
    protocol_hash = sha256(trial / "outputs/frozen_protocol.json")
    predictions = {"two_day": [], "jan26": []}
    member_checks, artifacts = {}, []
    for member_index, (key, dimension, folder, combiners, prefix) in enumerate(specs):
        cache = trial / "work" / f"{key}_predictions.npz"
        new_paths = {day: trial / "work" / f"{key}_{day}.cbm" for day in ["jan24", "jan25"]}
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                assert saved["protocol_sha256"].item() == protocol_hash
                for case, ref in refs.items():
                    assert np.array_equal(saved[f"{case}_idx"], ref["indices"])
                    p = saved[f"{case}_prediction"]
                    assert p.shape == ref["labels"].shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                    predictions[case].append(p)
                for day, path in new_paths.items():
                    assert saved[f"{day}_sha256"].item() == sha256(path)
                member_checks[key] = json.loads(saved["checks_json"].item())
            artifacts += [cache] + list(new_paths.values())
            print(f"Using completed new combiners: {key}", flush=True)
            continue
        print(f"\nFitting exposure-augmented combiners: {key}", flush=True)
        pipeline = neural_pipeline if dimension == 8 else helper16.neural_pipeline

        def features(fold, indices):
            with np.load(folder / fold / "data/cache" / f"split_{fold}.npz", allow_pickle=False) as saved:
                fit_idx, first_idx = saved["train_idx"], saved["valid_idx"]
            assert np.array_equal(fit_idx, np.flatnonzero(raw["t"] < CUTOFFS[fold]))
            assert len(np.unique(raw["u"][indices])) == len(indices)
            X, names, *_ = pipeline(folder / fold / "work/event_best.pt",
                                    {k: raw[k][fit_idx] for k in FIT_FIELDS},
                                    {k: raw[k][indices] for k in TARGET_FIELDS},
                                    folder / fold / "data/cache/preference_training.npz", device)
            return X, names, first_idx

        X24, names, first24 = features("jan24", meta24)
        assert np.array_equal(first24, meta24)
        X24plus, names_plus = augment(X24, names, extra["jan24"])
        # The original 37 feature columns are copied without alteration.
        assert np.array_equal(X24plus[:, :37], X24)
        model24 = fit_augmented(X24plus, raw["y"][meta24], names_plus)
        model24.save_model(str(new_paths["jan24"]))
        importance24 = float(model24.feature_importances_[37:].sum())
        del X24, X24plus
        Xprimary, primary_names, first25 = features("jan25", primary["indices"])
        assert primary_names == names and np.array_equal(first25, primary["indices"][day25])
        old = predict_combiner(load_combiner(combiners / f"{prefix}jan24.cbm"), Xprimary, names).reshape(-1, 10)
        diff_primary = check_close(old, primary["members"][member_index], f"{key} V5 primary", rtol=0, atol=1e-15)
        Xprimary_plus, _ = augment(Xprimary, names, extra["jan25"])
        new_primary = predict_combiner(model24, Xprimary_plus, names_plus).reshape(-1, 10)
        reloaded = predict_combiner(load_augmented(new_paths["jan24"], names_plus), Xprimary_plus, names_plus).reshape(-1, 10)
        check_close(new_primary, reloaded, f"{key} saved primary model", rtol=0, atol=1e-15)
        del model24, old, reloaded, Xprimary
        model25 = fit_augmented(Xprimary_plus[np.repeat(day25, 10)], primary["labels"][day25], names_plus)
        model25.save_model(str(new_paths["jan25"]))
        importance25 = float(model25.feature_importances_[37:].sum())
        del Xprimary_plus
        Xsecondary, secondary_names, first26 = features("jan26", secondary["indices"])
        assert secondary_names == names and np.array_equal(first26, secondary["indices"])
        old = predict_combiner(load_combiner(combiners / f"{prefix}jan25.cbm"), Xsecondary, names).reshape(-1, 10)
        diff_secondary = check_close(old, secondary["members"][member_index], f"{key} V5 secondary", rtol=0, atol=1e-15)
        Xsecondary_plus, _ = augment(Xsecondary, names, extra["jan26"])
        new_secondary = predict_combiner(model25, Xsecondary_plus, names_plus).reshape(-1, 10)
        reloaded = predict_combiner(load_augmented(new_paths["jan25"], names_plus), Xsecondary_plus, names_plus).reshape(-1, 10)
        check_close(new_secondary, reloaded, f"{key} saved secondary model", rtol=0, atol=1e-15)
        checks = {"primary_v5_max_abs_difference": diff_primary, "secondary_v5_max_abs_difference": diff_secondary,
                  "original_feature_columns_unchanged": True, "saved_combiners_reproduced": True,
                  "extra_feature_importance_percent_jan24": importance24,
                  "extra_feature_importance_percent_jan25": importance25}
        np.savez_compressed(cache, two_day_idx=primary["indices"], jan26_idx=secondary["indices"],
                            two_day_prediction=new_primary, jan26_prediction=new_secondary,
                            protocol_sha256=protocol_hash, checks_json=json.dumps(checks),
                            jan24_sha256=sha256(new_paths["jan24"]), jan25_sha256=sha256(new_paths["jan25"]))
        predictions["two_day"].append(new_primary)
        predictions["jan26"].append(new_secondary)
        member_checks[key] = checks
        artifacts += [cache] + list(new_paths.values())
        del Xsecondary, Xsecondary_plus, model25, reloaded, old
        torch.cuda.empty_cache()
    candidates, metrics = {}, {}
    for case, ref in refs.items():
        augmented = ensemble(predictions[case])
        candidate = candidate_blend(ref["baseline"], augmented)
        candidates[case] = candidate
        before, after = auc(ref["labels"], ref["baseline"]), auc(ref["labels"], candidate)
        metrics[case] = {"events": len(ref["indices"]), "baseline_v5_auc": before,
                         "augmented_ensemble_auc": auc(ref["labels"], augmented),
                         "fixed_50_50_candidate_auc": after, "auc_change": after - before}
        path = trial / "work" / f"{case}_ensemble_predictions.npz"
        np.savez_compressed(path, valid_idx=ref["indices"], labels=ref["labels"], baseline=ref["baseline"],
                            augmented=augmented, candidate=candidate, augmented_members=np.stack(predictions[case]),
                            member_weights=np.asarray(WEIGHTS))
        artifacts.append(path)
    print(json.dumps(metrics, indent=2), flush=True)
    interval = bootstrap(primary["labels"], primary["baseline"], candidates["two_day"], protocol)
    activity = np.bincount(raw["u"][raw["t"] < CUTOFFS["jan25"]], minlength=int(raw["u"].max()) + 1)[raw["u"][primary["indices"]]]
    length = (raw["h"][primary["indices"]] >= 0).sum(1)
    repeated_ui = extra["jan25"].reshape(-1, 10, 16)[:, :, 0].max(1) > 0
    groups = {}
    for name, mask in {"with_prior_user_item_exposure": repeated_ui, "without_prior_user_item_exposure": ~repeated_ui,
                       "activity_1_5": (activity >= 1) & (activity <= 5), "activity_16_plus": activity >= 16,
                       "unseen_with_history": (activity == 0) & (length > 0), "history_empty": length == 0,
                       "horizon_24_48h": ~day25}.items():
        a = auc(primary["labels"][mask], primary["baseline"][mask])
        b = auc(primary["labels"][mask], candidates["two_day"][mask])
        groups[name] = {"events": int(mask.sum()), "baseline_auc": a, "candidate_auc": b,
                        "auc_change": b - a if a is not None and b is not None else None}
    verify(protocol)
    report = {"protocol": {k: v for k, v in protocol.items() if k not in ["input_sha256", "dim16_helper"]},
              "two_day_primary": metrics["two_day"], "jan26_secondary": metrics["jan26"],
              "primary_paired_gain_95_percentile_interval": interval,
              "primary_groups_exploratory": groups, "exposure_coverage": coverage, "member_checks": member_checks,
              "v5_reproduction_passed": True, "neural_training_performed": False,
              "supported_for_five_day_check": bool(interval[0] > 0 and metrics["jan26"]["auc_change"] >= 0),
              "sampling_unit": "whole first event per user, preserving all ten candidates; global row AUC",
              "submission_created": False, "runtime_seconds": round(time.perf_counter() - start, 2),
              "artifact_sha256": {str(p.relative_to(trial)): sha256(p) for p in artifacts}}
    write_json(output, report)
    print("\nPAST EXPOSURE FEATURE TRIAL", flush=True)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["self-test", "setup", "evaluate"])
    args = parser.parse_args()
    if args.action == "self-test":
        from exposure_features import self_test
        self_test()
        a, b = np.full((2, 10), .1), np.full((2, 10), .2)
        np.testing.assert_array_equal(candidate_blend(a, b), .5 * a + .5 * b)
        print("Fixed 50/50 blend self-test passed.")
        return
    root = Path(__file__).resolve().parents[1]
    trial = root.parent / "techarena-exposure-trial"
    if args.action == "setup":
        setup(root, trial)
    else:
        protocol = read_json(trial / "outputs/frozen_protocol.json")
        verify(protocol)
        evaluate(root, trial, protocol)


if __name__ == "__main__":
    main()
