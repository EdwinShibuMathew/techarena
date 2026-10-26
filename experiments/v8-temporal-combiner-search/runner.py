"""One fitting-only temporal search, one frozen winner, unchanged V8 promotion gates."""
import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import search_core as core

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("closed_enriched_runner", ROOT / "experiments/v8-enriched-logloss/runner.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
prior = previous.prior
from experiment_runner import PairedBootstrapEvaluator, compute_roc_auc

WORK, REPORTS = core.WORK, core.REPORTS
save_npz, runtime = prior.save_npz, prior.runtime


def prepare(root):
    from catboost import CatBoostClassifier, Pool
    from predict_v8 import replay_v6
    core.require(not (root / WORK).exists(), "Preserve previous temporal search attempt")
    previous_budget = root / previous.WORK / "budget.json"
    budget_state = core.read_json(previous_budget)
    decision_path = root / previous.REPORTS / "decision.json"
    decision = core.read_json(decision_path)
    core.require(decision["status"] == "rejected_at_primary_gate" and not decision["submission_ready"]
                 and decision["total_compute_seconds"] == budget_state["used_seconds"], "Previous decision/budget differs")
    core.write_new(root / WORK / "budget.json", {"used_seconds": budget_state["used_seconds"], "phases": {},
        "carried_seconds": budget_state["used_seconds"], "carry_from": previous_budget.relative_to(root).as_posix(),
        "carry_sha256": core.sha256(previous_budget)})
    with core.Budget(root / WORK, "prepare") as budget:
        existing = core.prior.load_manifest(root)
        code = {name: core.sha256(root / name) for name in core.CODE_FILES}
        inputs = dict(existing["inputs_sha256"])
        for path in (previous_budget, decision_path, root / previous.WORK / "inputs.json",
                     root / previous.REPORTS / "preparation.json", root / previous.REPORTS / "screen.json"):
            inputs[path.relative_to(root).as_posix()] = core.sha256(path)
        parameters, resolved = {}, {}
        rng = np.random.default_rng(42)
        x = rng.normal(size=(80, 43)).astype(np.float32)
        y = np.zeros(80, dtype=np.int8)
        y[::10] = 1
        for case, relative in existing["combiners"].items():
            saved, _ = prior.original_parameters(root / relative, list(core.BASE_COLUMNS))
            parameters[case], resolved[case] = {}, {}
            for variant in ("A", "B", "C", "reference"):
                parameters[case][variant] = core.parameters(saved, variant)
                names = list(core.COLUMNS if variant == "B" else core.BASE_COLUMNS)
                model = CatBoostClassifier(**dict(parameters[case][variant], iterations=2))
                model.fit(Pool(x[:, :len(names)], label=y, feature_names=names), verbose=False)
                resolved[case][variant] = model.get_all_params()
        baseline, old, x_eval, names, rowids, errors = replay_v6(root, "cuda")
        with np.load(root / existing["baseline_cache"], allow_pickle=False) as data:
            core.require(np.array_equal(data["baseline"], baseline) and np.array_equal(data["old_member"], old)
                         and np.array_equal(data["X"], x_eval.astype(np.float32))
                         and np.array_equal(data["rowids"], rowids) and tuple(names) == core.BASE_COLUMNS,
                         "Fresh V6 replay differs from cache")
        manifest = {"protocol": core.PROTOCOL, "protocol_sha256": core.canonical(core.PROTOCOL),
            "code_sha256": code, "inputs_sha256": inputs, "parameters": parameters,
            "synthetic_resolved_parameters": resolved, "baseline_manifest": existing["baseline_manifest"],
            "reused_caches": existing["reused_caches"], "columns": existing["columns"],
            "models": existing["models"], "combiners": existing["combiners"], "cohorts": existing["cohorts"],
            "baseline_cache": existing["baseline_cache"], "runtime": runtime()}
        core.verify_files(root, inputs)
        core.verify_files(root, code)
        budget.check()
        core.write_new(root / WORK / "inputs.json", manifest)
        core.write_new(root / REPORTS / "preparation.json", {"complete": True, "submission_ready": False,
            "manifest_sha256": core.sha256(root / WORK / "inputs.json"), "protocol_sha256": core.canonical(core.PROTOCOL),
            "source_sha256": code, "runtime": runtime(), "v6_csv_sha256": core.V6_SHA,
            "v6_member_replay_errors": errors, "previous_compute_seconds": budget_state["used_seconds"],
            "competition_classifier_fitted": False})


def fit_recipe(root, manifest, case, variant, x_fit, labels, times, x_target, budget, parameter_case=None):
    from catboost import CatBoostClassifier, Pool
    labels = np.asarray(labels)
    core.require(labels.shape == (len(x_fit) // 10, 10) and np.isin(labels, [0, 1]).all()
                 and (labels.sum(1) <= 1).all() and len(np.unique(labels)) == 2, "Invalid fitting labels")
    core.require(np.asarray(times).shape == (len(labels),), "Fitting event timeline differs")
    original_x, original_y = x_fit, labels
    calibration_x = calibration_y = cutoff = None
    if variant == "C":
        base, calibration, cutoff = core.chronological_split(times)
        x_fit, labels = x_fit[np.repeat(base, 10)], labels[base]
        calibration_x, calibration_y = original_x[np.repeat(calibration, 10)], original_y[calibration]
        core.require(len(np.unique(labels)) == 2 and len(np.unique(calibration_y)) == 2, "Calibration split lacks label classes")
    x_fit, names = core.matrix(x_fit, variant)
    target, _ = core.matrix(x_target, variant)
    params = manifest["parameters"][parameter_case or case][variant]
    directory = root / WORK / "models" / case
    directory.mkdir(parents=True, exist_ok=False)
    model = CatBoostClassifier(**params)
    budget.check()
    print(f"Fitting {case} variant {variant}: {len(x_fit)} rows, {len(names)} features", flush=True)
    model.fit(Pool(x_fit, label=labels.ravel(), feature_names=names), verbose=False)
    budget.check()
    path = directory / "classifier.cbm"
    model.save_model(str(path))
    raw = model.predict(Pool(target, feature_names=names), prediction_type="RawFormulaVal", thread_count=4)
    correction = None
    files = {path.relative_to(root).as_posix(): core.sha256(path)}
    if variant == "C":
        raw_calibration = model.predict(Pool(calibration_x, feature_names=names), prediction_type="RawFormulaVal", thread_count=4)
        correction = core.fit_correction(raw_calibration, calibration_x, calibration_y)
        correction_path = directory / "correction.json"
        core.write_new(correction_path, correction)
        files[correction_path.relative_to(root).as_posix()] = core.sha256(correction_path)
    fitted = {"case": case, "variant": variant, "model": path.relative_to(root).as_posix(),
        "correction": (directory / "correction.json").relative_to(root).as_posix() if correction else None,
        "files_sha256": files, "requested_parameters": params, "resolved_parameters": model.get_all_params(),
        "feature_names": names, "fitting_events": len(original_y), "base_fitting_events": len(labels),
        "calibration_events": 0 if calibration_y is None else len(calibration_y), "calibration_cutoff": cutoff,
        "label_sha256": __import__("hashlib").sha256(original_y.tobytes()).hexdigest(),
        "timeline_sha256": __import__("hashlib").sha256(np.asarray(times).tobytes()).hexdigest()}
    core.write_new(directory / "fit.json", fitted)
    restored = load_predictor(root, fitted, x_target)
    direct = (core.apply_correction(raw, x_target, correction) if correction else
              core.prior.old.sigmoid(raw).reshape(-1, 10))
    core.require(np.allclose(restored, direct, rtol=0, atol=1e-15), "Saved recipe replay differs")
    budget.check()
    return restored, fitted


def load_predictor(root, fitted, x):
    from catboost import CatBoostClassifier, Pool
    core.verify_files(root, fitted["files_sha256"])
    variant = fitted["variant"]
    values, names = core.matrix(x, variant)
    model = CatBoostClassifier()
    model.load_model(str(core.inside(root, fitted["model"])))
    core.require(model.tree_count_ == 545 and model.feature_names_ == names and list(model.classes_) == [0, 1], "Saved model schema differs")
    saved = model.get_all_params()
    for key, expected in fitted["requested_parameters"].items():
        if key in {"allow_writing_files", "thread_count", "use_best_model"}:
            continue
        actual = saved.get(key)
        same = np.isclose(actual, expected, rtol=0, atol=1e-7) if isinstance(expected, float) else actual == expected
        core.require(bool(same), f"Saved model parameter differs: {key}")
    raw = model.predict(Pool(values, feature_names=names), prediction_type="RawFormulaVal", thread_count=4)
    if variant == "C":
        return core.apply_correction(raw, x, core.read_json(core.inside(root, fitted["correction"])))
    core.require(fitted["correction"] is None, "Unexpected correction")
    return core.prior.old.sigmoid(raw).reshape(-1, 10)


def select(root):
    manifest = core.load_manifest(root)
    core.require(not (root / REPORTS / "selection.json").exists(), "Preserve prior selection")
    with core.Budget(root / WORK, "selection") as budget:
        raw = prior.load_raw(root)
        idx = core.first_indices(raw["u"], raw["t"], core.FOLDS["jan24"], core.FOLDS["jan25"])
        x, _ = features(root, manifest, raw, "jan24", idx, "meta_jan24", budget)
        y, times = raw["y"][idx], raw["t"][idx]
        results = {variant: {"status": "complete", "folds": []} for variant in ("A", "B", "C")}
        for number, (train, valid) in enumerate(core.development_folds(times)):
            fit_x, valid_x = x[np.repeat(train, 10)], x[np.repeat(valid, 10)]
            reference, reference_fit = fit_recipe(root, manifest, f"development_{number}_reference", "reference",
                fit_x, y[train], times[train], valid_x, budget, "primary")
            before = compute_roc_auc(y[valid], reference)
            for variant in ("A", "B", "C"):
                if results[variant]["status"] != "complete":
                    continue
                try:
                    p, fitted = fit_recipe(root, manifest, f"development_{number}_{variant}", variant,
                        fit_x, y[train], times[train], valid_x, budget, "primary")
                except core.NonConvergence as error:
                    results[variant].update(status="rejected_nonconvergence", reason=str(error))
                    continue
                after = compute_roc_auc(y[valid], p)
                path = root / WORK / f"development_{number}_{variant}.npz"
                save_npz(path, fitting_idx=idx[train], valid_idx=idx[valid], labels=y[valid], reference=reference, candidate=p)
                results[variant]["folds"].append({"fold": number, "reference_auc": before, "candidate_auc": after,
                    "auc_diff": after - before, "fitting_events": int(train.sum()), "validation_events": int(valid.sum()),
                    "reference_fit": reference_fit, "fit": fitted, "predictions": path.relative_to(root).as_posix(),
                    "predictions_sha256": core.sha256(path)})
        winner = core.choose(results)
        core.load_manifest(root)
        path = root / REPORTS / "selection.json"
        core.write_new(path, {"complete": True, "protocol_sha256": core.canonical(core.PROTOCOL),
            "source_sha256": manifest["code_sha256"], "input_manifest_sha256": core.sha256(root / WORK / "inputs.json"),
            "development_day": "Jan24", "results": results, "winner": winner, "runtime": runtime(),
            "promotion_outcomes_used_for_selection": False})
        if winner is not None:
            core.write_new(root / REPORTS / "selected_recipe.json", {"winner": winner,
                "protocol_sha256": core.canonical(core.PROTOCOL), "selection_sha256": core.sha256(path),
                "parameters": {case: manifest["parameters"][case][winner] for case in manifest["parameters"]}})
        print(json.dumps({"winner": winner, "development_gains": {v: [f["auc_diff"] for f in r["folds"]] for v, r in results.items()}}), flush=True)
    return winner is not None


def fit_classifier(root, manifest, case, x_fit, labels, x_target, budget, times):
    selected = core.selection(root, manifest)
    return fit_recipe(root, manifest, case, selected["winner"], x_fit, labels, times, x_target, budget)


def score_case(root, manifest, raw, key, fit_idx, idx, y, baseline, old, x_fit, x_target, budget):
    error = prior.check_original(root, manifest, key, x_target, old)
    new, fitted = fit_classifier(root, manifest, key, x_fit, raw["y"][fit_idx], x_target, budget, raw["t"][fit_idx])
    candidate = core.replacement(baseline, old, new)
    path = root / WORK / (key + "_predictions.npz")
    save_npz(path, valid_idx=idx, labels=y, baseline=baseline, candidate=candidate, old_member=old, new_member=new)
    metrics = asdict(PairedBootstrapEvaluator().evaluate(candidate, baseline, y, raw["u"][idx]))
    metrics.update({"n_bootstrap": 1000, "valid_bootstrap_samples": metrics["resamples"], "events": len(idx),
        "fitting_events": len(fit_idx), "original_member_max_error": error, "predictions": path.relative_to(root).as_posix(),
        "predictions_sha256": core.sha256(path), "fit": fitted,
        "decomposition": prior.allocation.allocation_decomposition(y, baseline, candidate)})
    core.prior.validate_metrics(metrics)
    budget.check()
    return metrics, candidate


def report(root, manifest, stage, results, checks):
    core.load_manifest(root)
    core.write_new(root / REPORTS / (stage + ".json"), {"complete": True, "stage": stage, "submission_ready": False,
        "protocol_sha256": core.canonical(core.PROTOCOL), "source_sha256": manifest["code_sha256"],
        "input_manifest_sha256": core.sha256(root / WORK / "inputs.json"),
        "selected_recipe_sha256": core.canonical(core.selection(root, manifest)), "results": results,
        "decision_checks": checks, "passed": all(checks.values()), "runtime": runtime()})
    print(json.dumps({"stage": stage, "decision_checks": checks, "passed": all(checks.values())}), flush=True)


def features(root, manifest, raw, fold, indices, name, budget):
    """Read verified predecessor caches or create new caches in this trial only."""
    baseline = manifest["baseline_manifest"]
    reused = manifest["reused_caches"].get(name)
    if reused:
        path = root / reused
        record = core.read_json(path.with_suffix(".json"))
        expected = core.cache_identity(core.sha256(root / manifest["models"][fold]), core.FOLDS[fold], indices,
            core.canonical({"inputs": baseline["inputs_sha256"], "code": baseline["code_sha256"]}), list(core.BASE_COLUMNS))
        core.require(record["identity"] == expected and core.sha256(path) == record["sha256"], "Stale cache identity/hash")
        with np.load(path, allow_pickle=False) as data:
            core.require(np.array_equal(data["indices"], indices) and tuple(data["feature_names"]) == core.BASE_COLUMNS,
                         "Cached event order/schema differs")
            x = data["X"]
        core.enrich(x, core.BASE_COLUMNS)
        return x, path
    import torch
    sys.path.insert(0, str(root / "legacy"))
    from build_submission import neural_pipeline, FIT_FIELDS, TARGET_FIELDS
    checkpoint = root / manifest["models"][fold]
    identity = core.cache_identity(core.sha256(checkpoint), core.FOLDS[fold], indices,
                                  core.canonical(manifest["inputs_sha256"]), list(core.BASE_COLUMNS))
    path = root / WORK / "cache" / (name + ".npz")
    metadata = path.with_suffix(".json")
    if path.exists() or metadata.exists():
        record = core.read_json(metadata)
        core.require(record["identity"] == identity and core.sha256(path) == record["sha256"], "Stale new cache")
        with np.load(path, allow_pickle=False) as data:
            core.require(np.array_equal(data["indices"], indices) and tuple(data["feature_names"]) == core.BASE_COLUMNS,
                         "New cache order/schema differs")
            return data["X"], path
    header = torch.load(checkpoint, map_location="cpu", weights_only=True)
    full = prior.prefix.load_full_positive(root / "data/cache/preference_training.npz")
    positive, proof = prior.prefix.prefix_examples(raw, full, core.FOLDS[fold], int(header["n_users"]))
    positive_path = root / WORK / "cache" / (fold + "_prefix_clicks.npz")
    save_npz(positive_path, **positive)
    core.write_new(positive_path.with_suffix(".json"), {"proof": proof, "sha256": core.sha256(positive_path)})
    del full, positive, header
    fit = np.flatnonzero(raw["t"] < core.FOLDS[fold])
    budget.check()
    x, names, *_ = neural_pipeline(checkpoint, {k: raw[k][fit] for k in FIT_FIELDS},
        {k: raw[k][indices] for k in TARGET_FIELDS}, positive_path, torch.device("cuda"))
    x = x.astype(np.float32)
    core.require(x.shape == (len(indices) * 10, 37), "Generated feature shape differs")
    core.enrich(x, names)
    save_npz(path, X=x, indices=indices, feature_names=np.asarray(names))
    core.write_new(metadata, {"identity": identity, "sha256": core.sha256(path)})
    budget.check()
    return x, path


def screen(root):
    manifest = core.load_manifest(root)
    core.selection(root, manifest)
    core.require(not (root / REPORTS / "screen.json").exists(), "Preserve existing screen")
    with core.Budget(root / WORK, "screen") as budget:
        raw = prior.load_raw(root)
        idx25, y25, base25, old25 = prior.cohort(root, manifest, raw, "primary")
        idx26, y26, base26, old26 = prior.cohort(root, manifest, raw, "secondary")
        meta24 = core.first_indices(raw["u"], raw["t"], core.FOLDS["jan24"], core.FOLDS["jan25"])
        day25 = raw["t"][idx25] < core.FOLDS["jan26"]
        meta25 = idx25[day25]
        core.temporal_check(raw, meta24, idx25, core.FOLDS["jan24"], core.FOLDS["jan25"])
        core.temporal_check(raw, meta25, idx26, core.FOLDS["jan25"], core.FOLDS["jan26"])
        x24, _ = features(root, manifest, raw, "jan24", meta24, "meta_jan24", budget)
        x25, _ = features(root, manifest, raw, "jan25", idx25, "primary_jan25", budget)
        x26, _ = features(root, manifest, raw, "jan26", idx26, "secondary_jan26", budget)
        prior.check_original(root, manifest, "primary", x25, old25)
        prior.check_original(root, manifest, "secondary", x26, old26)
        results = {}
        results["primary"], _ = score_case(root, manifest, raw, "primary", meta24, idx25, y25, base25, old25, x24, x25, budget)
        results["secondary"], _ = score_case(root, manifest, raw, "secondary", meta25, idx26, y26, base26, old26,
                                            x25[np.repeat(day25, 10)], x26, budget)
        checks = core.screen_gates(results)
        report(root, manifest, "screen", results, checks)
    return all(checks.values())


def stress(root):
    manifest = core.load_manifest(root)
    core.validate_report(root, manifest, "screen")
    core.require(not (root / REPORTS / "stress.json").exists(), "Preserve existing stress")
    with core.Budget(root / WORK, "stress") as budget:
        raw = prior.load_raw(root)
        idx, y, baseline, old = prior.cohort(root, manifest, raw, "five_day")
        meta = core.first_indices(raw["u"], raw["t"], core.FOLDS["jan21"], core.FOLDS["jan22"])
        core.temporal_check(raw, meta, idx, core.FOLDS["jan21"], core.FOLDS["jan22"])
        x_fit, _ = features(root, manifest, raw, "jan21", meta, "meta_jan21", budget)
        x_target, _ = features(root, manifest, raw, "jan22", idx, "five_day_jan22", budget)
        result, candidate = score_case(root, manifest, raw, "five_day", meta, idx, y, baseline, old, x_fit, x_target, budget)
        hours = (raw["t"][idx] - core.FOLDS["jan22"]) / 3600
        late = (hours >= 48) & (hours < 120)
        core.require(int(late.sum()) == 7596, "Late stress cohort differs")
        before, after = compute_roc_auc(y[late], baseline[late]), compute_roc_auc(y[late], candidate[late])
        result["late_48_120h"] = {"events": int(late.sum()), "baseline_auc": before, "candidate_auc": after, "auc_diff": after - before}
        checks = core.stress_gates(result)
        report(root, manifest, "stress", result, checks)
    return all(checks.values())


def finalize(root):
    from predict_v8_selected import release_candidate
    manifest = core.load_manifest(root)
    core.validate_report(root, manifest, "screen")
    core.validate_report(root, manifest, "stress")
    core.require(not (root / "configs/v8.json").exists(), "Preserve existing release")
    with core.Budget(root / WORK, "finalize") as budget:
        raw = prior.load_raw(root)
        idx, _, _, _ = prior.cohort(root, manifest, raw, "secondary")
        core.require(np.array_equal(idx, core.first_indices(raw["u"], raw["t"], core.FOLDS["jan26"], core.FOLDS["jan27"])), "Final fitting cohort differs")
        x_fit, _ = features(root, manifest, raw, "jan26", idx, "secondary_jan26", budget)
        with np.load(root / manifest["baseline_cache"], allow_pickle=False) as cache:
            x, baseline, old, rowids = cache["X"], cache["baseline"], cache["old_member"], cache["rowids"]
        new, fitted = fit_classifier(root, manifest, "final", x_fit, raw["y"][idx], x, budget, raw["t"][idx])
        candidate = core.replacement(baseline, old, new)
    with core.Budget(root / WORK, "verify") as budget:
        release_candidate(root, manifest, fitted, rowids, candidate, budget)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "selection", "screen", "stress", "finalize"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(ROOT)
    elif not {"selection": select, "screen": screen, "stress": stress, "finalize": finalize}[args.action](ROOT):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
