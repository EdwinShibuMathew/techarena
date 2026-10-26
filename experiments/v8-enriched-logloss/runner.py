"""One fixed feature-enriched Logloss successor; stop at the first failed gate."""
import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import enriched_core as core

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("closed_v8_runner", ROOT / "experiments/v8-pairlogit/runner.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
from experiment_runner import PairedBootstrapEvaluator, compute_roc_auc

WORK, REPORTS = core.WORK, core.REPORTS
save_npz, runtime = prior.save_npz, prior.runtime


def prepare(root):
    from catboost import CatBoostClassifier, Pool
    from predict_v8 import replay_v6
    import torch
    core.require(not (root / WORK).exists(), "Preserve existing successor attempt")
    baseline = core.old.load_manifest(root)
    decision_path = root / prior.REPORTS / "decision.json"
    decision = core.read_json(decision_path)
    budget_path = root / prior.WORK / "budget.json"
    previous = core.read_json(budget_path)
    core.require(decision["status"] == "rejected_at_primary_gate" and decision["stopping_rule_honored"]
                 and not decision["submission_ready"], "Preserve the closed PairLogit decision")
    core.require(abs(previous["used_seconds"] - decision["compute_seconds"]) < 1e-6, "Previous compute differs")
    core.require(torch.cuda.is_available(), "CUDA unavailable in existing model environment")
    core.write_new(root / WORK / "budget.json", {"used_seconds": previous["used_seconds"], "phases": {},
        "carry_from": budget_path.relative_to(root).as_posix(), "carry_sha256": core.sha256(budget_path),
        "carried_seconds": previous["used_seconds"]})
    with core.Budget(root / WORK, "prepare") as budget:
        code = {name: core.sha256(root / name) for name in core.CODE_FILES}
        inputs = {**baseline["inputs_sha256"], **baseline["prepared_sha256"]}
        for path in (root / prior.WORK / "inputs.json", budget_path, decision_path,
                     root / prior.REPORTS / "screen.json", root / prior.REPORTS / "preparation.json"):
            inputs[path.relative_to(root).as_posix()] = core.sha256(path)
        # Bind the three existing caches and their generating metadata before any new fit.
        reused = {}
        for name in ("meta_jan24", "primary_jan25", "secondary_jan26"):
            path = root / prior.WORK / "cache" / (name + ".npz")
            record = core.read_json(path.with_suffix(".json"))
            core.require(core.sha256(path) == record["sha256"], "Existing feature cache changed")
            for p in (path, path.with_suffix(".json")):
                inputs[p.relative_to(root).as_posix()] = core.sha256(p)
            reused[name] = path.relative_to(root).as_posix()
        core.require(tuple(baseline["columns"]) == core.BASE_COLUMNS, "Baseline schema differs")
        parameters, resolved = {}, {}
        rng = np.random.default_rng(42)
        x = rng.normal(size=(80, 43)).astype(np.float32)
        y = np.zeros(80, dtype=np.int8)
        y[::10] = 1
        for case, path in baseline["combiners"].items():
            saved, _ = prior.original_parameters(root / path, list(core.BASE_COLUMNS))
            parameters[case] = core.classifier_parameters(saved)
            model = CatBoostClassifier(**dict(parameters[case], iterations=2))
            model.fit(Pool(x, label=y, feature_names=list(core.COLUMNS)), verbose=False)
            resolved[case] = model.get_all_params()
        replay, old_member, x_eval, columns, rowids, errors = replay_v6(root, "cuda")
        with np.load(root / baseline["baseline_cache"], allow_pickle=False) as cache:
            core.require(np.array_equal(cache["baseline"], replay) and np.array_equal(cache["old_member"], old_member)
                         and np.array_equal(cache["X"], x_eval.astype(np.float32))
                         and np.array_equal(cache["rowids"], rowids) and list(columns) == list(core.BASE_COLUMNS),
                         "Fresh V6 replay differs from the reusable evaluation cache")
        del replay, old_member, x_eval
        manifest = {"protocol": core.PROTOCOL, "protocol_sha256": core.canonical(core.PROTOCOL),
            "code_sha256": code, "inputs_sha256": inputs, "baseline_manifest": baseline,
            "classifier_parameters": parameters, "synthetic_resolved_parameters": resolved,
            "reused_caches": reused, "columns": list(core.BASE_COLUMNS),
            "models": baseline["models"], "combiners": baseline["combiners"], "cohorts": baseline["cohorts"],
            "baseline_cache": baseline["baseline_cache"], "runtime": runtime()}
        core.verify_files(root, code)
        core.verify_files(root, inputs)
        budget.check()
        core.write_new(root / WORK / "inputs.json", manifest)
        core.write_new(root / REPORTS / "preparation.json", {"complete": True, "submission_ready": False,
            "protocol_sha256": core.canonical(core.PROTOCOL), "source_sha256": code,
            "manifest_sha256": core.sha256(root / WORK / "inputs.json"), "runtime": runtime(),
            "previous_compute_seconds": previous["used_seconds"], "competition_model_fitted": False})


def features(root, manifest, raw, fold, indices, name, budget):
    """Read verified predecessor caches or create new caches in this trial only."""
    import torch
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


def fit_classifier(root, manifest, case, x_fit, labels, x_target, budget):
    from catboost import CatBoostClassifier, Pool
    y = np.asarray(labels).ravel()
    core.require(x_fit.shape == (len(y), 37) and np.isin(y, [0, 1]).all() and len(np.unique(y)) == 2,
                 "Invalid fitting rows/labels")
    x_fit, names = core.enrich(x_fit, core.BASE_COLUMNS)
    x_target, _ = core.enrich(x_target, core.BASE_COLUMNS)
    attempt = root / WORK / "models" / case
    attempt.mkdir(parents=True, exist_ok=False)
    parameters = manifest["classifier_parameters"][case]
    model = CatBoostClassifier(**parameters)
    budget.check()
    print(f"Fitting fixed {case} Logloss: {len(y)} rows, 43 features", flush=True)
    model.fit(Pool(x_fit, label=y, feature_names=names), verbose=100)
    budget.check()
    core.require(model.tree_count_ == 545 and model.get_all_params()["loss_function"] == "Logloss", "Fitted recipe differs")
    pool = Pool(x_target, feature_names=names)
    predictions = model.predict_proba(pool, thread_count=4)[:, 1]
    path = attempt / "classifier.cbm"
    model.save_model(str(path))
    restored = CatBoostClassifier()
    restored.load_model(str(path))
    core.require(np.allclose(restored.predict_proba(pool, thread_count=4)[:, 1], predictions, rtol=0, atol=1e-15), "Saved classifier replay differs")
    metadata = {"case": case, "model": path.relative_to(root).as_posix(), "model_sha256": core.sha256(path),
        "feature_names": names, "requested_parameters": parameters, "resolved_parameters": model.get_all_params(),
        "fitting_rows": len(y), "row_weights": "unit", "probability_link": "predict_proba positive class"}
    core.write_new(attempt / "fit.json", metadata)
    return predictions.reshape(-1, 10), metadata


def score_case(root, manifest, raw, key, fit_idx, idx, y, baseline, old, x_fit, x_target, budget):
    error = prior.check_original(root, manifest, key, x_target, old)
    new, fitted = fit_classifier(root, manifest, key, x_fit, raw["y"][fit_idx], x_target, budget)
    candidate = core.replacement(baseline, old, new)
    path = root / WORK / (key + "_predictions.npz")
    save_npz(path, valid_idx=idx, labels=y, baseline=baseline, candidate=candidate, old_member=old, new_member=new)
    metrics = asdict(PairedBootstrapEvaluator().evaluate(candidate, baseline, y, raw["u"][idx]))
    metrics.update({"n_bootstrap": 1000, "valid_bootstrap_samples": metrics["resamples"],
        "events": len(idx), "fitting_events": len(fit_idx), "original_member_max_error": error,
        "fitting_index_sha256": core.cache_identity("", 0, fit_idx, "", [])["event_index_sha256"],
        "predictions": path.relative_to(root).as_posix(), "predictions_sha256": core.sha256(path), "fit": fitted,
        "decomposition": prior.allocation.allocation_decomposition(y, baseline, candidate)})
    core.validate_metrics(metrics)
    budget.check()
    return metrics, candidate


def report(root, manifest, stage, results, checks):
    core.load_manifest(root)
    core.write_new(root / REPORTS / (stage + ".json"), {"complete": True, "stage": stage, "submission_ready": False,
        "protocol_sha256": core.canonical(core.PROTOCOL), "source_sha256": manifest["code_sha256"],
        "input_manifest_sha256": core.sha256(root / WORK / "inputs.json"), "results": results,
        "decision_checks": checks, "passed": all(checks.values()), "runtime": runtime()})
    print(json.dumps({"stage": stage, "decision_checks": checks, "passed": all(checks.values())}), flush=True)


def screen(root):
    manifest = core.load_manifest(root)
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
    from predict_v8_enriched import release_candidate
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
        new, fitted = fit_classifier(root, manifest, "final", x_fit, raw["y"][idx], x, budget)
        candidate = core.replacement(baseline, old, new)
    with core.Budget(root / WORK, "verify") as budget:
        release_candidate(root, manifest, fitted, rowids, candidate, budget)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "screen", "stress", "finalize"])
    args = parser.parse_args()
    root = ROOT
    if args.action == "prepare":
        prepare(root)
    else:
        passed = {"screen": screen, "stress": stress, "finalize": finalize}[args.action](root)
        if not passed:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
