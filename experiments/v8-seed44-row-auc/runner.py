"""Verify assets, train one seed44 recipe, enforce gates, and release only after replay."""
import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import seed44_core as core

ROOT = core.ROOT
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("seed44_closed_helpers", ROOT / "experiments/v8-pairlogit/runner.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
from experiment_runner import PairedBootstrapEvaluator, compute_roc_auc
from predict_v8 import replay_v6

WORK, REPORTS = core.WORK, core.REPORTS
save_npz, runtime = old.save_npz, old.runtime


def prefix_path(root, fold):
    return root / WORK / "prefix" / (fold + ".npz")


def feature_matrix(root, manifest, raw, fold, indices, checkpoint, name, budget, device="cuda"):
    import torch
    from build_submission import neural_pipeline, FIT_FIELDS, TARGET_FIELDS
    indices = np.asarray(indices, dtype=np.int64)
    cutoff = core.FOLDS[fold]
    core.require((raw["t"][indices] >= cutoff).all(), "Feature event predates neural cutoff")
    identity = {"checkpoint_sha256": core.sha256(checkpoint), "cutoff": cutoff,
        "indices_sha256": __import__("hashlib").sha256(indices.tobytes()).hexdigest(),
        "inputs_sha256": core.canonical(manifest["inputs_sha256"]), "columns": list(core.BASE_COLUMNS), "device": device}
    path = root / WORK / "cache" / (name + ".npz")
    metadata = path.with_suffix(".json")
    if path.exists() or metadata.exists():
        record = core.read_json(metadata)
        core.require(record["identity"] == identity and core.sha256(path) == record["sha256"], "Stale seed44 cache")
        with np.load(path, allow_pickle=False) as data:
            core.require(np.array_equal(data["indices"], indices) and tuple(data["feature_names"]) == core.BASE_COLUMNS, "Cache event/schema differs")
            return data["X"]
    fitting = np.flatnonzero(raw["t"] < cutoff)
    budget.check()
    x, names, *_ = neural_pipeline(checkpoint, {k: raw[k][fitting] for k in FIT_FIELDS},
        {k: raw[k][indices] for k in TARGET_FIELDS}, prefix_path(root, fold), torch.device(device))
    core.require(tuple(names) == core.BASE_COLUMNS and x.shape == (len(indices) * 10, 37)
                 and np.isfinite(x).all(), "Seed44 feature schema/values differ")
    x = x.astype(np.float32)
    save_npz(path, X=x, indices=indices, feature_names=np.asarray(names))
    core.write_new(metadata, {"identity": identity, "sha256": core.sha256(path)})
    budget.check()
    return x


def cohort(root, manifest, raw, case):
    indices, labels, baseline, _, _ = old.audit.load_cohort(root / manifest["cohorts"][case], old.audit.SPECS[case], raw)
    with np.load(root / manifest["original_predictions"][case], allow_pickle=False) as data:
        core.require(np.array_equal(data["indices"], indices), "Original seed44 event order differs")
        original = data["probabilities"]
    core.require(original.shape == labels.shape and np.isfinite(original).all(), "Invalid original seed44 predictions")
    return indices, labels, baseline, original


def prepare(root, source_root):
    import torch
    from catboost import CatBoostClassifier, Pool
    from build_submission import load_combiner, predict_combiner
    core.require(not (root / WORK).exists(), "Preserve previous seed44 attempt")
    budget_path = root / core.previous.WORK / "budget.json"
    carry = core.read_json(budget_path)
    decision_path = root / core.previous.REPORTS / "decision.json"
    decision = core.read_json(decision_path)
    core.require(decision["status"] == "rejected_at_primary_gate" and decision["total_compute_seconds"] == carry["used_seconds"], "Prior budget/decision differs")
    core.write_new(root / WORK / "budget.json", {"used_seconds": carry["used_seconds"], "phases": {},
        "carried_seconds": carry["used_seconds"], "carry_from": budget_path.relative_to(root).as_posix(), "carry_sha256": core.sha256(budget_path)})
    with core.Budget(root / WORK, "prepare") as budget:
        core.require(torch.cuda.is_available(), "Existing CUDA model environment required")
        from row_auc_objective import self_test
        self_test()
        prior_manifest = core.previous.load_manifest(root)
        inputs = dict(prior_manifest["inputs_sha256"])
        code = {path: core.sha256(root / path) for path in core.CODE_FILES}
        for path in (budget_path, decision_path, root / core.previous.WORK / "inputs.json", root / core.previous.REPORTS / "preparation.json"):
            inputs[path.relative_to(root).as_posix()] = core.sha256(path)

        def stage(relative, source, expected=None):
            original = core.inside(source_root, source)
            fingerprint = core.sha256(original)
            core.require(expected is None or fingerprint == expected, f"Reviewed asset changed: {source}")
            destination = WORK / "source" / relative
            old.copy_checked(original, core.inside(root, destination), fingerprint)
            inputs[destination.as_posix()] = fingerprint
            return destination.as_posix()

        reports = {}
        for name, source in (
            ("short_protocol", "local/archive/techarena-seed44-trial/outputs/frozen_protocol.json"),
            ("short_training", "local/archive/techarena-seed44-trial/outputs/neural_training_results.json"),
            ("five_protocol", "local/archive/techarena-v5-long-window/outputs/frozen_protocol.json"),
            ("five_training", "local/archive/techarena-v5-long-window/outputs/training_results.json"),
        ):
            reports[name] = stage("reports/" + name + ".json", source)
        short_report = core.read_json(root / reports["short_training"])
        five_report = core.read_json(root / reports["five_training"])
        preferences, originals, headers = {}, {}, {}
        for fold in ("jan24", "jan25", "jan26", "jan21", "jan22"):
            archive = "techarena-seed44-trial" if fold in core.SHORT_HASHES else "techarena-v5-long-window/dim8_seed44"
            for number, filename in enumerate(("preference_best.pt", "event_best.pt")):
                expected = (core.SHORT_HASHES[fold][number] if fold in core.SHORT_HASHES else
                            five_report["members"]["dim8_seed44"][fold][filename]["sha256"])
                relative = stage(f"{fold}/{filename}", f"local/archive/{archive}/{fold}/work/{filename}", expected)
                (preferences if number == 0 else originals)[fold] = relative
                model = torch.load(root / relative, map_location="cpu", weights_only=True)
                core.require(model["dimension"] == 8 and model["epoch"] == (260 if number == 0 else 8)
                             and model["n_items"] == 50000, "Historical model shape/stage differs")
                for key, value in (("seed", 44), ("cutoff", core.FOLDS[fold])):
                    core.require(key not in model or model[key] == value, "Conflicting historical prefix metadata")
                if number == 0:
                    core.require(model["fine_tuning_sampling_seed"] == 1044 and model["fine_tuning_initial_epoch"] == 196, "Wrong seed44 preference")
                if fold in core.SHORT_HASHES:
                    stage_name = "fine_tuned_preference" if number == 0 else "event"
                    core.require(model["validation_auc"] == short_report["folds"][fold][stage_name]["validation_auc"], "Historical training report differs")
                core.require(all(bool(torch.isfinite(v).all()) for v in model["model_state"].values() if v.is_floating_point()), "Nonfinite historical weights")
                headers[f"{fold}/{filename}"] = {k: model.get(k) for k in ("epoch", "dimension", "seed", "cutoff", "fine_tuning_sampling_seed", "n_users")}
                del model
        preferences["jan27"] = "artifacts/v6/dim8_seed44/preference.pt"
        combiners = {}
        for case, source in (
            ("primary", "local/archive/techarena-seed44-trial/work/combiner_jan24.cbm"),
            ("secondary", "local/archive/techarena-seed44-trial/work/combiner_jan25.cbm"),
            ("five_day", "local/archive/techarena-v5-long-window/work/dim8_seed44_combiner_jan21.cbm"),
        ):
            combiners[case] = stage(f"{case}/combiner.cbm", source, core.COMBINER_HASHES[case])
        core.require(core.sha256(root / "artifacts/v6/dim8_seed44/combiner.cbm") == core.COMBINER_HASHES["secondary"], "Final V6 combiner no longer equals Jan25 combiner")
        raw = old.load_raw(root)
        parameters = {}
        for case, path in combiners.items():
            saved, columns = old.original_parameters(root / path, list(core.BASE_COLUMNS))
            parameters[case] = core.previous.prior.classifier_parameters(saved)
            rng = np.random.default_rng(42)
            model = CatBoostClassifier(**dict(parameters[case], iterations=2))
            model.fit(Pool(rng.normal(size=(80, 37)).astype(np.float32), label=np.arange(80) % 10 == 0,
                           feature_names=columns), verbose=False)
        for fold, relative in preferences.items():
            checkpoint = torch.load(root / relative, map_location="cpu", weights_only=True)
            core.require(checkpoint["epoch"] == 260 and checkpoint["dimension"] == 8
                         and checkpoint["fine_tuning_sampling_seed"] == 1044, "Wrong preference initialization")
            full = old.prefix.load_full_positive(root / "data/cache/preference_training.npz")
            positive, proof = old.prefix.prefix_examples(raw, full, core.FOLDS[fold], int(checkpoint["n_users"]))
            path = prefix_path(root, fold)
            save_npz(path, **positive)
            core.write_new(path.with_suffix(".json"), {"proof": proof, "sha256": core.sha256(path)})
            for file in (path, path.with_suffix(".json")):
                inputs[file.relative_to(root).as_posix()] = core.sha256(file)
            del full, positive, checkpoint
            budget.check()
        manifest = {"protocol": core.PROTOCOL, "protocol_sha256": core.canonical(core.PROTOCOL), "code_sha256": code,
            "inputs_sha256": inputs, "preferences": preferences, "original_models": originals, "original_combiners": combiners,
            "parameters": parameters, "cohorts": prior_manifest["cohorts"], "original_predictions": {}, "runtime": runtime()}
        reconciliation = {}
        for case, (_, fold) in core.CASE_FOLDS.items():
            indices, labels, baseline, _, _ = old.audit.load_cohort(root / manifest["cohorts"][case], old.audit.SPECS[case], raw)
            x = feature_matrix(root, manifest, raw, fold, indices, root / originals[fold], "original_" + case, budget)
            p = predict_combiner(load_combiner(root / combiners[case]), x, list(core.BASE_COLUMNS)).reshape(-1, 10)
            auc = compute_roc_auc(labels, p)
            if case == "five_day":
                with np.load(root / manifest["cohorts"][case], allow_pickle=False) as data:
                    expected = data["candidate_members"][2]
                core.require(np.allclose(p, expected, rtol=0, atol=1e-15), "Five-day original seed44 vector differs")
            else:
                expected_auc = .810437327413879 if case == "primary" else .8200127829764116
                core.require(abs(auc - expected_auc) < 1e-12, "Original seed44 historical AUC differs")
            path = root / WORK / "cache" / ("original_" + case + "_predictions.npz")
            save_npz(path, indices=indices, probabilities=p)
            manifest["original_predictions"][case] = path.relative_to(root).as_posix()
            inputs[path.relative_to(root).as_posix()] = core.sha256(path)
            reconciliation[case] = {"events": len(indices), "original_member_auc": auc,
                "validation": "full vector at 1e-15" if case == "five_day" else "historical scalar AUC at 1e-12; checkpoint/combiner reviewed hashes"}
        baseline, _, _, _, rowids, errors = replay_v6(root, "cuda")
        with np.load(root / "artifacts/v6/dim8_seed44/reference_predictions.npz", allow_pickle=False) as data:
            original = data["combiner_predictions"].reshape(-1, 10)
        cache = root / WORK / "cache/baseline_evaluation.npz"
        save_npz(cache, baseline=baseline, old_member=original, rowids=rowids)
        manifest["baseline_cache"] = cache.relative_to(root).as_posix()
        inputs[manifest["baseline_cache"]] = core.sha256(cache)
        core.verify_files(root, inputs)
        core.verify_files(root, code)
        budget.check()
        core.write_new(root / WORK / "inputs.json", manifest)
        core.write_new(root / REPORTS / "preparation.json", {"complete": True, "submission_ready": False,
            "protocol_sha256": core.canonical(core.PROTOCOL), "manifest_sha256": core.sha256(root / WORK / "inputs.json"),
            "source_sha256": code, "headers": headers, "historical_replay": reconciliation,
            "v6_csv_sha256": core.V6_SHA, "v6_member_replay_errors": errors, "runtime": runtime(),
            "legacy_missing_metadata_policy": "reviewed file hashes plus source/fold/report lineage; absent fields stay null",
            "previous_compute_seconds": carry["used_seconds"], "new_neural_training_performed": False})


def train_fold(root, manifest, fold, budget):
    path = root / core.neural_path(fold)
    if path.exists():
        core.validate_neural(root, manifest, fold)
        return
    budget.run([sys.executable, "-u", str(root / "experiments/v8-seed44-row-auc/train_seed44.py"), fold], root)
    core.validate_neural(root, manifest, fold)


def score_case(root, manifest, raw, case, fit_idx, idx, labels, baseline, original, x_fit, x_target, budget):
    from catboost import CatBoostClassifier, Pool
    model = CatBoostClassifier(**manifest["parameters"][case])
    budget.check()
    model.fit(Pool(x_fit, label=raw["y"][fit_idx].ravel(), feature_names=list(core.BASE_COLUMNS)), verbose=False)
    path = root / WORK / "combiners" / (case + ".cbm")
    path.parent.mkdir(parents=True, exist_ok=True)
    core.require(not path.exists(), "Preserve existing combiner")
    model.save_model(str(path))
    pool = Pool(x_target, feature_names=list(core.BASE_COLUMNS))
    new = model.predict_proba(pool, thread_count=4)[:, 1].reshape(-1, 10)
    restored = CatBoostClassifier()
    restored.load_model(str(path))
    core.require(model.tree_count_ == 545 and np.allclose(restored.predict_proba(pool, thread_count=4)[:, 1], new.ravel(), rtol=0, atol=1e-15), "Combiner save/reload differs")
    candidate = core.replacement(baseline, original, new)
    fitted = {"case": case, "model": path.relative_to(root).as_posix(), "model_sha256": core.sha256(path),
        "parameters": manifest["parameters"][case], "resolved_parameters": model.get_all_params(),
        "neural_sha256": {core.neural_path(f).as_posix(): core.sha256(root / core.neural_path(f)) for f in core.CASE_FOLDS[case]}}
    predictions = root / WORK / (case + "_predictions.npz")
    save_npz(predictions, indices=idx, labels=labels, baseline=baseline, candidate=candidate, old_member=original, new_member=new)
    metrics = asdict(PairedBootstrapEvaluator().evaluate(candidate, baseline, labels, raw["u"][idx]))
    metrics.update({"n_bootstrap": 1000, "valid_bootstrap_samples": metrics["resamples"], "events": len(idx),
        "fitting_events": len(fit_idx), "fit": fitted, "predictions": predictions.relative_to(root).as_posix(),
        "predictions_sha256": core.sha256(predictions), "decomposition": old.allocation.allocation_decomposition(labels, baseline, candidate)})
    core.previous.prior.validate_metrics(metrics)
    budget.check()
    return metrics, candidate


def report(root, manifest, stage, results, checks):
    core.load_manifest(root)
    core.write_new(root / REPORTS / (stage + ".json"), {"complete": True, "submission_ready": False,
        "protocol_sha256": core.canonical(core.PROTOCOL), "source_sha256": manifest["code_sha256"],
        "input_manifest_sha256": core.sha256(root / WORK / "inputs.json"), "results": results,
        "decision_checks": checks, "passed": all(checks.values()), "runtime": runtime()})
    print(json.dumps({"stage": stage, "decision_checks": checks, "passed": all(checks.values())}), flush=True)


def screen(root):
    manifest = core.load_manifest(root)
    core.require(not (root / REPORTS / "screen.json").exists(), "Preserve prior screen")
    with core.Budget(root / WORK, "screen") as budget:
        raw = old.load_raw(root)
        idx25, y25, base25, old25 = cohort(root, manifest, raw, "primary")
        idx26, y26, base26, old26 = cohort(root, manifest, raw, "secondary")
        meta24 = core.first_indices(raw["u"], raw["t"], core.FOLDS["jan24"], core.FOLDS["jan25"])
        day25 = raw["t"][idx25] < core.FOLDS["jan26"]
        meta25 = idx25[day25]
        core.temporal_check(raw, meta24, idx25, core.FOLDS["jan24"], core.FOLDS["jan25"])
        core.temporal_check(raw, meta25, idx26, core.FOLDS["jan25"], core.FOLDS["jan26"])
        for fold in ("jan24", "jan25", "jan26"):
            train_fold(root, manifest, fold, budget)
        x24 = feature_matrix(root, manifest, raw, "jan24", meta24, root / core.neural_path("jan24"), "new_meta24", budget)
        x25 = feature_matrix(root, manifest, raw, "jan25", idx25, root / core.neural_path("jan25"), "new_primary", budget)
        x26 = feature_matrix(root, manifest, raw, "jan26", idx26, root / core.neural_path("jan26"), "new_secondary", budget)
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
    core.require(not (root / REPORTS / "stress.json").exists(), "Preserve prior stress")
    with core.Budget(root / WORK, "stress") as budget:
        raw = old.load_raw(root)
        idx, labels, baseline, original = cohort(root, manifest, raw, "five_day")
        meta = core.first_indices(raw["u"], raw["t"], core.FOLDS["jan21"], core.FOLDS["jan22"])
        core.temporal_check(raw, meta, idx, core.FOLDS["jan21"], core.FOLDS["jan22"])
        for fold in ("jan21", "jan22"):
            train_fold(root, manifest, fold, budget)
        x_fit = feature_matrix(root, manifest, raw, "jan21", meta, root / core.neural_path("jan21"), "new_meta21", budget)
        x_target = feature_matrix(root, manifest, raw, "jan22", idx, root / core.neural_path("jan22"), "new_five", budget)
        result, candidate = score_case(root, manifest, raw, "five_day", meta, idx, labels, baseline, original, x_fit, x_target, budget)
        hours = (raw["t"][idx] - core.FOLDS["jan22"]) / 3600
        late = (hours >= 48) & (hours < 120)
        core.require(int(late.sum()) == 7596, "Late cohort differs")
        before, after = compute_roc_auc(labels[late], baseline[late]), compute_roc_auc(labels[late], candidate[late])
        result["late_48_120h"] = {"events": int(late.sum()), "baseline_auc": before, "candidate_auc": after, "auc_diff": after - before}
        checks = core.stress_gates(result)
        report(root, manifest, "stress", result, checks)
    return all(checks.values())


def finalize(root):
    from predict_v8_seed44 import release_candidate
    manifest = core.load_manifest(root)
    core.validate_report(root, manifest, "screen")
    core.validate_report(root, manifest, "stress")
    core.require(not (root / "configs/v8.json").exists(), "Preserve existing release")
    with core.Budget(root / WORK, "finalize") as budget:
        train_fold(root, manifest, "jan27", budget)
    with core.Budget(root / WORK, "verify") as budget:
        release_candidate(root, manifest, budget)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "screen", "stress", "finalize"])
    parser.add_argument("--source-root", type=Path, default=ROOT.parents[2])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(ROOT, args.source_root.resolve())
    elif not {"screen": screen, "stress": stress, "finalize": finalize}[args.action](ROOT):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
