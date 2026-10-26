"""Staged, time-limited V8 PairLogit experiment. No sweeps or competition upload."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np

import v8_core as core

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "experiments/v6-pairlogit-v1-api-fix"))
import event_scale_diagnostic as audit
import conditional_choice_trial as prefix
import attention_allocation_test as allocation
from experiment_runner import PairedBootstrapEvaluator, compute_roc_auc

WORK = Path("artifacts/experiments/v8-pairlogit")
REPORTS = Path("docs/results/v8-pairlogit")
ROW_PROTOCOL_SHA = "5c1b6ebe3d2c513343d404878e316783fa9091b0b03825dbd787ebbf15b47a8c"
ROW_TRAIN_SHA = "777f576848a56ba5fd4a0053edebe49d742edd6128a4bc105b9b7d2ff8a15013"
FIVE_TRAIN_SHA = "386c3342204f02dcafc545c42ac3c6c627094780eb8b60cb5dff50f84167178a"


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def copy_checked(source, destination, expected):
    core.require(core.sha256(source) == expected, f"Original input hash differs: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as original, destination.open("xb") as target:
        shutil.copyfileobj(original, target)
    core.require(core.sha256(destination) == expected, "Staged input hash differs")


def runtime():
    import torch
    import catboost
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "catboost": catboost.__version__, "sklearn": sklearn.__version__}


def original_parameters(path, names=None):
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(str(path))
    parameters = model.get_all_params()
    core.require(model.tree_count_ == 545 and len(model.feature_names_) == 37, "Original combiner tree/schema differs")
    core.require(parameters["loss_function"] == "Logloss" and parameters["depth"] == 6 and parameters["random_seed"] == 42, "Original combiner recipe differs")
    core.require(abs(parameters["learning_rate"] - .05) < 1e-7 and abs(parameters["l2_leaf_reg"] - 20) < 1e-7, "Original learning rate/L2 differs")
    core.require(names is None or model.feature_names_ == names, "Original feature order differs")
    return parameters, model.feature_names_


def smoke(parameters, names):
    from catboost import CatBoostRanker, Pool
    rng = np.random.default_rng(42)
    x, y = rng.normal(size=(80, 37)).astype(np.float32), np.zeros(80, dtype=np.int8)
    y[[1, 12, 33, 48, 67]] = 1
    pairs = core.sample_pairs(y)
    model = CatBoostRanker(**dict(parameters, iterations=2))
    model.fit(Pool(x, label=y, group_id=np.zeros(80, dtype=np.int64), pairs=pairs,
                   pairs_weight=np.ones(len(pairs)), feature_names=names), verbose=False)
    return model.get_all_params()


def prepare(root, source_root):
    from predict_v8 import replay_v6
    import torch
    work = root / WORK
    core.require(not work.exists(), "Preserve existing V8 preparation and partial outputs")
    core.require(torch.cuda.is_available(), "CUDA unavailable; run in the verified existing GPU environment")
    work.mkdir(parents=True)
    with core.Budget(work, "prepare") as budget:
        code = {name: core.sha256(root / name) for name in core.CODE_FILES}
        core.write_new(work / "protocol.json", {"protocol": core.PROTOCOL,
                                               "protocol_sha256": core.canonical(core.PROTOCOL), "code_sha256": code})
        inputs = {}
        bundle = core.read_json(source_root / "artifacts/v6/manifest.json")
        core.require(bundle["config_sha256"] == core.sha256(root / "configs/v6.json")
                     and core.manifest_line_endings_match(root / "configs/source_manifest.json", bundle["source_manifest_sha256"])
                     and bundle["expected_submission_sha256"] == core.V6_SHA, "Source bundle is not the protected V6 bundle")
        for relative, info in bundle["files"].items():
            budget.check()
            core.require(Path(relative).parts[0] in {"data", "artifacts"}, "Unexpected bundle path")
            copy_checked(core.inside(source_root, relative), core.inside(root, relative), info["sha256"])
            inputs[relative] = info["sha256"]
        copy_checked(source_root / "artifacts/v6/manifest.json", root / "artifacts/v6/manifest.json",
                     core.sha256(source_root / "artifacts/v6/manifest.json"))
        inputs["artifacts/v6/manifest.json"] = core.sha256(root / "artifacts/v6/manifest.json")
        core.verify_v6_bundle(root)
        inputs.update(core.read_json(root / "configs/source_manifest.json")["files"])

        def stage(relative, source, expected):
            destination = WORK / "source" / relative
            copy_checked(core.inside(source_root, source), core.inside(root, destination), expected)
            inputs[destination.as_posix()] = expected
            return destination.as_posix()

        reports = {}
        for key, relative, fingerprint in (
            ("row_protocol", "local/archive/techarena-row-auc-trial/outputs/frozen_protocol.json", ROW_PROTOCOL_SHA),
            ("row_training", "local/archive/techarena-row-auc-trial/outputs/training_results.json", ROW_TRAIN_SHA),
            ("five_training", "local/archive/techarena-row-auc-five-day/outputs/training_results.json", FIVE_TRAIN_SHA),
        ):
            reports[key] = stage("reports/" + key + ".json", relative, fingerprint)
        row_training = core.read_json(root / reports["row_training"])
        row_result = core.read_json(root / "docs/results/techarena-row-auc-trial/ensemble_trial_results.json")
        five_result = core.read_json(root / "docs/results/techarena-row-auc-five-day/paired_results.json")
        models = {"jan27": "artifacts/v6/dim8_seed42/event.pt"}
        for fold in ("jan24", "jan25", "jan26"):
            relative = f"dim8_seed42/{fold}/work/event_best.pt"
            models[fold] = stage(relative, "local/archive/techarena-row-auc-trial/" + relative,
                                 row_training["members"]["dim8_seed42"][fold]["sha256"])
        for fold in ("jan21", "jan22"):
            relative = f"dim8_seed42/{fold}/work/event_best.pt"
            models[fold] = stage(relative, "local/archive/techarena-row-auc-five-day/" + relative,
                                 five_result["artifact_sha256"][relative])
        combiners, cohorts = {"final": "artifacts/v6/dim8_seed42/combiner.cbm"}, {}
        for key, day in (("primary", "jan24"), ("secondary", "jan25")):
            relative = f"work/dim8_seed42_{day}.cbm"
            combiners[key] = stage(relative, "local/archive/techarena-row-auc-trial/" + relative,
                                  row_result["artifact_sha256"][relative])
        relative = "work/dim8_seed42_combiner_jan21.cbm"
        combiners["five_day"] = stage(relative, "local/archive/techarena-row-auc-five-day/" + relative,
                                     five_result["artifact_sha256"][relative])
        for key, spec in audit.SPECS.items():
            folder = "techarena-row-auc-five-day" if key == "five_day" else "techarena-row-auc-trial"
            cohorts[key] = stage("predictions/" + spec["filename"],
                                  f"local/archive/{folder}/work/{spec['filename']}", spec["sha256"])
        parameters, resolved, names, headers = {}, {}, None, {}
        for key, relative in combiners.items():
            saved, columns = original_parameters(root / relative, names)
            names = list(columns)
            parameters[key] = core.ranker_parameters(saved)
            resolved[key] = smoke(parameters[key], names)
        for fold, relative in models.items():
            checkpoint = torch.load(root / relative, map_location="cpu", weights_only=True)
            core.require(checkpoint["dimension"] == 8 and checkpoint["epoch"] == 8 and checkpoint["seed"] == 42
                         and checkpoint["cutoff"] == core.FOLDS[fold], "Prefix checkpoint metadata differs")
            core.require(checkpoint["event_objective"] == "half_event_ce_half_global_row_pair_v1", "Wrong event objective")
            core.require(all(bool(torch.isfinite(v).all()) for v in checkpoint["model_state"].values() if v.is_floating_point()), "Nonfinite original checkpoint")
            headers[fold] = {k: checkpoint.get(k) for k in ("dimension", "epoch", "seed", "stage", "cutoff", "n_users", "n_items")}
            del checkpoint
        budget.check()
        baseline, old, x, columns, rowids, errors = replay_v6(root, "cuda")
        core.require(columns == names, "Full-data feature schema differs from prefix models")
        cache = WORK / "cache/baseline_evaluation.npz"
        save_npz(root / cache, X=x.astype(np.float32), baseline=baseline, old_member=old,
                 rowids=rowids, feature_names=np.asarray(names))
        prepared = {cache.as_posix(): core.sha256(root / cache),
                    (WORK / "protocol.json").as_posix(): core.sha256(work / "protocol.json")}
        core.verify_files(root, inputs)
        core.verify_files(root, code)
        manifest = {"protocol": core.PROTOCOL, "protocol_sha256": core.canonical(core.PROTOCOL),
                    "code_sha256": code, "inputs_sha256": inputs, "prepared_sha256": prepared,
                    "models": models, "combiners": combiners, "cohorts": cohorts,
                    "columns": names, "ranker_parameters": parameters, "synthetic_resolved_parameters": resolved,
                    "checkpoint_headers": headers, "v6_member_replay_errors": errors,
                    "baseline_cache": cache.as_posix(), "runtime": runtime()}
        core.write_new(work / "inputs.json", manifest)
        core.write_new(root / REPORTS / "preparation.json", {"complete": True, "submission_ready": False,
            "protocol_sha256": manifest["protocol_sha256"], "manifest_sha256": core.sha256(work / "inputs.json"),
            "v6_csv_sha256": core.V6_SHA, "v6_member_replay_errors": errors, "checkpoint_headers": headers,
            "source_manifest_binding": {"bundle_declared_sha256": bundle["source_manifest_sha256"],
                                        "checkout_sha256": core.sha256(root / "configs/source_manifest.json"),
                                        "compatibility": "JSON LF/CRLF only; all legacy source and model bytes checked exactly"},
            "source_sha256": code, "runtime": runtime(), "competition_data_ranker_fitted": False})


def load_raw(root):
    return prefix.load_raw(root / "data/cache/train_events.npz")


def features(root, manifest, raw, fold, indices, name, budget):
    import torch
    sys.path.insert(0, str(root / "legacy"))
    from build_submission import neural_pipeline, FIT_FIELDS, TARGET_FIELDS
    checkpoint_path = root / manifest["models"][fold]
    identity = core.cache_identity(core.sha256(checkpoint_path), core.FOLDS[fold], indices,
        core.canonical({"inputs": manifest["inputs_sha256"], "code": manifest["code_sha256"]}), manifest["columns"])
    cache = root / WORK / "cache" / (name + ".npz")
    metadata = cache.with_suffix(".json")
    if cache.exists() or metadata.exists():
        record = core.read_json(metadata)
        core.require(record["identity"] == identity and core.sha256(cache) == record["sha256"], "Feature cache identity/hash differs")
        with np.load(cache, allow_pickle=False) as data:
            core.require(np.array_equal(data["indices"], indices), "Cached target order differs")
            return data["X"], cache
    budget.check()
    positive_path = root / WORK / "cache" / (fold + "_prefix_clicks.npz")
    proof_path = positive_path.with_suffix(".json")
    if not positive_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        full = prefix.load_full_positive(root / "data/cache/preference_training.npz")
        positive, proof = prefix.prefix_examples(raw, full, core.FOLDS[fold], int(checkpoint["n_users"]))
        save_npz(positive_path, **positive)
        core.write_new(proof_path, {"proof": proof, "sha256": core.sha256(positive_path)})
        del checkpoint, full, positive
    core.require(core.sha256(positive_path) == core.read_json(proof_path)["sha256"], "Prefix clicks changed")
    fit = np.flatnonzero(raw["t"] < core.FOLDS[fold])
    print(f"Replaying {fold} original features for {name}: {len(indices)} events", flush=True)
    x, names, *_ = neural_pipeline(checkpoint_path, {k: raw[k][fit] for k in FIT_FIELDS},
        {k: raw[k][indices] for k in TARGET_FIELDS}, positive_path, torch.device("cuda"))
    core.require(names == manifest["columns"] and x.shape == (len(indices) * 10, 37) and np.isfinite(x).all(), "Original feature schema/values differ")
    x = x.astype(np.float32)
    save_npz(cache, X=x, indices=indices, feature_names=np.asarray(names))
    core.write_new(metadata, {"identity": identity, "sha256": core.sha256(cache), "cache_key": core.canonical(identity)})
    budget.check()
    return x, cache


def cohort(root, manifest, raw, key):
    path = root / manifest["cohorts"][key]
    idx, y, baseline, _, _ = audit.load_cohort(path, audit.SPECS[key], raw)
    with np.load(path, allow_pickle=False) as data:
        old = data["candidate_members"][0] if key == "five_day" else data["new_dim8_seed42"]
    core.require(old.shape == y.shape and np.isfinite(old).all(), "Original member cache differs")
    core.require(np.array_equal(core.replacement(baseline, old, old), baseline), "Identity replacement changed V6")
    return idx, y, baseline, old


def check_original(root, manifest, key, x, expected):
    sys.path.insert(0, str(root / "legacy"))
    from build_submission import load_combiner, predict_combiner
    replay = predict_combiner(load_combiner(root / manifest["combiners"][key]), x, manifest["columns"]).reshape(-1, 10)
    core.require(np.allclose(replay, expected, rtol=0, atol=1e-15), f"Original {key} member replay differs; no ranker fit permitted")
    return float(np.max(np.abs(replay - expected)))


def fit_ranker(root, manifest, case, x_fit, labels, x_target, budget):
    from catboost import CatBoostRanker, Pool
    y = np.asarray(labels).ravel()
    pairs = core.sample_pairs(y)
    core.require(np.any(pairs[:, 0] // 10 != pairs[:, 1] // 10), "Sampler lost cross-event pairs")
    names = manifest["columns"]
    pool = Pool(x_fit, label=y, group_id=np.zeros(len(y), dtype=np.int64), pairs=pairs,
                pairs_weight=np.ones(len(pairs)), feature_names=names)
    target = Pool(x_target, feature_names=names)
    attempt = root / WORK / "models" / case
    attempt.mkdir(parents=True, exist_ok=False)
    save_npz(attempt / "pairs.npz", pairs=pairs)
    model = CatBoostRanker(**manifest["ranker_parameters"][case])
    budget.check()
    print(f"Fitting fixed {case} PairLogit: {len(y)} rows, {len(pairs)} explicit pairs", flush=True)
    model.fit(pool, verbose=100)
    budget.check()
    core.require(model.tree_count_ == 545 and model.get_all_params()["loss_function"] == "PairLogit", "Ranker recipe differs")
    raw_train = model.predict(pool, thread_count=4)
    bias, link = core.intercept(raw_train, y)
    raw_target = model.predict(target, thread_count=4)
    model_path = attempt / "ranker.cbm"
    model.save_model(str(model_path))
    restored = CatBoostRanker()
    restored.load_model(str(model_path))
    replay = restored.predict(target, thread_count=4)
    core.require(np.allclose(replay, raw_target, rtol=0, atol=1e-15), "Saved ranker replay differs")
    link_path = attempt / "link.json"
    core.write_new(link_path, link)
    no_click = np.asarray(labels).reshape(-1, 10).sum(axis=1) == 0
    metadata = {"model": model_path.relative_to(root).as_posix(), "link": link_path.relative_to(root).as_posix(),
        "model_sha256": core.sha256(model_path), "link_sha256": core.sha256(link_path),
        "pairs_sha256": core.sha256(attempt / "pairs.npz"), "pair_count": len(pairs),
        "cross_event_pairs": int((pairs[:, 0] // 10 != pairs[:, 1] // 10).sum()),
        "pairs_with_no_click_negative": int(no_click[pairs[:, 1] // 10].sum()),
        "training_link": link, "resolved_parameters": model.get_all_params()}
    core.write_new(attempt / "fit.json", metadata)
    return core.sigmoid(raw_target + bias).reshape(-1, 10), metadata


def score_case(root, manifest, raw, key, fit_idx, idx, y, baseline, old, x_fit, x_target, budget):
    error = check_original(root, manifest, key, x_target, old)
    new, fitted = fit_ranker(root, manifest, key, x_fit, raw["y"][fit_idx], x_target, budget)
    candidate = core.replacement(baseline, old, new)
    path = root / WORK / (key + "_predictions.npz")
    save_npz(path, valid_idx=idx, labels=y, baseline=baseline, candidate=candidate, old_member=old, new_member=new)
    metrics = asdict(PairedBootstrapEvaluator().evaluate(candidate, baseline, y, raw["u"][idx]))
    metrics.update({"events": len(idx), "fitting_events": len(fit_idx), "original_member_max_error": error,
                    "fitting_index_sha256": core.cache_identity("", 0, fit_idx, "", [])["event_index_sha256"],
                    "predictions_sha256": core.sha256(path), "fit": fitted,
                    "decomposition": allocation.allocation_decomposition(y, baseline, candidate)})
    budget.check()
    return metrics, candidate


def screen(root):
    manifest = core.load_manifest(root)
    report = root / REPORTS / "screen.json"
    core.require(not report.exists(), "Preserve existing screen result")
    with core.Budget(root / WORK, "screen") as budget:
        raw = load_raw(root)
        idx25, y25, base25, old25 = cohort(root, manifest, raw, "primary")
        idx26, y26, base26, old26 = cohort(root, manifest, raw, "secondary")
        meta24 = core.first_indices(raw["u"], raw["t"], core.FOLDS["jan24"], core.FOLDS["jan25"])
        day25 = raw["t"][idx25] < core.FOLDS["jan26"]
        meta25 = idx25[day25]
        core.temporal_check(raw, meta24, idx25, core.FOLDS["jan24"], core.FOLDS["jan25"])
        core.temporal_check(raw, meta25, idx26, core.FOLDS["jan25"], core.FOLDS["jan26"])
        x24, _ = features(root, manifest, raw, "jan24", meta24, "meta_jan24", budget)
        x25, _ = features(root, manifest, raw, "jan25", idx25, "primary_jan25", budget)
        x26, _ = features(root, manifest, raw, "jan26", idx26, "secondary_jan26", budget)
        # Verify both comparators before fitting either competition ranker.
        check_original(root, manifest, "primary", x25, old25)
        check_original(root, manifest, "secondary", x26, old26)
        results = {}
        results["primary"], _ = score_case(root, manifest, raw, "primary", meta24, idx25, y25, base25, old25, x24, x25, budget)
        results["secondary"], _ = score_case(root, manifest, raw, "secondary", meta25, idx26, y26, base26, old26,
                                               x25[np.repeat(day25, 10)], x26, budget)
        gates = core.screen_gates(results)
        core.load_manifest(root)
        core.write_new(report, {"complete": True, "stage": "screen", "submission_ready": False,
            "protocol_sha256": core.canonical(core.PROTOCOL), "source_sha256": manifest["code_sha256"],
            "input_manifest_sha256": core.sha256(root / WORK / "inputs.json"),
            "results": results, "decision_checks": gates, "passed": all(gates.values()), "runtime": runtime()})
        print(json.dumps({"stage": "screen", "decision_checks": gates, "passed": all(gates.values())}), flush=True)
    return all(gates.values())


def prerequisite(root, stage):
    report = core.read_json(root / REPORTS / (stage + ".json"))
    core.require(report.get("complete") is True and report["protocol_sha256"] == core.canonical(core.PROTOCOL), "Incomplete or different prerequisite")
    checks = core.screen_gates(report["results"]) if stage == "screen" else core.stress_gates(report["results"])
    core.require(all(checks.values()) and report.get("passed") is True, f"{stage} gate failed: fixed recipe is closed")
    return report


def stress(root):
    prerequisite(root, "screen")
    manifest = core.load_manifest(root)
    report = root / REPORTS / "stress.json"
    core.require(not report.exists(), "Preserve existing stress result")
    with core.Budget(root / WORK, "stress") as budget:
        raw = load_raw(root)
        idx, y, baseline, old = cohort(root, manifest, raw, "five_day")
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
        result["daily_exploratory"] = {}
        for day in range(5):
            mask = (hours >= 24 * day) & (hours < 24 * (day + 1))
            a, b = compute_roc_auc(y[mask], baseline[mask]), compute_roc_auc(y[mask], candidate[mask])
            result["daily_exploratory"][str(day)] = {"events": int(mask.sum()), "baseline_auc": a, "candidate_auc": b, "auc_diff": b - a}
        gates = core.stress_gates(result)
        core.load_manifest(root)
        core.write_new(report, {"complete": True, "stage": "stress", "submission_ready": False,
            "protocol_sha256": core.canonical(core.PROTOCOL), "source_sha256": manifest["code_sha256"],
            "results": result, "decision_checks": gates, "passed": all(gates.values()), "runtime": runtime()})
        print(json.dumps({"stage": "stress", "decision_checks": gates, "passed": all(gates.values())}), flush=True)
    return all(gates.values())


def finalize(root):
    from predict_v8 import export_verified, compute_candidate
    prerequisite(root, "screen")
    prerequisite(root, "stress")
    manifest = core.load_manifest(root)
    core.require(not (root / "configs/v8.json").exists(), "Preserve existing V8 release configuration")
    with core.Budget(root / WORK, "finalize") as budget:
        raw = load_raw(root)
        idx, _, _, _ = cohort(root, manifest, raw, "secondary")
        core.require(np.array_equal(np.sort(idx), core.first_indices(raw["u"], raw["t"], core.FOLDS["jan26"], core.FOLDS["jan27"])), "Final meta cohort differs")
        x_fit, _ = features(root, manifest, raw, "jan26", idx, "secondary_jan26", budget)
        with np.load(root / manifest["baseline_cache"], allow_pickle=False) as cache:
            x_target, baseline, old, rowids = cache["X"], cache["baseline"], cache["old_member"], cache["rowids"]
        new, fitted = fit_ranker(root, manifest, "final", x_fit, raw["y"][idx], x_target, budget)
        p = core.replacement(baseline, old, new)
        draft = root / WORK / "draft.csv"
        fingerprint = export_verified(root, draft, rowids, p)
        files = {fitted["model"]: fitted["model_sha256"], fitted["link"]: fitted["link_sha256"]}
        for stage in ("screen", "stress"):
            relative = (REPORTS / (stage + ".json")).as_posix()
            files[relative] = core.sha256(root / relative)
        config = {"version": "v8_pairlogit_seed42", "submission_ready": False,
            "evaluation_rows": 529390, "evaluation_events": 52939, "member_weight": 2 / 9, "feature_count": 37,
            "model": fitted["model"], "link": fitted["link"], "feature_names": manifest["columns"],
            "protocol_sha256": core.canonical(core.PROTOCOL), "baseline_sha256": core.V6_SHA,
            "submission_sha256": fingerprint, "files_sha256": files, "code_sha256": manifest["code_sha256"],
            "screen_report": (REPORTS / "screen.json").as_posix(), "stress_report": (REPORTS / "stress.json").as_posix()}
        core.write_new(root / WORK / "candidate_config.json", config)
    with core.Budget(root / WORK, "verify") as budget:
        # Reconstruct all features from saved assets, without reusing final feature caches.
        replay_ids, replay_p, errors = compute_candidate(root, config, "cuda")
        core.require(np.array_equal(replay_ids, rowids) and np.allclose(replay_p, p, rtol=0, atol=1e-15), "Independent V8 prediction replay differs")
        export_verified(root, root / WORK / "independent_replay.csv", replay_ids, replay_p, fingerprint)
        budget.check()
        core.load_manifest(root)
        output = root / "artifacts/submissions/v8_candidate.csv"
        export_verified(root, output, replay_ids, replay_p, fingerprint,
                        protected=[root / name for name in manifest["inputs_sha256"]])
        config["submission_ready"] = True
        core.write_new(root / "configs/v8.json", config)
        core.write_new(root / REPORTS / "release.json", {"complete": True, "submission_ready": True,
            "submission_sha256": fingerprint, "configuration_sha256": core.sha256(root / "configs/v8.json"),
            "rows": len(rowids), "v6_member_replay_errors": errors, "independent_csv_replay": True,
            "protocol_sha256": core.canonical(core.PROTOCOL), "model": fitted, "runtime": runtime(),
            "competition_upload_performed": False})
        print(json.dumps({"submission_ready": True, "output": str(output), "sha256": fingerprint}), flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "screen", "stress", "finalize"):
        command = commands.add_parser(action)
        command.add_argument("--repo", type=Path, default=ROOT)
        if action == "prepare":
            command.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo.resolve(strict=True)
    if args.action == "prepare":
        prepare(root, args.source_root.resolve(strict=True))
        return
    passed = {"screen": screen, "stress": stress, "finalize": finalize}[args.action](root)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
