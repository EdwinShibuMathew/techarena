"""Verify original seed42 combiners, freeze one PairLogit screen, run separately.

GPU inference only for original feature replay; no neural fitting. Ranker fits
are CPU. All new files live in the isolated experiment. No submission export.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import event_scale_diagnostic as audit
import attention_allocation_test as shared
import conditional_choice_trial as prefix
import pairlogit_core as core


EXPERIMENT = "v6-pairlogit-v1"
HELPERS = {**prefix.HELPERS,
           "conditional_choice_trial.py": "f6e044172760accbaff9dd910cab2e6f967636a02b3ae8059d15696a1206a40c"}
LEGACY = {**prefix.LEGACY,
          "submission_features.py": "29d40f5715ba00242aa1b74013106eba0178ab80e279e33f0d56e357dd5a7788",
          "submission_export.py": "ea1c65280f70f293c6015b39fe508491ffa6d0cf4f6af3b6d3c50d98ff15036f",
          "evaluate_dim16_trial.py": "fe472e9c58a9220771ba8e30d5edcf914d9388cc03c2ea4dceb3054f15a51b8a"}
FULL_POSITIVE_SHA = "d50e41a312097f7758f7aab229879b6fa5618155121bcc7fefebfd7b5cc4bbfd"
ROW_PROTOCOL_SHA = "5c1b6ebe3d2c513343d404878e316783fa9091b0b03825dbd787ebbf15b47a8c"
FOLDS = {"jan24": 1737676800, "jan25": 1737763200, "jan26": 1737849600}
CASES = {
    "primary": {**audit.SPECS["primary"], "fit_day": "jan24", "target_fold": "jan25",
                "combiner_sha256": "813544ab4ae8a7a702ed4ef20180f34d231946ef31a890435a6303e8d20341a4"},
    "secondary": {**audit.SPECS["secondary"], "fit_day": "jan25", "target_fold": "jan26",
                  "combiner_sha256": "0382bc46dc70ce3e817e10736291a3d1ab5af88a068994fce1c7d489ca5cdebc"},
}
RECIPE = {
    "experiment": EXPERIMENT, "updated_member": "dim8_seed42", "member_weight": 2 / 9,
    "candidate_arithmetic": "cached V6 + (2/9)*(new_seed42_probability - cached_original_seed42_probability)",
    "unchanged": "all neural fits, original 37 features, four other members and ensemble weights",
    "training": {"loss_function": "PairLogit", "iterations": 545, "depth": 6, "learning_rate": .05,
                 "l2_leaf_reg": 20, "random_seed": 42, "thread_count": 4, "task_type": "CPU",
                 "other_settings": "copy applicable saved tree/sampling settings; inspect synthetic compatibility; freeze resolved parameters before target comparison"},
    "pairs": {"negative_rows_per_positive": 16, "seed": 3042,
              "sampling": "uniform across all fitting-day unclicked rows, without replacement per positive",
              "groups": "one group for all fitting rows; includes no-click events; explicit bounded pairs only"},
    "link": "unit-slope sigmoid; fitting-day-only intercept solves mean(sigmoid(raw_train+b))=meta_CTR; 80 bisections",
    "temporal": {"primary": "pre-Jan24 features / Jan24 first-user labels; pre-Jan25 features / Jan25-26 first-user targets",
                 "secondary": "Jan25 rows of primary features / Jan25 labels; pre-Jan26 features / Jan26 first-user targets"},
    "gate": "primary paired global row AUC interval lower > 0 AND secondary global row AUC point change >= 0",
    "bootstrap": {"resamples": 1000, "seed": 2026, "unit": "whole first event per user, all ten candidates",
                  "paired": True, "interval": "95% percentile"},
    "original_replay_tolerance": {"rtol": 0, "atol": 1e-15},
    "budget": "2700 seconds verification/preparation compute plus run compute, excluding human wait",
    "stop": "failure closes this fixed recipe; no objective/pair-count/tree/weight/calibration sweep",
    "next_stage": "same fixed recipe's earlier-cutoff five-day comparison before any promotion",
    "status": "retrospective reused development dates; intervals not selection-adjusted",
    "neural_training": False, "early_stopping": False, "submission_export": False,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_npz(path, **arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def clock(used=0.):
    start = time.monotonic()

    def check():
        if time.monotonic() - start + used > 2700:
            raise TimeoutError("Fixed 45-minute compute budget exceeded; preserve the log and stop")

    return start, check


def locate_split(repo, row_trial, expected):
    name = "split_jan24.npz"
    preferred = [row_trial / "dim8_seed42/jan24/data/cache" / name,
                 repo / "local/archive/techarena-check-jan24/data/cache" / name,
                 repo / "local/archive/techarena-confirm-jan24/data/cache" / name]
    for path in preferred:
        if path.is_file() and path.resolve().is_relative_to(repo) and audit.sha256(path) == expected:
            return path.resolve()
    # Search this small filename only, never hash all data/model archives.
    for directory, subdirs, files in os.walk(repo, followlinks=False):
        subdirs[:] = sorted(x for x in subdirs if x not in {".git", "team-export", "node_modules", ".venv"})
        if name in files:
            path = Path(directory) / name
            if path.is_file() and path.resolve().is_relative_to(repo) and audit.sha256(path) == expected:
                return path.resolve()
    raise FileNotFoundError(f"Missing original Jan24 split matching SHA256 {expected}; do not restore deleted data or synthesize a replacement")


def assert_combiner(model, wanted_names=None):
    p = model.get_all_params()
    audit.require(model.tree_count_ == 545 and len(model.feature_names_) == 37, "Saved combiner tree/feature count differs")
    audit.require(p["loss_function"] == "Logloss" and p["depth"] == 6 and p["random_seed"] == 42,
                  "Unexpected original combiner objective/depth/seed")
    audit.require(abs(p["learning_rate"] - .05) < 1e-7 and abs(p["l2_leaf_reg"] - 20) < 1e-7,
                  "Unexpected saved combiner learning rate/L2")
    audit.require(p.get("task_type", "CPU") == "CPU", "Saved combiner is not CPU")
    if wanted_names is not None:
        audit.require(model.feature_names_ == wanted_names, "Combiner feature order differs")
    return p


def compatibility_smoke(params, names):
    from catboost import CatBoostRanker, Pool
    rng = np.random.default_rng(42)
    x = rng.normal(size=(80, len(names))).astype(np.float32)
    y = np.zeros(80, dtype=np.int8)
    y[[1, 12, 33, 48, 67]] = 1
    pairs = core.sample_pairs(y)
    smoke = dict(params, iterations=2)
    model = CatBoostRanker(**smoke)
    model.fit(Pool(x, label=y, group_id=np.zeros(len(y), dtype=np.int64), pairs=pairs,
                   pairs_weight=np.ones(len(pairs)), feature_names=names), verbose=False)
    audit.require(model.tree_count_ == 2, "Synthetic compatibility fit failed")
    p = model.get_all_params()
    audit.require(p["loss_function"] == "PairLogit", "Synthetic ranker uses a different objective")
    return p


def verify(args):
    import torch
    import catboost
    import sklearn
    from catboost import CatBoostClassifier
    start, check = clock()
    repo = args.repo.resolve(strict=True)
    work = audit.inside(repo / "artifacts/experiments" / EXPERIMENT, repo)
    audit.require(not work.exists(), f"Verification already exists; preserve it: {work}")
    source = audit.inside(__file__, repo)
    inputs = {str(source): audit.sha256(source), str(source.with_name("pairlogit_core.py")): audit.sha256(source.with_name("pairlogit_core.py"))}
    for name, want in HELPERS.items():
        inputs[str(source.with_name(name))] = want
    for name, want in LEGACY.items():
        inputs[str(repo / "legacy" / name)] = want
    raw_path = repo / "data/cache/train_events.npz"
    full_path = repo / "data/cache/preference_training.npz"
    inputs[str(raw_path)], inputs[str(full_path)] = prefix.TRAIN_SHA, FULL_POSITIVE_SHA
    row = repo / "local/archive/techarena-row-auc-trial"
    row_protocol_path = row / "outputs/frozen_protocol.json"
    inputs[str(row_protocol_path)] = ROW_PROTOCOL_SHA
    prefix.verify_hashes(inputs, check)
    old_protocol = read_json(row_protocol_path)
    old_training_path = row / "outputs/training_results.json"
    old_result_path = row / "outputs/ensemble_trial_results.json"
    old_training, old_results = read_json(old_training_path), read_json(old_result_path)
    inputs[str(old_training_path)] = audit.sha256(old_training_path)
    inputs[str(old_result_path)] = audit.sha256(old_result_path)
    expected_split = None
    for k, v in old_protocol.get("input_sha256", {}).items():
        if k.endswith("split_jan24.npz"):
            expected_split = v
            break
    if expected_split is None:
        expected_split = "d8d15491975388917a8a68273c68c674c9b5d14900f05d10ad553a262c6e3ea7"
    split_path = locate_split(repo, row, expected_split)
    inputs[str(split_path)] = expected_split
    models, v6_paths, cb_paths, stored_params, target_params, smoke_params = {}, {}, {}, {}, {}, {}
    names = None
    for key, spec in CASES.items():
        cb = row / "work" / f"dim8_seed42_{spec['fit_day']}.cbm"
        audit.require(old_results["artifact_sha256"][f"work/{cb.name}"] == spec["combiner_sha256"], "Original combiner report hash differs")
        inputs[str(cb)] = spec["combiner_sha256"]
        p = row / "work" / spec["filename"]
        inputs[str(p)] = spec["sha256"]
        cb_paths[key], v6_paths[key] = str(cb), str(p)
    for fold in FOLDS:
        p = row / f"dim8_seed42/{fold}/work/event_best.pt"
        inputs[str(p)] = old_training["members"]["dim8_seed42"][fold]["sha256"]
        models[fold] = str(p)
        # Bind the original generated trainer to its frozen source and retain
        # its original temporal orchestration. Never run it here.
        for filename in ("train_events.py", "row_auc_objective.py"):
            relative = f"dim8_seed42/{fold}/src/{filename}"
            inputs[str(row / relative)] = old_protocol["prepared_sha256"][relative]
    for path in inputs:
        audit.inside(path, repo)
    prefix.verify_hashes(inputs, check)
    for key, path in cb_paths.items():
        model = CatBoostClassifier()
        model.load_model(path)
        p = assert_combiner(model, names)
        names = list(model.feature_names_)
        stored_params[key] = p
        target_params[key] = core.ranker_parameters(p)
        smoke_params[key] = compatibility_smoke(target_params[key], names)
        check()
        print(f"Saved {key} combiner and PairLogit synthetic compatibility passed", flush=True)
    raw, full = prefix.load_raw(raw_path), prefix.load_full_positive(full_path)
    _, full_proof = prefix.prefix_examples(raw, full, audit.END, int(full["n_users"]))
    with np.load(split_path, allow_pickle=False) as saved:
        fit24, meta24 = saved["train_idx"], saved["valid_idx"]
    audit.require(np.array_equal(fit24, np.flatnonzero(raw["t"] < FOLDS["jan24"])), "Original Jan24 fit indices differ")
    audit.require(meta24.dtype.kind in "iu" and meta24.ndim == 1, "Invalid Jan24 meta indices")
    expected_meta24 = audit.first_indices(raw["u"], raw["t"], FOLDS["jan24"])
    expected_meta24 = expected_meta24[raw["t"][expected_meta24] < FOLDS["jan25"]]
    audit.require(np.array_equal(np.sort(meta24), expected_meta24),
        "Original Jan24 meta cohort differs from first-user events on its fitting day")
    target = {}
    old_members = {}
    baseline = {}
    labels = {}
    for key, spec in CASES.items():
        idx, y, p, _, _ = audit.load_cohort(v6_paths[key], spec, raw)
        with np.load(v6_paths[key], allow_pickle=False) as saved:
            old = saved["new_dim8_seed42"]
        audit.require(old.shape == y.shape and np.isfinite(old).all(), "Invalid cached original seed42 member")
        target[key], labels[key], baseline[key], old_members[key] = idx, y, p, old
    day25 = raw["t"][target["primary"]] < FOLDS["jan26"]
    meta25 = target["primary"][day25]
    audit.require(np.intersect1d(meta24, target["primary"]).size == 0
                  and np.intersect1d(meta25, target["secondary"]).size == 0, "Meta/target cohorts overlap")
    audit.require(torch.cuda.is_available(), "CUDA unavailable for original feature replay; use the existing environment")
    torch.set_num_threads(4)
    sys.path.insert(0, str(repo / "legacy"))
    from build_submission import neural_pipeline, FIT_FIELDS, TARGET_FIELDS, predict_combiner, load_combiner
    prefix.verify_hashes(inputs, check)
    work.mkdir(parents=True)
    prefix_paths, prefix_proofs = {}, {}
    for fold, cutoff in FOLDS.items():
        cp = torch.load(models[fold], map_location="cpu", weights_only=True)
        audit.require(cp["epoch"] == 8 and cp["dimension"] == 8 and cp["seed"] == 42 and cp["cutoff"] == cutoff,
                      f"Original neural checkpoint metadata differs: {fold}")
        audit.require(cp["event_objective"] == "half_event_ce_half_global_row_pair_v1", "Wrong event objective")
        audit.require(all(bool(torch.isfinite(x).all()) for x in cp["model_state"].values() if x.is_floating_point()),
                      "Nonfinite original neural parameters")
        positive, proof = prefix.prefix_examples(raw, full, cutoff, int(cp["n_users"]))
        path = work / f"{fold}_prefix_clicks.npz"
        save_npz(path, **positive)
        inputs[str(path)] = audit.sha256(path)
        prefix_paths[fold], prefix_proofs[fold] = str(path), proof
        del cp

    def features(fold, indices):
        check()
        fit_idx = np.flatnonzero(raw["t"] < FOLDS[fold])
        x, columns, *_ = neural_pipeline(Path(models[fold]), {k: raw[k][fit_idx] for k in FIT_FIELDS},
            {k: raw[k][indices] for k in TARGET_FIELDS}, Path(prefix_paths[fold]), torch.device("cuda"))
        audit.require(columns == names and x.shape == (len(indices) * 10, 37), "Original 37-feature schema differs")
        check()
        return x

    # Preserve the historical row order. Secondary meta features are the actual
    # Jan25 rows of primary features, not independently re-queried cold users.
    x24 = features("jan24", meta24)
    x25 = features("jan25", target["primary"])
    x26 = features("jan26", target["secondary"])
    replay = {}
    for key, x in (("primary", x25), ("secondary", x26)):
        model = load_combiner(cb_paths[key])
        old = predict_combiner(model, x, names).reshape(-1, 10)
        error = float(np.max(np.abs(old - old_members[key])))
        audit.require(np.allclose(old, old_members[key], rtol=0, atol=1e-15),
                      f"Original {key} seed42 replay differs ({error}); no fit permitted")
        replay[key] = {"member_max_absolute_difference": error, "v6_auc": audit.auc(labels[key], baseline[key]),
                       "events": len(labels[key]), "passed": True}
        # New member == old member must preserve the cached V6 comparator exactly.
        audit.require(np.array_equal(core.replacement(baseline[key], old_members[key], old_members[key]), baseline[key]),
                      "Replacement identity failed")
    data_paths = {}
    for key, fit_idx, xfit, xtarget in (("primary", meta24, x24, x25),
                                       ("secondary", meta25, x25[np.repeat(day25, 10)], x26)):
        path = work / f"{key}_features.npz"
        save_npz(path, X_fit=xfit, y_fit=raw["y"][fit_idx], fit_idx=fit_idx,
                 X_target=xtarget, valid_idx=target[key], labels=labels[key], baseline=baseline[key],
                 old_member=old_members[key], feature_names=np.asarray(names))
        inputs[str(path)] = audit.sha256(path)
        data_paths[key] = str(path)
    prefix.verify_hashes(inputs, check)
    check()
    report = {"verified": True, "experiment": EXPERIMENT, "repo": str(repo), "inputs_sha256": inputs,
              "data_paths": data_paths, "feature_names": names, "feature_count": 37,
              "original_combiner_parameters": stored_params, "ranker_parameters": target_params,
              "synthetic_ranker_resolved_parameters": smoke_params, "original_replay": replay,
              "full_click_multiset_proof": full_proof, "prefix_click_proofs": prefix_proofs,
              "meta_cohorts": {"jan24_events": len(meta24), "jan25_events": len(meta25),
                               "jan24_split_original_hash_verified": True,
                               "jan25_rows_are_primary_subset": True, "meta_target_disjoint_per_comparison": True},
              "compute_seconds": round(time.monotonic() - start, 6), "neural_training_performed": False,
              "competition_ranker_fitting_performed": False, "original_artifacts_modified": False,
              "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                          "catboost": catboost.__version__, "sklearn": sklearn.__version__}}
    audit.write_new(work / "verification.json", report)
    print(f"Verification/replay passed: {work / 'verification.json'}. No competition ranker fitted.", flush=True)


def prepare(args):
    start = time.monotonic()
    repo = args.repo.resolve(strict=True)
    work = repo / "artifacts/experiments" / EXPERIMENT
    verified_path = work / "verification.json"
    verified = read_json(verified_path)
    audit.require(verified["verified"] is True and verified["repo"] == str(repo), "Wrong verification report")
    audit.require(not (work / "inputs.json").exists(), "Preparation already exists; do not overwrite")
    inputs = dict(verified["inputs_sha256"])
    audit.require(inputs.get(str(Path(__file__).resolve())) == audit.sha256(__file__), "Preparation source changed")
    prefix.verify_hashes(inputs)
    pair_paths, pair_counts = {}, {}
    for key, path in verified["data_paths"].items():
        with np.load(path, allow_pickle=False) as saved:
            y = saved["y_fit"]
        pairs = core.sample_pairs(y)
        pair_path = work / f"{key}_pairs.npz"
        save_npz(pair_path, pairs=pairs)
        inputs[str(pair_path)] = audit.sha256(pair_path)
        pair_paths[key] = str(pair_path)
        event_click = y.sum(1) > 0
        no_click_negative = ~event_click[pairs[:, 1] // 10]
        pair_counts[key] = {"rows": int(y.size), "positive_rows": int(y.sum()), "pairs": len(pairs),
                            "pairs_with_no_click_event_negative": int(no_click_negative.sum()),
                            "cross_event_pairs": int((pairs[:, 0] // 10 != pairs[:, 1] // 10).sum())}
    inputs[str(verified_path)] = audit.sha256(verified_path)
    prefix.verify_hashes(inputs)
    used = verified["compute_seconds"] + time.monotonic() - start
    audit.require(used < 2700, "Preparation exhausted the total compute budget")
    frozen = {"experiment": EXPERIMENT, "repo": str(repo), "protocol": RECIPE,
              "protocol_sha256": audit.canonical_hash(RECIPE), "inputs_sha256": inputs,
              "verification": str(verified_path), "data_paths": verified["data_paths"],
              "pair_paths": pair_paths, "pair_counts": pair_counts,
              "ranker_parameters": verified["ranker_parameters"], "preparation_compute_seconds": used,
              "competition_ranker_fitting_performed": False}
    audit.write_new(work / "inputs.json", frozen)
    print(json.dumps({"prepared": str(work / "inputs.json"), "pair_counts": pair_counts}, indent=2), flush=True)


def run(args):
    import catboost
    from catboost import CatBoostRanker, Pool
    path = args.manifest.resolve(strict=True)
    manifest_hash = audit.sha256(path)
    frozen = read_json(path)
    repo = Path(frozen["repo"]).resolve(strict=True)
    work = audit.inside(repo / "artifacts/experiments" / EXPERIMENT, repo)
    audit.inside(path, work)
    output = audit.inside(repo / "docs/results" / EXPERIMENT / "pairlogit.json", repo)
    audit.require(not output.exists(), f"Refusing to overwrite {output}")
    audit.require(frozen["experiment"] == EXPERIMENT and frozen["protocol"] == RECIPE
                  and frozen["protocol_sha256"] == audit.canonical_hash(RECIPE), "Frozen recipe changed")
    start, check = clock(frozen["preparation_compute_seconds"])
    inputs = frozen["inputs_sha256"]
    for p in inputs:
        audit.inside(p, repo)
    audit.require(inputs.get(str(Path(__file__).resolve())) == audit.sha256(__file__), "Unfrozen running source")
    for name in list(HELPERS) + ["pairlogit_core.py"]:
        p = Path(__file__).with_name(name).resolve()
        audit.require(inputs.get(str(p)) == audit.sha256(p), f"Running dependency not frozen: {name}")
    audit.require(str(audit.inside(frozen["verification"], repo)) in inputs, "Verification not frozen")
    prefix.verify_hashes(inputs, check)
    verified = read_json(frozen["verification"])
    audit.require(verified["verified"] is True and verified["repo"] == str(repo), "Wrong verification identity")
    for field in ("data_paths", "ranker_parameters"):
        audit.require(frozen[field] == verified[field], f"Prepared {field} differs from verification")
    audit.require(set(frozen["pair_paths"]) == set(frozen["data_paths"]) == set(CASES), "Wrong cases")
    attempt = work / "run"
    audit.require(not attempt.exists(), f"Run already started; preserve it: {attempt}")
    attempt.mkdir()
    results, artifacts = {}, {}
    for key in CASES:
        data_path = audit.inside(frozen["data_paths"][key], repo)
        pairs_path = audit.inside(frozen["pair_paths"][key], repo)
        audit.require(str(data_path) in inputs and str(pairs_path) in inputs, "Unfrozen features/pairs")
        with np.load(data_path, allow_pickle=False) as saved:
            data = {k: saved[k] for k in saved.files}
        with np.load(pairs_path, allow_pickle=False) as saved:
            pairs = saved["pairs"]
        audit.require(np.array_equal(pairs, core.sample_pairs(data["y_fit"])), "Pair recipe differs from frozen sampler")
        labels = data["y_fit"].ravel()
        names = data["feature_names"].tolist()
        model = CatBoostRanker(**frozen["ranker_parameters"][key])
        pool = Pool(data["X_fit"], label=labels, group_id=np.zeros(len(labels), dtype=np.int64),
                    pairs=pairs, pairs_weight=np.ones(len(pairs)), feature_names=names)
        print(f"Fitting {key} CPU PairLogit: {len(labels)} rows, {len(pairs)} explicit pairs", flush=True)
        check()
        model.fit(pool, verbose=100)
        check()
        audit.require(model.tree_count_ == 545, "Ranker tree count differs")
        raw_train = model.predict(pool, thread_count=4)
        bias, link = core.intercept(raw_train, labels)
        raw_target = model.predict(Pool(data["X_target"], feature_names=names),
                                   thread_count=4)
        new_member = core.sigmoid(raw_target + bias).reshape(-1, 10)
        candidate = core.replacement(data["baseline"], data["old_member"], new_member)
        model_path = attempt / f"{key}_pairlogit.cbm"
        model.save_model(str(model_path))
        reloaded = CatBoostRanker()
        reloaded.load_model(str(model_path))
        replay = reloaded.predict(Pool(data["X_target"], feature_names=names),
                                  thread_count=4)
        audit.require(np.allclose(raw_target, replay, rtol=0, atol=1e-15), "Saved ranker replay failed")
        pred_path = attempt / f"{key}_predictions.npz"
        save_npz(pred_path, valid_idx=data["valid_idx"], labels=data["labels"], baseline=data["baseline"],
                 candidate=candidate, new_member=new_member, old_member=data["old_member"], intercept=np.asarray(bias))
        artifacts[str(model_path)], artifacts[str(pred_path)] = audit.sha256(model_path), audit.sha256(pred_path)
        metrics = audit.metrics(data["labels"], data["baseline"], candidate)
        metrics["candidate_auc"] = metrics.pop("normalized_auc")
        old_auc = audit.auc(data["labels"], data["old_member"])
        new_auc = audit.auc(data["labels"], new_member)
        metrics["seed42_member_auc"] = {"original": old_auc, "pairlogit": new_auc,
                                        "change": new_auc - old_auc}
        metrics["training_only_link"] = link
        metrics["resolved_ranker_parameters"] = model.get_all_params()
        metrics["decomposition"] = shared.allocation_decomposition(data["labels"], data["baseline"], candidate)
        print(json.dumps({key: {k: v for k, v in metrics.items()
                              if k not in {"decomposition", "resolved_ranker_parameters"}}}, indent=2), flush=True)
        metrics.update(audit.bootstrap(data["labels"], data["baseline"], candidate,
            {"overall": np.ones(len(candidate), dtype=bool)}, check, resamples=1000)["overall"])
        results[key] = metrics
        del model, reloaded, pool, data
        check()
    gate = {"primary_interval_entirely_positive": results["primary"]["paired_gain_95_percentile_interval"][0] > 0,
            "secondary_point_change_nonnegative": results["secondary"]["auc_change"] >= 0}
    prefix.verify_hashes(inputs, check)
    prefix.verify_hashes(artifacts, check)
    audit.require(audit.sha256(path) == manifest_hash, "Frozen manifest changed")
    check()
    report = {"complete": True, "protocol": RECIPE, "protocol_sha256": frozen["protocol_sha256"],
              "manifest_sha256": manifest_hash, "source_sha256": audit.sha256(__file__),
              "inputs_sha256": inputs, "new_artifacts_sha256": artifacts, "results": results,
              "decision_checks": gate, "supports_five_day_review": all(gate.values()),
              "input_hashes_rechecked_unchanged": True, "neural_training_performed": False,
              "original_artifacts_modified": False, "evaluation_data_read": False,
              "submission_created": False, "automatic_promotion_performed": False,
              "runtime": {"run_seconds": round(time.monotonic() - start, 3),
                          "preparation_compute_seconds": frozen["preparation_compute_seconds"],
                          "python": sys.version, "numpy": np.__version__, "catboost": catboost.__version__}}
    audit.write_new(output, report)
    print(json.dumps({"decision_checks": gate, "supports_five_day_review": all(gate.values()),
                      "report": str(output)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("verify", "prepare"):
        commands.add_parser(action).add_argument("--repo", type=Path, required=True)
    commands.add_parser("run").add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    {"verify": verify, "prepare": prepare, "run": run}[args.action](args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TimeoutError) as error:
        raise SystemExit(f"STOPPED: {error}") from error
