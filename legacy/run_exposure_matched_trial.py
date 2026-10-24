"""One exposure-matched preference expert, fixed 85% V6 / 15% expert blend.

In techarena-final: self-test, setup, train, evaluate. New files are confined to
techarena-exposure-matched-trial. No full-data training, CSV export or upload.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from exposure_matched_preference import (RECIPE, distribution, alias_table, preference_source,
    event_source, make_pipeline, numeric_self_test, file_hash)
from row_auc_objective import CONFIG
from run_row_auc_trial import read_json, write_json
from run_expanded_combiner_trial import CUTOFFS, WEIGHTS, ensemble, load_references, read_raw
from audit_v5 import auc, AucOrder

STAGES = [("initial", "train_preferences.py", "preference_initial.pt", 196),
          ("fine", "train_preference_finetune.py", "preference_best.pt", 260),
          ("event", "train_events.py", "event_best.pt", 8)]
REPORT = "outputs/ensemble_trial_results.json"


def hashes(paths):
    return {str(Path(p).resolve(strict=True)): file_hash(p) for p in paths}


def verify_files(base, mapping):
    for name, expected in mapping.items():
        assert file_hash(base / name) == expected, f"Changed frozen file: {base / name}"


def verify(trial, protocol):
    assert protocol["recipe"] == RECIPE and protocol["event_objective"] == CONFIG
    assert protocol["v6_member_weights"] == WEIGHTS
    verify_files(Path(), protocol["input_sha256"])
    verify_files(Path(), protocol["protected_v5_v6_sha256"])
    verify_files(trial, protocol["prepared_sha256"])


def load_v6(root, raw):
    """Reconstruct the five-member V6 from honest, aligned historical caches."""
    refs = load_references(root, raw)
    short = root.parent / "techarena-row-auc-trial"
    study = read_json(short / "outputs/ensemble_trial_results.json")
    for case, ref in refs.items():
        path = short / "work" / f"{case}_ensemble_predictions.npz"
        with np.load(path, allow_pickle=False) as saved:
            assert np.array_equal(saved["valid_idx"], ref["indices"])
            assert np.array_equal(saved["labels"], ref["labels"])
            assert np.array_equal(saved["weights"], WEIGHTS)
            members = [saved["new_dim8_seed42"], saved["new_dim8_seed43"]] + ref["members"][2:]
            baseline = ensemble(members)
            np.testing.assert_allclose(baseline, saved["candidate"], rtol=0, atol=1e-15)
        metric = "two_day_primary" if case == "two_day" else "jan26_secondary"
        assert abs(auc(ref["labels"], baseline) - study[metric]["candidate_row_ranking_auc"]) < 1e-12
        cutoff = CUTOFFS["jan25" if case == "two_day" else "jan26"]
        candidates = np.flatnonzero((raw["t"] >= cutoff) & (raw["t"] < 1737936000))
        chronological = candidates[np.argsort(raw["t"][candidates], kind="stable")]
        _, first = np.unique(raw["u"][chronological], return_index=True)
        assert np.array_equal(np.sort(chronological[first]), np.sort(ref["indices"]))
        ref["v5"] = ref["baseline"]
        ref["baseline"], ref["members"] = baseline, members
    return refs


def prepare(root, fold):
    source = root.parent / "techarena-seed43/src/train_preferences.py"
    original = source.read_text(encoding="utf-8")
    event = root.parent / f"techarena-row-auc-trial/dim8_seed42/{fold}/src/train_events.py"
    return {"train_preferences.py": preference_source(original, fold),
            "train_preference_finetune.py": preference_source(original, fold, fine=True),
            "train_events.py": event_source(event.read_text(encoding="utf-8")),
            "row_auc_objective.py": (root / "src/row_auc_objective.py").read_text(encoding="utf-8"),
            "exposure_matched_preference.py": (root / "src/exposure_matched_preference.py").read_text(encoding="utf-8")}


def write_sampling(path, items, times, cutoff, n_items):
    counts, q = distribution(items, times, cutoff, n_items)
    accept, alias = alias_table(q)
    np.savez_compressed(path, counts=counts, probabilities=q, alias_accept=accept,
                        alias_index=alias, cutoff=cutoff, n_items=n_items,
                        fitting_events=int((times < cutoff).sum()))
    return {"fitting_events": int((times < cutoff).sum()), "total_candidate_displays": int(counts.sum()),
            "minimum_sampling_probability": float(q.min()), "maximum_sampling_probability": float(q.max())}


def setup(root, trial):
    if trial.exists():
        verify(trial, read_json(trial / "outputs/frozen_protocol.json"))
        print("Exposure-matched trial already prepared; continue with train/evaluate.", flush=True)
        return
    from run_row_auc_final import verify as verify_final, verify_training as verify_final_training, validation_replay, check_v5
    from run_row_auc_five_day import verify_artifacts
    final = root.parent / "techarena-row-auc-final"
    final_protocol = read_json(final / "outputs/frozen_protocol.json")
    verify_final(final, final_protocol)
    verify_final_training(final, final_protocol, read_json(final / "outputs/training_results.json"))
    final_checks = read_json(final / "outputs/candidate_submission_checks.json")
    assert final_checks["submission_ready"] and final_checks["v5_files_unchanged"]
    assert final_checks["version"] == "v6_candidate_row_ranking" and final_checks["validation_pipeline_check_passed"]
    assert file_hash(final / final_protocol["candidate_csv"]) == final_checks["submission_sha256"]
    verify_artifacts(final, final_checks["artifact_sha256"])
    check_v5(root)
    print("Replaying V6 historical predictions before any new training...", flush=True)
    raw, replay = validation_replay(root)
    assert all(v["passed"] and max(v["member_max_abs_differences"].values()) == 0 for v in replay.values())
    refs = load_v6(root, raw)
    deps = [Path(__file__), root.parent / "techarena-seed43/src/train_preferences.py"]
    deps += [root / "src" / name for name in ["exposure_matched_preference.py", "row_auc_objective.py",
        "run_row_auc_trial.py", "run_row_auc_final.py", "row_auc_final_utils.py", "run_row_auc_five_day.py",
        "run_expanded_combiner_trial.py", "expanded_combiner_features.py", "prepare_preference_finetune_trial.py",
        "build_submission.py", "submission_features.py", "train_final_models.py", "evaluate_dim16_trial.py",
        "audit_v5.py", "submission_export.py", "build_preference_finetuned_submission.py"]]
    deps += [Path(p) for p in final_protocol["input_sha256"]]
    deps += [final / p for p in final_protocol["prepared_sha256"]]
    protected = [Path(p) for p in final_protocol["v5_protected_sha256"]]
    protected += [final / p for p in final_checks["artifact_sha256"]]
    protected += [final / final_protocol["candidate_csv"], final / "outputs/candidate_submission_checks.json",
                  final / "outputs/frozen_protocol.json", final / "outputs/training_results.json"]
    deps += protected
    folds, training_inputs = {}, [root / "data/cache/train_events.npz"]
    for fold, cutoff in CUTOFFS.items():
        original = root.parent / f"techarena-row-auc-trial/dim8_seed42/{fold}"
        paths = {name: (original / "data/cache" / name).resolve(strict=True) for name in
                 ["train_events.npz", "preference_training.npz", f"split_{fold}.npz"]}
        assert paths["train_events.npz"] == (root / "data/cache/train_events.npz").resolve()
        with np.load(paths[f"split_{fold}.npz"], allow_pickle=False) as data:
            fitting, valid = data["train_idx"], data["valid_idx"]
            assert np.array_equal(fitting, np.flatnonzero(raw["t"] < cutoff))
            assert len(np.unique(raw["u"][valid])) == len(valid)
            assert (raw["t"][valid] >= cutoff).all() and (raw["t"][valid] < cutoff + 86400).all()
            if fold != "jan24":
                idx = refs["two_day" if fold == "jan25" else "jan26"]["indices"]
                expected = idx[raw["t"][idx] < cutoff + 86400]
                assert np.array_equal(valid, expected)
        with np.load(paths["preference_training.npz"], allow_pickle=False) as data:
            n_users, n_items = int(data["n_users"]), int(data["n_items"])
            assert int(data["cutoff"]) == cutoff and (data["timestamps"] < cutoff).all()
            assert len(data["users"]) == len(data["items"]) == len(data["timestamps"]) > 0
            assert np.isin(data["users"], np.unique(raw["u"][fitting])).all()
            assert ((data["items"] >= 0) & (data["items"] < n_items)).all()
            assert ((data["users"] >= 0) & (data["users"] < n_users)).all()
        deps += list(paths.values()) + [original / "src/train_events.py"]
        training_inputs += list(paths.values())
        folds[fold] = {"paths": paths, "n_users": n_users, "n_items": n_items, "scripts": prepare(root, fold)}
    protocol = {
        "experiment": "exposure-matched preference sampling expert added to unchanged V6",
        "recipe": RECIPE, "event_objective": CONFIG, "v6_member_weights": WEIGHTS,
        "preference_positive_examples": "same prefix-only initial histories and observed clicks as V5/V6",
        "sampling_rule": "all prefix candidate displays; independent draws with replacement; no log-q score correction",
        "warmup_limit": "warm-up impressions unavailable; observed-period exposure is an approximation",
        "inference": "unchanged event softmax, power0 exposure step, original ridge fallback and 37 features",
        "combiner": {"trees": 545, "depth": 6, "learning_rate": .05, "l2_leaf_reg": 20, "seed": 42, "early_stopping": False},
        "blend_rule": "85% V6 plus 15% expert for all candidate rows, including empty histories",
        "primary_check": "Jan24 meta-training; Jan25 prefix; first-user Jan25-26 evaluation",
        "secondary_check": "Jan25 meta-training; Jan26 prefix; first-user Jan26 evaluation",
        "bootstrap_resamples": 300, "bootstrap_seed": 2026,
        "support_rule": "global primary paired gain interval entirely positive and global secondary AUC change nonnegative",
        "next_gate": "identical recipe versus V6 in five-day paired comparison before full-data fitting or submission",
        "early_stopping": False, "weight_search_performed": False,
        "status": "retrospective development; reused dates; intervals not selection-adjusted",
        "v6_replay": replay, "sampling": {},
        "input_sha256": hashes(deps), "protected_v5_v6_sha256": hashes(protected),
        "prepared_sha256": {}, "training_input_sha256": hashes(training_inputs),
    }
    # Transactional setup: an interrupted preparation cannot leave a half-ready trial.
    with tempfile.TemporaryDirectory(prefix=".exposure_matched_setup_", dir=root.parent) as temporary:
        staged = Path(temporary)
        for part in ["outputs", "work"]:
            (staged / part).mkdir()
        for fold, data in folds.items():
            folder = staged / fold
            for part in ["src", "work", "outputs", "data/cache"]:
                (folder / part).mkdir(parents=True)
            for name, text in data["scripts"].items():
                path = folder / "src" / name
                path.write_text(text, encoding="utf-8")
                protocol["prepared_sha256"][str(path.relative_to(staged))] = file_hash(path)
            for name, path in data["paths"].items():
                (folder / "data/cache" / name).symlink_to(path)
            path = folder / "data/cache/exposure_sampling.npz"
            protocol["sampling"][fold] = write_sampling(path, raw["i"], raw["t"], CUTOFFS[fold], data["n_items"])
            protocol["sampling"][fold].update(n_users=data["n_users"], n_items=data["n_items"])
            protocol["prepared_sha256"][str(path.relative_to(staged))] = file_hash(path)
            protocol["training_input_sha256"][str(trial / path.relative_to(staged))] = file_hash(path)
        write_json(staged / "outputs/frozen_protocol.json", protocol)
        staged.rename(trial)
    verify(trial, protocol)
    print("EXPOSURE-MATCHED SETUP PASSED", flush=True)
    print(json.dumps({"workspace": str(trial), "recipe": RECIPE, "sampling": protocol["sampling"],
                      "v6_reproduction_passed": True, "v5_v6_protected": True}, indent=2), flush=True)


def check_checkpoint(trial, protocol, fold, stage, filename, epoch):
    import torch
    folder = trial / fold
    path = folder / "work" / filename
    saved = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"epoch": epoch, "dimension": 8, "seed": 42, "cutoff": CUTOFFS[fold],
                "n_users": protocol["sampling"][fold]["n_users"], "n_items": protocol["sampling"][fold]["n_items"],
                "sampler_recipe": RECIPE, "frozen_protocol_sha256": file_hash(trial / "outputs/frozen_protocol.json")}
    for key, value in expected.items():
        assert saved[key] == value, (fold, stage, key)
    if stage == "event":
        assert saved["event_objective"] == CONFIG["objective"] and saved["pair_sampling_seed"] == 3042
        assert saved["preference_checkpoint_sha256"] == file_hash(folder / "work/preference_best.pt")
    else:
        assert saved["stage"] == "preference"
        assert saved["sampling_sha256"] == file_hash(folder / "data/cache/exposure_sampling.npz")
        lr = .012 if stage == "initial" else .003
        assert all(g["lr"] == lr for g in saved["optimizer_state"]["param_groups"])
        if stage == "fine":
            assert saved["initial_checkpoint_sha256"] == file_hash(folder / "work/preference_initial.pt")
            assert saved["fine_tuning_sampling_seed"] == 1042
    assert all(torch.isfinite(v).all() for v in saved["model_state"].values() if v.is_floating_point())
    for state in saved["optimizer_state"]["state"].values():
        assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v) and v.is_floating_point())
    return file_hash(path)


def verify_training(trial, protocol, report):
    assert report["protocol_sha256"] == file_hash(trial / "outputs/frozen_protocol.json")
    assert set(report["folds"]) == set(CUTOFFS)
    for fold in CUTOFFS:
        for stage, _, filename, epoch in STAGES:
            assert check_checkpoint(trial, protocol, fold, stage, filename, epoch) == report["folds"][fold][stage]


def train(trial, protocol):
    output = trial / "outputs/training_results.json"
    if output.exists():
        verify_training(trial, protocol, read_json(output))
        print("All three preference/event fits already complete; continue with evaluate.", flush=True)
        return
    start, records = time.perf_counter(), {}
    for fold in CUTOFFS:
        records[fold] = {}
        for stage, script, filename, epoch in STAGES:
            folder = trial / fold
            if not (folder / "work" / filename).exists():
                print(f"\nTraining {fold}: {stage}, fixed final epoch {epoch}", flush=True)
                subprocess.run([sys.executable, str(folder / "src" / script)], cwd=folder, check=True)
            records[fold][stage] = check_checkpoint(trial, protocol, fold, stage, filename, epoch)
    verify(trial, protocol)
    report = {"folds": records, "protocol_sha256": file_hash(trial / "outputs/frozen_protocol.json"),
              "runtime_seconds": round(time.perf_counter() - start, 2)}
    verify_training(trial, protocol, report)
    write_json(output, report)
    print("EXPOSURE-MATCHED TRAINING COMPLETE", flush=True)


def save_npz(path, **arrays):
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def paired_interval(labels, old, new, resamples, seed):
    n = len(labels)
    ids = np.repeat(np.arange(n), 10)
    before, after = [AucOrder(labels.ravel(), p.ravel(), ids) for p in [old, new]]
    rng, gains = np.random.default_rng(seed), []
    for repeat in range(resamples):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        a, b = before.value(counts), after.value(counts)
        if a is not None and b is not None:
            gains.append(b - a)
        if (repeat + 1) % 50 == 0:
            print(f"Paired whole-user bootstrap {repeat + 1}/{resamples}", flush=True)
    assert len(gains) >= .99 * resamples
    return np.quantile(gains, [.025, .975]).tolist(), len(gains)


def evaluate(root, trial, protocol):
    output = trial / REPORT
    training = read_json(trial / "outputs/training_results.json")
    verify_training(trial, protocol, training)
    if output.exists():
        report = read_json(output)
        verify_files(trial, report["artifact_sha256"])
        print("\nEXPOSURE-MATCHED PREFERENCE TRIAL", flush=True)
        print(json.dumps(report, indent=2), flush=True)
        return
    import torch
    from build_submission import FIT_FIELDS, TARGET_FIELDS, predict_combiner, load_combiner, check_close
    from evaluate_dim16_trial import fit_combiner
    start = time.perf_counter()
    raw = read_raw(root)
    refs = load_v6(root, raw)
    pipeline = make_pipeline()
    with np.load(trial / "jan24/data/cache/split_jan24.npz", allow_pickle=False) as data:
        meta24 = data["valid_idx"]
    primary = refs["two_day"]
    day25 = raw["t"][primary["indices"]] < CUTOFFS["jan26"]
    assert not np.intersect1d(meta24, primary["indices"]).size
    assert not np.intersect1d(primary["indices"][day25], refs["jan26"]["indices"]).size

    def features(fold, idx):
        cutoff = CUTOFFS[fold]
        fit = np.flatnonzero(raw["t"] < cutoff)
        assert (raw["t"][idx] >= cutoff).all() and len(np.unique(raw["u"][idx])) == len(idx)
        X, names, raw_p, pipeline_p, *_ = pipeline(trial / fold / "work/event_best.pt",
            {k: raw[k][fit] for k in FIT_FIELDS}, {k: raw[k][idx] for k in TARGET_FIELDS},
            trial / fold / "data/cache/preference_training.npz", torch.device("cuda"))
        assert X.shape == (len(idx) * 10, 37) and len(names) == 37 and np.isfinite(X).all()
        # For seen/empty users there is no fallback; power0 must be exact identity.
        count = np.bincount(raw["u"][fit], minlength=protocol["sampling"][fold]["n_users"])
        cold = (count[raw["u"][idx]] == 0) & (raw["h"][idx] >= 0).any(1)
        np.testing.assert_array_equal(pipeline_p[~cold], raw_p[~cold])
        return X, names, pipeline_p

    print("Building honest meta-training features and two new expert combiners...", flush=True)
    X24, names, _ = features("jan24", meta24)
    X25, names25, raw25 = features("jan25", primary["indices"])
    assert names25 == names
    model24 = fit_combiner(X24, raw["y"][meta24], names)
    path24 = trial / "work/combiner_jan24.cbm"
    model24.save_model(str(path24))
    p25 = predict_combiner(model24, X25, names).reshape(-1, 10)
    check_close(p25, predict_combiner(load_combiner(path24), X25, names).reshape(-1, 10), "Expert Jan24 reload", rtol=0, atol=1e-15)
    del X24, model24
    model25 = fit_combiner(X25[np.repeat(day25, 10)], primary["labels"][day25], names)
    path25 = trial / "work/combiner_jan25.cbm"
    model25.save_model(str(path25))
    del X25
    X26, names26, raw26 = features("jan26", refs["jan26"]["indices"])
    assert names26 == names
    p26 = predict_combiner(model25, X26, names).reshape(-1, 10)
    check_close(p26, predict_combiner(load_combiner(path25), X26, names).reshape(-1, 10), "Expert Jan25 reload", rtol=0, atol=1e-15)
    del X26, model25
    metrics, candidates, artifacts = {}, {}, [path24, path25, trial / "outputs/training_results.json"]
    for case, expert, neural in [("two_day", p25, raw25), ("jan26", p26, raw26)]:
        ref = refs[case]
        candidate = RECIPE["baseline_weight"] * ref["baseline"] + RECIPE["expert_weight"] * expert
        assert candidate.shape == ref["labels"].shape and np.isfinite(candidate).all()
        assert ((candidate >= 0) & (candidate <= 1)).all()
        candidates[case] = candidate
        before, after = auc(ref["labels"], ref["baseline"]), auc(ref["labels"], candidate)
        metrics[case] = {"events": len(candidate), "baseline_v6_auc": before,
                         "exposure_matched_expert_auc": auc(ref["labels"], expert),
                         "expert_neural_pipeline_auc": auc(ref["labels"], neural),
                         "candidate_auc": after, "auc_change": after - before}
        path = trial / "work" / f"{case}_ensemble_predictions.npz"
        save_npz(path, valid_idx=ref["indices"], labels=ref["labels"], baseline=ref["baseline"],
                 expert=expert, candidate=candidate, v6_members=np.stack(ref["members"]),
                 weights=np.array([.85, .15]), protocol_sha256=file_hash(trial / "outputs/frozen_protocol.json"))
        artifacts.append(path)
    interval, valid = paired_interval(primary["labels"], primary["baseline"], candidates["two_day"],
                                      protocol["bootstrap_resamples"], protocol["bootstrap_seed"])
    verify(trial, protocol)
    verify_training(trial, protocol, training)
    report = {"protocol": {k: v for k, v in protocol.items() if not k.endswith("sha256")},
        "two_day_primary": metrics["two_day"], "jan26_secondary": metrics["jan26"],
        "primary_paired_gain_95_percentile_interval": interval, "valid_bootstrap_resamples": valid,
        "v6_reproduction_passed": True, "saved_expert_combiners_reproduced": True,
        "v5_files_unchanged": True, "v6_files_unchanged": True, "all_five_v6_members_unchanged": True,
        "supported_for_five_day_check": bool(interval[0] > 0 and metrics["jan26"]["auc_change"] >= 0),
        "sampling_unit": "whole first event per user, preserving all ten candidates; global row AUC",
        "submission_created": False, "upload_performed": False,
        "runtime_seconds": round(time.perf_counter() - start, 2),
        "artifact_sha256": {str(p.relative_to(trial)): file_hash(p) for p in artifacts}}
    write_json(output, report)
    print("\nEXPOSURE-MATCHED PREFERENCE TRIAL", flush=True)
    print(json.dumps(report, indent=2), flush=True)


def self_test(root):
    import torch
    from exposure_matched_preference import ExposureSampler
    from build_submission import neural_pipeline, FIT_FIELDS, TARGET_FIELDS, adjust_exposure
    from row_auc_objective import self_test as objective_test
    numeric_self_test()
    assert torch.cuda.is_available(), "Activate techarena with working CUDA."
    objective_test()
    cutoff = CUTOFFS["jan24"]
    rng = np.random.default_rng(2034)
    raw = {"u": np.array([0, 1, 2, 0, 1, 2, 3, 4]), "i": rng.integers(0, 30, (8, 10)),
           "t": np.r_[cutoff - 3600 + np.arange(6) * 60, [cutoff, cutoff + 60]],
           "tb": np.zeros(8, dtype=np.int8), "dt": np.zeros(8, dtype=np.int8),
           "h": np.full((8, 20), -1, dtype=np.int32), "ht": np.zeros((8, 20), dtype=np.int64),
           "y": np.zeros((8, 10), dtype=np.int8)}
    raw["y"][[0, 2, 4, 6], [1, 2, 3, 4]] = 1
    raw["h"][6, :2], raw["ht"][6, :2] = [1, 2], [cutoff - 7200, cutoff - 3600]
    texts = prepare(root, "jan24")
    with tempfile.TemporaryDirectory(prefix="exposure_matched_selftest_", dir=root) as temporary:
        trial, protocol = Path(temporary), {"recipe": RECIPE, "event_objective": CONFIG, "prepared_sha256": {}, "training_input_sha256": {}}
        folder = trial / "jan24"
        for part in ["src", "work", "outputs", "data/cache"]:
            (folder / part).mkdir(parents=True)
        (trial / "outputs").mkdir()
        for name, text in texts.items():
            path = folder / "src" / name
            path.write_text(text, encoding="utf-8")
            protocol["prepared_sha256"][str(path.relative_to(trial))] = file_hash(path)
        cache = folder / "data/cache"
        np.savez(cache / "train_events.npz", **raw)
        np.savez(cache / "split_jan24.npz", train_idx=np.arange(6), valid_idx=[6, 7])
        positive = cache / "preference_training.npz"
        np.savez(positive, users=[0, 0, 1, 1, 2, 2], items=[1, 2, 3, 4, 5, 6],
                 timestamps=np.full(6, cutoff - 7200), n_users=8, n_items=30, cutoff=cutoff)
        write_sampling(cache / "exposure_sampling.npz", raw["i"], raw["t"], cutoff, 30)
        protocol["training_input_sha256"] = hashes(list(cache.iterdir()))
        write_json(trial / "outputs/frozen_protocol.json", protocol)
        for device in [torch.device("cpu"), torch.device("cuda")]:
            sampler = ExposureSampler(cache / "exposure_sampling.npz", device)
            torch.manual_seed(2035)
            observed = np.bincount(sampler.sample((200000,)).cpu().numpy(), minlength=30) / 200000
            with np.load(cache / "exposure_sampling.npz") as data:
                q = data["probabilities"]
            assert (np.abs(observed - q) < 6 * np.sqrt(q * (1 - q) / 200000) + .0005).all()
        # Real generated trainers on a tiny fixture, without shortening the schedule.
        protocol["sampling"] = {"jan24": {"n_users": 8, "n_items": 30}}
        write_json(trial / "outputs/frozen_protocol.json", protocol)
        for stage, script, filename, epoch in STAGES:
            subprocess.run([sys.executable, str(folder / "src" / script)], cwd=folder, check=True)
            check_checkpoint(trial, protocol, "jan24", stage, filename, epoch)
        training = {k: raw[k][:6] for k in FIT_FIELDS}
        target = {k: raw[k][6:] for k in TARGET_FIELDS}
        # Rebuild the power0 pipeline, and independently verify it did not patch V5/V6.
        pipeline = make_pipeline()
        assert neural_pipeline.__globals__["adjust_exposure"] is adjust_exposure
        X, names, raw_p, pipeline_p, *_ = pipeline(folder / "work/event_best.pt", training, target, positive, torch.device("cuda"))
        assert X.shape == (20, 37) and len(names) == 37 and np.isfinite(X).all()
        np.testing.assert_array_equal(pipeline_p[1], raw_p[1])
        assert neural_pipeline.__globals__["adjust_exposure"] is adjust_exposure
    print("Exposure-matched self-test passed: CPU/CUDA sampling, exact source updates, 196+64 Adam schedule, event8 and cold/empty inference.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["self-test", "setup", "train", "evaluate"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.action == "self-test":
        self_test(root)
        return
    import torch
    assert torch.cuda.is_available(), "Activate techarena with working CUDA."
    torch.set_num_threads(4)
    trial = root.parent / "techarena-exposure-matched-trial"
    if args.action == "setup":
        setup(root, trial)
        return
    protocol = read_json(trial / "outputs/frozen_protocol.json")
    verify(trial, protocol)
    if args.action == "train":
        train(trial, protocol)
    else:
        evaluate(root, trial, protocol)


if __name__ == "__main__":
    main()
