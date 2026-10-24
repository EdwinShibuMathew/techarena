"""Replace only V6 seed42 with a time-of-day bilinear event extension.

In techarena-final: self-test, setup, train, evaluate. All output is separate
in techarena-time-bilinear-trial. No preference retraining or submission.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from time_bilinear_model import RECIPE, file_hash, adapt_event, make_pipeline, model_self_test
from row_auc_objective import CONFIG
from run_row_auc_trial import read_json, write_json
from run_expanded_combiner_trial import CUTOFFS, WEIGHTS, ensemble, read_raw
from run_exposure_matched_trial import hashes, verify_files, load_v6, paired_interval, save_npz
from audit_v5 import auc

REPORT = "outputs/ensemble_trial_results.json"


def verify(trial, protocol):
    assert protocol["recipe"] == RECIPE and protocol["event_objective"] == CONFIG
    assert protocol["v6_member_weights"] == WEIGHTS
    verify_files(Path(), protocol["input_sha256"])
    verify_files(Path(), protocol["protected_v5_v6_sha256"])
    verify_files(trial, protocol["prepared_sha256"])


def prepare(root, fold):
    source = root.parent / f"techarena-row-auc-trial/dim8_seed42/{fold}/src"
    return {"train_events.py": adapt_event((source / "train_events.py").read_text(encoding="utf-8")),
            "train_preferences.py": (source / "train_preferences.py").read_text(encoding="utf-8"),
            "row_auc_objective.py": (root / "src/row_auc_objective.py").read_text(encoding="utf-8"),
            "time_bilinear_model.py": (root / "src/time_bilinear_model.py").read_text(encoding="utf-8")}


def setup(root, trial):
    if trial.exists():
        verify(trial, read_json(trial / "outputs/frozen_protocol.json"))
        print("Time-bilinear setup already complete; continue with train/evaluate.", flush=True)
        return
    import torch
    from run_row_auc_final import verify as verify_final, verify_training as verify_final_training, validation_replay, check_v5
    from run_row_auc_five_day import verify_artifacts
    final = root.parent / "techarena-row-auc-final"
    fp = read_json(final / "outputs/frozen_protocol.json")
    verify_final(final, fp)
    verify_final_training(final, fp, read_json(final / "outputs/training_results.json"))
    checks = read_json(final / "outputs/candidate_submission_checks.json")
    assert checks["submission_ready"] and checks["validation_pipeline_check_passed"] and checks["v5_files_unchanged"]
    assert checks["version"] == "v6_candidate_row_ranking"
    assert file_hash(final / fp["candidate_csv"]) == checks["submission_sha256"]
    verify_artifacts(final, checks["artifact_sha256"])
    check_v5(root)
    print("Replaying V6's historical predictions before any new training...", flush=True)
    raw, replay = validation_replay(root)
    assert all(v["passed"] and max(v["member_max_abs_differences"].values()) == 0 for v in replay.values())
    refs = load_v6(root, raw)
    protected = [Path(p) for p in fp["v5_protected_sha256"]]
    protected += [final / p for p in checks["artifact_sha256"]]
    protected += [final / fp["candidate_csv"], final / "outputs/candidate_submission_checks.json",
                  final / "outputs/frozen_protocol.json", final / "outputs/training_results.json"]
    dependencies = [Path(__file__)] + [root / "src" / name for name in ["time_bilinear_model.py",
        "run_exposure_matched_trial.py", "exposure_matched_preference.py", "row_auc_objective.py",
        "run_row_auc_trial.py", "run_row_auc_final.py", "row_auc_final_utils.py", "run_row_auc_five_day.py",
        "run_expanded_combiner_trial.py", "expanded_combiner_features.py", "prepare_preference_finetune_trial.py",
        "build_submission.py", "submission_features.py", "train_final_models.py", "evaluate_dim16_trial.py",
        "audit_v5.py", "submission_export.py", "build_preference_finetuned_submission.py"]]
    dependencies += [Path(p) for p in fp["input_sha256"]] + [final / p for p in fp["prepared_sha256"]] + protected
    folds, training_inputs = {}, []
    for fold, cutoff in CUTOFFS.items():
        original = root.parent / f"techarena-row-auc-trial/dim8_seed42/{fold}"
        paths = {name: (original / "data/cache" / name).resolve(strict=True) for name in
                 ["train_events.npz", "preference_training.npz", f"split_{fold}.npz"]}
        assert paths["train_events.npz"] == (root / "data/cache/train_events.npz").resolve()
        with np.load(paths[f"split_{fold}.npz"], allow_pickle=False) as saved:
            fitting, valid = saved["train_idx"], saved["valid_idx"]
            assert np.array_equal(fitting, np.flatnonzero(raw["t"] < cutoff))
            assert (raw["t"][valid] >= cutoff).all() and (raw["t"][valid] < cutoff + 86400).all()
            assert len(np.unique(raw["u"][valid])) == len(valid)
            if fold != "jan24":
                idx = refs["two_day" if fold == "jan25" else "jan26"]["indices"]
                assert np.array_equal(valid, idx[raw["t"][idx] < cutoff + 86400])
        preference = (original / "work/preference_best.pt").resolve(strict=True)
        checkpoint = torch.load(preference, map_location="cpu", weights_only=True)
        assert checkpoint["dimension"] == 8 and checkpoint["epoch"] == 260 and checkpoint["fine_tuning_sampling_seed"] == 1042
        with np.load(paths["preference_training.npz"], allow_pickle=False) as saved:
            assert int(saved["cutoff"]) == cutoff and (saved["timestamps"] < cutoff).all()
            n_users, n_items = int(saved["n_users"]), int(saved["n_items"])
            assert checkpoint["n_users"] == n_users and checkpoint["n_items"] == n_items
            assert np.isin(saved["users"], np.unique(raw["u"][fitting])).all()
        counts = np.bincount(raw["tb"][fitting], minlength=4)
        assert len(counts) == 4 and (counts > 0).all()
        folds[fold] = {"paths": paths, "preference": preference, "scripts": prepare(root, fold),
                       "metadata": {"n_users": n_users, "n_items": n_items, "fitting_events": len(fitting),
                                    "fitting_time_bucket_events": counts.tolist(), "preference_sha256": file_hash(preference)}}
        dependencies += list(paths.values()) + [preference, original / "src/train_events.py", original / "src/train_preferences.py"]
        training_inputs += list(paths.values()) + [preference]
        del checkpoint
    protocol = {"experiment": "time-of-day bilinear interaction in V6 seed42 only",
        "recipe": RECIPE, "event_objective": CONFIG, "v6_member_weights": WEIGHTS,
        "unchanged_members": ["dim8_seed43", "dim8_seed44", "dim16_seed42", "dim16_seed43"],
        "preference_training_performed": False, "inference": "original softmax, exposure adjustment, ridge history fallback and 37 features; extended event model only",
        "combiner": {"trees": 545, "depth": 6, "learning_rate": .05, "l2_leaf_reg": 20, "seed": 42, "early_stopping": False},
        "primary_check": "Jan24 meta-training; Jan25 prefix; first-user Jan25-26 evaluation",
        "secondary_check": "Jan25 meta-training; Jan26 prefix; first-user Jan26 evaluation",
        "support_rule": "global primary paired gain interval entirely positive and global secondary AUC change nonnegative",
        "next_gate": "identical fixed recipe versus V6 in five-day paired comparison before full-data training or submission",
        "bootstrap_resamples": 300, "bootstrap_seed": 2026, "weight_search_performed": False,
        "status": "retrospective development experiment; reused dates; intervals not selection-adjusted",
        "research_context": "https://doi.org/10.1145/1864708.1864727; motivation only, not a replication or predicted lift",
        "v6_replay": replay, "folds": {fold: data["metadata"] for fold, data in folds.items()},
        "input_sha256": hashes(dependencies), "protected_v5_v6_sha256": hashes(protected),
        "training_input_sha256": hashes(training_inputs), "prepared_sha256": {}}
    with tempfile.TemporaryDirectory(prefix=".time_bilinear_setup_", dir=root.parent) as temporary:
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
            (folder / "work/preference_best.pt").symlink_to(data["preference"])
        write_json(staged / "outputs/frozen_protocol.json", protocol)
        staged.rename(trial)
    verify(trial, protocol)
    print("TIME-BILINEAR SETUP PASSED", flush=True)
    print(json.dumps({"workspace": str(trial), "recipe": RECIPE, "folds": protocol["folds"],
                      "v6_reproduction_passed": True, "v5_v6_protected": True}, indent=2), flush=True)


def check_checkpoint(trial, protocol, fold):
    import torch
    folder = trial / fold
    path = folder / "work/event_best.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    for key, expected in {"epoch": 8, "dimension": 8, "seed": 42, "cutoff": CUTOFFS[fold],
        "time_bilinear_recipe": RECIPE, "event_objective": CONFIG["objective"], "pair_sampling_seed": 3042,
        "n_users": protocol["folds"][fold]["n_users"], "n_items": protocol["folds"][fold]["n_items"],
        "frozen_protocol_sha256": file_hash(trial / "outputs/frozen_protocol.json"),
        "preference_checkpoint_sha256": protocol["folds"][fold]["preference_sha256"]}.items():
        assert saved[key] == expected, (fold, key)
    assert file_hash(folder / "work/preference_best.pt") == saved["preference_checkpoint_sha256"]
    delta = saved["model_state"]["time_interaction"]
    assert tuple(delta.shape) == (4, 8, 8) and torch.isfinite(delta).all()
    assert all(torch.isfinite(v).all() for v in saved["model_state"].values() if v.is_floating_point())
    for state in saved["optimizer_state"]["state"].values():
        assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v) and v.is_floating_point())
    return {"sha256": file_hash(path), "epoch": 8, "interaction_rms": float(delta.square().mean().sqrt()),
            "interaction_max_absolute": float(delta.abs().max())}


def verify_training(trial, protocol, report):
    assert report["protocol_sha256"] == file_hash(trial / "outputs/frozen_protocol.json")
    assert set(report["folds"]) == set(CUTOFFS)
    for fold in CUTOFFS:
        assert report["folds"][fold] == check_checkpoint(trial, protocol, fold)


def train(trial, protocol):
    output = trial / "outputs/training_results.json"
    if output.exists():
        verify_training(trial, protocol, read_json(output))
        print("Three event fits already complete; continue with evaluate.", flush=True)
        return
    start, records = time.perf_counter(), {}
    for fold in CUTOFFS:
        folder = trial / fold
        if not (folder / "work/event_best.pt").exists():
            print(f"\nTraining time-bilinear {fold}: eight event epochs, preference reused", flush=True)
            subprocess.run([sys.executable, str(folder / "src/train_events.py")], cwd=folder, check=True)
        records[fold] = check_checkpoint(trial, protocol, fold)
    verify(trial, protocol)
    report = {"folds": records, "protocol_sha256": file_hash(trial / "outputs/frozen_protocol.json"),
              "runtime_seconds": round(time.perf_counter() - start, 2)}
    verify_training(trial, protocol, report)
    write_json(output, report)
    print("TIME-BILINEAR EVENT TRAINING RESULTS", flush=True)
    print(json.dumps(report, indent=2), flush=True)


def evaluate(root, trial, protocol):
    output = trial / REPORT
    training = read_json(trial / "outputs/training_results.json")
    verify_training(trial, protocol, training)
    if output.exists():
        report = read_json(output)
        verify_files(trial, report["artifact_sha256"])
        print("\nTIME-BILINEAR INTERACTION TRIAL", flush=True)
        print(json.dumps(report, indent=2), flush=True)
        return
    import torch
    from build_submission import FIT_FIELDS, TARGET_FIELDS, predict_combiner, load_combiner, check_close
    from evaluate_dim16_trial import fit_combiner
    start = time.perf_counter()
    raw, pipeline = read_raw(root), make_pipeline()
    refs = load_v6(root, raw)
    primary = refs["two_day"]
    day25 = raw["t"][primary["indices"]] < CUTOFFS["jan26"]
    with np.load(trial / "jan24/data/cache/split_jan24.npz", allow_pickle=False) as saved:
        meta24 = saved["valid_idx"]
    assert not np.intersect1d(meta24, primary["indices"]).size
    assert not np.intersect1d(primary["indices"][day25], refs["jan26"]["indices"]).size

    def features(fold, idx):
        fitting = np.flatnonzero(raw["t"] < CUTOFFS[fold])
        assert (raw["t"][idx] >= CUTOFFS[fold]).all() and len(np.unique(raw["u"][idx])) == len(idx)
        X, names, _, p, *_ = pipeline(trial / fold / "work/event_best.pt", {k: raw[k][fitting] for k in FIT_FIELDS},
            {k: raw[k][idx] for k in TARGET_FIELDS}, trial / fold / "data/cache/preference_training.npz", torch.device("cuda"))
        assert X.shape == (len(idx) * 10, 37) and len(names) == 37 and np.isfinite(X).all()
        return X, names, p

    print("Fitting two honest 37-feature time-bilinear combiners...", flush=True)
    X24, names, _ = features("jan24", meta24)
    X25, names25, neural25 = features("jan25", primary["indices"])
    assert names25 == names
    model24 = fit_combiner(X24, raw["y"][meta24], names)
    path24 = trial / "work/combiner_jan24.cbm"
    model24.save_model(str(path24))
    p25 = predict_combiner(model24, X25, names).reshape(-1, 10)
    check_close(p25, predict_combiner(load_combiner(path24), X25, names).reshape(-1, 10), "Jan24 combiner reload", rtol=0, atol=1e-15)
    del X24, model24
    model25 = fit_combiner(X25[np.repeat(day25, 10)], primary["labels"][day25], names)
    path25 = trial / "work/combiner_jan25.cbm"
    model25.save_model(str(path25))
    del X25
    X26, names26, neural26 = features("jan26", refs["jan26"]["indices"])
    assert names26 == names
    p26 = predict_combiner(model25, X26, names).reshape(-1, 10)
    check_close(p26, predict_combiner(load_combiner(path25), X26, names).reshape(-1, 10), "Jan25 combiner reload", rtol=0, atol=1e-15)
    del X26, model25
    artifacts = [path24, path25, trial / "outputs/training_results.json"]
    metrics, candidates = {}, {}
    for case, updated, neural in [("two_day", p25, neural25), ("jan26", p26, neural26)]:
        ref = refs[case]
        members = [updated] + ref["members"][1:]
        candidate = ensemble(members)
        assert candidate.shape == ref["labels"].shape and np.isfinite(candidate).all() and ((candidate >= 0) & (candidate <= 1)).all()
        candidates[case] = candidate
        before, after = auc(ref["labels"], ref["baseline"]), auc(ref["labels"], candidate)
        groups = {}
        for bucket in range(4):
            mask = raw["tb"][ref["indices"]] == bucket
            old, new = auc(ref["labels"][mask], ref["baseline"][mask]), auc(ref["labels"][mask], candidate[mask])
            groups[str(bucket)] = {"events": int(mask.sum()), "baseline_auc": old, "candidate_auc": new,
                                  "auc_change": None if old is None or new is None else new - old}
        metrics[case] = {"events": len(candidate), "baseline_v6_auc": before, "candidate_auc": after, "auc_change": after - before,
            "original_seed42_auc": auc(ref["labels"], ref["members"][0]), "time_bilinear_seed42_auc": auc(ref["labels"], updated),
            "time_bilinear_neural_pipeline_auc": auc(ref["labels"], neural), "time_buckets_exploratory": groups}
        path = trial / "work" / f"{case}_ensemble_predictions.npz"
        save_npz(path, valid_idx=ref["indices"], labels=ref["labels"], baseline=ref["baseline"], candidate=candidate,
                 new_dim8_seed42=updated, v6_members=np.stack(ref["members"]), candidate_members=np.stack(members),
                 weights=np.asarray(WEIGHTS), protocol_sha256=file_hash(trial / "outputs/frozen_protocol.json"))
        artifacts.append(path)
    interval, valid = paired_interval(primary["labels"], primary["baseline"], candidates["two_day"],
                                      protocol["bootstrap_resamples"], protocol["bootstrap_seed"])
    verify(trial, protocol)
    verify_training(trial, protocol, training)
    report = {"protocol": {k: v for k, v in protocol.items() if not k.endswith("sha256")},
        "two_day_primary": metrics["two_day"], "jan26_secondary": metrics["jan26"],
        "primary_paired_gain_95_percentile_interval": interval, "valid_bootstrap_resamples": valid,
        "v6_reproduction_passed": True, "saved_combiners_reproduced": True,
        "unchanged_four_members_reused_exactly": True, "v5_files_unchanged": True, "v6_files_unchanged": True,
        "supported_for_five_day_check": bool(interval[0] > 0 and metrics["jan26"]["auc_change"] >= 0),
        "sampling_unit": "whole first event per user, preserving all ten candidates; global row AUC",
        "submission_created": False, "upload_performed": False, "runtime_seconds": round(time.perf_counter() - start, 2),
        "artifact_sha256": {str(p.relative_to(trial)): file_hash(p) for p in artifacts}}
    write_json(output, report)
    print("\nTIME-BILINEAR INTERACTION TRIAL", flush=True)
    print(json.dumps(report, indent=2), flush=True)


def self_test(root):
    import torch
    from train_final_models import PreferenceModel
    from build_submission import neural_pipeline, FIT_FIELDS, TARGET_FIELDS
    from row_auc_objective import self_test as objective_self_test
    assert torch.cuda.is_available(), "Activate techarena with working CUDA."
    model_self_test()
    objective_self_test()
    cutoff, rng = CUTOFFS["jan24"], np.random.default_rng(2039)
    raw = {"u": np.array([0, 1, 2, 0, 1, 2, 3, 4]), "i": rng.integers(0, 30, (8, 10)),
           "t": np.r_[cutoff - 3600 + np.arange(6) * 60, [cutoff, cutoff + 60]],
           "tb": np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int8), "dt": np.zeros(8, dtype=np.int8),
           "h": np.full((8, 20), -1, dtype=np.int32), "ht": np.zeros((8, 20), dtype=np.int64), "y": np.zeros((8, 10), dtype=np.int8)}
    raw["y"][[0, 2, 4, 6], [1, 2, 3, 4]] = 1
    raw["h"][6, :2], raw["ht"][6, :2] = [1, 2], [cutoff - 7200, cutoff - 3600]
    with tempfile.TemporaryDirectory(prefix="time_bilinear_selftest_", dir=root) as temporary:
        trial = Path(temporary)
        folder = trial / "jan24"
        for part in ["src", "work", "outputs", "data/cache"]:
            (folder / part).mkdir(parents=True)
        (trial / "outputs").mkdir()
        protocol = {"recipe": RECIPE, "event_objective": CONFIG, "prepared_sha256": {}, "training_input_sha256": {}}
        for name, text in prepare(root, "jan24").items():
            path = folder / "src" / name
            path.write_text(text, encoding="utf-8")
            protocol["prepared_sha256"][str(path.relative_to(trial))] = file_hash(path)
        cache = folder / "data/cache"
        np.savez(cache / "train_events.npz", **raw)
        np.savez(cache / "split_jan24.npz", train_idx=np.arange(6), valid_idx=[6, 7])
        positive = cache / "preference_training.npz"
        np.savez(positive, users=[0, 0, 1, 1, 2, 2], items=[1, 2, 3, 4, 5, 6],
                 timestamps=np.full(6, cutoff - 7200), n_users=8, n_items=30, cutoff=cutoff)
        preference = folder / "work/preference_best.pt"
        torch.manual_seed(42)
        model = PreferenceModel(8, 30, 8, torch.tensor([True, True, True, False, False, False, False, False]))
        torch.save({"model_state": model.state_dict(), "epoch": 260, "dimension": 8, "n_users": 8, "n_items": 30,
                    "fine_tuning_sampling_seed": 1042}, preference)
        protocol["training_input_sha256"] = hashes(list(cache.iterdir()) + [preference])
        protocol["folds"] = {"jan24": {"n_users": 8, "n_items": 30, "preference_sha256": file_hash(preference)}}
        write_json(trial / "outputs/frozen_protocol.json", protocol)
        subprocess.run([sys.executable, str(folder / "src/train_events.py")], cwd=folder, check=True)
        check_checkpoint(trial, protocol, "jan24")
        original_class = neural_pipeline.__globals__["EventModel"]
        pipeline = make_pipeline()
        X, names, raw_p, p, *_ = pipeline(folder / "work/event_best.pt", {k: raw[k][:6] for k in FIT_FIELDS},
            {k: raw[k][6:] for k in TARGET_FIELDS}, positive, torch.device("cuda"))
        assert X.shape == (20, 37) and len(names) == 37 and np.isfinite(X).all()
        assert np.isfinite(raw_p).all() and np.isfinite(p).all()
        assert neural_pipeline.__globals__["EventModel"] is original_class
    print("Generated time-bilinear trainer handoff passed: exact event8 updates plus new penalty, compatible checkpoint and original cold/empty history pipeline.", flush=True)


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
    trial = root.parent / "techarena-time-bilinear-trial"
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
