"""Frozen global row-ranking recipe versus V5 on the same five-day events.

In techarena-final: self-test, setup, train, evaluate.
Reuses earlier V5 preference checkpoints; four new event fits, two combiners.
No full-data training, evaluation.csv access, weight search or submission.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from audit_v5 import AucOrder, auc
from row_auc_objective import CONFIG
from run_row_auc_trial import KEYS, read_json, write_json, verify as verify_short, verify_training as verify_short_training
from run_expanded_combiner_trial import ensemble, WEIGHTS
from run_v5_long_window import (CUTOFFS, END, MEMBERS, first_per_user, load_raw,
                                verify as verify_baseline)
from prepare_preference_finetune_trial import replace_once, classes
from submission_export import sha256

OLD_CUTOFF = 1737676800
RESAMPLES, BOOTSTRAP_SEED = 1000, 2026


def adapt_trainer(source, fold):
    """Only redirect the monitoring split and checkpoint cutoff metadata."""
    assert fold in CUTOFFS
    text = replace_once(source, "split_jan24.npz", f"split_{fold}.npz")
    text = replace_once(text, f'            "cutoff": {OLD_CUTOFF},',
                         f'            "cutoff": {CUTOFFS[fold]},')
    restored = replace_once(text, f"split_{fold}.npz", "split_jan24.npz")
    restored = replace_once(restored, f'            "cutoff": {CUTOFFS[fold]},',
                             f'            "cutoff": {OLD_CUTOFF},')
    assert restored == source and classes(text) == classes(source)
    assert "for epoch in range(9):" in text and "if epoch != 8:" in text
    assert text.count("mixed_row_auc_loss(") == 1
    assert "loss += 0.1 * (" in text and "loss += 0.05 * (" in text
    ast.parse(text)
    return text


def verify_artifacts(folder, hashes):
    for relative, expected in hashes.items():
        assert sha256(folder / relative) == expected, f"Changed saved artifact: {folder / relative}"


def input_fingerprint(hashes):
    canonical = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def verify_report_reconciliation(ref, proof, input_hashes):
    """A replay is reusable only with the exact source/model/data bytes checked."""
    assert proof["input_fingerprint"] == input_fingerprint(input_hashes), "Replay proof belongs to different frozen inputs"
    assert proof["training_report_references"] == ref["training_report_references"], "Baseline report references changed"
    required = set(ref["training_report_references"]["mismatched_members"])
    assert set(proof["replayed_members"]) == required, "Missing baseline prediction replay"
    assert all(0 <= difference <= 1e-15 for difference in proof["replayed_members"].values())


def replay_affected_baseline(root, raw, baseline, ref):
    """Replace a stale report-file link with direct checkpoint/combiner replay.

    The original baseline trainer rewrites its runtime-bearing training report
    on every train action. Never relabel cached hashes or assume this is benign:
    score each affected member again using the current, hash-verified artifacts.
    """
    required = ref["training_report_references"]["mismatched_members"]
    if not required:
        return {}
    import torch
    from build_submission import TARGET_FIELDS, FIT_FIELDS, neural_pipeline, predict_combiner, load_combiner, check_close
    assert torch.cuda.is_available(), "CUDA is required to verify baseline predictions after a training-report hash mismatch."
    torch.set_num_threads(4)
    with np.load(baseline / "data/cache/jan22/split_jan22.npz", allow_pickle=False) as saved:
        fitting = saved["train_idx"]
    assert np.array_equal(fitting, np.flatnonzero(raw["t"] < CUTOFFS["jan22"]))
    helper16 = None
    differences = {}
    print("Baseline training-report hashes differ; verifying the actual model predictions before accepting the cache.", flush=True)
    for number, (key, dimension, _) in enumerate(MEMBERS):
        if key not in required:
            continue
        if dimension == 16 and helper16 is None:
            original = read_json(baseline / "outputs/frozen_protocol.json")
            spec = importlib.util.spec_from_file_location("row_auc_report_replay_dim16", original["dim16_helper"])
            helper16 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper16)
        pipeline = neural_pipeline if dimension == 8 else helper16.neural_pipeline
        print(f"Replaying saved five-day baseline: {key}", flush=True)
        idx = ref["indices"]
        X, names, *_ = pipeline(baseline / key / "jan22/work/event_best.pt",
            {k: raw[k][fitting] for k in FIT_FIELDS}, {k: raw[k][idx] for k in TARGET_FIELDS},
            baseline / "data/cache/jan22/preference_training.npz", torch.device("cuda"))
        assert X.shape == (len(idx) * 10, 37) and len(names) == 37
        model = load_combiner(baseline / "work" / f"{key}_combiner_jan21.cbm")
        prediction = predict_combiner(model, X, names).reshape(-1, 10)
        differences[key] = check_close(prediction, ref["members"][number],
            f"Baseline replay after report hash mismatch: {key}", rtol=0, atol=1e-15)
        print(f"Baseline replay passed: {key}; maximum difference {differences[key]:.3g}", flush=True)
        del X, model, prediction
        torch.cuda.empty_cache()
    assert set(differences) == set(required)
    return differences


def reference(root, raw, baseline):
    path = baseline / "work/v5_five_day_predictions.npz"
    report = read_json(baseline / "outputs/baseline_results.json")
    assert report["baseline_established"] and report["data_and_saved_combiner_checks_passed"]
    with np.load(path, allow_pickle=False) as saved:
        idx, labels, p, members = [saved[k] for k in ["valid_idx", "labels", "predictions", "members"]]
        assert np.array_equal(saved["weights"], np.asarray(WEIGHTS))
    assert idx.dtype.kind in "iu" and idx.ndim == 1
    assert np.array_equal(idx, first_per_user(raw, CUTOFFS["jan22"], END))
    assert len(np.unique(raw["u"][idx])) == len(idx)
    assert np.array_equal(labels, raw["y"][idx]) and labels.shape == p.shape == (len(idx), 10)
    assert members.shape == (5, len(idx), 10)
    assert np.isfinite(members).all() and ((members >= 0) & (members <= 1)).all()
    np.testing.assert_allclose(ensemble(list(members)), p, rtol=0, atol=1e-15)
    assert len(idx) == report["overall"]["events"] and abs(auc(labels, p) - report["overall"]["auc"]) < 1e-12
    training_hash = sha256(baseline / "outputs/training_results.json")
    cached_training_hashes = {}
    mismatched = []
    for number, (key, _, _) in enumerate(MEMBERS):
        with np.load(baseline / "work" / f"{key}_predictions.npz", allow_pickle=False) as saved:
            assert np.array_equal(saved["valid_idx"], idx) and np.array_equal(saved["labels"], labels)
            np.testing.assert_array_equal(saved["predictions"], members[number])
            cached_hash = str(saved["training_report_sha256"].item())
            assert len(cached_hash) == 64 and all(c in "0123456789abcdef" for c in cached_hash)
            cached_training_hashes[key] = cached_hash
            if cached_hash != training_hash:
                mismatched.append(key)
            assert saved["combiner_sha256"].item() == sha256(baseline / "work" / f"{key}_combiner_jan21.cbm")
    # A mismatch is NOT accepted by this structural loader. prior_evidence must
    # replay affected members before setup; evaluate must validate that frozen proof.
    return {"indices": idx, "labels": labels, "baseline": p, "members": members,
            "training_report_references": {"current_report_sha256": training_hash,
                "cached_report_sha256_by_member": cached_training_hashes, "mismatched_members": mismatched}}


def prior_evidence(root, baseline, short):
    """Require both completed studies and unchanged original inputs/models."""
    short_protocol = read_json(short / "outputs/frozen_protocol.json")
    short_report = read_json(short / "outputs/ensemble_trial_results.json")
    verify_short(short, short_protocol)
    verify_short_training(short, read_json(short / "outputs/training_results.json"))
    verify_artifacts(short, short_report["artifact_sha256"])
    assert short_protocol["event_objective"] == CONFIG and short_protocol["updated_members"] == KEYS
    assert short_report["supported_for_five_day_check"] is True
    assert short_report["primary_paired_gain_95_percentile_interval"][0] > 0
    assert short_report["jan26_secondary"]["auc_change"] >= 0
    assert short_report["cached_v5_ensemble_and_updated_members_reproduction_passed"] is True
    assert short_report["unchanged_three_members_reused_exactly"] is True
    original = read_json(baseline / "outputs/frozen_protocol.json")
    verify_baseline(baseline, original)
    training = read_json(baseline / "outputs/training_results.json")
    assert original["weights"] == WEIGHTS and original["event_epochs"] == 8
    assert original["target_neural_cutoff"] == "2025-01-22T00:00:00+00:00"
    assert original["combiner_fitting_day"] == "2025-01-21"
    assert original["members"] == [k for k, _, _ in MEMBERS]
    paths = [Path(__file__), root / "src/run_row_auc_trial.py", root / "src/row_auc_objective.py",
             root / "src/run_expanded_combiner_trial.py", root / "src/expanded_combiner_features.py",
             root / "src/run_v5_long_window.py", root / "src/prepare_preference_finetune_trial.py",
             root / "src/build_submission.py", root / "src/submission_features.py",
             root / "src/train_final_models.py", root / "src/evaluate_dim16_trial.py",
             root / "src/audit_v5.py", root / "src/submission_export.py", root / "data/cache/train_events.npz",
             baseline / "outputs/frozen_protocol.json", baseline / "outputs/baseline_results.json",
             baseline / "outputs/training_results.json", baseline / "work/v5_five_day_predictions.npz",
             baseline / "data/cache/stress_indices.npy", short / "outputs/frozen_protocol.json",
             short / "outputs/training_results.json", short / "outputs/ensemble_trial_results.json"]
    # Keep both earlier experiments' existing provenance, including generated
    # scripts/caches. No original report, checkpoint or prediction is rewritten.
    for folder, protocol in [(short, short_protocol), (baseline, original)]:
        paths += [Path(p) for p in protocol["input_sha256"]]
        paths += [folder / p for p in protocol["prepared_sha256"]]
    for key, dimension, seed in MEMBERS:
        for fold in CUTOFFS:
            stages = training["members"][key][fold]
            for filename, info in stages.items():
                path = baseline / key / fold / "work" / filename
                assert sha256(path) == info["sha256"]
                paths.append(path)
        paths += [baseline / "work" / f"{key}_predictions.npz", baseline / "work" / f"{key}_combiner_jan21.cbm"]
    raw = load_raw(root)
    ref = reference(root, raw, baseline)
    frozen_inputs = {str(p.resolve(strict=True)): sha256(p) for p in paths}
    differences = replay_affected_baseline(root, raw, baseline, ref)
    for path, expected in frozen_inputs.items():
        assert sha256(Path(path)) == expected, f"Input changed during baseline verification: {path}"
    ref["report_reconciliation"] = {
        "status": "report_hash_mismatch_resolved_by_exact_model_replay" if differences else "report_hashes_match",
        "training_report_references": ref["training_report_references"],
        "replayed_members": differences, "input_fingerprint": input_fingerprint(frozen_inputs)}
    verify_report_reconciliation(ref, ref["report_reconciliation"], frozen_inputs)
    ref["verified_input_sha256"] = frozen_inputs
    return paths, raw, ref


def verify(trial, protocol):
    assert protocol["event_objective"] == CONFIG and protocol["updated_members"] == KEYS
    assert protocol["member_weights"] == WEIGHTS and protocol["bootstrap_resamples"] == RESAMPLES
    for name, expected in protocol["input_sha256"].items():
        assert sha256(Path(name)) == expected, f"Changed frozen input: {name}"
    verify_artifacts(trial, protocol["prepared_sha256"])


def setup(root, trial):
    if trial.exists():
        verify(trial, read_json(trial / "outputs/frozen_protocol.json"))
        print("Five-day row-ranking setup already complete; continue with train/evaluate.", flush=True)
        return
    import torch
    baseline, short = root.parent / "techarena-v5-long-window", root.parent / "techarena-row-auc-trial"
    paths, raw, ref = prior_evidence(root, baseline, short)
    with np.load(baseline / "data/cache/jan21/split_jan21.npz", allow_pickle=False) as saved:
        meta_idx = saved["valid_idx"]
    assert np.array_equal(meta_idx, first_per_user(raw, CUTOFFS["jan21"], CUTOFFS["jan22"]))
    assert np.intersect1d(meta_idx, ref["indices"]).size == 0
    prepared = {}
    for key in KEYS:
        seed = int(key.rsplit("seed", 1)[1])
        source = short / key / "jan24/src"
        event_source = (source / "train_events.py").read_text(encoding="utf-8")
        pref_source = (source / "train_preferences.py").read_text(encoding="utf-8")
        helper_source = (source / "row_auc_objective.py").read_text(encoding="utf-8")
        # Windows-to-WSL copies can differ only in line endings. Both files'
        # original bytes are frozen separately; compare their Python text here.
        assert helper_source == (root / "src/row_auc_objective.py").read_text(encoding="utf-8")
        for fold, cutoff in CUTOFFS.items():
            pref = baseline / key / fold / "work/preference_best.pt"
            saved = torch.load(pref, map_location="cpu", weights_only=True)
            assert saved["seed"] == seed and saved["cutoff"] == cutoff
            assert saved["epoch"] == 260 and saved["dimension"] == 8 and saved["fine_tuning_sampling_seed"] == seed + 1000
            positive = baseline / "data/cache" / fold / "preference_training.npz"
            split = baseline / "data/cache" / fold / f"split_{fold}.npz"
            with np.load(positive, allow_pickle=False) as data:
                assert int(data["cutoff"]) == cutoff and (data["timestamps"] < cutoff).all()
                assert int(data["n_users"]) == saved["n_users"] and int(data["n_items"]) == saved["n_items"]
            with np.load(split, allow_pickle=False) as data:
                assert np.array_equal(data["train_idx"], np.flatnonzero(raw["t"] < cutoff))
                assert np.array_equal(data["valid_idx"], first_per_user(raw, cutoff, cutoff + 86400))
            prepared[(key, fold)] = {"scripts": {"train_events.py": adapt_trainer(event_source, fold),
                "train_preferences.py": pref_source, "row_auc_objective.py": helper_source},
                "links": {"work/preference_best.pt": pref.resolve(strict=True),
                    "data/cache/preference_training.npz": positive.resolve(strict=True),
                    f"data/cache/split_{fold}.npz": split.resolve(strict=True),
                    "data/cache/train_events.npz": (root / "data/cache/train_events.npz").resolve(strict=True)}}
            del saved
    protocol = {
        "experiment": "five-day paired test of the frozen global row-ranking recipe versus unchanged V5",
        "event_objective": CONFIG, "updated_members": KEYS, "unchanged_members": [m[0] for m in MEMBERS[2:]],
        "member_weights": WEIGHTS, "preference_epoch": 260, "preference_checkpoints_reused": True,
        "meta_neural_cutoff": "2025-01-21T00:00:00+00:00", "combiner_fitting_day": "2025-01-21",
        "target_neural_cutoff": "2025-01-22T00:00:00+00:00",
        "target_window": "Jan22 through Jan26 inclusive; first event per user across the entire window",
        "target_events": len(ref["indices"]), "meta_events": len(meta_idx),
        "temporal_rule": "weights, pairs, ridge and aggregates fit strictly before their prefix cutoff; targets only query frozen models",
        "combiner": {"trees": 545, "depth": 6, "learning_rate": .05, "l2_leaf_reg": 20,
                     "seed": 42, "features": 37, "early_stopping": False},
        "bootstrap_resamples": RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
        "support_rule": "overall five-day paired gain interval entirely positive AND pooled 48-120h global AUC change nonnegative; all reproduction checks pass",
        "late_window": "first-user events 48 through 120 hours after Jan22 cutoff, i.e. Jan24-26",
        "daily_and_subgroup_interpretation": "exploratory diagnostics; no subgroup selection, no per-day promotion thresholds",
        "next_gate": "review the paired stress results before deciding on full-data training; this script never promotes or exports",
        "weight_or_objective_search_performed": False,
        "status": "retrospective stress test; dates reused; smaller fitting set; intervals not selection-adjusted or independent confirmation",
        "baseline_report_reconciliation": ref["report_reconciliation"],
        "input_sha256": ref["verified_input_sha256"], "prepared_sha256": {},
    }
    (trial / "outputs").mkdir(parents=True)
    (trial / "work").mkdir()
    for (key, fold), data in prepared.items():
        folder = trial / key / fold
        for part in ["src", "work", "outputs", "data/cache"]:
            (folder / part).mkdir(parents=True)
        for name, source in data["scripts"].items():
            path = folder / "src" / name
            path.write_text(source, encoding="utf-8")
            protocol["prepared_sha256"][str(path.relative_to(trial))] = sha256(path)
        for relative, original in data["links"].items():
            (folder / relative).symlink_to(original)
    write_json(trial / "outputs/frozen_protocol.json", protocol)
    verify(trial, protocol)
    print("ROW-RANKING FIVE-DAY SETUP", flush=True)
    print(json.dumps({k: v for k, v in protocol.items() if k not in ["input_sha256", "prepared_sha256"]}, indent=2), flush=True)


def verify_training(trial, report):
    assert set(report["members"]) == set(KEYS)
    for key, folds in report["members"].items():
        assert set(folds) == set(CUTOFFS)
        for fold, info in folds.items():
            assert sha256(trial / key / fold / "work/event_best.pt") == info["sha256"], f"Changed trained model: {key}/{fold}"


def train(trial, protocol):
    import torch
    output = trial / "outputs/training_results.json"
    if output.exists():
        verify_training(trial, read_json(output))
        print("Four earlier-cutoff event fits complete; continue with evaluate.", flush=True)
        return
    start, records = time.perf_counter(), {}
    for key in KEYS:
        seed = int(key.rsplit("seed", 1)[1])
        records[key] = {}
        for fold, cutoff in CUTOFFS.items():
            folder = trial / key / fold
            path = folder / "work/event_best.pt"
            if not path.exists():
                print(f"\nEight event epochs, frozen row-ranking loss: {key}/{fold}", flush=True)
                subprocess.run([sys.executable, str(folder / "src/train_events.py")], cwd=folder, check=True)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            assert saved["epoch"] == 8 and saved["dimension"] == 8 and saved["seed"] == seed and saved["cutoff"] == cutoff
            assert saved["event_objective"] == CONFIG["objective"] and saved["pair_sampling_seed"] == seed + 3000
            assert all(torch.isfinite(v).all() for v in saved["model_state"].values() if v.is_floating_point())
            records[key][fold] = {"sha256": sha256(path), "epoch": 8, "final_epoch_auc_diagnostic_only": float(saved["validation_auc"])}
            del saved
    verify(trial, protocol)
    report = {"members": records, "runtime_seconds": round(time.perf_counter() - start, 2)}
    verify_training(trial, report)
    write_json(output, report)
    print("\nROW-RANKING FIVE-DAY TRAINING RESULTS", flush=True)
    print(json.dumps(report, indent=2), flush=True)


def metrics(labels, baseline, candidate):
    a, b = auc(labels, baseline), auc(labels, candidate)
    def log_loss(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-(labels * np.log(p) + (1 - labels) * np.log1p(-p)).mean()) if labels.size else None
    return {"events": len(labels), "rows": labels.size, "clicks": int(labels.sum()),
            "baseline_v5_auc": a, "candidate_auc": b, "auc_change": b - a if a is not None and b is not None else None,
            "baseline_log_loss": log_loss(baseline), "candidate_log_loss": log_loss(candidate)}


def paired_intervals(labels, baseline, candidate, late, resamples=RESAMPLES, progress=True):
    n = len(labels)
    assert labels.shape == baseline.shape == candidate.shape == (n, 10)
    assert late.dtype == np.bool_ and late.shape == (n,) and late.any()
    masks = {"overall": np.ones(n, dtype=bool), "late_48_120h": late}
    orders = {name: [AucOrder(labels[mask].ravel(), p[mask].ravel(), np.repeat(np.flatnonzero(mask), 10))
                     for p in [baseline, candidate]] for name, mask in masks.items()}
    samples = {name: [] for name in masks}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for repetition in range(resamples):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        for name, (before, after) in orders.items():
            a, b = before.value(counts), after.value(counts)
            if a is not None and b is not None:
                samples[name].append(b-a)
        if progress and (repetition + 1) % 100 == 0:
            print(f"Paired whole-user bootstrap: {repetition + 1}/{resamples}", flush=True)
    result = {}
    for name, values in samples.items():
        assert len(values) >= .99 * resamples, f"Too few valid bootstrap samples: {name}"
        result[name] = {"paired_gain_95_percentile_interval": np.quantile(values, [.025, .975]).tolist(),
                        "valid_resamples": len(values)}
    return result


def evaluate(root, trial, protocol):
    output = trial / "outputs/paired_results.json"
    if output.exists():
        report = read_json(output)
        verify_artifacts(trial, report["artifact_sha256"])
        print("Saved five-day comparison complete; displaying it.", flush=True)
        print("\nGLOBAL ROW-RANKING FIVE-DAY PAIRED COMPARISON", flush=True)
        print(output.read_text(encoding="utf-8"), flush=True)
        return
    import torch
    from build_submission import TARGET_FIELDS, FIT_FIELDS, neural_pipeline, predict_combiner, load_combiner, check_close
    from evaluate_dim16_trial import fit_combiner
    start = time.perf_counter()
    baseline = root.parent / "techarena-v5-long-window"
    training_path = trial / "outputs/training_results.json"
    training_hash = sha256(training_path)
    verify_training(trial, read_json(training_path))
    raw = load_raw(root)
    ref = reference(root, raw, baseline)
    verify_report_reconciliation(ref, protocol["baseline_report_reconciliation"], protocol["input_sha256"])
    idx, labels = ref["indices"], ref["labels"]
    with np.load(baseline / "data/cache/jan21/split_jan21.npz", allow_pickle=False) as saved:
        meta = saved["valid_idx"]
    assert np.array_equal(meta, first_per_user(raw, CUTOFFS["jan21"], CUTOFFS["jan22"]))
    assert np.intersect1d(meta, idx).size == 0
    assert len(idx) == protocol["target_events"] and len(meta) == protocol["meta_events"]
    new_members, checks = [], {}
    artifacts = [training_path] + [trial / key / fold / "work/event_best.pt" for key in KEYS for fold in CUTOFFS]
    protocol_hash = sha256(trial / "outputs/frozen_protocol.json")
    for number, key in enumerate(KEYS):
        cache = trial / "work" / f"{key}_predictions.npz"
        model_path = trial / "work" / f"{key}_combiner_jan21.cbm"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                assert saved["protocol_sha256"].item() == protocol_hash and saved["training_report_sha256"].item() == training_hash
                assert saved["combiner_sha256"].item() == sha256(model_path)
                assert np.array_equal(saved["valid_idx"], idx) and np.array_equal(saved["labels"], labels)
                p = saved["predictions"]
                checks[key] = json.loads(saved["checks_json"].item())
            print(f"Using completed new combiner and predictions: {key}", flush=True)
        else:
            def features(fold, targets, old=False):
                folder = (baseline if old else trial) / key / fold
                cache_folder = baseline / "data/cache" / fold
                with np.load(cache_folder / f"split_{fold}.npz", allow_pickle=False) as saved:
                    fitting = saved["train_idx"]
                assert np.array_equal(fitting, np.flatnonzero(raw["t"] < CUTOFFS[fold]))
                assert (raw["t"][targets] >= CUTOFFS[fold]).all() and (raw["t"][targets] < END).all()
                assert len(np.unique(raw["u"][targets])) == len(targets)
                X, names, *_ = neural_pipeline(folder / "work/event_best.pt",
                    {k: raw[k][fitting] for k in FIT_FIELDS}, {k: raw[k][targets] for k in TARGET_FIELDS},
                    cache_folder / "preference_training.npz", torch.device("cuda"))
                assert len(names) == 37 and X.shape == (len(targets) * 10, 37)
                return X, names

            print(f"Reproducing original five-day V5 member: {key}", flush=True)
            Xold, names = features("jan22", idx, old=True)
            old_p = predict_combiner(load_combiner(baseline / "work" / f"{key}_combiner_jan21.cbm"), Xold, names).reshape(-1, 10)
            difference = check_close(old_p, ref["members"][number], f"Original five-day member {key}", rtol=0, atol=1e-15)
            del Xold, old_p
            X21, names21 = features("jan21", meta)
            assert names21 == names
            model = fit_combiner(X21, raw["y"][meta], names)
            model.save_model(str(model_path))
            del X21
            X22, names22 = features("jan22", idx)
            assert names22 == names
            p = predict_combiner(model, X22, names).reshape(-1, 10)
            reloaded = predict_combiner(load_combiner(model_path), X22, names).reshape(-1, 10)
            check_close(p, reloaded, f"Saved new five-day model {key}", rtol=0, atol=1e-15)
            checks[key] = {"original_v5_max_abs_difference": difference, "new_saved_combiner_reproduced": True}
            np.savez_compressed(cache, valid_idx=idx, labels=labels, predictions=p,
                protocol_sha256=protocol_hash, training_report_sha256=training_hash,
                combiner_sha256=sha256(model_path), checks_json=json.dumps(checks[key]))
            del X22, model, reloaded
            torch.cuda.empty_cache()
        assert p.shape == labels.shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        new_members.append(p)
        artifacts += [cache, model_path]
    candidate_members = np.stack(new_members + list(ref["members"][2:]))
    assert np.array_equal(candidate_members[2:], ref["members"][2:])
    candidate = ensemble(list(candidate_members))
    path = trial / "work/five_day_paired_predictions.npz"
    np.savez_compressed(path, valid_idx=idx, labels=labels, baseline=ref["baseline"], candidate=candidate,
                        baseline_members=ref["members"], candidate_members=candidate_members, weights=np.asarray(WEIGHTS))
    artifacts.append(path)
    elapsed = (raw["t"][idx] - CUTOFFS["jan22"]) / 3600
    late = (elapsed >= 48) & (elapsed < 120)
    overall = metrics(labels, ref["baseline"], candidate)
    late_result = metrics(labels[late], ref["baseline"][late], candidate[late])
    print(json.dumps({"five_day_overall": overall, "late_48_120h": late_result}, indent=2), flush=True)
    intervals = paired_intervals(labels, ref["baseline"], candidate, late)
    overall.update(intervals["overall"])
    late_result.update(intervals["late_48_120h"])
    daily = {}
    for day in range(5):
        mask = (elapsed >= 24 * day) & (elapsed < 24 * (day + 1))
        date = str(np.datetime64("2025-01-22") + np.timedelta64(day, "D"))
        daily[date] = metrics(labels[mask], ref["baseline"][mask], candidate[mask])
    activity = np.bincount(raw["u"][raw["t"] < CUTOFFS["jan22"]], minlength=int(raw["u"].max()) + 1)[raw["u"][idx]]
    length = (raw["h"][idx] >= 0).sum(1)
    groups = {}
    for name, mask in {"seen_users": activity > 0, "unseen_with_history": (activity == 0) & (length > 0),
                       "history_empty": length == 0, "history_1_5": (length >= 1) & (length <= 5),
                       "activity_1_5": (activity >= 1) & (activity <= 5), "activity_16_plus": activity >= 16}.items():
        groups[name] = metrics(labels[mask], ref["baseline"][mask], candidate[mask])
    verify(trial, protocol)
    assert sha256(training_path) == training_hash
    verify_training(trial, read_json(training_path))
    overall_pass = bool(overall["paired_gain_95_percentile_interval"][0] > 0)
    late_pass = bool(late_result["auc_change"] is not None and late_result["auc_change"] >= 0)
    report = {"protocol": {k: v for k, v in protocol.items() if k not in ["input_sha256", "prepared_sha256"]},
              "five_day_overall": overall, "late_48_120h": late_result, "per_day_exploratory": daily,
              "groups_exploratory": groups, "member_checks": checks,
              "baseline_ensemble_and_original_members_reproduced": True,
              "unchanged_three_members_reused_exactly": True, "overall_gain_gate_passed": overall_pass,
              "late_window_gate_passed": late_pass, "supported_for_full_data_review": overall_pass and late_pass,
              "sampling_unit": "whole first event per user across Jan22-26, preserving all ten candidates; same draw for both models",
              "submission_created": False, "automatic_promotion_performed": False,
              "runtime_seconds": round(time.perf_counter() - start, 2),
              "artifact_sha256": {str(p.relative_to(trial)): sha256(p) for p in artifacts}}
    write_json(output, report)
    print("\nGLOBAL ROW-RANKING FIVE-DAY PAIRED COMPARISON", flush=True)
    print(json.dumps(report, indent=2), flush=True)


def dense_auc(labels, predictions):
    y, p = labels.ravel(), predictions.ravel()
    positives, negatives = p[y == 1], p[y == 0]
    return float(((positives[:, None] > negatives) + .5 * (positives[:, None] == negatives)).mean())


def self_test(root):
    short = root.parent / "techarena-row-auc-trial"
    for key in KEYS:
        source = (short / key / "jan24/src/train_events.py").read_text(encoding="utf-8")
        for fold in CUTOFFS:
            adapt_trainer(source, fold)
    rng = np.random.default_rng(16)
    labels = np.zeros((8, 10), dtype=np.int8)
    labels[:, 0] = 1
    old, new = np.round(rng.uniform(size=(2, 8, 10)), 1)
    late = np.arange(8) >= 3
    result = paired_intervals(labels, old, new, late, resamples=30, progress=False)
    draws = np.random.default_rng(BOOTSTRAP_SEED)
    samples = {"overall": [], "late_48_120h": []}
    for _ in range(30):
        idx = draws.integers(0, 8, 8)
        for name, selected in [("overall", idx), ("late_48_120h", idx[late[idx]])]:
            assert len(selected) > 0
            samples[name].append(dense_auc(labels[selected], new[selected]) - dense_auc(labels[selected], old[selected]))
    for name, values in samples.items():
        np.testing.assert_allclose(result[name]["paired_gain_95_percentile_interval"], np.quantile(values, [.025, .975]), rtol=0, atol=1e-14)
    members = rng.uniform(size=(5, 8, 10))
    updated = [members[0].copy(), members[1].copy()] + list(members[2:])
    np.testing.assert_array_equal(ensemble(updated), ensemble(list(members)))
    np.testing.assert_array_equal(np.stack(updated)[2:], members[2:])
    print("Five-day self-test passed: only split/cutoff metadata adapted, original objective/architecture preserved, tie-aware paired bootstrap matches explicit user duplication, and three retained members stay exact.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["self-test", "setup", "train", "evaluate"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.action == "self-test":
        self_test(root)
        return
    trial = root.parent / "techarena-row-auc-five-day"
    if args.action == "setup":
        setup(root, trial)
        return
    protocol = read_json(trial / "outputs/frozen_protocol.json")
    verify(trial, protocol)
    import torch
    assert torch.cuda.is_available(), "Activate techarena with working CUDA."
    torch.set_num_threads(4)
    if args.action == "train":
        train(trial, protocol)
    else:
        evaluate(root, trial, protocol)


if __name__ == "__main__":
    main()
