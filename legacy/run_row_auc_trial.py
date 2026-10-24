"""Fixed mixed event-likelihood/global-row-ranking experiment for V5.

In techarena-final: self-test, setup, train, evaluate. Reuses the two fine-tuned
dim8 preference checkpoints. Six new eight-epoch event fits and four combiners
are separate from V5. No full-data fitting or competition submission.
"""

import argparse
import ast
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from row_auc_objective import CONFIG
from prepare_preference_finetune_trial import adapt_events, classes, replace_once
from run_expanded_combiner_trial import member_specs, ensemble, reference_paths, load_references, read_raw, CUTOFFS, WEIGHTS
from audit_v5 import AucOrder, auc
from submission_export import sha256

KEYS = ["dim8_seed42", "dim8_seed43"]


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_sources(preference_source, event_source, seed, fold):
    pref_class = next(n for n in ast.parse(preference_source).body if isinstance(n, ast.ClassDef) and n.name == "PreferenceModel")
    shim = '"""Original preference architecture; checkpoint is reused read-only."""\nimport torch\nfrom torch import nn\n\n' + ast.get_source_segment(preference_source, pref_class) + '\n'
    event = adapt_events(event_source, seed, fold)
    event = replace_once(event, "from train_preferences import PreferenceModel",
                         "from train_preferences import PreferenceModel\nfrom row_auc_objective import mixed_row_auc_loss")
    event = replace_once(event, "    optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)",
                         "    optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)\n"
                         f"    pair_generator = torch.Generator(device=device).manual_seed({seed + CONFIG['pair_sampling_seed_offset']})")
    event = replace_once(event, '''                loss = nn.functional.cross_entropy(
                    include_no_click(scores),
                    train_targets[q],
                )''', '''                loss = mixed_row_auc_loss(
                    scores, train_targets[q], pair_generator
                )''')
    marker = '            "validation_auc": auc,\n        }\n\n        torch.save(state, "work/event_last.pt")'
    event = replace_once(event, marker,
                         '            "validation_auc": auc,\n'
                         f'            "seed": {seed},\n            "cutoff": {CUTOFFS[fold]},\n'
                         f'            "event_objective": "{CONFIG["objective"]}",\n'
                         f'            "pair_sampling_seed": {seed + CONFIG["pair_sampling_seed_offset"]},\n'
                         '        }\n\n        torch.save(state, "work/event_last.pt")')
    assert classes(shim) == classes(preference_source) and classes(event) == classes(event_source)
    assert "if epoch != 8:" in event and "for epoch in range(9):" in event
    assert "loss += 0.1 * (" in event and "loss += 0.05 * (" in event
    ast.parse(event)
    return {"train_preferences.py": shim, "train_events.py": event}


def verify(trial, protocol):
    assert protocol["event_objective"] == CONFIG and protocol["member_weights"] == WEIGHTS
    assert protocol["updated_members"] == KEYS
    for name, expected in protocol["input_sha256"].items():
        assert sha256(Path(name)) == expected, f"Changed frozen input: {name}"
    for name, expected in protocol["prepared_sha256"].items():
        assert sha256(trial / name) == expected, f"Changed generated trainer: {name}"


def setup(root, trial):
    if trial.exists():
        verify(trial, read_json(trial / "outputs/frozen_protocol.json"))
        print("Row-ranking trial already prepared; continue with train/evaluate.", flush=True)
        return
    import torch
    source = root.parent / "techarena-seed43/src"
    pref_source = (source / "train_preferences.py").read_text(encoding="utf-8")
    event_source = (source / "train_events.py").read_text(encoding="utf-8")
    objective = (root / "src/row_auc_objective.py").read_text(encoding="utf-8")
    inputs = [Path(__file__), source / "train_preferences.py", source / "train_events.py",
              root / "data/cache/train_events.npz"]
    inputs += [root / "src" / name for name in ["row_auc_objective.py", "prepare_preference_finetune_trial.py",
               "run_expanded_combiner_trial.py", "expanded_combiner_features.py", "audit_v5.py", "submission_export.py",
               "build_submission.py", "submission_features.py", "train_final_models.py", "evaluate_dim16_trial.py"]]
    checks_path = root / "outputs/five_model_submission_checks.json"
    checks = read_json(checks_path)
    assert checks["version"] == "v5" and checks["submission_ready"] is True and checks["weights"] == WEIGHTS
    inputs.append(checks_path)
    raw = read_raw(root)
    load_references(root, raw)
    for case in ["two_day", "jan26"]:
        inputs += reference_paths(root.parent, case)
    prepared = {}
    for key, dimension, folder, combiners, prefix in member_specs(root.parent)[:2]:
        assert key in KEYS and dimension == 8
        seed = int(key.rsplit("seed", 1)[1])
        for fold, cutoff in CUTOFFS.items():
            original = folder / fold
            pref_path = original / "work/preference_best.pt"
            pref = torch.load(pref_path, map_location="cpu", weights_only=True)
            assert pref["dimension"] == 8 and pref["epoch"] == 260 and pref["fine_tuning_sampling_seed"] == seed + 1000
            positive = original / "data/cache/preference_training.npz"
            split = original / "data/cache" / f"split_{fold}.npz"
            with np.load(positive, allow_pickle=False) as saved:
                assert int(saved["cutoff"]) == cutoff and (saved["timestamps"] < cutoff).all()
                assert int(saved["n_users"]) == pref["n_users"] and int(saved["n_items"]) == pref["n_items"]
            with np.load(split, allow_pickle=False) as saved:
                assert np.array_equal(saved["train_idx"], np.flatnonzero(raw["t"] < cutoff))
                valid = saved["valid_idx"]
                assert (raw["t"][valid] >= cutoff).all() and (raw["t"][valid] < cutoff + 86400).all()
                assert len(np.unique(raw["u"][valid])) == len(valid)
            paths = {"preference_training.npz": positive.resolve(strict=True), f"split_{fold}.npz": split.resolve(strict=True),
                     "train_events.npz": (root / "data/cache/train_events.npz").resolve(strict=True)}
            scripts = prepare_sources(pref_source, event_source, seed, fold)
            scripts["row_auc_objective.py"] = objective
            prepared[(key, fold)] = {"scripts": scripts, "paths": paths, "preference": pref_path.resolve(strict=True)}
            inputs += [pref_path, positive, split, original / "work/event_best.pt"]
            del pref
        inputs += [combiners / f"{prefix}{day}.cbm" for day in ["jan24", "jan25"]]
    protocol = {
        "experiment": "replace two V5 dim8 event members with mixed event CE and global row-pair ranking training",
        "updated_members": KEYS, "unchanged_members": [s[0] for s in member_specs(root.parent)[2:]],
        "event_objective": CONFIG, "preference_epoch": 260, "preference_checkpoints_reused": True,
        "dimension": 8, "seeds": [42, 43], "member_weights": WEIGHTS,
        "candidate": "replace dim8 seed42/43 only; retain seed44 and both dim16 members, with the same V5 weights",
        "inference": "original 11-outcome softmax, exposure adjustment, cold-history ridge fallback and 37 feature formulas",
        "combiner": {"trees": 545, "depth": 6, "learning_rate": .05, "l2_leaf_reg": 20,
                     "seed": 42, "early_stopping": False, "fitting_selection": "first event per user on preceding meta day"},
        "temporal_rule": "event updates, negative-row sampling, ridge and aggregate fitting strictly before each prefix cutoff",
        "primary_check": "Jan24 meta-training, Jan25 prefix, first-user Jan25-26 evaluation",
        "secondary_check": "Jan25 meta-training, Jan26 prefix, first-user Jan26 evaluation",
        "bootstrap_resamples": 300, "bootstrap_seed": 2026,
        "support_rule": "primary global paired gain interval entirely positive and secondary global AUC change nonnegative",
        "next_gate": "same fixed recipe versus V5 in earlier-cutoff five-day paired test before full-data promotion",
        "weight_search_performed": False,
        "status": "retrospective development experiment; reused validation dates; intervals not selection-adjusted",
        "input_sha256": {str(p.resolve(strict=True)): sha256(p) for p in inputs}, "prepared_sha256": {},
    }
    (trial / "outputs").mkdir(parents=True)
    (trial / "work").mkdir()
    for (key, fold), data in prepared.items():
        folder = trial / key / fold
        for name in ["src", "work", "outputs", "data/cache"]:
            (folder / name).mkdir(parents=True)
        for name, text in data["scripts"].items():
            path = folder / "src" / name
            path.write_text(text, encoding="utf-8")
            protocol["prepared_sha256"][str(path.relative_to(trial))] = sha256(path)
        for name, path in data["paths"].items():
            (folder / "data/cache" / name).symlink_to(path)
        (folder / "work/preference_best.pt").symlink_to(data["preference"])
    write_json(trial / "outputs/frozen_protocol.json", protocol)
    verify(trial, protocol)
    print("ROW-RANKING TRIAL SETUP", flush=True)
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
        print("Row-ranking event training complete; continue with evaluate.", flush=True)
        return
    start, records = time.perf_counter(), {}
    for key in KEYS:
        seed = int(key.rsplit("seed", 1)[1])
        records[key] = {}
        for fold, cutoff in CUTOFFS.items():
            folder = trial / key / fold
            path = folder / "work/event_best.pt"
            if not path.exists():
                print(f"\nEight event epochs with mixed row-ranking loss: {key}/{fold}", flush=True)
                subprocess.run([sys.executable, str(folder / "src/train_events.py")], cwd=folder, check=True)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            assert saved["epoch"] == 8 and saved["dimension"] == 8 and saved["seed"] == seed and saved["cutoff"] == cutoff
            assert saved["event_objective"] == CONFIG["objective"] and saved["pair_sampling_seed"] == seed + 3000
            assert all(torch.isfinite(value).all() for value in saved["model_state"].values() if value.is_floating_point())
            records[key][fold] = {"sha256": sha256(path), "epoch": 8,
                                  "final_epoch_auc_diagnostic_only": float(saved["validation_auc"])}
            del saved
    verify(trial, protocol)
    report = {"members": records, "runtime_seconds": round(time.perf_counter() - start, 2)}
    verify_training(trial, report)
    write_json(output, report)
    print("\nROW-RANKING EVENT TRAINING RESULTS", flush=True)
    print(json.dumps(report, indent=2), flush=True)


def evaluate(root, trial, protocol):
    output = trial / "outputs/ensemble_trial_results.json"
    if output.exists():
        report = read_json(output)
        for relative, expected in report["artifact_sha256"].items():
            assert sha256(trial / relative) == expected, f"Changed trial output: {relative}"
        print("Saved row-ranking trial complete; displaying it.", flush=True)
        print("\nGLOBAL ROW-RANKING TRIAL", flush=True)
        print(output.read_text(encoding="utf-8"), flush=True)
        return
    import torch
    from build_submission import TARGET_FIELDS, FIT_FIELDS, neural_pipeline, predict_combiner, load_combiner, check_close
    from evaluate_dim16_trial import fit_combiner
    start = time.perf_counter()
    training_path = trial / "outputs/training_results.json"
    training_hash = sha256(training_path)
    verify_training(trial, read_json(training_path))
    raw = read_raw(root)
    refs = load_references(root, raw)
    primary, secondary = refs["two_day"], refs["jan26"]
    day25 = raw["t"][primary["indices"]] < CUTOFFS["jan26"]
    with np.load(trial / KEYS[0] / "jan24/data/cache/split_jan24.npz", allow_pickle=False) as saved:
        meta24 = saved["valid_idx"]
    assert (raw["t"][meta24] >= CUTOFFS["jan24"]).all() and (raw["t"][meta24] < CUTOFFS["jan25"]).all()
    assert np.intersect1d(meta24, primary["indices"]).size == 0
    assert np.intersect1d(primary["indices"][day25], secondary["indices"]).size == 0
    updated = {"two_day": [], "jan26": []}
    member_checks, artifacts = {}, []
    protocol_hash = sha256(trial / "outputs/frozen_protocol.json")
    for number, (key, dimension, origin, combiners, prefix) in enumerate(member_specs(root.parent)[:2]):
        cache = trial / "work" / f"{key}_predictions.npz"
        model_paths = {day: trial / "work" / f"{key}_{day}.cbm" for day in ["jan24", "jan25"]}
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                assert saved["protocol_sha256"].item() == protocol_hash
                for case, ref in refs.items():
                    assert np.array_equal(saved[f"{case}_idx"], ref["indices"])
                    p = saved[f"{case}_prediction"]
                    assert p.shape == ref["labels"].shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                    updated[case].append(p)
                for day, path in model_paths.items():
                    assert saved[f"{day}_sha256"].item() == sha256(path)
                member_checks[key] = json.loads(saved["checks_json"].item())
            artifacts += [cache] + list(model_paths.values())
            continue

        def features(fold, indices, old=False):
            folder = (origin if old else trial / key) / fold
            with np.load(folder / "data/cache" / f"split_{fold}.npz", allow_pickle=False) as saved:
                fit_idx, first_idx = saved["train_idx"], saved["valid_idx"]
            assert np.array_equal(fit_idx, np.flatnonzero(raw["t"] < CUTOFFS[fold]))
            assert (raw["t"][indices] >= CUTOFFS[fold]).all() and (raw["t"][indices] < 1737936000).all()
            assert len(np.unique(raw["u"][indices])) == len(indices)
            X, names, *_ = neural_pipeline(folder / "work/event_best.pt", {k: raw[k][fit_idx] for k in FIT_FIELDS},
                                           {k: raw[k][indices] for k in TARGET_FIELDS},
                                           folder / "data/cache/preference_training.npz", torch.device("cuda"))
            assert len(names) == 37
            return X, names, first_idx

        differences = {}
        for case, fold, day in [("two_day", "jan25", "jan24"), ("jan26", "jan26", "jan25")]:
            Xold, names, _ = features(fold, refs[case]["indices"], old=True)
            reproduced = predict_combiner(load_combiner(combiners / f"{prefix}{day}.cbm"), Xold, names).reshape(-1, 10)
            differences[case] = check_close(reproduced, refs[case]["members"][number], f"Original V5 {key}/{case}", rtol=0, atol=1e-15)
            del Xold, reproduced
        print(f"Fitting new 37-feature combiners: {key}", flush=True)
        X24, names, first24 = features("jan24", meta24)
        assert np.array_equal(first24, meta24)
        X25, names25, first25 = features("jan25", primary["indices"])
        assert names25 == names and np.array_equal(first25, primary["indices"][day25])
        model24 = fit_combiner(X24, raw["y"][meta24], names)
        model24.save_model(str(model_paths["jan24"]))
        p_primary = predict_combiner(model24, X25, names).reshape(-1, 10)
        check_close(p_primary, predict_combiner(load_combiner(model_paths["jan24"]), X25, names).reshape(-1, 10),
                    f"Saved {key} primary combiner", rtol=0, atol=1e-15)
        del X24, model24
        model25 = fit_combiner(X25[np.repeat(day25, 10)], primary["labels"][day25], names)
        model25.save_model(str(model_paths["jan25"]))
        del X25
        X26, names26, first26 = features("jan26", secondary["indices"])
        assert names26 == names and np.array_equal(first26, secondary["indices"])
        p_secondary = predict_combiner(model25, X26, names).reshape(-1, 10)
        check_close(p_secondary, predict_combiner(load_combiner(model_paths["jan25"]), X26, names).reshape(-1, 10),
                    f"Saved {key} secondary combiner", rtol=0, atol=1e-15)
        checks = {"primary_v5_max_abs_difference": differences["two_day"], "secondary_v5_max_abs_difference": differences["jan26"],
                  "saved_combiners_reproduced": True}
        np.savez_compressed(cache, two_day_idx=primary["indices"], jan26_idx=secondary["indices"],
                            two_day_prediction=p_primary, jan26_prediction=p_secondary,
                            protocol_sha256=protocol_hash, checks_json=json.dumps(checks),
                            jan24_sha256=sha256(model_paths["jan24"]), jan25_sha256=sha256(model_paths["jan25"]))
        member_checks[key] = checks
        updated["two_day"].append(p_primary)
        updated["jan26"].append(p_secondary)
        artifacts += [cache] + list(model_paths.values())
        del X26, model25
        torch.cuda.empty_cache()
    candidates, metrics = {}, {}
    for case, ref in refs.items():
        candidate = ensemble(updated[case] + ref["members"][2:])
        assert candidate.shape == ref["labels"].shape and np.isfinite(candidate).all() and ((candidate >= 0) & (candidate <= 1)).all()
        candidates[case] = candidate
        before, after = auc(ref["labels"], ref["baseline"]), auc(ref["labels"], candidate)
        metrics[case] = {"events": len(ref["indices"]), "baseline_v5_auc": before, "candidate_row_ranking_auc": after,
                         "auc_change": after - before, "new_dim8_seed42_auc": auc(ref["labels"], updated[case][0]),
                         "new_dim8_seed43_auc": auc(ref["labels"], updated[case][1])}
        path = trial / "work" / f"{case}_ensemble_predictions.npz"
        np.savez_compressed(path, valid_idx=ref["indices"], labels=ref["labels"], baseline=ref["baseline"], candidate=candidate,
                            new_dim8_seed42=updated[case][0], new_dim8_seed43=updated[case][1], weights=np.asarray(WEIGHTS))
        artifacts.append(path)
    print(json.dumps(metrics, indent=2), flush=True)
    n = len(primary["indices"])
    ids = np.repeat(np.arange(n), 10)
    old_order = AucOrder(primary["labels"].ravel(), primary["baseline"].ravel(), ids)
    new_order = AucOrder(primary["labels"].ravel(), candidates["two_day"].ravel(), ids)
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    gains = []
    for repetition in range(protocol["bootstrap_resamples"]):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        gains.append(new_order.value(counts) - old_order.value(counts))
        if (repetition + 1) % 50 == 0:
            print(f"Global paired user bootstrap: {repetition + 1}/300", flush=True)
    interval = np.quantile(gains, [.025, .975]).tolist()
    verify(trial, protocol)
    assert sha256(training_path) == training_hash
    verify_training(trial, read_json(training_path))
    report = {"protocol": {k: v for k, v in protocol.items() if k not in ["input_sha256", "prepared_sha256"]},
              "two_day_primary": metrics["two_day"], "jan26_secondary": metrics["jan26"],
              "primary_paired_gain_95_percentile_interval": interval, "member_checks": member_checks,
              "cached_v5_ensemble_and_updated_members_reproduction_passed": True,
              "unchanged_three_members_reused_exactly": True,
              "supported_for_five_day_check": bool(interval[0] > 0 and metrics["jan26"]["auc_change"] >= 0),
              "sampling_unit": "whole first event per user, preserving all ten candidates; global row AUC",
              "submission_created": False, "runtime_seconds": round(time.perf_counter() - start, 2),
              "artifact_sha256": {str(p.relative_to(trial)): sha256(p) for p in artifacts}}
    write_json(output, report)
    print("\nGLOBAL ROW-RANKING TRIAL", flush=True)
    print(json.dumps(report, indent=2), flush=True)


def handoff_self_test(root):
    """Run the real generated trainer and original inference on a tiny fixture."""
    import torch
    from train_final_models import PreferenceModel
    from build_submission import TARGET_FIELDS, FIT_FIELDS, neural_pipeline
    assert torch.cuda.is_available(), "Activate techarena with working CUDA."
    source = root.parent / "techarena-seed43/src"
    scripts = prepare_sources((source / "train_preferences.py").read_text(encoding="utf-8"),
                              (source / "train_events.py").read_text(encoding="utf-8"), 42, "jan24")
    scripts["row_auc_objective.py"] = (root / "src/row_auc_objective.py").read_text(encoding="utf-8")
    rng = np.random.default_rng(2031)
    cutoff = CUTOFFS["jan24"]
    with tempfile.TemporaryDirectory(prefix="row_ranking_selftest_", dir=root) as temporary:
        folder = Path(temporary)
        for name in ["src", "work", "outputs", "data/cache"]:
            (folder / name).mkdir(parents=True)
        for name, text in scripts.items():
            (folder / "src" / name).write_text(text, encoding="utf-8")
        torch.manual_seed(42)
        pref = PreferenceModel(8, 30, 8, torch.tensor([True, True, True, False, False, False, False, False]))
        torch.save({"model_state": pref.state_dict(), "n_users": 8, "n_items": 30, "dimension": 8, "epoch": 260},
                   folder / "work/preference_best.pt")
        positive = folder / "data/cache/preference_training.npz"
        np.savez(positive, users=[0, 0, 1, 1, 2, 2], items=[1, 2, 3, 4, 5, 6],
                 timestamps=np.full(6, cutoff - 7200), n_users=8, n_items=30, cutoff=cutoff)
        y = np.zeros((8, 10), dtype=np.int8)
        y[[0, 2, 4, 6], [1, 2, 3, 4]] = 1
        raw = {"u": np.array([0, 1, 2, 0, 1, 2, 3, 4]), "i": rng.integers(0, 30, (8, 10)),
               "t": np.r_[cutoff - 3600 + np.arange(6) * 60, [cutoff, cutoff + 60]],
               "tb": np.zeros(8, dtype=np.int8), "dt": np.zeros(8, dtype=np.int8), "y": y,
               "h": np.full((8, 20), -1, dtype=np.int32), "ht": np.zeros((8, 20), dtype=np.int64)}
        raw["h"][6, :2] = [1, 2]
        raw["ht"][6, :2] = [cutoff - 7200, cutoff - 3600]
        np.savez(folder / "data/cache/train_events.npz", **raw)
        np.savez(folder / "data/cache/split_jan24.npz", train_idx=np.arange(6), valid_idx=[6, 7])
        subprocess.run([sys.executable, str(folder / "src/train_events.py")], cwd=folder, check=True)
        saved = torch.load(folder / "work/event_best.pt", map_location="cpu", weights_only=True)
        assert saved["epoch"] == 8 and saved["event_objective"] == CONFIG["objective"]
        X, names, raw_p, corrected, *_ = neural_pipeline(folder / "work/event_best.pt",
            {k: raw[k][:6] for k in FIT_FIELDS}, {k: raw[k][6:] for k in TARGET_FIELDS}, positive, torch.device("cuda"))
        assert X.shape == (20, 37) and len(names) == 37 and np.isfinite(X).all()
        assert np.isfinite(raw_p).all() and np.isfinite(corrected).all()
    print("Generated trainer handoff passed: eight epochs, compatible checkpoint, cold/empty history and original 37-feature pipeline.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["self-test", "setup", "train", "evaluate"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.action == "self-test":
        from row_auc_objective import self_test
        self_test()
        handoff_self_test(root)
        return
    trial = root.parent / "techarena-row-auc-trial"
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
