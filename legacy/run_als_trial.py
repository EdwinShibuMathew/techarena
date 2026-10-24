"""One independent ALS-initialized event expert, blended 15% with V5.

In techarena-final: self-test, setup, train, evaluate.
All new artifacts live in techarena-als-trial. No evaluation files or submission
export are used. This is a fixed retrospective development experiment.
"""

import argparse
import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from implicit_als_model import CONFIG
from audit_v5 import AucOrder, auc
from submission_export import sha256

CUTOFFS = {"jan24": 1737676800, "jan25": 1737763200, "jan26": 1737849600}
WEIGHTS = [2 / 9, 2 / 9, 2 / 9, 1 / 6, 1 / 6]
EXPERT_WEIGHT = .15


def blend(baseline, expert):
    assert baseline.shape == expert.shape and baseline.ndim == 2 and baseline.shape[1] == 10
    assert np.isfinite(baseline).all() and np.isfinite(expert).all()
    assert ((baseline >= 0) & (baseline <= 1)).all() and ((expert >= 0) & (expert <= 1)).all()
    return (1 - EXPERT_WEIGHT) * baseline + EXPERT_WEIGHT * expert


def reference_paths(projects, case):
    return [projects / "techarena-preference-finetune-trial/work" / f"{case}_predictions.npz",
            projects / "techarena-seed44-trial/work" / f"{case}_ensemble_predictions.npz",
            projects / "techarena-dim16-trial/work" / f"{case}_ensemble_predictions.npz",
            projects / "techarena-dim16-seed43-trial/work" / f"{case}_ensemble_predictions.npz"]


def references(root, raw):
    result = {}
    for case, expected in [("two_day", 0.8183722315263706), ("jan26", 0.8269940935263441)]:
        paths = reference_paths(root.parent, case)
        with np.load(paths[1]) as saved:
            idx, y, p, third8 = [saved[k] for k in ["valid_idx", "labels", "candidate", "new_dim8_seed44"]]
        members = []
        for path, keys in [(paths[0], ["new_dim8_seed42", "new_dim8_seed43"]),
                           (paths[2], ["dim16"]), (paths[3], ["new_dim16"])]:
            with np.load(path) as saved:
                assert np.array_equal(saved["valid_idx"], idx) and np.array_equal(saved["labels"], y)
                members.extend([saved[k] for k in keys])
        reconstructed = (2 / 3) * ((members[0] + members[1] + third8) / 3) + (1 / 3) * (.5 * members[2] + .5 * members[3])
        np.testing.assert_allclose(reconstructed, p, rtol=0, atol=1e-15)
        assert abs(auc(y, p) - expected) < 1e-12
        assert np.array_equal(y, raw["y"][idx]) and len(np.unique(raw["u"][idx])) == len(idx)
        assert (raw["t"][idx] >= CUTOFFS["jan25" if case == "two_day" else "jan26"]).all()
        assert (raw["t"][idx] < 1737936000).all()
        result[case] = {"indices": idx, "labels": y, "baseline": p}
    return result


def raw_data(root):
    with np.load(root / "data/cache/train_events.npz") as saved:
        return {k: saved[k] for k in ["u", "i", "t", "tb", "dt", "h", "ht", "y"]}


def prepare_sources(preference_source, event_source, fold):
    from prepare_preference_finetune_trial import adapt_events, replace_once, classes
    pref_class = next(n for n in ast.parse(preference_source).body if isinstance(n, ast.ClassDef) and n.name == "PreferenceModel")
    shim = '"""Original PreferenceModel architecture; ALS training is orchestrated separately."""\nimport torch\nfrom torch import nn\n\n' + ast.get_source_segment(preference_source, pref_class) + '\n'
    event = adapt_events(event_source, CONFIG["seed"], fold)
    marker = '            "dimension": dimension,\n            "epoch": epoch,\n            "validation_auc": auc,\n'
    event = replace_once(event, marker, marker +
                         f'            "seed": {CONFIG["seed"]},\n            "cutoff": {CUTOFFS[fold]},\n'
                         '            "stage": "event",\n            "preference_algorithm": "implicit_als_v1",\n')
    assert classes(event) == classes(event_source)
    assert classes(shim) == classes(preference_source)
    assert "if epoch != 8:" in event
    ast.parse(shim)
    ast.parse(event)
    return {"train_preferences.py": shim, "train_events.py": event}


def verify(trial, protocol):
    assert protocol["als_model"] == CONFIG and protocol["expert_weight"] == EXPERT_WEIGHT
    for name, expected in protocol["input_sha256"].items():
        assert sha256(Path(name)) == expected, f"Changed frozen input: {name}"
    for name, expected in protocol["prepared_sha256"].items():
        assert sha256(trial / name) == expected, f"Changed generated trainer: {name}"


def setup(root, trial):
    if trial.exists():
        frozen = json.loads((trial / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
        verify(trial, frozen)
        print("ALS trial already prepared; continue with train or evaluate.", flush=True)
        return
    source = root.parent / "techarena-seed43/src"
    pref_text = (source / "train_preferences.py").read_text(encoding="utf-8")
    event_text = (source / "train_events.py").read_text(encoding="utf-8")
    prepared = {fold: prepare_sources(pref_text, event_text, fold) for fold in CUTOFFS}
    helper = root.parent / "techarena-dim16-trial/src/dim16_pipeline.py"
    inputs = [Path(__file__), source / "train_preferences.py", source / "train_events.py", helper,
              root / "data/cache/train_events.npz"]
    inputs += [root / "src" / name for name in ["implicit_als_model.py", "prepare_preference_finetune_trial.py",
               "train_final_models.py", "build_submission.py", "submission_features.py", "submission_export.py",
               "audit_v5.py", "evaluate_dim16_trial.py"]]
    checks_path = root / "outputs/five_model_submission_checks.json"
    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    assert checks["submission_ready"] is True and checks["version"] == "v5" and checks["weights"] == WEIGHTS
    inputs.append(checks_path)
    raw = raw_data(root)
    references(root, raw)
    for case in ["two_day", "jan26"]:
        inputs += reference_paths(root.parent, case)
    sources, sizes = {}, {}
    for fold, cutoff in CUTOFFS.items():
        cache = root.parent / "techarena-seed44-trial" / fold / "data/cache"
        positive_path, split_path = cache / "preference_training.npz", cache / f"split_{fold}.npz"
        with np.load(positive_path) as saved:
            assert int(saved["cutoff"]) == cutoff and (saved["timestamps"] < cutoff).all()
            assert saved["users"].shape == saved["items"].shape == saved["timestamps"].shape
            n_users, n_items = int(saved["n_users"]), int(saved["n_items"])
            sizes[fold] = {"click_examples": len(saved["users"]), "allocated_users": n_users, "allocated_items": n_items}
        with np.load(split_path) as saved:
            idx, valid = saved["train_idx"], saved["valid_idx"]
        assert np.array_equal(idx, np.flatnonzero(raw["t"] < cutoff))
        assert (raw["t"][valid] >= cutoff).all() and (raw["t"][valid] < cutoff + 86400).all()
        assert len(np.unique(raw["u"][valid])) == len(valid)
        assert raw["u"].max() < n_users and raw["i"].max() < n_items
        sizes[fold]["fitting_events"] = len(idx)
        sources[fold] = {"preference_training.npz": positive_path.resolve(strict=True),
                         f"split_{fold}.npz": split_path.resolve(strict=True),
                         "train_events.npz": (root / "data/cache/train_events.npz").resolve(strict=True)}
        inputs += [positive_path, split_path]
    protocol = {
        "experiment": "independent implicit-ALS initialized event expert added to unchanged V5",
        "als_model": CONFIG.copy(), "v5_weight": 1 - EXPERT_WEIGHT, "expert_weight": EXPERT_WEIGHT,
        "blend_rule": "85% V5 plus 15% ALS expert for every candidate row, including empty histories",
        "v5_member_weights": WEIGHTS, "original_v5_members_unchanged": True,
        "combiner": {"trees": 545, "depth": 6, "learning_rate": .05, "l2_leaf_reg": 20,
                     "seed": 42, "features": 37, "early_stopping": False,
                     "fitting_selection": "first event per user on the meta-training day"},
        "primary_check": "ALS prefix Jan25, combiner trained on Jan24, first-user Jan25-26 evaluation",
        "secondary_check": "ALS prefix Jan26, combiner trained on Jan25, first-user Jan26 evaluation",
        "feature_fitting_rule": "each prefix fits ALS, event parameters, ridge encoder and aggregate statistics strictly before that prefix cutoff; Jan24 meta features use the Jan24 prefix",
        "bootstrap_resamples": 300, "bootstrap_seed": 2026,
        "support_rule": "primary paired gain interval entirely positive and secondary AUC change nonnegative",
        "next_gate": "five-day paired check before full-data training, promotion or submission",
        "status": "retrospective development experiment; reused validation dates; interval not selection-adjusted",
        "source_sizes": sizes, "dim16_helper": str(helper),
        "input_sha256": {str(path.resolve(strict=True)): sha256(path) for path in inputs},
        "prepared_sha256": {},
    }
    (trial / "outputs").mkdir(parents=True)
    (trial / "work").mkdir()
    for fold, texts in prepared.items():
        folder = trial / fold
        for name in ["src", "work", "outputs", "data/cache"]:
            (folder / name).mkdir(parents=True)
        for name, text in texts.items():
            path = folder / "src" / name
            path.write_text(text, encoding="utf-8")
            protocol["prepared_sha256"][str(path.relative_to(trial))] = sha256(path)
        for name, path in sources[fold].items():
            (folder / "data/cache" / name).symlink_to(path)
    (trial / "outputs/frozen_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    verify(trial, protocol)
    print("ALS TRIAL SETUP", flush=True)
    print(json.dumps({k: v for k, v in protocol.items() if k not in ["input_sha256", "prepared_sha256", "dim16_helper"]}, indent=2), flush=True)


def verify_training(trial, report):
    for fold, stages in report["folds"].items():
        for filename, info in stages.items():
            assert sha256(trial / fold / "work" / filename) == info["sha256"], "Trained checkpoint changed"


def train_all(trial, protocol, device):
    import torch
    from implicit_als_model import train_preference
    output = trial / "outputs/training_results.json"
    if output.exists():
        report = json.loads(output.read_text(encoding="utf-8"))
        verify_training(trial, report)
        print("ALS training already completed; continue with evaluate.", flush=True)
        return
    start = time.perf_counter()
    reports = {}
    for fold, cutoff in CUTOFFS.items():
        folder = trial / fold
        positive = folder / "data/cache/preference_training.npz"
        pref = folder / "work/preference_best.pt"
        if not pref.exists():
            print(f"\nFitting ALS preference factors: {fold}", flush=True)
            train_preference(positive, cutoff, pref, device)
        saved = torch.load(pref, map_location="cpu", weights_only=True)
        assert saved["config"] == CONFIG and saved["cutoff"] == cutoff and saved["epoch"] == CONFIG["iterations"]
        assert saved["positive_cache_sha256"] == sha256(positive)
        reports[fold] = {"preference_best.pt": {"sha256": sha256(pref), "last_iteration": saved["records"][-1],
                                               "unique_user_item_pairs": saved["unique_user_item_pairs"]}}
        del saved
        event = folder / "work/event_best.pt"
        if not event.exists():
            print(f"Fitting eight event epochs: {fold}", flush=True)
            subprocess.run([sys.executable, str(folder / "src/train_events.py")], cwd=folder, check=True)
        saved = torch.load(event, map_location="cpu", weights_only=True)
        assert saved["dimension"] == 16 and saved["epoch"] == 8 and saved["seed"] == CONFIG["seed"]
        assert saved["cutoff"] == cutoff and saved["preference_algorithm"] == CONFIG["algorithm"]
        reports[fold]["event_best.pt"] = {"sha256": sha256(event), "epoch": 8}
        del saved
        torch.cuda.empty_cache()
    verify(trial, protocol)
    report = {"folds": reports, "runtime_seconds": round(time.perf_counter() - start, 2)}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nALS EVENT EXPERT TRAINING RESULTS", flush=True)
    print(json.dumps(report, indent=2), flush=True)


def evaluate(root, trial, protocol, device):
    from build_submission import TARGET_FIELDS, FIT_FIELDS, predict_combiner, load_combiner, check_close
    from evaluate_dim16_trial import fit_combiner
    output = trial / "outputs/ensemble_trial_results.json"
    if output.exists():
        print("Saved ALS trial already complete; displaying it.", flush=True)
        print("\nIMPLICIT ALS ENSEMBLE TRIAL", flush=True)
        print(output.read_text(encoding="utf-8"), flush=True)
        return
    training_path = trial / "outputs/training_results.json"
    training_hash = sha256(training_path)
    verify_training(trial, json.loads(training_path.read_text(encoding="utf-8")))
    start = time.perf_counter()
    raw = raw_data(root)
    refs = references(root, raw)
    primary, secondary = refs["two_day"], refs["jan26"]
    spec = importlib.util.spec_from_file_location("als_dim16_pipeline", protocol["dim16_helper"])
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    with np.load(trial / "jan24/data/cache/split_jan24.npz") as saved:
        meta_idx = saved["valid_idx"]
    assert (raw["t"][meta_idx] >= CUTOFFS["jan24"]).all() and (raw["t"][meta_idx] < CUTOFFS["jan25"]).all()
    assert np.intersect1d(meta_idx, primary["indices"]).size == 0

    def features(fold, indices):
        with np.load(trial / fold / "data/cache" / f"split_{fold}.npz") as saved:
            fitting = saved["train_idx"]
        assert np.array_equal(fitting, np.flatnonzero(raw["t"] < CUTOFFS[fold]))
        assert (raw["t"][indices] >= CUTOFFS[fold]).all() and (raw["t"][indices] < 1737936000).all()
        X, names, *_ = helper.neural_pipeline(trial / fold / "work/event_best.pt",
                                             {k: raw[k][fitting] for k in FIT_FIELDS},
                                             {k: raw[k][indices] for k in TARGET_FIELDS},
                                             trial / fold / "data/cache/preference_training.npz", device)
        assert len(names) == 37
        return X, names

    print("Building ALS expert meta-training features...", flush=True)
    X24, names = features("jan24", meta_idx)
    X25, names25 = features("jan25", primary["indices"])
    assert names == names25
    model = fit_combiner(X24, raw["y"][meta_idx], names)
    model.save_model(str(trial / "work/combiner_jan24.cbm"))
    p_primary = predict_combiner(model, X25, names).reshape(-1, 10)
    check_close(p_primary, predict_combiner(load_combiner(trial / "work/combiner_jan24.cbm"), X25, names).reshape(-1, 10),
                "ALS saved primary combiner", rtol=0, atol=1e-15)
    del X24, model
    day25 = raw["t"][primary["indices"]] < CUTOFFS["jan26"]
    with np.load(trial / "jan25/data/cache/split_jan25.npz") as saved:
        assert np.array_equal(primary["indices"][day25], saved["valid_idx"])
    with np.load(trial / "jan26/data/cache/split_jan26.npz") as saved:
        assert np.array_equal(secondary["indices"], saved["valid_idx"])
    assert np.intersect1d(primary["indices"][day25], secondary["indices"]).size == 0
    model = fit_combiner(X25[np.repeat(day25, 10)], primary["labels"][day25], names)
    model.save_model(str(trial / "work/combiner_jan25.cbm"))
    del X25
    X26, names26 = features("jan26", secondary["indices"])
    assert names26 == names
    p_secondary = predict_combiner(model, X26, names).reshape(-1, 10)
    check_close(p_secondary, predict_combiner(load_combiner(trial / "work/combiner_jan25.cbm"), X26, names).reshape(-1, 10),
                "ALS saved secondary combiner", rtol=0, atol=1e-15)
    del X26, model
    candidates, metrics = {}, {}
    for case, expert in [("two_day", p_primary), ("jan26", p_secondary)]:
        ref = refs[case]
        candidate = blend(ref["baseline"], expert)
        candidates[case] = candidate
        before, after = auc(ref["labels"], ref["baseline"]), auc(ref["labels"], candidate)
        metrics[case] = {"events": len(ref["indices"]), "baseline_v5_auc": before,
                         "als_expert_auc": auc(ref["labels"], expert), "candidate_auc": after,
                         "auc_change": after - before}
        np.savez_compressed(trial / "work" / f"{case}_ensemble_predictions.npz", valid_idx=ref["indices"],
                            labels=ref["labels"], baseline=ref["baseline"], als_expert=expert, candidate=candidate)
    print(json.dumps(metrics, indent=2), flush=True)
    n = len(primary["indices"])
    ids = np.repeat(np.arange(n), 10)
    old_order = AucOrder(primary["labels"].ravel(), primary["baseline"].ravel(), ids)
    new_order = AucOrder(primary["labels"].ravel(), candidates["two_day"].ravel(), ids)
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    gains = []
    for iteration in range(protocol["bootstrap_resamples"]):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        gains.append(new_order.value(counts) - old_order.value(counts))
        if (iteration + 1) % 50 == 0:
            print(f"Paired whole-user bootstrap: {iteration + 1}/300", flush=True)
    interval = np.quantile(gains, [.025, .975]).tolist()
    idx = primary["indices"]
    activity = np.bincount(raw["u"][raw["t"] < CUTOFFS["jan25"]], minlength=int(raw["u"].max()) + 1)[raw["u"][idx]]
    length = (raw["h"][idx] >= 0).sum(1)
    groups = {}
    for name, mask in {"activity_1_5": (activity >= 1) & (activity <= 5), "activity_16_plus": activity >= 16,
                       "unseen_with_history": (activity == 0) & (length > 0), "history_empty": length == 0,
                       "history_1_5": (length >= 1) & (length <= 5), "horizon_24_48h": ~day25}.items():
        a, b = auc(primary["labels"][mask], primary["baseline"][mask]), auc(primary["labels"][mask], candidates["two_day"][mask])
        groups[name] = {"events": int(mask.sum()), "baseline_auc": a, "candidate_auc": b,
                        "auc_change": b - a if a is not None and b is not None else None}
    verify(trial, protocol)
    assert sha256(training_path) == training_hash
    verify_training(trial, json.loads(training_path.read_text(encoding="utf-8")))
    report = {
        "protocol": {k: v for k, v in protocol.items() if k not in ["input_sha256", "prepared_sha256", "dim16_helper"]},
        "two_day_primary": metrics["two_day"], "jan26_secondary": metrics["jan26"],
        "primary_paired_gain_95_percentile_interval": interval, "primary_groups": groups,
        "v5_cached_ensemble_reproduction_passed": True, "saved_expert_combiners_reproduced": True,
        "supported_for_long_window_check": bool(interval[0] > 0 and metrics["jan26"]["auc_change"] >= 0),
        "sampling_unit": "whole first event per user, preserving all 10 candidate rows",
        "submission_created": False, "runtime_seconds": round(time.perf_counter() - start, 2),
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nIMPLICIT ALS ENSEMBLE TRIAL", flush=True)
    print(json.dumps(report, indent=2), flush=True)


def handoff_self_test(root):
    import torch
    from implicit_als_model import train_preference
    source = root.parent / "techarena-seed43/src"
    texts = prepare_sources((source / "train_preferences.py").read_text(encoding="utf-8"),
                            (source / "train_events.py").read_text(encoding="utf-8"), "jan24")
    rng = np.random.default_rng(2028)
    cutoff = CUTOFFS["jan24"]
    with tempfile.TemporaryDirectory(prefix="als_handoff_selftest_", dir=root) as temporary:
        folder = Path(temporary)
        for name in ["src", "data/cache", "work", "outputs"]:
            (folder / name).mkdir(parents=True)
        for name, text in texts.items():
            (folder / "src" / name).write_text(text, encoding="utf-8")
        pu, pi = np.array([0, 0, 1, 1, 2, 2]), np.array([1, 2, 3, 4, 5, 6])
        positive = folder / "data/cache/preference_training.npz"
        np.savez(positive, users=pu, items=pi, timestamps=np.full(6, cutoff - 7200), n_users=8, n_items=30, cutoff=cutoff)
        y = np.zeros((8, 10), dtype=np.int8)
        y[[0, 2, 4, 6], [1, 2, 3, 4]] = 1
        raw = {"u": np.array([0, 1, 2, 0, 1, 2, 3, 4]), "i": rng.integers(0, 30, (8, 10)),
               "t": np.r_[np.arange(6) * 60 + cutoff - 3600, [cutoff, cutoff + 60]],
               "tb": np.zeros(8, dtype=np.int8), "dt": np.zeros(8, dtype=np.int8), "y": y,
               "h": np.full((8, 20), -1, dtype=np.int32), "ht": np.zeros((8, 20), dtype=np.int64)}
        raw["h"][6, :2], raw["ht"][6, :2] = [1, 2], [cutoff - 7200, cutoff - 3600]
        np.savez(folder / "data/cache/train_events.npz", **raw)
        np.savez(folder / "data/cache/split_jan24.npz", train_idx=np.arange(6), valid_idx=np.array([6, 7]))
        train_preference(positive, cutoff, folder / "work/preference_best.pt", torch.device("cuda"))
        subprocess.run([sys.executable, str(folder / "src/train_events.py")], cwd=folder, check=True)
        state = torch.load(folder / "work/event_best.pt", map_location="cpu", weights_only=True)
        assert state["dimension"] == 16 and state["epoch"] == 8 and state["preference_algorithm"] == CONFIG["algorithm"]
        helper_path = root.parent / "techarena-dim16-trial/src/dim16_pipeline.py"
        spec = importlib.util.spec_from_file_location("als_handoff_dim16", helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        fitting = {k: raw[k][:6] for k in helper.FIT_FIELDS}
        target = {k: raw[k][6:] for k in helper.TARGET_FIELDS}
        X, names, p, adjusted, *_ = helper.neural_pipeline(folder / "work/event_best.pt", fitting, target, positive, torch.device("cuda"))
        assert X.shape == (20, 37) and len(names) == 37 and np.isfinite(X).all()
        assert np.isfinite(p).all() and np.isfinite(adjusted).all()
    print("ALS handoff self-test passed: exported preference state, original eight event epochs, cold/empty histories and original 37-feature pipeline.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["self-test", "setup", "train", "evaluate"])
    args = parser.parse_args()
    if args.action == "self-test":
        from implicit_als_model import self_test
        self_test()
        handoff_self_test(Path(__file__).resolve().parents[1])
        return
    root = Path(__file__).resolve().parents[1]
    trial = root.parent / "techarena-als-trial"
    if args.action == "setup":
        setup(root, trial)
        return
    protocol = json.loads((trial / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    verify(trial, protocol)
    import torch
    assert torch.cuda.is_available(), "Activate techarena with working CUDA."
    torch.set_num_threads(4)
    device = torch.device("cuda")
    if args.action == "train":
        train_all(trial, protocol, device)
    else:
        evaluate(root, trial, protocol, device)


if __name__ == "__main__":
    main()
