"""Rebuild the frozen V5 recipe for a five-day historical baseline.

Run in techarena-final: self-test, setup, train, evaluate.
Meta-models: neural fit before Jan21, combiner labels Jan21.
Stress predictions: neural fit before Jan22, first event/user Jan22-26.
All artifacts live in techarena-v5-long-window. No submission is generated.
"""

import argparse
import ast
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DAY = 86400
JAN20 = 1737331200
CUTOFFS = {"jan21": JAN20 + DAY, "jan22": JAN20 + 2 * DAY}
END = JAN20 + 7 * DAY
MEMBERS = [("dim8_seed42", 8, 42), ("dim8_seed43", 8, 43), ("dim8_seed44", 8, 44),
           ("dim16_seed42", 16, 42), ("dim16_seed43", 16, 43)]
WEIGHTS = [2 / 9, 2 / 9, 2 / 9, 1 / 6, 1 / 6]


def first_per_user(raw, begin, end):
    idx = np.flatnonzero((raw["t"] >= begin) & (raw["t"] < end))
    idx = idx[np.argsort(raw["t"][idx], kind="stable")]
    _, positions = np.unique(raw["u"][idx], return_index=True)
    return idx[np.sort(positions)]


def sorted_examples(users, items, timestamps):
    assert users.shape == items.shape == timestamps.shape
    order = np.lexsort((timestamps, items, users))
    return np.column_stack([users[order], items[order], timestamps[order]])


def prefix_examples(raw, source, cutoff):
    """Keep original ordering, but independently prove the fitting examples.

    Filtering timestamps alone would leak histories belonging to future-only
    users. Require that the user has an observed fitting event as well.
    """
    n_users, n_items = int(source["n_users"]), int(source["n_items"])
    fitting = np.flatnonzero(raw["t"] < cutoff)
    assert len(fitting) and cutoff < int(source["cutoff"])
    assert raw["u"].max() < n_users and raw["i"].max() < n_items
    present_users = np.bincount(raw["u"][fitting], minlength=n_users) > 0
    keep = present_users[source["users"]] & (source["timestamps"] < cutoff)
    result = {k: source[k][keep] for k in ["users", "items", "timestamps"]}
    first = first_per_user(raw, JAN20, cutoff)
    h, ht = raw["h"][first], raw["ht"][first]
    present = h >= 0
    assert (ht[present] < np.broadcast_to(raw["t"][first, None], ht.shape)[present]).all()
    warm_users = np.broadcast_to(raw["u"][first, None], h.shape)[present]
    clicked_event, clicked_column = np.nonzero(raw["y"][fitting])
    click_rows = fitting[clicked_event]
    expected_users = np.r_[warm_users, raw["u"][click_rows]]
    expected_items = np.r_[h[present], raw["i"][click_rows, clicked_column]]
    expected_times = np.r_[ht[present], raw["t"][click_rows]]
    # Multiset comparison preserves genuine repeated clicks, including duplicates.
    assert np.array_equal(sorted_examples(result["users"], result["items"], result["timestamps"]),
                          sorted_examples(expected_users, expected_items, expected_times)), "Prefix examples differ from raw fitting histories/clicks"
    assert (result["timestamps"] < cutoff).all()
    assert present_users[result["users"]].all()
    result.update(n_users=n_users, n_items=n_items, cutoff=cutoff)
    report = {"training_events": len(fitting), "training_users": int(present_users.sum()),
              "initial_history_clicks": int(present.sum()), "observed_training_clicks": len(click_rows),
              "total_click_examples": len(result["users"]), "examples_match_raw_multiset": True,
              "future_only_users_excluded": True}
    return result, report


def annotate(text, seed, fold, stage):
    from prepare_preference_finetune_trial import replace_once
    old = '            "dimension": dimension,\n            "epoch": epoch,\n            "validation_auc": auc,\n'
    return replace_once(text, old, old + f'            "seed": {seed},\n            "cutoff": {CUTOFFS[fold]},\n            "stage": "{stage}",\n')


def scripts(preference, events, seed, dimension, fold):
    from prepare_preference_finetune_trial import adapt_preference, adapt_events, replace_once, classes
    assert "if epoch != max_epochs:" in preference and "if epoch != 8:" in events
    initial = replace_once(preference, "torch.manual_seed(43)", f"torch.manual_seed({seed})")
    initial = replace_once(initial, "split_jan26.npz", f"split_{fold}.npz")
    initial = replace_once(initial, "dimension = 8", f"dimension = {dimension}")
    assert initial.count("max_epochs = 196") == 1
    progress = '''        if epoch % 40 == 0 and epoch != max_epochs:
            print(json.dumps({"stage": "initial_preference", "epoch": epoch,
                              "training_loss": float(loss_total.item() / len(users))}), flush=True)

'''
    initial = replace_once(initial, "        if epoch != max_epochs:\n            continue",
                           progress + "        if epoch != max_epochs:\n            continue")
    initial = annotate(initial, seed, fold, "preference")
    result = {}
    if dimension == 8:
        for old, new in [("work/preference_best.pt", "work/preference_initial.pt"),
                         ("work/preference_last.pt", "work/preference_initial_last.pt"),
                         ("work/preference_validation.npz", "work/preference_initial_validation.npz"),
                         ("outputs/preference_metrics.json", "outputs/preference_initial_metrics.json")]:
            initial = replace_once(initial, old, new)
        result["train_initial_preferences.py"] = initial
        result["train_preferences.py"] = annotate(adapt_preference(preference, seed, fold), seed, fold, "preference")
    else:
        assert dimension == 16
        result["train_preferences.py"] = initial
    result["train_events.py"] = annotate(adapt_events(events, seed, fold), seed, fold, "event")
    for name, text in result.items():
        ast.parse(text)
        assert classes(text) == classes(events if name == "train_events.py" else preference)
    return result


def load_raw(root):
    with np.load(root / "data/cache/train_events.npz") as saved:
        raw = {name: saved[name] for name in ["u", "i", "t", "tb", "dt", "h", "ht", "y"]}
    assert raw["i"].shape == raw["y"].shape == (len(raw["t"]), 10)
    assert np.isin(raw["y"], [0, 1]).all() and (raw["y"].sum(1) <= 1).all()
    assert raw["t"].min() >= JAN20 and raw["t"].max() < END
    return raw


def setup(root, trial):
    from submission_export import sha256
    if trial.exists():
        raise FileExistsError(f"Already prepared: {trial}; continue with train or evaluate.")
    source = root.parent / "techarena-seed43/src"
    preference = (source / "train_preferences.py").read_text(encoding="utf-8")
    events = (source / "train_events.py").read_text(encoding="utf-8")
    template_path = root.parent / "techarena-seed44-trial/jan24/data/cache/preference_training.npz"
    with np.load(template_path) as saved:
        template = {k: saved[k] for k in ["users", "items", "timestamps", "n_users", "n_items", "cutoff"]}
    assert int(template["cutoff"]) == JAN20 + 4 * DAY
    checks = root / "outputs/five_model_submission_checks.json"
    checked = json.loads(checks.read_text(encoding="utf-8"))
    assert checked["version"] == "v5" and checked["submission_ready"] is True
    assert checked["weights"] == WEIGHTS
    helper16 = root.parent / "techarena-dim16-trial/src/dim16_pipeline.py"
    input_paths = [Path(__file__), root / "data/cache/train_events.npz", template_path,
                   source / "train_preferences.py", source / "train_events.py", checks, helper16]
    input_paths += [root / "src" / name for name in ["prepare_preference_finetune_trial.py", "build_submission.py",
                    "train_final_models.py", "submission_features.py", "submission_export.py", "audit_v5.py", "evaluate_dim16_trial.py"]]
    hashes = {str(p.resolve(strict=True)): sha256(p) for p in input_paths}
    raw = load_raw(root)
    prepared = {}
    for fold, cutoff in CUTOFFS.items():
        examples, report = prefix_examples(raw, template, cutoff)
        fitting = np.flatnonzero(raw["t"] < cutoff)
        valid = first_per_user(raw, cutoff, cutoff + DAY)
        assert len(valid) and np.intersect1d(fitting, valid).size == 0
        prepared[fold] = (examples, report, fitting, valid)
    target = first_per_user(raw, CUTOFFS["jan22"], END)
    assert len(target) == len(np.unique(raw["u"][target]))
    assert np.array_equal(target[raw["t"][target] < CUTOFFS["jan22"] + DAY], prepared["jan22"][3])
    assert np.intersect1d(target, prepared["jan21"][3]).size == 0
    generated = {(key, fold): scripts(preference, events, seed, d, fold)
                 for key, d, seed in MEMBERS for fold in CUTOFFS}
    protocol = {
        "experiment": "V5 recipe five-day historical baseline; no new candidate",
        "members": [k for k, _, _ in MEMBERS], "weights": WEIGHTS,
        "dim8_schedule": {"initial_preference_epochs": 196, "fine_tuning_epochs": 64,
                          "initial_learning_rate": 0.012, "fine_tuning_learning_rate": 0.003,
                          "adam_moments_reused": True, "fine_tuning_sampling_seeds": [1042, 1043, 1044]},
        "dim16_schedule": {"preference_epochs": 196, "learning_rate": 0.012},
        "event_epochs": 8, "event_learning_rate": 0.0015,
        "combiner": {"trees": 545, "depth": 6, "learning_rate": 0.05,
                     "l2_leaf_reg": 20, "seed": 42, "features": 37, "early_stopping": False},
        "meta_neural_cutoff": "2025-01-21T00:00:00+00:00", "combiner_fitting_day": "2025-01-21",
        "target_neural_cutoff": "2025-01-22T00:00:00+00:00",
        "target_window": "2025-01-22T00:00:00+00:00 through 2025-01-27T00:00:00+00:00 exclusive",
        "target_selection": "first event per user across the entire five-day window; stable original order for equal times",
        "target_history_policy": "use each target's supplied prior history; never update fitted weights/statistics with window events",
        "early_stopping_or_weight_search": False, "bootstrap_resamples": 300, "bootstrap_seed": 2026,
        "bootstrap_interpretation": "sampling uncertainty conditional on these fitted models; excludes training variability",
        "status": "retrospective diagnostic; reused dates, smaller fitting set, not independent confirmation",
        "interpretation": "horizon groups contain different users; their AUC differences do not isolate temporal decay",
        "promotion_or_qualification_threshold": None,
        "folds": {f: {"cutoff": CUTOFFS[f], **prepared[f][1], "fixed_day_monitoring_events": len(prepared[f][3])} for f in CUTOFFS},
        "target_events": len(target), "dim16_helper": str(helper16),
        "input_sha256": hashes, "prepared_sha256": {},
    }
    (trial / "data/cache").mkdir(parents=True)
    (trial / "work").mkdir()
    (trial / "outputs").mkdir()
    for fold, (examples, _, fitting, valid) in prepared.items():
        cache = trial / "data/cache" / fold
        cache.mkdir()
        np.savez_compressed(cache / "preference_training.npz", **examples)
        np.savez_compressed(cache / f"split_{fold}.npz", train_idx=fitting, valid_idx=valid)
    np.save(trial / "data/cache/stress_indices.npy", target, allow_pickle=False)
    for (key, fold), texts in generated.items():
        folder = trial / key / fold
        for part in ["src", "work", "outputs", "data/cache"]:
            (folder / part).mkdir(parents=True)
        for filename, text in texts.items():
            (folder / "src" / filename).write_text(text, encoding="utf-8")
        for filename in ["preference_training.npz", f"split_{fold}.npz"]:
            (folder / "data/cache" / filename).symlink_to((trial / "data/cache" / fold / filename).resolve(strict=True))
        (folder / "data/cache/train_events.npz").symlink_to((root / "data/cache/train_events.npz").resolve(strict=True))
    for path in [*(trial / "data/cache").rglob("*.np*"), *trial.glob("dim*/jan*/src/*.py")]:
        protocol["prepared_sha256"][str(path.relative_to(trial))] = sha256(path)
    (trial / "outputs/frozen_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    verify(trial, protocol)
    print("\nV5 FIVE-DAY BASELINE SETUP", flush=True)
    print(json.dumps({k: v for k, v in protocol.items() if k not in ["input_sha256", "prepared_sha256", "dim16_helper"]}, indent=2), flush=True)


def verify(trial, protocol):
    from submission_export import sha256
    assert protocol["weights"] == WEIGHTS and protocol["event_epochs"] == 8
    assert protocol["combiner"]["trees"] == 545 and protocol["combiner"]["features"] == 37
    for name, expected in protocol["input_sha256"].items():
        assert sha256(Path(name)) == expected, f"Changed source input: {name}"
    for name, expected in protocol["prepared_sha256"].items():
        assert sha256(trial / name) == expected, f"Changed prepared input: {name}"


def train_all(trial, protocol):
    import torch
    from submission_export import sha256
    start = time.perf_counter()
    assert torch.cuda.is_available(), "Activate techarena with working CUDA."
    reports = {}
    for key, dimension, seed in MEMBERS:
        reports[key] = {}
        for fold, cutoff in CUTOFFS.items():
            folder = trial / key / fold
            stages = [("train_initial_preferences.py", "preference_initial.pt", 196)] if dimension == 8 else []
            stages += [("train_preferences.py", "preference_best.pt", 260 if dimension == 8 else 196),
                       ("train_events.py", "event_best.pt", 8)]
            reports[key][fold] = {}
            for script, filename, epoch in stages:
                print(f"\nTraining frozen V5 recipe: {key}, {fold}, {script}", flush=True)
                path = folder / "work" / filename
                if not path.exists():
                    subprocess.run([sys.executable, str(folder / "src" / script)], cwd=folder, check=True)
                saved = torch.load(path, map_location="cpu", weights_only=True)
                assert saved["dimension"] == dimension and saved["seed"] == seed
                assert saved["epoch"] == epoch and saved["cutoff"] == cutoff
                if dimension == 8 and epoch == 260:
                    assert saved["fine_tuning_sampling_seed"] == seed + 1000
                    assert saved["fine_tuning_learning_rate"] == 0.003
                reports[key][fold][filename] = {"epoch": epoch, "sha256": sha256(path)}
                del saved
    verify(trial, protocol)
    report = {"members": reports, "runtime_seconds": round(time.perf_counter() - start, 1)}
    (trial / "outputs/training_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nV5 FIVE-DAY BASELINE TRAINING", flush=True)
    print(json.dumps(report, indent=2), flush=True)


def metric(labels, predictions, mask=None):
    from audit_v5 import auc, choice_metrics
    if mask is not None:
        labels, predictions = labels[mask], predictions[mask]
    if labels.size == 0:
        return {"events": 0, "rows": 0, "auc": None}
    p = np.clip(predictions, 1e-12, 1 - 1e-12)
    return {"events": len(labels), "rows": int(labels.size), "clicks": int(labels.sum()),
            "auc": auc(labels, predictions), "actual_click_rate": float(labels.mean()),
            "mean_prediction": float(predictions.mean()),
            "log_loss": float(-(labels * np.log(p) + (1 - labels) * np.log1p(-p)).mean()),
            "choice_diagnostics": choice_metrics(labels, predictions)}


def evaluate(root, trial, protocol):
    import torch
    from audit_v5 import AucOrder, auc
    from build_submission import TARGET_FIELDS, FIT_FIELDS, neural_pipeline, predict_combiner, load_combiner, check_close
    from evaluate_dim16_trial import fit_combiner
    from submission_export import sha256
    start = time.perf_counter()
    output = trial / "outputs/baseline_results.json"
    if output.exists():
        raise FileExistsError(f"Baseline already reported: {output}")
    training_path = trial / "outputs/training_results.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    for key, dimension, seed in MEMBERS:
        for fold in CUTOFFS:
            for name, info in training["members"][key][fold].items():
                assert sha256(trial / key / fold / "work" / name) == info["sha256"]
    training_hash = sha256(training_path)
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    raw = load_raw(root)
    target_idx = np.load(trial / "data/cache/stress_indices.npy", allow_pickle=False)
    assert np.array_equal(target_idx, first_per_user(raw, CUTOFFS["jan22"], END))
    with np.load(trial / "data/cache/jan21/split_jan21.npz") as saved:
        meta_idx = saved["valid_idx"]
    assert np.array_equal(meta_idx, first_per_user(raw, CUTOFFS["jan21"], CUTOFFS["jan22"]))
    assert np.intersect1d(meta_idx, target_idx).size == 0
    spec = importlib.util.spec_from_file_location("long_window_dim16", protocol["dim16_helper"])
    helper16 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper16)
    members, member_metrics = [], {}
    for key, dimension, seed in MEMBERS:
        path = trial / "work" / f"{key}_predictions.npz"
        combiner_path = trial / "work" / f"{key}_combiner_jan21.cbm"
        if path.exists():
            with np.load(path) as saved:
                assert np.array_equal(saved["valid_idx"], target_idx)
                assert np.array_equal(saved["labels"], raw["y"][target_idx])
                assert str(saved["training_report_sha256"].item()) == training_hash
                assert str(saved["combiner_sha256"].item()) == sha256(combiner_path)
                p = saved["predictions"]
            print(f"Using completed predictions: {key}", flush=True)
        else:
            print(f"\nBuilding frozen 37-feature baseline: {key}", flush=True)
            pipeline = neural_pipeline if dimension == 8 else helper16.neural_pipeline

            def features(fold, indices):
                with np.load(trial / "data/cache" / fold / f"split_{fold}.npz") as saved:
                    fit_idx = saved["train_idx"]
                assert np.array_equal(fit_idx, np.flatnonzero(raw["t"] < CUTOFFS[fold]))
                assert (raw["t"][indices] >= CUTOFFS[fold]).all()
                return pipeline(trial / key / fold / "work/event_best.pt",
                                {name: raw[name][fit_idx] for name in FIT_FIELDS},
                                {name: raw[name][indices] for name in TARGET_FIELDS},
                                trial / "data/cache" / fold / "preference_training.npz", device)

            X21, names, *_ = features("jan21", meta_idx)
            assert len(names) == 37
            model = fit_combiner(X21, raw["y"][meta_idx], names)
            model.save_model(str(combiner_path))
            del X21
            X22, names22, *_ = features("jan22", target_idx)
            assert names22 == names
            p = predict_combiner(model, X22, names).reshape(-1, 10)
            # Verify serialized calibration gives identical target predictions.
            reloaded = predict_combiner(load_combiner(combiner_path), X22, names).reshape(-1, 10)
            check_close(p, reloaded, f"Saved combiner reproduction: {key}", rtol=0, atol=1e-15)
            np.savez_compressed(path, valid_idx=target_idx, labels=raw["y"][target_idx], predictions=p,
                                training_report_sha256=training_hash, combiner_sha256=sha256(combiner_path))
            del X22, model, reloaded
            torch.cuda.empty_cache()
        assert p.shape == raw["y"][target_idx].shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        members.append(p)
        member_metrics[key] = metric(raw["y"][target_idx], p)
    prediction = (2 / 3) * ((members[0] + members[1] + members[2]) / 3) + (1 / 3) * (0.5 * members[3] + 0.5 * members[4])
    labels = raw["y"][target_idx]
    np.savez_compressed(trial / "work/v5_five_day_predictions.npz", valid_idx=target_idx, labels=labels,
                        predictions=prediction, members=np.stack(members), weights=np.asarray(WEIGHTS))
    elapsed = (raw["t"][target_idx] - CUTOFFS["jan22"]) / 3600
    fitting = np.flatnonzero(raw["t"] < CUTOFFS["jan22"])
    activity = np.bincount(raw["u"][fitting], minlength=int(raw["u"].max()) + 1)[raw["u"][target_idx]]
    length = (raw["h"][target_idx] >= 0).sum(1)
    groups = {"seen_users": activity > 0, "unseen_with_history": (activity == 0) & (length > 0),
              "unseen_empty_history": (activity == 0) & (length == 0), "history_empty": length == 0,
              "history_1_5": (length >= 1) & (length <= 5), "history_6_19": (length >= 6) & (length <= 19),
              "history_20": length == 20, "activity_1_5": (activity >= 1) & (activity <= 5),
              "activity_6_15": (activity >= 6) & (activity <= 15), "activity_16_plus": activity >= 16}
    per_day = {}
    for offset in range(5):
        mask = (elapsed >= 24 * offset) & (elapsed < 24 * (offset + 1))
        date = str(np.datetime64("2025-01-22") + np.timedelta64(offset, "D"))
        entry = metric(labels, prediction, mask)
        entry.update(unseen_user_fraction=float((activity[mask] == 0).mean()) if mask.any() else None,
                     history_1_5_fraction=float(((length[mask] >= 1) & (length[mask] <= 5)).mean()) if mask.any() else None)
        per_day[date] = entry
    for begin, end in [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120)]:
        groups[f"horizon_{begin}_{end}h"] = (elapsed >= begin) & (elapsed < end)
    group_metrics = {name: metric(labels, prediction, mask) for name, mask in groups.items()}
    print(json.dumps({"overall": metric(labels, prediction), "per_day": per_day}, indent=2), flush=True)
    order = AucOrder(labels.ravel(), prediction.ravel(), np.repeat(np.arange(len(labels)), 10))
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    scores = []
    for iteration in range(protocol["bootstrap_resamples"]):
        counts = np.bincount(rng.integers(0, len(labels), len(labels)), minlength=len(labels))
        value = order.value(counts)
        if value is not None:
            scores.append(value)
        if (iteration + 1) % 50 == 0:
            print(f"Whole-user AUC bootstrap: {iteration + 1}/300", flush=True)
    assert len(scores) >= 290
    verify(trial, protocol)
    report = {
        "protocol": {k: v for k, v in protocol.items() if k not in ["input_sha256", "prepared_sha256", "dim16_helper"]},
        "overall": metric(labels, prediction), "auc_95_percentile_interval": np.quantile(scores, [0.025, 0.975]).tolist(),
        "per_day": per_day, "groups": group_metrics, "member_metrics": member_metrics,
        "population": {"users": len(labels), "first_event_per_user_verified": True,
                       "unseen_user_fraction": float((activity == 0).mean()), "empty_history_fraction": float((length == 0).mean()),
                       "horizon_hours_p50_p90_max": np.quantile(elapsed, [0.5, 0.9, 1]).tolist()},
        "sampling_unit": "whole first event per user, preserving all 10 candidates",
        "data_and_saved_combiner_checks_passed": True, "candidate_comparison_performed": False,
        "baseline_established": True, "submission_created": False,
        "runtime_seconds": round(time.perf_counter() - start, 2),
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    short_metric = lambda entry: {k: entry[k] for k in ["events", "auc", "actual_click_rate", "mean_prediction"] if k in entry}
    summary = {
        "diagnostic": "frozen V5 recipe, earlier fitting cutoffs, first event per user across Jan22-26",
        "meta_neural_cutoff": protocol["meta_neural_cutoff"], "combiner_fitting_day": protocol["combiner_fitting_day"],
        "target_neural_cutoff": protocol["target_neural_cutoff"], "weights": WEIGHTS,
        "fitting_data": protocol["folds"], "overall": report["overall"],
        "auc_95_percentile_interval": report["auc_95_percentile_interval"],
        "per_day": {date: {**short_metric(entry), "unseen_user_fraction": entry["unseen_user_fraction"],
                            "history_1_5_fraction": entry["history_1_5_fraction"]} for date, entry in per_day.items()},
        "groups": {name: short_metric(entry) for name, entry in group_metrics.items() if not name.startswith("horizon_")},
        "member_auc": {name: entry["auc"] for name, entry in member_metrics.items()},
        "population": report["population"], "sampling_unit": report["sampling_unit"],
        "data_and_saved_combiner_checks_passed": True, "candidate_comparison_performed": False,
        "baseline_established": True, "submission_created": False,
        "interpretation": protocol["interpretation"], "status": protocol["status"],
        "full_report": str(output), "runtime_seconds": report["runtime_seconds"],
    }
    (trial / "outputs/baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nV5 FIVE-DAY HISTORICAL BASELINE", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


def self_test():
    # An early-timestamp history for a future-only user must still be excluded.
    raw = {"u": np.array([0, 1, 0, 2, 1]),
           "t": JAN20 + np.array([10, 20, DAY + 10, 2 * DAY + 10, 3 * DAY + 10]),
           "i": np.tile(np.arange(10), (5, 1)), "y": np.zeros((5, 10), dtype=np.int8),
           "h": np.full((5, 20), -1), "ht": np.zeros((5, 20), dtype=np.int64)}
    raw["h"][0, :2], raw["ht"][0, :2] = [3, 3], [JAN20 - 20, JAN20 - 20]
    raw["h"][3, 0], raw["ht"][3, 0] = 9, JAN20 - 30
    raw["y"][0, 2] = raw["y"][2, 4] = 1
    source = {"users": np.array([0, 0, 2, 0, 0]), "items": np.array([3, 3, 9, 2, 4]),
              "timestamps": np.array([JAN20 - 20, JAN20 - 20, JAN20 - 30, JAN20 + 10, JAN20 + DAY + 10]),
              "n_users": 3, "n_items": 10, "cutoff": JAN20 + 4 * DAY}
    p1, _ = prefix_examples(raw, source, CUTOFFS["jan21"])
    p2, _ = prefix_examples(raw, source, CUTOFFS["jan22"])
    assert p1["users"].tolist() == [0, 0, 0]
    assert p1["items"].tolist() == [3, 3, 2]  # Genuine duplicates preserved.
    assert p2["items"].tolist() == [3, 3, 2, 4]
    assert first_per_user(raw, JAN20 + DAY, END).tolist() == [2, 3, 4]
    assert first_per_user(raw, JAN20 + 2 * DAY, END).tolist() == [3, 4]
    # Every generated trainer must retain the original architecture and guards.
    root = Path(__file__).resolve().parents[1]
    source_folder = root.parent / "techarena-seed43/src"
    pref = (source_folder / "train_preferences.py").read_text(encoding="utf-8")
    evt = (source_folder / "train_events.py").read_text(encoding="utf-8")
    for _, d, seed in MEMBERS:
        for fold in CUTOFFS:
            texts = scripts(pref, evt, seed, d, fold)
            assert "if epoch != 8:" in texts["train_events.py"]
            assert f'"cutoff": {CUTOFFS[fold]}' in texts["train_events.py"]
            assert "if epoch != max_epochs:" in texts["train_preferences.py"]
    print("Five-day baseline self-test passed: causal examples, future-only-user exclusion, repeated clicks, first-user selection, all trainer adaptations.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["self-test", "setup", "train", "evaluate"])
    args = parser.parse_args()
    if args.action == "self-test":
        self_test()
        return
    root = Path(__file__).resolve().parents[1]
    trial = root.parent / "techarena-v5-long-window"
    if args.action == "setup":
        setup(root, trial)
        return
    protocol = json.loads((trial / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    verify(trial, protocol)
    if args.action == "train":
        train_all(trial, protocol)
    else:
        evaluate(root, trial, protocol)


if __name__ == "__main__":
    main()
