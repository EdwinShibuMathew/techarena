"""One fixed integration check: update dim8 combiners, retain V5 dim16.

Reuses existing trained models; fits nothing and creates no submission.
Run from techarena-final: python src/diagnose_history_ensemble.py
"""

import importlib.util
import json
import time
from pathlib import Path

import numpy as np


def selected_members(old, new):
    assert len(old) == len(new) == 5
    return list(new[:3]) + list(old[3:])


def cross_dimension_correlation(members):
    """Descriptive score similarity, not proof of a causal mechanism."""
    log_scores = np.stack([np.log(np.clip(p.ravel(), 1e-8, 1)) for p in members])
    correlation = np.corrcoef(log_scores)
    return float(correlation[:3, 3:].mean())


def main():
    import torch
    from catboost import CatBoostClassifier
    from audit_v5 import AucOrder, auc
    from build_submission import (TARGET_FIELDS, FIT_FIELDS, neural_pipeline,
                                  load_combiner, predict_combiner, check_close)
    from run_history_trial import member_specs, verify, ensemble, FEATURE, CUTOFFS, read_raw
    from submission_export import sha256

    start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    trial = root.parent / "techarena-history-trial"
    output = trial / "outputs/dim8_only_check"
    if output.exists():
        raise FileExistsError(f"Check already started: {output}; preserve its frozen protocol/results.")
    protocol_path = trial / "outputs/frozen_protocol.json"
    original = json.loads(protocol_path.read_text(encoding="utf-8"))
    verify(original)
    failed_report_path = trial / "outputs/ensemble_trial_results.json"
    failed_report = json.loads(failed_report_path.read_text(encoding="utf-8"))
    assert failed_report["v5_reproduction_passed"] is True
    assert failed_report["supported_for_long_window_check"] is False
    assert torch.cuda.is_available(), "Activate techarena with working CUDA."
    torch.set_num_threads(4)
    specs = member_specs(root.parent)
    inputs = [Path(__file__), protocol_path, failed_report_path]
    inputs += [root / "src" / name for name in
               ["build_submission.py", "submission_features.py", "audit_v5.py", "submission_export.py"]]
    for name in ["two_day", "jan26"]:
        inputs.append(trial / "work" / f"{name}_predictions.npz")
    for key, *_ in specs:
        for day in ["jan24", "jan25"]:
            inputs.append(trial / "work" / f"{key}_combiner_{day}.cbm")
    protocol = {
        "experiment": "history feature in three dim8 combiners only; original dim16 combiners retained",
        "members": [s[0] for s in specs],
        "weights": [2 / 9, 2 / 9, 2 / 9, 1 / 6, 1 / 6],
        "new_history_combiners": [s[0] for s in specs[:3]],
        "original_v5_combiners": [s[0] for s in specs[3:]],
        "training_performed": False, "weight_search_performed": False,
        "primary_check": "first event per user across Jan25-26; combiners trained on Jan24",
        "secondary_check": "first event per user on Jan26; combiners trained on Jan25",
        "bootstrap_resamples": 300, "bootstrap_seed": 2026,
        "support_rule": "primary paired gain interval entirely positive and secondary AUC change nonnegative",
        "next_gate": "longer-window paired stress check before full-data promotion or submission",
        "status": "separate retrospective follow-up chosen after failed all-five integration; reused validation dates",
        "input_sha256": {str(p.resolve(strict=True)): sha256(p) for p in inputs},
    }
    # Freeze the one comparison before computing its predictions or AUC.
    output.mkdir()
    (output / "frozen_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    raw = read_raw(root)
    device = torch.device("cuda")
    spec = importlib.util.spec_from_file_location("history_integration_dim16", original["dim16_helper"])
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    cases = {}
    for filename, fold, combiner_day, report_key in [
        ("two_day", "jan25", "jan24", "two_day_primary"),
        ("jan26", "jan26", "jan25", "jan26_secondary"),
    ]:
        with np.load(trial / "work" / f"{filename}_predictions.npz") as saved:
            data = {key: saved[key] for key in
                    ["valid_idx", "labels", "baseline", "candidate", "history_model_probability"]}
        indices = data["valid_idx"]
        assert np.array_equal(data["labels"], raw["y"][indices])
        assert len(np.unique(raw["u"][indices])) == len(indices)
        assert data["labels"].shape == data["history_model_probability"].shape == (len(indices), 10)
        assert (raw["t"][indices] >= CUTOFFS[fold]).all()
        assert (raw["t"][indices] < 1737936000).all()
        old_members, new_members = [], []
        member_metrics = {}
        for key, dimension, folder, combiners, prefix in specs:
            print(f"Reusing trained combiners: {filename}, {key}", flush=True)
            with np.load(folder / fold / "data/cache" / f"split_{fold}.npz") as saved:
                fitting = saved["train_idx"]
            assert np.array_equal(fitting, np.flatnonzero(raw["t"] < CUTOFFS[fold]))
            pipeline = neural_pipeline if dimension == 8 else helper.neural_pipeline
            X, names, *_ = pipeline(
                folder / fold / "work/event_best.pt",
                {k: raw[k][fitting] for k in FIT_FIELDS},
                {k: raw[k][indices] for k in TARGET_FIELDS},
                folder / fold / "data/cache/preference_training.npz", device)
            assert len(names) == 37 and FEATURE not in names
            old_model = load_combiner(combiners / f"{prefix}{combiner_day}.cbm")
            old_members.append(predict_combiner(old_model, X, names).reshape(-1, 10))
            new_model = CatBoostClassifier()
            new_model.load_model(str(trial / "work" / f"{key}_combiner_{combiner_day}.cbm"))
            assert new_model.tree_count_ == 545 and len(new_model.feature_names_) == 38
            augmented = np.column_stack([X, data["history_model_probability"].ravel()])
            new_members.append(predict_combiner(new_model, augmented, list(names) + [FEATURE]).reshape(-1, 10))
            member_metrics[key] = {"v5_auc": auc(data["labels"], old_members[-1]),
                                   "history_feature_auc": auc(data["labels"], new_members[-1])}
            del X, augmented, old_model, new_model
            torch.cuda.empty_cache()
        baseline, failed = ensemble(old_members), ensemble(new_members)
        check_close(baseline, data["baseline"], f"V5 reproduction {filename}", rtol=0, atol=1e-15)
        check_close(failed, data["candidate"], f"Failed all-five reproduction {filename}", rtol=0, atol=1e-15)
        assert abs(auc(data["labels"], failed) - failed_report[report_key]["candidate_history_auc"]) < 1e-12
        candidate = ensemble(selected_members(old_members, new_members))
        baseline_auc, candidate_auc = auc(data["labels"], baseline), auc(data["labels"], candidate)
        metrics = {
            "events": len(indices), "baseline_v5_auc": baseline_auc,
            "candidate_dim8_only_auc": candidate_auc, "auc_change": candidate_auc - baseline_auc,
            "member_auc": member_metrics,
            "mean_dim8_dim16_log_score_correlation_v5": cross_dimension_correlation(old_members),
            "mean_dim8_dim16_log_score_correlation_all_five_updated": cross_dimension_correlation(new_members),
        }
        np.savez_compressed(output / f"{filename}_predictions.npz", valid_idx=indices,
                            labels=data["labels"], baseline=baseline, candidate=candidate,
                            failed_all_five=failed, old_members=np.stack(old_members),
                            new_members=np.stack(new_members), weights=np.asarray(protocol["weights"]))
        cases[filename] = {"metrics": metrics, "labels": data["labels"],
                           "baseline": baseline, "candidate": candidate}
        print(json.dumps({report_key: metrics}, indent=2), flush=True)
    primary = cases["two_day"]
    n = len(primary["labels"])
    event_ids = np.repeat(np.arange(n), 10)
    orders = [AucOrder(primary["labels"].ravel(), primary[k].ravel(), event_ids)
              for k in ["baseline", "candidate"]]
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    gains = []
    for iteration in range(protocol["bootstrap_resamples"]):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
        gains.append(orders[1].value(counts) - orders[0].value(counts))
        if (iteration + 1) % 50 == 0:
            print(f"Whole-event paired bootstrap: {iteration + 1}/300", flush=True)
    interval = np.quantile(gains, [0.025, 0.975]).tolist()
    verify(original)
    for path, expected in protocol["input_sha256"].items():
        assert sha256(Path(path)) == expected, f"Input changed during check: {path}"
    report = {
        "protocol": {k: v for k, v in protocol.items() if k != "input_sha256"},
        "two_day_primary": primary["metrics"], "jan26_secondary": cases["jan26"]["metrics"],
        "primary_paired_gain_95_percentile_interval": interval,
        "sampling_unit": "whole first event per user, preserving all 10 candidates",
        "v5_and_failed_trial_reproduction_passed": True,
        "supported_for_long_window_check": bool(interval[0] > 0 and cases["jan26"]["metrics"]["auc_change"] >= 0),
        "original_all_five_trial_remains_rejected": True,
        "submission_created": False, "runtime_seconds": round(time.perf_counter() - start, 2),
    }
    (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nHISTORY ENSEMBLE INTEGRATION CHECK", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
