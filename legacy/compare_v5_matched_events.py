"""CPU-only paired comparison of two frozen V5 fitting cutoffs.

Run in techarena-final:
    python src/compare_v5_matched_events.py --self-test
    python src/compare_v5_matched_events.py

Fits nothing, changes no models, creates no competition submission.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from audit_v5 import AucOrder, auc, choice_metrics
from submission_export import sha256

JAN22, JAN25, JAN26, JAN27 = 1737504000, 1737763200, 1737849600, 1737936000
WEIGHTS = [2 / 9, 2 / 9, 2 / 9, 1 / 6, 1 / 6]
MEMBERS = ["dim8_seed42", "dim8_seed43", "dim8_seed44", "dim16_seed42", "dim16_seed43"]


def ensemble(members):
    assert len(members) == 5 and all(p.shape == members[0].shape for p in members)
    return (2 / 3) * ((members[0] + members[1] + members[2]) / 3) + (1 / 3) * (0.5 * members[3] + 0.5 * members[4])


def aligned(old_idx, old_labels, new_idx, new_labels):
    for idx, labels in [(old_idx, old_labels), (new_idx, new_labels)]:
        assert np.issubdtype(idx.dtype, np.integer) and idx.ndim == 1
        assert len(np.unique(idx)) == len(idx), "Duplicate event index"
        assert labels.shape == (len(idx), 10) and np.isin(labels, [0, 1]).all()
        assert (labels.sum(1) <= 1).all()
    common, left, right = np.intersect1d(old_idx, new_idx, return_indices=True)
    assert len(common) > 0, "No common events"
    assert np.array_equal(old_idx[left], new_idx[right])
    assert np.array_equal(old_labels[left], new_labels[right]), "Matched labels disagree"
    return common, left, right


def metrics(labels, old, new):
    if labels.size == 0:
        return {"events": 0, "rows": 0, "older_auc": None, "later_auc": None, "auc_change": None}
    a, b = auc(labels, old), auc(labels, new)
    loss = lambda p: float(-(labels * np.log(np.clip(p, 1e-12, 1 - 1e-12)) +
                            (1 - labels) * np.log1p(-np.clip(p, 1e-12, 1 - 1e-12))).mean())
    return {"events": len(labels), "rows": int(labels.size), "clicks": int(labels.sum()),
            "older_auc": a, "later_auc": b, "auc_change": b - a if a is not None and b is not None else None,
            "actual_click_rate": float(labels.mean()), "older_mean_prediction": float(old.mean()),
            "later_mean_prediction": float(new.mean()), "older_log_loss": loss(old), "later_log_loss": loss(new)}


def paired_bootstrap(labels, old, new, masks, resamples=1000, seed=2026, progress=True):
    """Resample matched users once per iteration; use identical weights for both models."""
    n = len(labels)
    reports, orders, samples = {}, {}, {}
    for name, mask in masks.items():
        y = labels[mask]
        reports[name] = metrics(y, old[mask], new[mask])
        if mask.sum() >= 100 and y.sum() >= 30 and y.size - y.sum() >= 100:
            event_ids = np.repeat(np.flatnonzero(mask), 10)
            orders[name] = [AucOrder(y.ravel(), p[mask].ravel(), event_ids) for p in [old, new]]
            samples[name] = [[], [], []]
        else:
            reports[name]["interval_skipped_reason"] = "fewer than 100 events, 30 clicks or 100 nonclick rows"
    rng = np.random.default_rng(seed)
    for iteration in range(resamples):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        for name, (a, b) in orders.items():
            aa, bb = a.value(counts), b.value(counts)
            if aa is not None and bb is not None:
                for values, value in zip(samples[name], [aa, bb, bb - aa]):
                    values.append(value)
        if progress and (iteration + 1) % 100 == 0:
            print(f"Paired whole-user bootstrap: {iteration + 1}/{resamples}", flush=True)
    for name, values in samples.items():
        reports[name]["valid_bootstrap_resamples"] = len(values[0])
        assert len(values[0]) >= 0.9 * resamples
        for key, numbers in zip(["older_auc_95_percentile_interval", "later_auc_95_percentile_interval",
                                 "paired_auc_change_95_percentile_interval"], values):
            reports[name][key] = np.quantile(numbers, [0.025, 0.975]).tolist()
    return reports


def check_hashes(protocol):
    for name, expected in protocol["input_sha256"].items():
        assert sha256(Path(name)) == expected, f"Changed comparison input: {name}"


def run(root):
    start = time.perf_counter()
    projects = root.parent
    baseline = projects / "techarena-v5-long-window"
    seed44 = projects / "techarena-seed44-trial"
    output = baseline / "outputs/matched_event_comparison"
    older_path = baseline / "work/v5_five_day_predictions.npz"
    later_path = seed44 / "work/two_day_ensemble_predictions.npz"
    inputs = [Path(__file__), root / "src/audit_v5.py", root / "src/submission_export.py",
              root / "data/cache/train_events.npz", older_path, later_path,
              baseline / "outputs/frozen_protocol.json", baseline / "outputs/baseline_results.json",
              seed44 / "outputs/frozen_protocol.json", seed44 / "outputs/ensemble_trial_results.json",
              baseline / "data/cache/jan22/split_jan22.npz",
              seed44 / "jan25/data/cache/split_jan25.npz",
              projects / "techarena-preference-finetune-trial/work/two_day_predictions.npz",
              projects / "techarena-dim16-trial/work/two_day_ensemble_predictions.npz",
              projects / "techarena-dim16-seed43-trial/work/two_day_ensemble_predictions.npz"]
    if (output / "results.json").exists():
        protocol = json.loads((output / "frozen_protocol.json").read_text(encoding="utf-8"))
        check_hashes(protocol)
        print("Saved comparison already complete; displaying it without recomputing.", flush=True)
        print("\nV5 MATCHED-EVENT COMPARISON", flush=True)
        print((output / "results.json").read_text(encoding="utf-8"), flush=True)
        return
    if output.exists():
        protocol = json.loads((output / "frozen_protocol.json").read_text(encoding="utf-8"))
        check_hashes(protocol)
        print("Resuming the existing frozen comparison.", flush=True)
    else:
        protocol = {
            "diagnostic": "same events, unchanged V5 recipe, two different fitting cutoffs",
            "older_neural_cutoff": "2025-01-22T00:00:00+00:00", "older_combiner_fitting_day": "2025-01-21",
            "later_neural_cutoff": "2025-01-25T00:00:00+00:00", "later_combiner_fitting_day": "2025-01-24",
            "target_days": ["2025-01-25", "2025-01-26"], "expected_events": 3531,
            "weights": WEIGHTS, "bootstrap_resamples": 1000, "bootstrap_seed": 2026,
            "sampling_unit": "whole matched first event per user, preserving all 10 candidate rows",
            "interpretation": "compares more recently fitted pipelines on identical targets; additional data, fitted parameters, exposure statistics and combiner fitting day all change",
            "limitations": "selected late-arriving subset, not representative of all evaluation users; retrospective reused dates; subgroup intervals exploratory and unadjusted for multiple comparisons; no isolated causal estimate of time decay or sample size",
            "training_performed": False, "candidate_or_weight_search_performed": False,
            "input_sha256": {str(p.resolve(strict=True)): sha256(p) for p in inputs},
        }
        output.mkdir()
        (output / "frozen_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    baseline_report = json.loads((baseline / "outputs/baseline_results.json").read_text(encoding="utf-8"))
    baseline_protocol = json.loads((baseline / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    assert baseline_report["baseline_established"] and baseline_report["data_and_saved_combiner_checks_passed"]
    assert baseline_protocol["weights"] == WEIGHTS
    raw_path = (root / "data/cache/train_events.npz").resolve(strict=True)
    assert baseline_protocol["input_sha256"][str(raw_path)] == sha256(raw_path), "Raw cache changed since five-day baseline"
    assert baseline_report["protocol"]["target_neural_cutoff"] == protocol["older_neural_cutoff"]
    assert baseline_report["protocol"]["combiner_fitting_day"] == protocol["older_combiner_fitting_day"]
    later_report = json.loads((seed44 / "outputs/ensemble_trial_results.json").read_text(encoding="utf-8"))
    later_protocol = json.loads((seed44 / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    assert later_protocol["weights"] == WEIGHTS and later_protocol["combiner_trees"] == 545
    with np.load(root / "data/cache/train_events.npz") as saved:
        raw = {k: saved[k] for k in ["u", "t", "h", "ht", "y"]}
    with np.load(older_path) as saved:
        old_idx, old_labels, old_pred, old_members = [saved[k] for k in ["valid_idx", "labels", "predictions", "members"]]
        assert np.array_equal(saved["weights"], np.asarray(WEIGHTS))
    with np.load(later_path) as saved:
        new_idx, new_labels, new_pred, third8 = [saved[k] for k in ["valid_idx", "labels", "candidate", "new_dim8_seed44"]]
    paths = inputs[-3:]
    more_members = []
    for path, keys in [(paths[0], ["new_dim8_seed42", "new_dim8_seed43"]),
                       (paths[1], ["dim16"]), (paths[2], ["new_dim16"])]:
        with np.load(path) as saved:
            assert np.array_equal(saved["valid_idx"], new_idx) and np.array_equal(saved["labels"], new_labels)
            more_members.extend([saved[k] for k in keys])
    new_members = np.stack([more_members[0], more_members[1], third8, more_members[2], more_members[3]])
    for idx, labels, p, members in [(old_idx, old_labels, old_pred, old_members), (new_idx, new_labels, new_pred, new_members)]:
        assert p.shape == (len(idx), 10) and members.shape == (5, len(idx), 10)
        assert np.isfinite(members).all() and ((members >= 0) & (members <= 1)).all()
        np.testing.assert_allclose(ensemble(members), p, rtol=0, atol=1e-15)
        assert np.array_equal(labels, raw["y"][idx])
        assert len(np.unique(raw["u"][idx])) == len(idx)
    assert abs(auc(old_labels, old_pred) - baseline_report["overall"]["auc"]) < 1e-12
    assert abs(auc(new_labels, new_pred) - later_report["two_day_primary"]["candidate_five_model_auc"]) < 1e-12
    idx, left, right = aligned(old_idx, old_labels, new_idx, new_labels)
    assert len(idx) == protocol["expected_events"], f"Expected 3531 matched events, found {len(idx)}"
    assert np.array_equal(idx, np.sort(old_idx[raw["t"][old_idx] >= JAN25]))
    assert (raw["t"][idx] >= JAN25).all() and (raw["t"][idx] < JAN27).all()
    assert len(np.unique(raw["u"][idx])) == len(idx)
    y, old, new = old_labels[left], old_pred[left], new_pred[right]
    present = raw["h"][idx] >= 0
    assert (raw["ht"][idx][present] < np.broadcast_to(raw["t"][idx, None], present.shape)[present]).all()
    counts = []
    for split, cutoff in [(inputs[10], JAN22), (inputs[11], JAN25)]:
        with np.load(split) as saved:
            fit_idx = saved["train_idx"]
        assert np.array_equal(fit_idx, np.flatnonzero(raw["t"] < cutoff))
        counts.append(np.bincount(raw["u"][fit_idx], minlength=int(raw["u"].max()) + 1)[raw["u"][idx]])
    # These users have no events in Jan22-24, or their first five-day target
    # would have been earlier. Their fitting event counts must therefore match.
    assert np.array_equal(counts[0], counts[1]), "Matched users' activity unexpectedly changed between cutoffs"
    activity, length = counts[0], present.sum(1)
    jan25, jan26 = raw["t"][idx] < JAN26, raw["t"][idx] >= JAN26
    assert int(jan25.sum()) == 2165 and int(jan26.sum()) == 1366
    masks = {"all": np.ones(len(idx), dtype=bool), "jan25": jan25, "jan26": jan26,
             "seen_users": activity > 0, "unseen_with_history": (activity == 0) & (length > 0),
             "unseen_empty_history": (activity == 0) & (length == 0), "history_empty": length == 0,
             "history_1_5": (length >= 1) & (length <= 5), "history_6_19": (length >= 6) & (length <= 19),
             "history_20": length == 20, "activity_1_5": (activity >= 1) & (activity <= 5),
             "activity_6_15": (activity >= 6) & (activity <= 15), "activity_16_plus": activity >= 16}
    result = paired_bootstrap(y, old, new, masks, protocol["bootstrap_resamples"], protocol["bootstrap_seed"])
    for name, date in [("jan25", "2025-01-25"), ("jan26", "2025-01-26")]:
        assert abs(result[name]["older_auc"] - baseline_report["per_day"][date]["auc"]) < 1e-12
    result["all"]["older_choice_diagnostics"] = choice_metrics(y, old)
    result["all"]["later_choice_diagnostics"] = choice_metrics(y, new)
    member_metrics = {name: metrics(y, old_members[i, left], new_members[i, right]) for i, name in enumerate(MEMBERS)}
    np.savez_compressed(output / "matched_predictions.npz", valid_idx=idx, user_ids=raw["u"][idx], labels=y,
                        older_predictions=old, later_predictions=new,
                        older_members=old_members[:, left], later_members=new_members[:, right], weights=np.asarray(WEIGHTS))
    check_hashes(protocol)
    report = {
        "protocol": {k: v for k, v in protocol.items() if k != "input_sha256"},
        "overall": result.pop("all"), "per_day": {"2025-01-25": result.pop("jan25"), "2025-01-26": result.pop("jan26")},
        "groups": result, "member_metrics": member_metrics,
        "population": {"matched_users": len(idx), "unseen_user_fraction_both_cutoffs": float((activity == 0).mean()),
                       "empty_history_fraction": float((length == 0).mean()),
                       "history_1_5_fraction": float(((length >= 1) & (length <= 5)).mean()),
                       "fitting_activity_unchanged_for_all_matched_users": True},
        "event_and_label_alignment_passed": True, "both_cached_ensembles_reproduced": True,
        "training_performed": False, "submission_created": False,
        "runtime_seconds": round(time.perf_counter() - start, 2),
    }
    (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nV5 MATCHED-EVENT COMPARISON", flush=True)
    print(json.dumps(report, indent=2), flush=True)


def self_test():
    rng = np.random.default_rng(2026)
    labels = np.zeros((160, 10), dtype=np.int8)
    labels[np.arange(0, 160, 2), 0] = 1
    idx = np.arange(160) * 2 + 10
    shuffled = rng.permutation(160)
    common, left, right = aligned(idx, labels, idx[shuffled], labels[shuffled])
    assert np.array_equal(common, idx) and np.array_equal(labels[left], labels[shuffled][right])
    try:
        corrupted = labels[shuffled].copy()
        corrupted[0] = 0
        corrupted[0, 9] = 1
        aligned(idx, labels, idx[shuffled], corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("Bad labels were accepted")
    tied = np.round(rng.uniform(0.01, 0.1, labels.shape), 2)
    identical = paired_bootstrap(labels, tied, tied, {"all": np.ones(160, bool)}, resamples=30, progress=False)
    assert identical["all"]["paired_auc_change_95_percentile_interval"] == [0.0, 0.0]
    # Verify weighted event bootstrap equals explicit event duplication, including ties.
    counts = rng.integers(0, 4, 160)
    repeated = np.repeat(np.arange(160), counts)
    order = AucOrder(labels.ravel(), tied.ravel(), np.repeat(np.arange(160), 10))
    assert abs(order.value(counts) - auc(labels[repeated], tied[repeated])) < 1e-15
    improved = tied.copy()
    improved[labels == 1] = 0.9
    comparison = paired_bootstrap(labels, tied, improved, {"all": np.ones(160, bool)}, resamples=30, progress=False)
    assert comparison["all"]["auc_change"] > 0
    assert comparison["all"]["paired_auc_change_95_percentile_interval"][0] > 0
    print("Matched-event self-test passed: shuffled alignment, label rejection, ties, paired resampling and gain direction.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
