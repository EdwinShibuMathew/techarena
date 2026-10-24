"""One fixed, retrospective ALS routing check using saved predictions only.

In techarena-final:
    python src/check_als_cold_routing.py --self-test
    python src/check_als_cold_routing.py

Requires NumPy and the existing audit_v5.py/submission_export.py helpers.
No torch, fitting, evaluation data, weight search or submission export.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from audit_v5 import AucOrder, auc
from submission_export import sha256


CASES = {"two_day": (1737763200, 0.8183722315263706),
         "jan26": (1737849600, 0.8269940935263441)}
END = 1737936000
EXPERT_WEIGHT = .15
RESAMPLES = 300
SEED = 2026


def probabilities(values, shape):
    values = np.asarray(values)
    assert values.shape == shape and values.dtype.kind == "f"
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    return values


def eligibility(raw, indices, cutoff):
    """Membership depends only on fitting exposure and target history presence."""
    users, times, histories = raw["u"], raw["t"], raw["h"]
    assert users.ndim == times.ndim == 1 and len(users) == len(times) == len(histories)
    assert users.dtype.kind in "iu" and (users >= 0).all()
    assert histories.ndim == 2 and (times[indices] >= cutoff).all()
    counts = np.bincount(users[times < cutoff], minlength=int(users.max()) + 1)
    activity = counts[users[indices]]
    length = (histories[indices] >= 0).sum(1)
    return (activity == 0) & (length > 0), activity, length


def route(baseline, expert, eligible):
    shape = baseline.shape
    assert len(shape) == 2 and shape[1] == 10
    probabilities(baseline, shape)
    probabilities(expert, shape)
    assert eligible.dtype == np.bool_ and eligible.shape == (shape[0],)
    candidate = baseline.copy()
    candidate[eligible] = ((1 - EXPERT_WEIGHT) * baseline[eligible]
                           + EXPERT_WEIGHT * expert[eligible])
    assert np.array_equal(candidate[~eligible], baseline[~eligible])
    return probabilities(candidate, shape)


def metrics(labels, baseline, candidate):
    before, after = auc(labels, baseline), auc(labels, candidate)
    return {"events": len(labels), "baseline_v5_auc": before, "candidate_auc": after,
            "auc_change": after - before if before is not None and after is not None else None}


def paired_interval(labels, baseline, candidate, resamples=RESAMPLES, progress=True):
    n = len(labels)
    assert labels.shape == baseline.shape == candidate.shape == (n, 10) and n > 0
    ids = np.repeat(np.arange(n), 10)
    orders = [AucOrder(labels.ravel(), p.ravel(), ids) for p in [baseline, candidate]]
    rng = np.random.default_rng(SEED)
    gains = []
    for iteration in range(resamples):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        before, after = [order.value(counts) for order in orders]
        if before is not None and after is not None:
            gains.append(after - before)
        if progress and (iteration + 1) % 50 == 0:
            print(f"Global paired user bootstrap: {iteration + 1}/{resamples}", flush=True)
    assert len(gains) >= .99 * resamples, "Too many resamples with only one label class"
    return np.quantile(gains, [.025, .975]).tolist(), len(gains)


def reference_paths(projects, case):
    return [projects / "techarena-preference-finetune-trial/work" / f"{case}_predictions.npz",
            projects / "techarena-seed44-trial/work" / f"{case}_ensemble_predictions.npz",
            projects / "techarena-dim16-trial/work" / f"{case}_ensemble_predictions.npz",
            projects / "techarena-dim16-seed43-trial/work" / f"{case}_ensemble_predictions.npz"]


def reproduce_v5(paths, indices, labels, baseline):
    arrays = []
    for path, keys in zip(paths, [["new_dim8_seed42", "new_dim8_seed43"],
                                 ["new_dim8_seed44"], ["dim16"], ["new_dim16"]]):
        with np.load(path, allow_pickle=False) as saved:
            assert np.array_equal(saved["valid_idx"], indices)
            assert np.array_equal(saved["labels"], labels)
            arrays.extend([probabilities(saved[key], labels.shape) for key in keys])
    rebuilt = ((2 / 3) * ((arrays[0] + arrays[1] + arrays[2]) / 3)
               + (1 / 3) * (.5 * arrays[3] + .5 * arrays[4]))
    np.testing.assert_allclose(rebuilt, baseline, rtol=0, atol=1e-15)


def check_hashes(hashes):
    for name, expected in hashes.items():
        assert sha256(Path(name)) == expected, f"Changed frozen input: {name}"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def source_inputs(root, trial):
    """Verify the completed ALS trial and retain its original input provenance."""
    old_protocol = read_json(trial / "outputs/frozen_protocol.json")
    old_report = read_json(trial / "outputs/ensemble_trial_results.json")
    training = read_json(trial / "outputs/training_results.json")
    assert set(training["folds"]) == {"jan24", "jan25", "jan26"}
    assert all(set(stages) == {"preference_best.pt", "event_best.pt"}
               for stages in training["folds"].values())
    assert old_protocol["expert_weight"] == EXPERT_WEIGHT
    assert old_report["v5_cached_ensemble_reproduction_passed"] is True
    assert old_report["saved_expert_combiners_reproduced"] is True
    assert old_report["supported_for_long_window_check"] is False
    check_hashes(old_protocol["input_sha256"])
    check_hashes({str(trial / p): h for p, h in old_protocol["prepared_sha256"].items()})
    paths = [Path(__file__), root / "src/audit_v5.py", root / "src/submission_export.py",
             trial / "outputs/frozen_protocol.json", trial / "outputs/ensemble_trial_results.json",
             trial / "outputs/training_results.json", root / "data/cache/train_events.npz"]
    # Hash checkpoints without importing torch or allocating GPU memory.
    for fold, stages in training["folds"].items():
        for filename, info in stages.items():
            path = trial / fold / "work" / filename
            assert sha256(path) == info["sha256"], f"ALS checkpoint changed: {path}"
            paths.append(path)
    for day in ["jan24", "jan25"]:
        paths.append(trial / "work" / f"combiner_{day}.cbm")
    for case in CASES:
        paths.append(trial / "work" / f"{case}_ensemble_predictions.npz")
        paths.extend(reference_paths(root.parent, case))
    return paths, old_report


def evaluate(root):
    start = time.perf_counter()
    trial = root.parent / "techarena-als-trial"
    output = trial / "outputs/cold_routing_check"
    paths, old_report = source_inputs(root, trial)
    hashes = {str(p.resolve(strict=True)): sha256(p) for p in paths}
    protocol = {
        "experiment": "ALS expert only for unseen users with nonempty target history",
        "eligibility": "zero fitting events strictly before the case cutoff, and at least one history item",
        "eligible_v5_weight": 1 - EXPERT_WEIGHT, "eligible_als_weight": EXPERT_WEIGHT,
        "ineligible_rule": "retain original V5 probabilities exactly for every row",
        "eligibility_uses_target_labels": False,
        "training_performed": False, "weight_search_performed": False,
        "primary_check": "first-user Jan25-26; fitting activity cutoff Jan25; existing ALS Jan25 / combiner Jan24",
        "secondary_check": "first-user Jan26; fitting activity cutoff Jan26; existing ALS Jan26 / combiner Jan25",
        "bootstrap_resamples": RESAMPLES, "bootstrap_seed": SEED,
        "support_rule": "global primary paired gain interval entirely positive and global secondary AUC change nonnegative",
        "next_gate": "five-day paired test of the same routing rule before full-data fitting or submission",
        "status": "post-hoc follow-up to failed global ALS blend; reused validation dates; intervals not selection-adjusted",
        "input_sha256": hashes,
    }
    frozen_path, result_path = output / "frozen_protocol.json", output / "results.json"
    if frozen_path.exists():
        assert read_json(frozen_path) == protocol, "Routing protocol or frozen inputs changed; preserve the original check"
        if result_path.exists():
            report = read_json(result_path)
            for name, expected in report["output_sha256"].items():
                assert sha256(output / name) == expected, f"Changed saved routing predictions: {name}"
            print("Saved routing check already complete; displaying it.", flush=True)
            print("\nALS COLD-USER ROUTING CHECK", flush=True)
            print(result_path.read_text(encoding="utf-8"), flush=True)
            return
    else:
        output.mkdir(parents=True, exist_ok=True)
        assert not result_path.exists(), "Results exist without a protocol"
        write_json(frozen_path, protocol)

    with np.load(root / "data/cache/train_events.npz", allow_pickle=False) as saved:
        raw = {key: saved[key] for key in ["u", "t", "h", "y"]}
    cases = {}
    outputs = {}
    for case, (cutoff, expected_auc) in CASES.items():
        with np.load(trial / "work" / f"{case}_ensemble_predictions.npz", allow_pickle=False) as saved:
            indices, labels, baseline, expert, old_candidate = [saved[key] for key in
                ["valid_idx", "labels", "baseline", "als_expert", "candidate"]]
        assert indices.ndim == 1 and indices.dtype.kind in "iu"
        assert (indices >= 0).all() and (indices < len(raw["u"])).all()
        window = np.flatnonzero((raw["t"] >= cutoff) & (raw["t"] < END))
        ordered = window[np.argsort(raw["t"][window], kind="stable")]
        _, first = np.unique(raw["u"][ordered], return_index=True)
        assert np.array_equal(indices, np.sort(ordered[first])), "First-user selection differs"
        assert np.array_equal(labels, raw["y"][indices]) and labels.shape == (len(indices), 10)
        assert np.isin(labels, [0, 1]).all() and (labels.sum(1) <= 1).all()
        probabilities(baseline, labels.shape)
        probabilities(expert, labels.shape)
        reproduce_v5(reference_paths(root.parent, case), indices, labels, baseline)
        assert abs(auc(labels, baseline) - expected_auc) < 1e-12
        np.testing.assert_allclose(.85 * baseline + .15 * expert, old_candidate, rtol=0, atol=1e-15)
        report_key = "two_day_primary" if case == "two_day" else "jan26_secondary"
        assert abs(auc(labels, old_candidate) - old_report[report_key]["candidate_auc"]) < 1e-12
        assert abs(auc(labels, expert) - old_report[report_key]["als_expert_auc"]) < 1e-12
        eligible, activity, length = eligibility(raw, indices, cutoff)
        candidate = route(baseline, expert, eligible)
        assert eligible.any() and (~eligible).any()
        assert not eligible[length == 0].any() and not eligible[activity > 0].any()
        result = metrics(labels, baseline, candidate)
        result.update({"eligible_events": int(eligible.sum()),
                       "eligible_event_percent": float(100 * eligible.mean()),
                       "changed_candidate_rows": int(np.count_nonzero(candidate != baseline)),
                       "ineligible_predictions_unchanged": np.array_equal(candidate[~eligible], baseline[~eligible]),
                       "empty_history_predictions_unchanged": np.array_equal(candidate[length == 0], baseline[length == 0]),
                       "eligible_group": metrics(labels[eligible], baseline[eligible], candidate[eligible]),
                       "ineligible_group": metrics(labels[~eligible], baseline[~eligible], candidate[~eligible])})
        path = output / f"{case}_predictions.npz"
        np.savez_compressed(path, valid_idx=indices, labels=labels, baseline=baseline,
                            als_expert=expert, candidate=candidate, eligible=eligible)
        outputs[path.name] = sha256(path)
        cases[case] = {"metrics": result, "labels": labels, "baseline": baseline, "candidate": candidate}
        print(json.dumps({report_key: result}, indent=2), flush=True)
    primary = cases["two_day"]
    # ALL users are resampled together: this includes changed-vs-unchanged row pairs.
    interval, valid_resamples = paired_interval(primary["labels"], primary["baseline"], primary["candidate"])
    check_hashes(hashes)
    report = {"protocol": {k: v for k, v in protocol.items() if k != "input_sha256"},
              "two_day_primary": primary["metrics"], "jan26_secondary": cases["jan26"]["metrics"],
              "global_primary_paired_gain_95_percentile_interval": interval,
              "valid_bootstrap_resamples": valid_resamples,
              "sampling_unit": "whole first event per user, preserving all ten candidates; all users included",
              "v5_and_original_als_blend_reproduction_passed": True,
              "supported_for_five_day_check": bool(interval[0] > 0 and cases["jan26"]["metrics"]["auc_change"] >= 0),
              "original_global_als_blend_remains_rejected": True,
              "submission_created": False, "output_sha256": outputs,
              "runtime_seconds": round(time.perf_counter() - start, 2)}
    write_json(result_path, report)
    print("\nALS COLD-USER ROUTING CHECK", flush=True)
    print(json.dumps(report, indent=2), flush=True)


def dense_auc(labels, predictions):
    """Independent tiny pairwise oracle, including ties."""
    y, p = labels.ravel(), predictions.ravel()
    positive, negative = p[y == 1], p[y == 0]
    return float(((positive[:, None] > negative).astype(float)
                  + .5 * (positive[:, None] == negative)).mean())


def self_test():
    # A fitting user with no clicks still counts as seen. A user appearing only
    # after the cutoff remains cold; an empty target history is ineligible.
    raw = {"u": np.array([0, 1, 0, 2, 3, 1, 4]),
           "t": np.array([1, 2, 10, 11, 12, 13, 14]),
           "h": np.array([[-1, -1], [-1, -1], [3, -1], [4, 5], [-1, -1], [8, -1], [9, -1]])}
    idx = np.arange(2, 7)
    eligible, activity, _ = eligibility(raw, idx, 10)
    np.testing.assert_array_equal(activity, [1, 0, 0, 1, 0])
    np.testing.assert_array_equal(eligible, [False, True, False, False, True])
    # Routing must not depend on the order in which targets are scored.
    rng = np.random.default_rng(19)
    baseline, expert = rng.uniform(.01, .15, (2, 5, 10))
    candidate = route(baseline, expert, eligible)
    shuffled = np.array([4, 1, 0, 3, 2])
    np.testing.assert_array_equal(route(baseline[shuffled], expert[shuffled], eligible[shuffled]), candidate[shuffled])
    np.testing.assert_array_equal(route(baseline, expert, np.zeros(5, dtype=bool)), baseline)
    assert not np.shares_memory(candidate, baseline)
    # Compare the full paired bootstrap to explicit event duplication and an
    # independent pairwise AUC oracle; include tied scores.
    labels = np.zeros((5, 10), dtype=np.int8)
    labels[:, 0] = 1
    baseline, expert = np.round(baseline, 2), np.round(expert, 2)
    candidate = route(baseline, expert, eligible)
    interval, valid = paired_interval(labels, baseline, candidate, resamples=25, progress=False)
    rng = np.random.default_rng(SEED)
    oracle = []
    for _ in range(25):
        draw = rng.integers(0, 5, 5)
        oracle.append(dense_auc(labels[draw], candidate[draw]) - dense_auc(labels[draw], baseline[draw]))
    np.testing.assert_allclose(interval, np.quantile(oracle, [.025, .975]), rtol=0, atol=1e-14)
    assert valid == 25
    # A subgroup can retain perfect internal ranking while global AUC falls.
    y = np.tile(np.r_[1, np.zeros(9)], (2, 1))
    a = np.array([[.9] + [.2] * 9, [.8] + [.1] * 9])
    b = a.copy()
    b[0] = [.05] + [.04] * 9
    assert dense_auc(y[:1], a[:1]) == dense_auc(y[:1], b[:1]) == 1
    assert dense_auc(y, b) < dense_auc(y, a)
    print("Self-test passed: cutoff routing, seen/empty protection, target order, ties, global cross-group AUC and whole-user paired bootstrap.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        evaluate(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
