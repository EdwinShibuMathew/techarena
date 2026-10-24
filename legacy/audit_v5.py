"""CPU-only V5 error and history-support audit; no training or submission.

In techarena-final:
    python src/audit_v5.py --self-test
    python src/audit_v5.py

Uses NumPy and the existing caches. Saves full JSON/CSV plus a compact summary.
Pair counts come only from fitting clicks; the target user's contribution is
removed. Queries use distinct nonidentical history items, not repeated positions.
"""

import argparse
import csv
import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

DAY = 86400
END_TRAIN = 1737936000
THRESHOLDS = (1, 2, 3, 5, 10)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class AucOrder:
    """Exact tie-aware weighted AUC after sorting scores once.

    A bootstrap weight belongs to an event, hence to all selected rows from it.
    """

    def __init__(self, labels, predictions, event_ids):
        order = np.argsort(predictions, kind="stable")
        scores = np.asarray(predictions)[order]
        self.y = np.asarray(labels, dtype=bool)[order]
        self.events = np.asarray(event_ids, dtype=np.int32)[order]
        self.starts = None
        if len(scores) > 1 and np.any(scores[1:] == scores[:-1]):
            self.starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]

    def value(self, counts):
        weights = counts[self.events].astype(np.float64)
        positive = weights * self.y
        negative = weights * ~self.y
        if self.starts is not None:
            positive = np.add.reduceat(positive, self.starts)
            negative = np.add.reduceat(negative, self.starts)
        total_positive, total_negative = positive.sum(), negative.sum()
        if total_positive == 0 or total_negative == 0:
            return None
        lower_negative = np.cumsum(negative) - 0.5 * negative
        return float(np.sum(positive * lower_negative) / (total_positive * total_negative))


def auc(labels, predictions):
    y, p = np.asarray(labels).ravel(), np.asarray(predictions).ravel()
    return AucOrder(y, p, np.arange(len(y), dtype=np.int32)).value(np.ones(len(y), dtype=np.int32))


def require_probabilities(values, shape):
    result = np.asarray(values, dtype=np.float64).reshape(shape)
    assert np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()
    return result


def unique_history(history):
    ordered = np.sort(history, axis=1)
    keep = ordered >= 0
    keep[:, 1:] &= ordered[:, 1:] != ordered[:, :-1]
    return ordered, keep


def membership(sorted_keys, queries):
    queries = np.asarray(queries, dtype=np.uint64)
    if not len(sorted_keys):
        return np.zeros(queries.shape, dtype=bool)
    locations = np.searchsorted(sorted_keys, queries)
    return (locations < len(sorted_keys)) & (sorted_keys[np.minimum(locations, len(sorted_keys) - 1)] == queries)


@lru_cache(maxsize=32)
def triangle_indices(length):
    return np.triu_indices(length, k=1)


def count_queried_pairs(edges, query_keys, n_items, pair_buffer=500000, progress=False):
    """Exact distinct-user co-click counts, storing only requested item pairs.

    Each user-item edge and each unordered pair within a user occurs once.
    Emit pairs in bounded chunks, then match them against sorted queried keys.
    No dense item-item matrix or SciPy installation is required.
    """
    support = np.zeros(len(query_keys), dtype=np.int32)
    if not len(edges) or not len(query_keys):
        return support, 0
    users = edges // np.uint64(n_items)
    starts = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1]
    ends = np.r_[starts[1:], len(edges)]
    pending, buffered, emitted = [], 0, 0

    def flush():
        nonlocal pending, buffered
        if not pending:
            return
        keys, frequencies = np.unique(np.concatenate(pending), return_counts=True)
        locations = np.searchsorted(query_keys, keys)
        inside = locations < len(query_keys)
        inside[inside] &= query_keys[locations[inside]] == keys[inside]
        # Keys and matched positions are unique, so indexed addition is exact.
        support[locations[inside]] += frequencies[inside].astype(np.int32)
        pending, buffered = [], 0

    def append(keys):
        nonlocal buffered, emitted
        pending.append(keys)
        buffered += len(keys)
        emitted += len(keys)
        if buffered >= pair_buffer:
            flush()

    for user_number, (begin, end) in enumerate(zip(starts, ends)):
        items = edges[begin:end] % np.uint64(n_items)
        size = len(items)
        if 2 <= size <= 256:
            left, right = triangle_indices(size)
            append(items[left] * np.uint64(n_items) + items[right])
        elif size > 256:
            # Long profiles use slices instead of allocating a quadratic index grid.
            for left in range(size - 1):
                for begin_right in range(left + 1, size, pair_buffer):
                    right = items[begin_right:begin_right + pair_buffer]
                    append(items[left] * np.uint64(n_items) + right)
        if progress and (user_number + 1) % 5000 == 0:
            print(f"  Pair support: {user_number + 1:,}/{len(starts):,} fitting users", flush=True)
    flush()
    return support, emitted


def history_support(positive_users, positive_items, target, n_items, labels=None,
                    pair_limit=100000000, progress=False):
    """Return maximum other-user support per candidate across distinct history items."""
    click_edges = positive_users.astype(np.uint64) * np.uint64(n_items) + positive_items.astype(np.uint64)
    edges = np.unique(click_edges)
    user_ids, profile_sizes = np.unique(edges // np.uint64(n_items), return_counts=True)
    potential_pairs = int(np.sum(profile_sizes.astype(np.int64) * (profile_sizes - 1) // 2))
    if potential_pairs > pair_limit:
        raise RuntimeError(
            f"Exact pair audit needs {potential_pairs:,} fitting-user pairs, above the "
            f"declared {pair_limit:,} work limit. No sampling was performed. "
            "Share this message and the completed error report before increasing the limit."
        )
    history, keep = unique_history(target["h"])
    n_events, width = history.shape
    n_candidates = target["i"].shape[1]
    valid = keep[:, None, :] & (target["i"][:, :, None] != history[:, None, :])
    positions = np.flatnonzero(valid.ravel())
    del valid
    candidate_rows = positions // width
    event_rows = candidate_rows // n_candidates
    candidate_slots = candidate_rows % n_candidates
    history_slots = positions % width
    h_items = history[event_rows, history_slots]
    c_items = target["i"][event_rows, candidate_slots]
    keys = np.minimum(h_items, c_items).astype(np.uint64) * np.uint64(n_items) + np.maximum(h_items, c_items).astype(np.uint64)
    query_keys, inverse = np.unique(keys, return_inverse=True)
    if progress:
        print(f"  Probing {len(query_keys):,} distinct cross-item pairs; {potential_pairs:,} fitting-user pairs", flush=True)
    counts, emitted = count_queried_pairs(edges, query_keys, n_items, progress=progress)
    own_candidate = membership(edges, target["u"].astype(np.uint64)[:, None] * np.uint64(n_items) + target["i"].astype(np.uint64))
    safe_history = history.clip(min=0)
    own_history = membership(edges, target["u"].astype(np.uint64)[:, None] * np.uint64(n_items) + safe_history.astype(np.uint64)) & keep
    own = own_candidate[event_rows, candidate_slots] & own_history[event_rows, history_slots]
    external = counts[inverse] - own.astype(np.int32)
    assert (external >= 0).all()
    maximum = np.zeros((n_events, n_candidates), dtype=np.int32)
    np.maximum.at(maximum.ravel(), candidate_rows, external)
    has_query = np.zeros(maximum.size, dtype=bool)
    has_query[candidate_rows] = True
    has_query = has_query.reshape(maximum.shape)
    repeat = ((target["i"][:, :, None] == target["h"][:, None, :]) & (target["h"][:, None, :] >= 0)).any(axis=2)
    summary = {
        "definition": "distinct fitting users who clicked both different items; target user's contribution excluded",
        "relationship": "unordered co-click relationship, not a directed transition",
        "fitting_click_records_including_genuine_repeats": len(positive_users),
        "distinct_user_item_edges_for_support": len(edges),
        "fitting_users_with_clicks": len(user_ids),
        "maximum_distinct_items_per_fitting_user": int(profile_sizes.max(initial=0)),
        "fitting_user_pairs_enumerated": emitted,
        "queried_distinct_cross_item_pairs": len(query_keys),
        "queried_candidate_history_pairs_after_deduplicating_history_items": len(positions),
        "self_contributions_removed": int(own.sum()),
        "candidate_rows_with_nonidentical_history_item_fraction": float(has_query.mean()),
        "exact_repeat_candidate_fraction": float(repeat.mean()),
        "coverage_by_minimum_other_users": {},
        "query_pairs_with_any_fitting_support_fraction": float((counts > 0).mean()) if len(counts) else None,
    }
    for threshold in THRESHOLDS:
        covered = maximum >= threshold
        entry = {
            "candidate_row_fraction": float(covered.mean()),
            "fraction_among_rows_with_cross_item_history": float(covered[has_query].mean()) if has_query.any() else None,
            "nonrepeat_candidate_row_fraction": float(covered[~repeat].mean()) if (~repeat).any() else None,
            "event_fraction_with_any_supported_candidate": float(covered.any(axis=1).mean()),
            "candidate_history_pair_fraction": float((external >= threshold).mean()) if len(external) else None,
        }
        if labels is not None:
            clicked = labels.astype(bool)
            entry["clicked_row_fraction"] = float(covered[clicked].mean()) if clicked.any() else None
            entry["nonclicked_row_fraction"] = float(covered[~clicked].mean()) if (~clicked).any() else None
        summary["coverage_by_minimum_other_users"][str(threshold)] = entry
    return maximum, has_query, summary


def group_definitions(target, fit_user_counts, item_exposure, cutoff, pair_max=None, pair_has=None):
    length = (target["h"] >= 0).sum(axis=1)
    _, unique = unique_history(target["h"])
    activity = fit_user_counts[target["u"]]
    age = np.where(target["h"] >= 0, (target["t"][:, None] - target["ht"]) / 3600, np.inf).min(axis=1)
    elapsed = (target["t"] - cutoff) / 3600
    repeat = ((target["i"][:, :, None] == target["h"][:, None, :]) & (target["h"][:, None, :] >= 0)).any(axis=2)
    low, high = np.quantile(item_exposure, [1 / 3, 2 / 3])
    exposure = item_exposure[target["i"]]
    groups = {
        "all": np.ones(len(length), dtype=bool),
        "seen_users": activity > 0, "unseen_users": activity == 0,
        "unseen_with_history": (activity == 0) & (length > 0),
        "unseen_empty_history": (activity == 0) & (length == 0),
        "activity_1_5": (activity >= 1) & (activity <= 5),
        "activity_6_15": (activity >= 6) & (activity <= 15),
        "activity_16_plus": activity >= 16,
        "history_empty": length == 0,
        "history_1_5": (length >= 1) & (length <= 5),
        "history_6_10": (length >= 6) & (length <= 10),
        "history_11_19": (length >= 11) & (length <= 19),
        "history_20": length == 20,
        "history_contains_repeats": unique.sum(axis=1) < length,
        "last_click_under_1h": age < 1,
        "last_click_1_6h": (age >= 1) & (age < 6),
        "last_click_6_24h": (age >= 6) & (age < 24),
        "last_click_24_72h": (age >= 24) & (age < 72),
        "last_click_72h_plus_nonempty": (age >= 72) & (length > 0),
        "repeat_candidates": repeat, "nonrepeat_candidates": ~repeat,
        "exposure_low": exposure <= low,
        "exposure_middle": (exposure > low) & (exposure <= high),
        "exposure_high": exposure > high,
    }
    for begin, end in [(0, 6), (6, 12), (12, 24), (24, 48), (48, 96), (96, 168)]:
        groups[f"horizon_{begin}_{end}h"] = (elapsed >= begin) & (elapsed < end)
    for day in np.unique(target["t"] // DAY):
        date = str(np.datetime64(int(day), "D"))
        groups[f"target_day_{date}"] = target["t"] // DAY == day
    if pair_max is not None:
        groups.update({
            "cross_item_history_available": pair_has,
            "cross_item_support_0": pair_has & (pair_max == 0),
            "cross_item_support_1": pair_max == 1,
            "cross_item_support_2": pair_max == 2,
            "cross_item_support_3_plus": pair_max >= 3,
            "cross_item_support_5_plus": pair_max >= 5,
        })
    return groups, {"low_max_inclusive": float(low), "middle_max_inclusive": float(high),
                    "definition": "terciles of fitting display counts over the full item catalogue"}


def selected_rows(mask, n_candidates):
    if mask.ndim == 1:
        return (np.flatnonzero(mask)[:, None] * n_candidates + np.arange(n_candidates)).ravel()
    return np.flatnonzero(mask.ravel())


def choice_metrics(labels, probabilities):
    clicked = labels.sum(axis=1) > 0
    if not clicked.any():
        return None
    y, p = labels[clicked], probabilities[clicked]
    score = (p * y).sum(axis=1)
    not_clicked = ~y.astype(bool)
    lower = ((score[:, None] > p) & not_clicked).sum(axis=1)
    equal = ((score[:, None] == p) & not_clicked).sum(axis=1)
    return {
        "clicked_events": int(clicked.sum()),
        "within_event_auc_clicked_events": float(((lower + 0.5 * equal) / (p.shape[1] - 1)).mean()),
        "clicked_candidate_top1_with_fractional_ties": float(((score == p.max(axis=1)) / (p == p.max(axis=1, keepdims=True)).sum(axis=1)).mean()),
        "mean_clicked_rank_with_average_ties": float((p.shape[1] - lower - 0.5 * equal).mean()),
        "note": "choice diagnostics, not the competition's global row AUC",
    }


def group_reports(labels, p4, p5, groups, resamples, seed):
    n_events, n_candidates = labels.shape
    y, old, new = labels.ravel(), p4.ravel(), p5.ravel()
    reports, ranked = {}, {}
    ones = np.ones(n_events, dtype=np.int32)
    for name, mask in groups.items():
        rows = selected_rows(mask, n_candidates)
        if not len(rows):
            continue
        event_ids = (rows // n_candidates).astype(np.int32)
        events = len(np.unique(event_ids))
        yy = y[rows]
        order4, order5 = AucOrder(yy, old[rows], event_ids), AucOrder(yy, new[rows], event_ids)
        a4, a5 = order4.value(ones), order5.value(ones)
        report = {
            "unit": "event_cohort" if mask.ndim == 1 else "candidate_row_cohort",
            "events_with_selected_rows": events, "rows": len(rows), "clicks": int(yy.sum()),
            "actual_click_rate": float(yy.mean()), "v4_mean_prediction": float(old[rows].mean()),
            "v5_mean_prediction": float(new[rows].mean()), "v4_auc": a4, "v5_auc": a5,
            "auc_change": a5 - a4 if a4 is not None and a5 is not None else None,
            "v5_auc_95_percentile_interval": None, "paired_gain_95_percentile_interval": None,
        }
        if events >= 300 and yy.sum() >= 30 and len(yy) - yy.sum() >= 100 and resamples:
            ranked[name] = (order4, order5, [], [])
        else:
            report["bootstrap_skipped_reason"] = "disabled or fewer than 300 events / 30 clicks / 100 nonclicks"
        if mask.ndim == 1:
            report["v5_choice_diagnostics"] = choice_metrics(labels[mask], p5[mask])
        reports[name] = report
    rng = np.random.default_rng(seed)
    for iteration in range(resamples):
        counts = np.bincount(rng.integers(0, n_events, size=n_events), minlength=n_events)
        for order4, order5, gains, scores in ranked.values():
            a4, a5 = order4.value(counts), order5.value(counts)
            if a4 is not None and a5 is not None:
                gains.append(a5 - a4)
                scores.append(a5)
        if (iteration + 1) % 50 == 0:
            print(f"  Whole-event paired bootstrap: {iteration + 1}/{resamples}", flush=True)
    for name, (_, _, gains, scores) in ranked.items():
        reports[name]["valid_bootstrap_resamples"] = len(gains)
        if len(gains) >= max(20, int(0.9 * resamples)):
            reports[name]["paired_gain_95_percentile_interval"] = np.quantile(gains, [0.025, 0.975]).tolist()
            reports[name]["v5_auc_95_percentile_interval"] = np.quantile(scores, [0.025, 0.975]).tolist()
    return reports


def population(target, fit_counts, p4, p5, cutoff):
    length = (target["h"] >= 0).sum(axis=1)
    activity = fit_counts[target["u"]]
    elapsed = (target["t"] - cutoff) / 3600
    return {
        "events": len(length), "candidate_rows": target["i"].size,
        "unseen_user_fraction": float((activity == 0).mean()),
        "activity_0_5_fraction": float((activity <= 5).mean()),
        "empty_history_fraction": float((length == 0).mean()),
        "history_1_5_fraction": float(((length >= 1) & (length <= 5)).mean()),
        "mean_history_length": float(length.mean()),
        "horizon_hours_p50_p90_max": np.quantile(elapsed, [0.5, 0.9, 1]).tolist(),
        "v4_mean_prediction": float(p4.mean()), "v5_mean_prediction": float(p5.mean()),
    }


def first_indices(raw, cutoff, end):
    eligible = np.flatnonzero((raw["t"] >= cutoff) & (raw["t"] < end))
    ordered = eligible[np.argsort(raw["t"][eligible], kind="stable")]
    _, positions = np.unique(raw["u"][ordered], return_index=True)
    return np.sort(ordered[positions])


def read_positive_cache(path, cutoff):
    with np.load(path) as saved:
        users, items, timestamps = saved["users"], saved["items"], saved["timestamps"]
        n_users, n_items = int(saved["n_users"]), int(saved["n_items"])
        assert int(saved["cutoff"]) == cutoff
    assert users.shape == items.shape == timestamps.shape
    assert (timestamps < cutoff).all()
    assert ((users >= 0) & (users < n_users)).all() and ((items >= 0) & (items < n_items)).all()
    return users, items, n_users, n_items


def check_target(target, n_users, n_items):
    assert target["i"].shape == (len(target["u"]), 10)
    assert target["h"].shape == target["ht"].shape == (len(target["u"]), 20)
    assert ((target["u"] >= 0) & (target["u"] < n_users)).all()
    assert ((target["i"] >= 0) & (target["i"] < n_items)).all()
    present = target["h"] >= 0
    assert ((target["h"] == -1) | (present & (target["h"] < n_items))).all()
    assert (target["ht"][present] < np.broadcast_to(target["t"][:, None], present.shape)[present]).all()


def compact_support(report):
    return {
        "fitting_click_records": report["fitting_click_records_including_genuine_repeats"],
        "queried_distinct_cross_item_pairs": report["queried_distinct_cross_item_pairs"],
        "self_contributions_removed": report["self_contributions_removed"],
        "rows_with_cross_item_history_fraction": report["candidate_rows_with_nonidentical_history_item_fraction"],
        "coverage_by_minimum_other_users": {
            str(threshold): {key: value for key, value in report["coverage_by_minimum_other_users"][str(threshold)].items()
                             if key in ["candidate_row_fraction", "nonrepeat_candidate_row_fraction", "clicked_row_fraction", "nonclicked_row_fraction"]}
            for threshold in [1, 3, 5]
        },
    }


def write_cohort_csv(path, cases):
    fields = ["case", "group", "unit", "events_with_selected_rows", "rows", "clicks",
              "actual_click_rate", "v4_mean_prediction", "v5_mean_prediction", "v4_auc", "v5_auc",
              "auc_change", "gain_ci_low", "gain_ci_high", "v5_auc_ci_low", "v5_auc_ci_high"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case, data in cases.items():
            for name, metrics in data["groups"].items():
                row = {k: metrics.get(k) for k in fields}
                gain = metrics.get("paired_gain_95_percentile_interval") or [None, None]
                score = metrics.get("v5_auc_95_percentile_interval") or [None, None]
                row.update(case=case, group=name, gain_ci_low=gain[0], gain_ci_high=gain[1],
                           v5_auc_ci_low=score[0], v5_auc_ci_high=score[1])
                writer.writerow(row)


def self_test():
    # Independent pairwise AUC reference, including ties and event weights.
    y = np.array([1, 0, 1, 0, 1, 0])
    p = np.array([0.5, 0.5, 0.8, 0.2, 0.1, 0.1])
    events = np.array([0, 0, 1, 1, 2, 2])
    for counts in [np.array([1, 1, 1]), np.array([3, 0, 2]), np.array([0, 4, 1])]:
        weights = counts[events]
        numerator = denominator = 0.0
        for pos in np.flatnonzero(y):
            for neg in np.flatnonzero(1 - y):
                weight = weights[pos] * weights[neg]
                numerator += weight * ((p[pos] > p[neg]) + 0.5 * (p[pos] == p[neg]))
                denominator += weight
        assert abs(AucOrder(y, p, events).value(counts) - numerator / denominator) < 1e-14
    positive_users = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4])
    positive_items = np.array([0, 1, 1, 2, 0, 1, 0, 2, 1, 2, 3])
    target = {"u": np.array([0, 4, 5]), "i": np.array([[1, 3], [1, 2], [1, 3]]),
              "h": np.array([[0, 0, 1], [0, 0, -1], [0, -1, -1]])}
    maximum, has, summary = history_support(positive_users, positive_items, target, 4)
    assert np.array_equal(maximum, [[1, 0], [2, 2], [2, 0]])
    assert has.all() and summary["distinct_user_item_edges_for_support"] == 10
    # Compare pair engine to an independently constructed dense binary toy matrix.
    edges = np.unique(positive_users.astype(np.uint64) * 4 + positive_items.astype(np.uint64))
    queries = np.array([1, 2, 3, 6, 7, 11], dtype=np.uint64)
    support, _ = count_queried_pairs(edges, queries, 4, pair_buffer=2)
    binary = np.zeros((6, 4), dtype=np.int64)
    binary[positive_users, positive_items] = 1
    dense = binary.T @ binary
    assert np.array_equal(support, dense[(queries // 4).astype(int), (queries % 4).astype(int)])
    empty = {"u": np.array([5]), "i": np.array([[0, 1]]), "h": np.array([[-1, -1, -1]])}
    maximum, has, _ = history_support(positive_users, positive_items, empty, 4)
    assert not maximum.any() and not has.any()
    labels = np.array([[0, 1, 0], [0, 0, 0]])
    choice = choice_metrics(labels, np.array([[0.2, 0.2, 0.1], [0.1, 0.1, 0.1]]))
    assert choice["mean_clicked_rank_with_average_ties"] == 1.5
    assert choice["clicked_candidate_top1_with_fractional_ties"] == 0.5
    print("V5 AUDIT SELF-TEST PASSED: weighted AUC/ties; distinct-user pair counts; target-user exclusion; duplicate/empty histories; event-choice ties.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--resamples", type=int, default=300)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    assert args.resamples >= 0
    start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    projects = root.parent
    trial = projects / "techarena-seed44-trial"
    output = root / "outputs/v5_audit"
    if output.exists():
        raise FileExistsError(f"Audit output already exists: {output}. Keep the previous report; share it before rerunning.")
    input_paths = set()
    raw_path = root / "data/cache/train_events.npz"
    input_paths.add(raw_path.resolve())
    with np.load(raw_path) as saved:
        raw = {name: saved[name] for name in ["u", "i", "t", "h", "ht", "y"]}
    assert len(raw["u"]) == 686824 and raw["i"].shape == raw["y"].shape == (686824, 10)
    assert np.isin(raw["y"], [0, 1]).all() and (raw["y"].sum(axis=1) <= 1).all()
    checks_path = root / "outputs/five_model_submission_checks.json"
    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    assert checks["version"] == "v5" and checks["submission_ready"] is True
    assert checks["validation_pipeline_check_passed"] is True
    assert checks["weights"] == [2 / 9, 2 / 9, 2 / 9, 1 / 6, 1 / 6]
    input_paths.add(checks_path.resolve())
    trial_report_path = trial / "outputs/ensemble_trial_results.json"
    trial_report = json.loads(trial_report_path.read_text(encoding="utf-8"))
    input_paths.add(trial_report_path.resolve())
    output.mkdir(parents=True)
    report = {
        "diagnostic_type": "retrospective_V5_error_and_history_support",
        "models_changed": False, "submission_created": False,
        "public_score_last_reported_by_user": 0.819402,
        "protocol": {
            "resamples": args.resamples, "bootstrap_seed": 2026,
            "sampling_unit": "whole first event per user; retain all 10 candidate rows before cohort selection",
            "minimum_cohort_bootstrap": "300 events, 30 clicks, 100 nonclicks",
            "pair_work_limit_per_case": 100000000,
            "support_thresholds_other_fitting_users": list(THRESHOLDS),
            "cohorts_overlap": True,
            "caveat": "subgroup AUCs are not additive contributions to global AUC; repeated development is not corrected by these intervals",
            "pair_caveat": "co-click coverage is descriptive, not validation of a new predictor or a directed transition model",
        },
        "cases": {},
    }
    for case, filename, fold, cutoff, expected_v5, expected_v4 in [
        ("two_day_primary", "two_day", "jan25", 1737763200, 0.8183722315263706, 0.8168650717272858),
        ("jan26_secondary", "jan26", "jan26", 1737849600, 0.8269940935263441, 0.8260509325478992),
    ]:
        print(f"\nAuditing {case}...", flush=True)
        predictions_path = trial / "work" / f"{filename}_ensemble_predictions.npz"
        input_paths.add(predictions_path.resolve())
        with np.load(predictions_path) as saved:
            indices, labels = saved["valid_idx"], saved["labels"]
            p4 = require_probabilities(saved["baseline"], labels.shape)
            p5 = require_probabilities(saved["candidate"], labels.shape)
        assert np.array_equal(indices, first_indices(raw, cutoff, END_TRAIN))
        assert np.array_equal(labels, raw["y"][indices])
        assert abs(auc(labels, p5) - expected_v5) < 1e-10
        assert abs(auc(labels, p4) - expected_v4) < 1e-10
        assert abs(expected_v5 - trial_report[case]["candidate_five_model_auc"]) < 1e-10
        folder = trial / fold / "data/cache"
        split_path, positive_path = folder / f"split_{fold}.npz", folder / "preference_training.npz"
        input_paths.update([split_path.resolve(), positive_path.resolve()])
        with np.load(split_path) as saved:
            fit_idx = saved["train_idx"]
        assert np.array_equal(fit_idx, np.flatnonzero(raw["t"] < cutoff))
        positive_users, positive_items, n_users, n_items = read_positive_cache(positive_path, cutoff)
        target = {name: raw[name][indices] for name in ["u", "i", "t", "h", "ht"]}
        check_target(target, n_users, n_items)
        fit_counts = np.bincount(raw["u"][fit_idx], minlength=n_users)
        exposure = np.bincount(raw["i"][fit_idx].ravel(), minlength=n_items)
        # Save error diagnostics first, so an unusually expensive pair audit cannot lose them.
        groups, exposure_thresholds = group_definitions(target, fit_counts, exposure, cutoff)
        print("  Computing error groups and paired intervals...", flush=True)
        metrics = group_reports(labels, p4, p5, groups, args.resamples, 2026)
        report["cases"][case] = {
            "fitting_cutoff_unix": cutoff, "fitting_events": len(fit_idx),
            "evaluation_labels_available": True,
            "population": population(target, fit_counts, p4, p5, cutoff),
            "exposure_thresholds": exposure_thresholds, "groups": metrics,
        }
        (output / "error_report_partial.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("  Computing exact other-user history support...", flush=True)
        maximum, has_query, support_report = history_support(positive_users, positive_items, target, n_items, labels, progress=True)
        support_groups, _ = group_definitions(target, fit_counts, exposure, cutoff, maximum, has_query)
        support_groups = {name: mask for name, mask in support_groups.items() if name.startswith("cross_item_")}
        metrics.update(group_reports(labels, p4, p5, support_groups, args.resamples, 2026))
        report["cases"][case]["history_support"] = support_report
        (output / "error_report_partial.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        del maximum, positive_users, positive_items

    print("\nProfiling official evaluation (no labels)...", flush=True)
    evaluation_path = root / "data/cache/evaluation_events.npz"
    v4_path, v5_path = root / "work/preference_finetuned_evaluation.npz", root / "work/five_model_evaluation.npz"
    positive_path = root / "data/cache/preference_training.npz"
    input_paths.update(path.resolve() for path in [evaluation_path, v4_path, v5_path, positive_path])
    with np.load(evaluation_path) as saved:
        assert "y" not in saved.files
        evaluation = {name: saved[name] for name in ["u", "i", "t", "h", "ht"]}
    assert len(evaluation["u"]) == len(np.unique(evaluation["u"])) == 52939
    assert (evaluation["t"] >= END_TRAIN).all() and (evaluation["t"] < END_TRAIN + 7 * DAY).all()
    with np.load(v4_path) as saved:
        p4 = require_probabilities(saved["predictions"], (52939, 10))
    with np.load(v5_path) as saved:
        p5 = require_probabilities(saved["predictions"], (52939, 10))
        assert np.array_equal(saved["weights"], checks["weights"])
    positive_users, positive_items, n_users, n_items = read_positive_cache(positive_path, END_TRAIN)
    check_target(evaluation, n_users, n_items)
    maximum, has_query, support_report = history_support(positive_users, positive_items, evaluation, n_items, progress=True)
    fit_counts = np.bincount(raw["u"], minlength=n_users)
    exposure = np.bincount(raw["i"].ravel(), minlength=n_items)
    groups, thresholds = group_definitions(evaluation, fit_counts, exposure, END_TRAIN, maximum, has_query)
    profile = population(evaluation, fit_counts, p4, p5, END_TRAIN)
    report["official_evaluation"] = {
        "evaluation_labels_available": False, "auc": None,
        "population": profile, "history_support": support_report,
        "exposure_thresholds": thresholds,
        "cohort_row_fractions": {name: len(selected_rows(mask, 10)) / evaluation["i"].size for name, mask in groups.items()},
    }
    report["input_sha256"] = {str(path): digest(path) for path in sorted(input_paths)}
    report["script_sha256"] = digest(Path(__file__))
    report["runtime_seconds"] = round(time.perf_counter() - start, 2)
    full_path, csv_path, summary_path = output / "audit_full.json", output / "cohort_metrics.csv", output / "audit_summary.json"
    full_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_cohort_csv(csv_path, report["cases"])
    key_groups = ["unseen_with_history", "unseen_empty_history", "activity_1_5", "activity_6_15",
                  "activity_16_plus", "history_empty", "history_1_5", "history_20",
                  "last_click_72h_plus_nonempty", "nonrepeat_candidates", "horizon_24_48h",
                  "cross_item_support_0", "cross_item_support_3_plus"]
    primary_groups = report["cases"]["two_day_primary"]["groups"]
    summary = {
        "diagnostic_type": report["diagnostic_type"], "models_changed": False,
        "validation_source_checks_passed": True,
        "validation_summary": {case: data["groups"]["all"] for case, data in report["cases"].items()},
        "primary_key_groups": {name: {k: primary_groups[name][k] for k in [
            "events_with_selected_rows", "rows", "clicks", "v5_auc", "auc_change", "paired_gain_95_percentile_interval"]}
            for name in key_groups if name in primary_groups},
        "history_support": {case: compact_support(data["history_support"]) for case, data in report["cases"].items()},
        "official_evaluation_history_support": compact_support(support_report),
        "population_comparison": {**{case: data["population"] for case, data in report["cases"].items()}, "official_evaluation": profile},
        "reports": {"full_json": str(full_path), "cohort_csv": str(csv_path), "summary_json": str(summary_path)},
        "interpretation_note": "retrospective diagnostics; coverage is not predictive lift; evaluation has no AUC; cohort AUCs cannot be added",
        "runtime_seconds": report["runtime_seconds"],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nV5 ERROR AND HISTORY-SUPPORT AUDIT", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
