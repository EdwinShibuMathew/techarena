"""Frozen, CPU-only V6 event-scale diagnostic. Requires only existing NumPy.

Read prediction archives and train_events.npz; never fit, infer, or export a
submission. Commands: self-test, locate, run. See RUN_EVENT_SCALE.md.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


EXPERIMENT = "v6-event-scale-v1"
RESAMPLES, SEED, BUDGET_SECONDS = 1000, 2026, 1800
TRAIN_EVENTS, TRAIN_CLICKS, LATE_EVENTS = 686824, 432023, 7596
JAN20, DAY = 1737331200, 86400
END = JAN20 + 7 * DAY
SPECS = {
    "primary": {
        "filename": "two_day_ensemble_predictions.npz",
        "sha256": "a84d3ee0775b33ec6be78cf8f9a5e2a9695d020920d2647a0e52fc1a92649c12",
        "begin": JAN20 + 5 * DAY,
        "events": 43964,
        "auc": 0.8184592533262289,
    },
    "secondary": {
        "filename": "jan26_ensemble_predictions.npz",
        "sha256": "507b3c0a24574283fee9291540ce00e248e98b3704376f0b0eb51039012adeb2",
        "begin": JAN20 + 6 * DAY,
        "events": 35138,
        "auc": 0.8271277122502162,
    },
    "five_day": {
        "filename": "five_day_paired_predictions.npz",
        "sha256": "f854b8980f6455c6f33de4684d4692c5540652bf1cce7398d06e5e5dbe5be446",
        "begin": JAN20 + 2 * DAY,
        "events": 51488,
        "auc": 0.7818190858952342,
    },
}
PROTOCOL = {
    "experiment": EXPERIMENT,
    "comparator": "candidate array in the three hash-verified historical archives = V6",
    "diagnostic": "q[e,i] = p[e,i] / sum_j(p[e,j]); no clipping or fitted parameters",
    "metric": "global row ROC AUC, half credit for ties; all ten candidates",
    "bootstrap": {"resamples": RESAMPLES, "seed": SEED,
                  "unit": "whole first event per user", "interval": "95% percentile",
                  "pairing": "identical event multiplicities for V6 and normalized scores",
                  "five_day_late": "same whole-cohort draws, restricted to 48 <= hours < 120"},
    "gate": "primary interval lower > 0; secondary point change >= 0; five-day interval lower > 0; late point change >= 0",
    "gate_meaning": "support investigating event-scale correction, never automatic promotion",
    "stop": "one fixed normalization; no sweeps, routing, training, inference or submission; 1800 seconds from run start",
    "validation_status": "reused retrospective development dates; intervals not selection-adjusted",
    "expected_prediction_inputs": SPECS,
    "training_cache_contract": {"events": TRAIN_EVENTS, "clicks": TRAIN_CLICKS,
                                "five_day_late_events": LATE_EVENTS},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def inside(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    require(path.is_relative_to(parent), f"Path must stay inside {parent}: {path}")
    return path


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


class AucOrder:
    """Exact, tie-aware global row AUC with event multiplicities.

    Sorting once makes whole-event bootstrap feasible without pair matrices.
    The implementation is independently checked against explicit pair counting.
    """

    def __init__(self, labels, scores, event_ids):
        y, p, ids = [np.asarray(x).ravel() for x in (labels, scores, event_ids)]
        require(y.shape == p.shape == ids.shape, "AUC input shapes differ")
        require(np.isfinite(p).all() and np.isin(y, [0, 1]).all(), "Invalid AUC inputs")
        order = np.argsort(p, kind="stable")
        p = p[order]
        self.y = y[order].astype(bool)
        self.ids = ids[order].astype(np.int64)
        self.starts = None
        if len(p) > 1 and np.any(p[1:] == p[:-1]):
            self.starts = np.r_[0, np.flatnonzero(p[1:] != p[:-1]) + 1]

    def value(self, counts):
        weight = counts[self.ids].astype(np.float64)
        positive, negative = weight * self.y, weight * ~self.y
        if self.starts is not None:
            positive = np.add.reduceat(positive, self.starts)
            negative = np.add.reduceat(negative, self.starts)
        total_positive, total_negative = positive.sum(), negative.sum()
        if total_positive == 0 or total_negative == 0:
            return None
        return float(np.sum(positive * (np.cumsum(negative) - .5 * negative))
                     / (total_positive * total_negative))


def auc(y, p):
    y, p = np.asarray(y).ravel(), np.asarray(p).ravel()
    return AucOrder(y, p, np.zeros(len(y), dtype=np.int64)).value(np.ones(1))


def normalize(p):
    p = np.asarray(p, dtype=np.float64)
    require(p.ndim == 2 and p.shape[1] == 10, "Expected ten scores per event")
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid V6 probabilities")
    sums = p.sum(axis=1)
    require((sums > 0).all(), "Zero event sum; no fallback is allowed")
    q = p / sums[:, None]
    require(np.isfinite(q).all(), "Nonfinite normalized scores")
    require(np.allclose(q.sum(1), 1, rtol=0, atol=1e-14), "Normalization failed")
    # Include ties, not just argmax: finite precision must not change any order.
    for i in range(10):
        for j in range(i):
            require(np.array_equal(np.sign(p[:, i] - p[:, j]), np.sign(q[:, i] - q[:, j])),
                    "Normalization changed an ordering or tie within an event")
    return q, sums


def metrics(y, p, q):
    before, after = auc(y, p), auc(y, q)
    require(before is not None and after is not None, "Cohort needs positives and negatives")
    return {"events": len(y), "rows": int(y.size), "clicks": int(y.sum()),
            "v6_auc": before, "normalized_auc": after, "auc_change": after - before}


def decomposition(y, p, q):
    """Partition all positive-negative pairs by the negative event's outcome.

    Every positive is compared with every negative in each partition, including
    other events. These are not per-event AUC averages.
    """
    positive = y.astype(bool)
    clicked_event = np.broadcast_to(y.sum(1)[:, None] > 0, y.shape)
    negative_count = int((~positive).sum())
    parts = {}
    for name, neg in (("negative_from_clicked_event", ~positive & clicked_event),
                      ("negative_from_no_click_event", ~positive & ~clicked_event)):
        n = int(neg.sum())
        mask = positive | neg
        weight = n / negative_count
        before, after = (auc(y[mask], x[mask]) if n else None for x in (p, q))
        parts[name] = {"negative_rows": n, "positive_negative_pairs": int(y.sum()) * n,
                       "pair_weight": weight, "v6_auc": before, "normalized_auc": after,
                       "weighted_auc_change": weight * (after - before) if n else 0.0}
    for name, x in (("v6", p), ("normalized", q)):
        rebuilt = sum(v["pair_weight"] * v[f"{name}_auc"]
                      for v in parts.values() if v["negative_rows"])
        require(abs(rebuilt - auc(y, x)) < 1e-12, "AUC decomposition does not reconstruct global AUC")
    # These pairs are a tiny subset of the global metric. Report their exact
    # contribution separately so clicked-event negatives are not misread as
    # exclusively within-event comparisons.
    rows, columns = np.nonzero(y)
    negative_p, negative_q = p[rows], q[rows]
    hit_p, hit_q = p[rows, columns, None], q[rows, columns, None]
    mask = ~y[rows].astype(bool)
    win_p = ((hit_p > negative_p) + .5 * (hit_p == negative_p))[mask].mean()
    win_q = ((hit_q > negative_q) + .5 * (hit_q == negative_q))[mask].mean()
    require(win_p == win_q, "Same-event concordance changed")
    return {"negative_event_partitions": parts,
            "same_event_pairs_subset": {"auc": float(win_p),
                "pair_weight_in_global_auc": 9 / negative_count,
                "unchanged_by_normalization": True}}


def bootstrap(y, p, q, masks, check, resamples=None, progress=True):
    if resamples is None:
        resamples = RESAMPLES
    n = len(y)
    ids = np.broadcast_to(np.arange(n)[:, None], y.shape)
    orders = {name: [AucOrder(y[mask], x[mask], ids[mask]) for x in (p, q)]
              for name, mask in masks.items()}
    changes = {name: [] for name in masks}
    rng = np.random.default_rng(SEED)
    for repetition in range(resamples):
        check()
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        for name, (before, after) in orders.items():
            a, b = before.value(counts), after.value(counts)
            if a is not None and b is not None:
                changes[name].append(b - a)
        if progress and (repetition + 1) % 100 == 0:
            print(f"  paired user resamples {repetition + 1}/{resamples}", flush=True)
    result = {}
    for name, values in changes.items():
        require(len(values) >= .99 * resamples, f"Too few valid resamples: {name}")
        result[name] = {"paired_gain_95_percentile_interval": np.quantile(values, [.025, .975]).tolist(),
                        "valid_resamples": len(values)}
    return result


def first_indices(users, times, begin):
    eligible = np.flatnonzero((times >= begin) & (times < END))
    ordered = eligible[np.argsort(times[eligible], kind="stable")]
    _, positions = np.unique(users[ordered], return_index=True)
    return np.sort(ordered[positions])


def load_cohort(path, spec, raw):
    require(sha256(path) == spec["sha256"], f"Historical prediction hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as saved:
        idx, y, p = [saved[k].copy() for k in ("valid_idx", "labels", "candidate")]
    require(idx.ndim == 1 and idx.dtype.kind in "iu", "Invalid event indices")
    require(len(idx) == spec["events"] and len(np.unique(idx)) == len(idx), "Wrong cohort size/duplicate indices")
    require(((idx >= 0) & (idx < len(raw["u"]))).all(), "Event indices out of bounds")
    require(np.array_equal(np.sort(idx), first_indices(raw["u"], raw["t"], spec["begin"])),
            "Cohort differs from first events per user in the frozen window")
    require(len(np.unique(raw["u"][idx])) == len(idx), "A user contributes multiple events")
    require(y.shape == p.shape == (len(idx), 10), "Wrong labels/prediction shape")
    require(np.array_equal(y, raw["y"][idx]), "Prediction labels do not align with training cache")
    require(np.isin(y, [0, 1]).all() and (y.sum(1) <= 1).all(), "Invalid event labels")
    q, sums = normalize(p)
    require(abs(auc(y, p) - spec["auc"]) < 1e-12, "V6 AUC does not match the handoff")
    return idx, y, p, q, sums


def locate(args):
    repo = args.repo.resolve(strict=True)
    manifest = inside(args.manifest, repo / "artifacts/experiments" / EXPERIMENT)
    require(not manifest.exists(), f"Manifest exists; inspect and reuse it: {manifest}")
    candidates = {key: [] for key in SPECS}
    names = {s["filename"]: key for key, s in SPECS.items()}
    # Scan the consolidated repository only; never search WSL sibling projects.
    for directory, subdirs, files in os.walk(repo, followlinks=False):
        subdirs[:] = sorted(d for d in subdirs if d not in {".git", ".venv", "node_modules", "team-export"})
        for name in sorted(set(files) & set(names)):
            path = Path(directory) / name
            try:
                resolved = inside(path, repo)
                found_hash = sha256(resolved)
                candidates[names[name]].append({"path": str(resolved), "sha256": found_hash})
                print(f"{names[name]}: {path}\n  SHA256 {found_hash}", flush=True)
            except (OSError, ValueError) as error:
                print(f"Unavailable candidate {path}: {error}", file=sys.stderr, flush=True)
    selections = {}
    for key, spec in SPECS.items():
        matches = sorted({v["path"] for v in candidates[key] if v["sha256"] == spec["sha256"]})
        require(matches, f"No hash-matching {key} archive found inside {repo}. Expected {spec['filename']} / {spec['sha256']}")
        selections[key] = {"path": matches[0], "sha256": spec["sha256"], "identical_copies_found": matches}
    events = inside(repo / "data/cache/train_events.npz", repo)
    require(events.is_file(), f"Missing active training cache: {events}")
    selections["training_events"] = {"path": str(events), "sha256": sha256(events)}
    record = {"experiment": EXPERIMENT, "repo": str(repo), "script_sha256": sha256(__file__),
              "protocol_sha256": canonical_hash(PROTOCOL), "inputs": selections}
    write_new(manifest, record)
    print(f"Hash-matching inputs located. Manifest: {manifest}", flush=True)


def run(args):
    start = time.monotonic()

    def check():
        if time.monotonic() - start > BUDGET_SECONDS:
            raise TimeoutError("Fixed 30-minute budget exceeded; no complete result or decision written")

    manifest_path = args.manifest.resolve(strict=True)
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo = Path(manifest["repo"]).resolve(strict=True)
    inside(manifest_path, repo / "artifacts/experiments" / EXPERIMENT)
    output = inside(args.output, repo / "docs/results" / EXPERIMENT)
    require(not output.exists(), f"Result exists; refusing to overwrite: {output}")
    require(manifest["experiment"] == EXPERIMENT and manifest["protocol_sha256"] == canonical_hash(PROTOCOL),
            "Manifest protocol differs")
    require(manifest["script_sha256"] == sha256(__file__), "Diagnostic source changed after locate")
    require(set(manifest["inputs"]) == set(SPECS) | {"training_events"}, "Unexpected input manifest entries")
    paths = {}
    for key, info in manifest["inputs"].items():
        path = inside(info["path"], repo)
        require(sha256(path) == info["sha256"], f"Input changed since locate: {path}")
        if key in SPECS:
            require(info["sha256"] == SPECS[key]["sha256"], "Manifest cannot replace a historical prediction hash")
        paths[key] = path
        check()
    # Read only the three fields needed for cohort, labels and late-window checks.
    with np.load(paths["training_events"], allow_pickle=False) as saved:
        raw = {k: saved[k] for k in ("u", "t", "y")}
    require(raw["u"].shape == raw["t"].shape == (TRAIN_EVENTS,), "Wrong training cache length")
    require(raw["y"].shape == (TRAIN_EVENTS, 10) and int(raw["y"].sum()) == TRAIN_CLICKS, "Wrong training labels")
    require(np.isin(raw["y"], [0, 1]).all() and (raw["y"].sum(1) <= 1).all(), "Invalid training labels")
    require(((raw["t"] >= JAN20) & (raw["t"] < END)).all(), "Wrong training timeline")
    check()
    results = {}
    for key, spec in SPECS.items():
        print(f"Checking and evaluating {key}...", flush=True)
        idx, y, p, q, sums = load_cohort(paths[key], spec, raw)
        check()
        masks = {"overall": np.ones(len(idx), dtype=bool)}
        if key == "five_day":
            hours = (raw["t"][idx] - spec["begin"]) / 3600
            masks["late_48_120h"] = (hours >= 48) & (hours < 120)
            require(int(masks["late_48_120h"].sum()) == LATE_EVENTS, "Late-window cohort does not match handoff")
        result = {}
        for name, mask in masks.items():
            result[name] = metrics(y[mask], p[mask], q[mask])
            result[name]["decomposition"] = decomposition(y[mask], p[mask], q[mask])
        result["event_score_sums"] = {
            "quantile_levels": [0, .01, .1, .5, .9, .99, 1],
            "quantiles": np.quantile(sums, [0, .01, .1, .5, .9, .99, 1]).tolist(),
            "mean": float(sums.mean()), "fraction_above_one": float((sums > 1).mean()),
            "within_event_order_and_ties_unchanged": True,
        }
        print(json.dumps({name: {k: v for k, v in value.items() if k != "decomposition"}
                          for name, value in result.items() if name in masks}, indent=2), flush=True)
        intervals = bootstrap(y, p, q, masks, check)
        for name, interval in intervals.items():
            result[name].update(interval)
        results[key] = result
        del y, p, q, sums
        check()
    checks = {
        "primary_interval_entirely_positive": results["primary"]["overall"]["paired_gain_95_percentile_interval"][0] > 0,
        "secondary_point_change_nonnegative": results["secondary"]["overall"]["auc_change"] >= 0,
        "five_day_interval_entirely_positive": results["five_day"]["overall"]["paired_gain_95_percentile_interval"][0] > 0,
        "late_point_change_nonnegative": results["five_day"]["late_48_120h"]["auc_change"] >= 0,
    }
    # Detect changes during the run; never update hashes to accept them.
    for key, path in paths.items():
        require(sha256(path) == manifest["inputs"][key]["sha256"], f"Input changed during run: {path}")
        check()
    require(sha256(manifest_path) == manifest_hash, "Manifest changed during run")
    require(sha256(__file__) == manifest["script_sha256"], "Source changed during run")
    check()
    report = {
        "complete": True, "protocol": PROTOCOL, "protocol_sha256": canonical_hash(PROTOCOL),
        "source_sha256": sha256(__file__), "manifest_sha256": manifest_hash,
        "inputs": manifest["inputs"], "runtime": {"python": sys.version, "numpy": np.__version__,
            "elapsed_seconds": round(time.monotonic() - start, 3)},
        "results": results, "decision_checks": checks,
        "supports_investigating_event_scale_correction": all(checks.values()),
        "interpretation": "A passed gate supports an honest temporal correction experiment, not a submission. A failed gate is insufficient support; mixed results are inconclusive. Harm from full normalization does not rule out partial scale misspecification.",
        "input_hashes_rechecked_unchanged": True,
        "training_performed": False, "evaluation_cache_read": False, "submission_created": False,
    }
    write_new(output, report)
    print(json.dumps({"decision_checks": checks,
                      "supports_investigating_event_scale_correction": all(checks.values()),
                      "report": str(output)}, indent=2), flush=True)


def dense_auc(y, p):
    y, p = np.asarray(y).ravel(), np.asarray(p).ravel()
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg) + .5 * (pos[:, None] == neg)).mean())


def self_test():
    rng = np.random.default_rng(18)
    y = np.zeros((8, 10), dtype=np.int8)
    y[[0, 2, 4, 6], [1, 2, 3, 4]] = 1
    # Deliberate score ties and no-click events.
    p = rng.integers(1, 10, size=y.shape) / 20
    q, _ = normalize(p)
    ids = np.broadcast_to(np.arange(len(y))[:, None], y.shape)
    order = AucOrder(y, p, ids)
    counts = np.array([2, 1, 0, 3, 1, 1, 2, 1])
    duplicated = np.repeat(np.arange(len(y)), counts)
    require(abs(order.value(counts) - dense_auc(y[duplicated], p[duplicated])) < 1e-14,
            "Tie-aware weighted AUC disagrees with explicit user duplication")
    require(abs(auc(y, p) - dense_auc(y, p)) < 1e-14, "AUC oracle mismatch")
    require(auc(np.zeros_like(y), p) is None, "Single-class AUC should be undefined")
    decomposition(y, p, q)
    late = np.arange(len(y)) >= 2
    masks = {"overall": np.ones(len(y), dtype=bool), "late": late}
    result = bootstrap(y, p, q, masks, lambda: None, resamples=40, progress=False)
    expected = {name: [] for name in masks}
    draw = np.random.default_rng(SEED)
    for _ in range(40):
        chosen = draw.integers(0, len(y), len(y))
        for name, mask in masks.items():
            selected = chosen[mask[chosen]]
            a, b = dense_auc(y[selected], p[selected]), dense_auc(y[selected], q[selected])
            if a is not None and b is not None:
                expected[name].append(b - a)
    for name, values in expected.items():
        np.testing.assert_allclose(result[name]["paired_gain_95_percentile_interval"],
                                   np.quantile(values, [.025, .975]), rtol=0, atol=1e-14)
    times = JAN20 + np.array([5 * DAY + 20, 5 * DAY + 10, 5 * DAY + 5, 6 * DAY + 5])
    np.testing.assert_array_equal(first_indices(np.array([1, 1, 2, 2]), times, JAN20 + 5 * DAY), [1, 2])
    try:
        normalize(np.zeros((1, 10)))
    except ValueError:
        pass
    else:
        raise AssertionError("Zero event sum accepted")
    print("Self-test passed: tie-aware AUC, exact decomposition, paired whole-event bootstrap, shared late draws, normalization and first-user selection.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    find = commands.add_parser("locate")
    find.add_argument("--repo", type=Path, required=True)
    find.add_argument("--manifest", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "locate":
        locate(args)
    else:
        run(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TimeoutError) as error:
        print(f"STOPPED: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
