"""Descriptive strata for completed V8 predictions; never select a recipe or gate."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import threading
import time
import numpy as np

import v8_core as core

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from experiment_runner import FastAuc


def remaining_verification_seconds(state):
    used = state["phases"].get("verify", {}).get("seconds", 0.) + state.get("supplementary_verification_seconds", 0.)
    return min(core.PHASE_LIMITS["verify"] - used, core.PROTOCOL["total_compute_seconds"] - state["used_seconds"])


@contextmanager
def verification_budget(work):
    """Charge supplementary diagnostics to the same verification/global allowance."""
    path = work / "budget.json"
    state = core.read_json(path)
    core.require(all(value["status"] != "running" for value in state["phases"].values()), "Wait for model stages to finish")
    limit = remaining_verification_seconds(state)
    core.require(limit > 0, "Verification/global budget exhausted")
    start = time.monotonic()
    lock = threading.Lock()
    def record(status):
        elapsed = time.monotonic() - start
        state["used_seconds"] += elapsed
        state["supplementary_verification_seconds"] = state.get("supplementary_verification_seconds", 0.) + elapsed
        state["supplementary_verification_status"] = status
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(path)
    def expire():
        with lock:
            record("timeout")
        os._exit(124)
    timer = threading.Timer(limit, expire)
    timer.daemon = True
    timer.start()
    status = "failed"
    try:
        yield
        status = "complete"
    finally:
        timer.cancel()
        with lock:
            record(status)


def metrics(labels, baseline, candidate, mask):
    y = labels[mask].ravel()
    result = {"events": int(mask.sum()), "rows": int(y.size), "clicks": int(y.sum())}
    if not len(y) or len(np.unique(y)) != 2:
        return {**result, "auc_defined": False}
    ids = np.zeros(len(y), dtype=np.int64)
    a = FastAuc(y, baseline[mask].ravel(), ids).value(np.ones(1))
    b = FastAuc(y, candidate[mask].ravel(), ids).value(np.ones(1))
    return {**result, "auc_defined": True, "baseline_auc": a, "candidate_auc": b, "auc_diff": b - a}


def run(root=ROOT):
    root = Path(root).resolve()
    work = root / "artifacts/experiments/v8-pairlogit"
    output = root / "docs/results/v8-pairlogit/diagnostics.json"
    core.require(not output.exists(), "Preserve existing diagnostic report")
    with verification_budget(work):
        manifest = core.load_manifest(root)
        with np.load(root / "data/cache/train_events.npz", allow_pickle=False) as data:
            raw = {name: data[name] for name in ("u", "t", "h", "y")}
        results, hashes = {}, {}
        for key, cutoff in (("primary", core.FOLDS["jan25"]), ("secondary", core.FOLDS["jan26"]), ("five_day", core.FOLDS["jan22"])):
            path = work / (key + "_predictions.npz")
            if not path.exists():
                continue
            stage = "stress" if key == "five_day" else "screen"
            report_path = root / "docs/results/v8-pairlogit" / (stage + ".json")
            report = core.read_json(report_path)
            core.require(report.get("complete") is True and report["protocol_sha256"] == core.canonical(core.PROTOCOL), "Unfinished/different model stage")
            result = report["results"] if stage == "stress" else report["results"][key]
            core.require(core.sha256(path) == result["predictions_sha256"], "Predictions changed after evaluation")
            hashes[key] = core.sha256(path)
            with np.load(path, allow_pickle=False) as data:
                idx, y, baseline, candidate = [data[name] for name in ("valid_idx", "labels", "baseline", "candidate")]
            core.require(np.array_equal(y, raw["y"][idx]), "Diagnostic labels do not align")
            warm_users = np.unique(raw["u"][raw["t"] < cutoff])
            cold = ~np.isin(raw["u"][idx], warm_users)
            history = (raw["h"][idx] >= 0).sum(axis=1)
            days = ((raw["t"][idx] - cutoff) // core.DAY).astype(int)
            masks = {"cold_users": cold, "warm_users": ~cold, "history_empty": history == 0,
                     "history_1_5": (history >= 1) & (history <= 5), "history_6_20": history >= 6}
            masks.update({f"day_{day}": days == day for day in np.unique(days)})
            results[key] = {name: metrics(y, baseline, candidate, mask) for name, mask in masks.items()}
        core.require(bool(results), "No completed predictions available")
        core.load_manifest(root)
        core.write_new(output, {"complete": True, "descriptive_only": True, "used_for_selection_or_routing": False,
            "protocol_sha256": core.canonical(core.PROTOCOL), "source_sha256": core.sha256(Path(__file__)),
            "prediction_sha256": hashes, "results": results,
            "unrun_cohorts": sorted(set(manifest["cohorts"]) - set(results))})
    print("Descriptive diagnostics saved:", output, flush=True)


if __name__ == "__main__":
    run()
