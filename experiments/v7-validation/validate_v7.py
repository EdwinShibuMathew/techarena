"""Freeze and reproduce the single V7 sharpening recipe; no fitting or upload."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "experiments/v6-pairlogit-v1-api-fix"))
import event_scale_diagnostic as audit
from experiment_runner import PairedBootstrapEvaluator, compute_roc_auc
from predict_v7 import calibrate_v7_probabilities, generate

TRAIN_SHA = "a8fcada9cf0d08f900824594302b2ac45b90254d1780c9d915206ec96cc062b8"
V6_SHA = "c386d545b3746e05dd515020c6185afda06cefdbcdf7d48df022454762fb378c"
V7_SHA = "9aebf08df53b9b1d0661479151c37118859831b3b4c9d371d533c3bf203b141a"
GAMMA, RESAMPLES, SEED = 1.12, 1000, 2026
SOURCE_FILES = (
    "experiments/v7-validation/validate_v7.py", "tools/experiment_runner.py",
    "tools/predict_v7.py", "configs/v6.json", "configs/source_manifest.json",
    "experiments/v6-pairlogit-v1-api-fix/event_scale_diagnostic.py",
)
ARCHIVES = {
    "primary": "local/archive/techarena-row-auc-trial/work/two_day_ensemble_predictions.npz",
    "secondary": "local/archive/techarena-row-auc-trial/work/jan26_ensemble_predictions.npz",
    "five_day": "local/archive/techarena-row-auc-five-day/work/five_day_paired_predictions.npz",
}
PROTOCOL = {
    "experiment": "v7-validation", "gamma": GAMMA,
    "hypothesis": "Fixed power sharpening improves global row AUC while preserving V6 event sums and within-event order",
    "comparator": "Unchanged hash-verified V6 candidate arrays, first event per user",
    "method": "p**gamma / sum(p**gamma, axis=1) * sum(p, axis=1)",
    "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "unit": "whole user",
                  "sampling": "multinomial user multiplicities, shared by candidate and V6",
                  "interval": "95% percentile", "minimum_valid_fraction": .99},
    "gates": "primary CI lower > 0; secondary change >= 0; five-day CI lower > 0; late 48-120h change >= 0",
    "stop": "One fixed recipe; hash mismatch or failed gate does not permit tuning gamma, seeds, weights or cohorts",
    "status": "reused retrospective development dates; not independent holdout or selection-adjusted evidence",
    "model_fitting": False, "competition_upload": False,
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def verify_inputs(repo, inputs):
    for relative, expected in inputs.items():
        if audit.sha256(repo / relative) != expected:
            raise ValueError(f"Frozen input changed: {relative}")


def prepare(repo, source_root):
    """Read only the three explicitly approved archive paths, without discovery."""
    repo, source_root = Path(repo).resolve(strict=True), Path(source_root).resolve(strict=True)
    work = repo / "artifacts/experiments/v7-validation"
    if work.exists():
        raise FileExistsError(f"Preserve existing validation inputs: {work}")
    for key, relative in ARCHIVES.items():
        path = source_root / relative
        if path.resolve(strict=True) != path or audit.sha256(path) != audit.SPECS[key]["sha256"]:
            raise ValueError(f"Original archive path/hash differs: {relative}")
    if audit.sha256(repo / "data/cache/train_events.npz") != TRAIN_SHA:
        raise ValueError("Training cache differs from the frozen historical input")
    inputs = {name: audit.sha256(repo / name) for name in SOURCE_FILES}
    inputs["data/cache/train_events.npz"] = TRAIN_SHA
    for name in ("data/cache/evaluation_rowids.npy", "data/raw/example_submission.csv"):
        inputs[name] = audit.sha256(repo / name)
    baseline = repo / "artifacts/submissions/v6.csv"
    if not baseline.is_file() or audit.sha256(baseline) != V6_SHA:
        raise ValueError("Stage the protected V6 CSV at artifacts/submissions/v6.csv")
    inputs["artifacts/submissions/v6.csv"] = V6_SHA
    source = work / "source"
    source.mkdir(parents=True)
    cohorts = {}
    for key, relative in ARCHIVES.items():
        destination = source / audit.SPECS[key]["filename"]
        with (source_root / relative).open("rb") as original, destination.open("xb") as target:
            shutil.copyfileobj(original, target)
        staged = destination.relative_to(repo).as_posix()
        inputs[staged] = audit.SPECS[key]["sha256"]
        cohorts[key] = staged
    verify_inputs(repo, inputs)
    audit.write_new(work / "inputs.json", {"protocol": PROTOCOL, "protocol_sha256": canonical_hash(PROTOCOL),
                                          "inputs_sha256": inputs, "cohorts": cohorts})
    print("Frozen the one V7 recipe and three verified cohorts; no candidate was scored.", flush=True)


def evaluate_cohort(raw, predictions, spec):
    idx, labels, baseline, _, _ = audit.load_cohort(predictions, spec, raw)
    candidate = calibrate_v7_probabilities(baseline, GAMMA)
    metrics = asdict(PairedBootstrapEvaluator(RESAMPLES, SEED).evaluate(
        candidate, baseline, labels, raw["u"][idx]))
    error = float(np.max(np.abs(candidate.sum(axis=1) - baseline.sum(axis=1))))
    if error > 1e-14:
        raise ValueError("V7 changed event sums")
    for i in range(10):
        for j in range(i):
            if not np.array_equal(np.sign(candidate[:, i] - candidate[:, j]),
                                  np.sign(baseline[:, i] - baseline[:, j])):
                raise ValueError("V7 changed a within-event ordering or tie")
    result = {"events": len(idx), "rows": int(labels.size), **metrics,
              "maximum_event_sum_error": error, "within_event_order_preserved": True,
              "candidate_minimum": float(candidate.min()), "candidate_maximum": float(candidate.max())}
    return result, idx, labels, baseline, candidate


def run(repo, output):
    repo = Path(repo).resolve(strict=True)
    output = audit.inside(output, repo / "docs/results/v7-validation")
    if output.exists():
        raise FileExistsError(f"Preserve existing validation report: {output}")
    manifest_path = repo / "artifacts/experiments/v7-validation/inputs.json"
    manifest_hash = audit.sha256(manifest_path)
    frozen = json.loads(manifest_path.read_text())
    if frozen["protocol"] != PROTOCOL or frozen["protocol_sha256"] != canonical_hash(PROTOCOL):
        raise ValueError("V7 recipe differs from the frozen protocol")
    if set(frozen["cohorts"]) != set(audit.SPECS):
        raise ValueError("Validation requires exactly the three fixed cohorts")
    inputs = frozen["inputs_sha256"]
    for name in SOURCE_FILES:
        if name not in inputs:
            raise ValueError(f"Unfrozen validation source: {name}")
    if inputs.get("data/cache/train_events.npz") != TRAIN_SHA:
        raise ValueError("Unfrozen training data")
    for key, relative in frozen["cohorts"].items():
        audit.inside(repo / relative, repo / "artifacts/experiments/v7-validation/source")
        if inputs.get(relative) != audit.SPECS[key]["sha256"]:
            raise ValueError("Cohort is not bound to the historical hash")
    verify_inputs(repo, inputs)
    start = time.monotonic()
    with np.load(repo / "data/cache/train_events.npz", allow_pickle=False) as data:
        raw = {name: data[name] for name in ("u", "t", "y")}
    if (raw["u"].shape != (audit.TRAIN_EVENTS,) or raw["t"].shape != (audit.TRAIN_EVENTS,)
            or raw["y"].shape != (audit.TRAIN_EVENTS, 10) or int(raw["y"].sum()) != audit.TRAIN_CLICKS):
        raise ValueError("Historical training cache shape/click contract differs")
    results = {}
    for key, spec in audit.SPECS.items():
        result, idx, y, baseline, candidate = evaluate_cohort(raw, repo / frozen["cohorts"][key], spec)
        if key == "five_day":
            hours = (raw["t"][idx] - spec["begin"]) / 3600
            late = (hours >= 48) & (hours < 120)
            if int(late.sum()) != audit.LATE_EVENTS:
                raise ValueError("Historical five-day late cohort differs")
            before, after = compute_roc_auc(y[late], baseline[late]), compute_roc_auc(y[late], candidate[late])
            result["late_48_120h"] = {"events": int(late.sum()), "baseline_auc": before,
                                       "candidate_auc": after, "auc_diff": after - before}
        results[key] = result
        print(json.dumps({key: result}, indent=2), flush=True)
    gates = {"primary_ci_lower_positive": results["primary"]["ci_lower"] > 0,
             "secondary_change_nonnegative": results["secondary"]["auc_diff"] >= 0,
             "five_day_ci_lower_positive": results["five_day"]["ci_lower"] > 0,
             "five_day_late_change_nonnegative": results["five_day"]["late_48_120h"]["auc_diff"] >= 0}
    generated = repo / "artifacts/experiments/v7-validation/v7_candidate.csv"
    receipt = generate(generated, root=repo, gamma=GAMMA)
    if receipt["submission_sha256"] != V7_SHA or receipt["baseline_sha256"] != V6_SHA:
        raise ValueError("Generated V7 fingerprint does not match the fixed candidate")
    verify_inputs(repo, inputs)
    if audit.sha256(manifest_path) != manifest_hash:
        raise ValueError("Frozen manifest changed during evaluation")
    report = {"complete": True, "protocol": PROTOCOL, "protocol_sha256": canonical_hash(PROTOCOL),
              "inputs_sha256": inputs, "manifest_sha256": manifest_hash, "results": results,
              "decision_checks": gates, "promotion_supported": all(gates.values()),
              "submission_verification": {**receipt, "fingerprint_verified": True,
                                           "output_relative": generated.relative_to(repo).as_posix()},
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "seconds": round(time.monotonic() - start, 3)},
              "model_fitting_performed": False, "competition_upload_performed": False}
    audit.write_new(output, report)
    print(json.dumps({"promotion_supported": report["promotion_supported"], "report": str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "run"):
        command = commands.add_parser(action)
        command.add_argument("--repo", type=Path, default=ROOT)
        if action == "prepare":
            command.add_argument("--source-root", type=Path, required=True,
                                 help="Repository containing the three explicitly approved archive paths")
        else:
            command.add_argument("--output", type=Path, default=ROOT / "docs/results/v7-validation/validation.json")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.repo, args.source_root)
    else:
        run(args.repo, args.output)


if __name__ == "__main__":
    main()
