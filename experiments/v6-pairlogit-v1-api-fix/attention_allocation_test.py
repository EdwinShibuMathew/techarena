"""One frozen 15% attention-allocation screen against V6. CPU/NumPy only.

prepare verifies transferred historical evidence and freezes inputs before the
new candidate is scored. run changes item allocation while preserving V6 sums.
No fitting, neural replay, evaluation cache, model writes or submission export.
"""

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

import event_scale_diagnostic as audit


EXPERIMENT = "v6-attention-allocation-v1"
WEIGHT, RESAMPLES, SEED, BUDGET = .15, 1000, 2026, 1800
ZIP_NAME = "attention-trial-handoff-20261025T172654Z-5adf3fc8.zip"
ZIP_SHA = "8f423f70f7035e78f80ab19987c1b7137cfe7aa061e3653813d1454787f5d6f0"
HELPER_SHA = "0007bc7db9fda6ab689bed23d44b4b06bb0502ff3eee0c0bd3da04f8902293d5"
TRAIN_SHA = "a8fcada9cf0d08f900824594302b2ac45b90254d1780c9d915206ec96cc062b8"
TRAIN_EVENTS, TRAIN_CLICKS = 686824, 432023
PREFIX = "attention-trial-evidence/"
ATT_FILES = {
    "outputs/frozen_protocol.json": "ed849eac078e8a093dce37ac8ea45783bde03fd5fa82aff81053327267e351ba",
    "outputs/ensemble_trial_results.json": "ae1a9f3b09021a527a030adb20da352e0c8f7ea40e2c1b543daaa906b701a5fb",
    "work/two_day_predictions.npz": "95e4929a0b55c9820eb29ecd145ce7932ebdc2cccbcf891a2b4efe3b9e75ffa7",
    "work/jan26_predictions.npz": "c674bdc396dd501c416f4fc6ac0d4e0ba30a684c2963393ac4cd54c2836261be",
}
CASES = {
    "primary": {**audit.SPECS["primary"], "attention_file": "work/two_day_predictions.npz",
                "report_key": "two_day_primary", "empty_events": 361},
    "secondary": {**audit.SPECS["secondary"], "attention_file": "work/jan26_predictions.npz",
                  "report_key": "jan26_secondary", "empty_events": 198},
}
FOLDS = {"jan24": 1737676800, "jan25": 1737763200, "jan26": 1737849600}
LEGACY_NAMES = ("attention_history_model.py", "run_attention_trial.py", "audit_v5.py",
                "submission_export.py", "train_final_models.py", "evaluate_dim16_trial.py")
RECIPE = {
    "experiment": EXPERIMENT,
    "attention_weight": WEIGHT,
    "candidate": "0.85*p_v6 + 0.15*sum(p_v6,axis=1)*attention_expert/sum(attention_expert,axis=1)",
    "eligibility": "nonempty actual event history; empty-history rows remain exactly V6",
    "expert": "calibrated attention_expert array only; attention_raw is historical verification only",
    "comparator": "unchanged V6 candidate arrays in hash-verified row-AUC archives",
    "event_sum_tolerance": {"rtol": 0, "atol": 1e-14},
    "metrics": "global row ROC AUC, all ten candidates, half credit for ties",
    "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "unit": "whole first event per user",
                  "paired": True, "interval": "95% percentile"},
    "gate": "primary paired interval lower > 0 AND secondary global AUC change >= 0",
    "stop": "one recipe; failure ends it; no weight/raw-expert/routing variants; 1800 second budget",
    "next_stage": "only a separate review for the identical recipe's earlier-cutoff five-day test; not promotion",
    "score_semantics": "AUC ranking scores; preserve event sums even if above one; never clip/export",
    "status": "post-failure mechanistic follow-up on reused development dates; intervals not selection-adjusted",
    "expected_attention_files": ATT_FILES,
    "expected_v6_cases": CASES,
}


def byte_hash(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def allocation(p, expert, eligible):
    p = np.asarray(p, dtype=np.float64)
    expert = np.asarray(expert, dtype=np.float64)
    eligible = np.asarray(eligible)
    audit.require(p.shape == expert.shape and p.ndim == 2 and p.shape[1] == 10,
                  "Expected aligned ten-candidate arrays")
    audit.require(eligible.dtype == bool and eligible.shape == (len(p),), "Invalid eligibility mask")
    for x in (p, expert):
        audit.require(np.isfinite(x).all() and ((x >= 0) & (x <= 1)).all(), "Invalid input probabilities")
    audit.require((p.sum(1) > 0).all(), "Zero V6 event sum")
    expert_sum = expert[eligible].sum(1)
    audit.require((expert_sum > 0).all(), "Zero eligible expert sum; no fallback allowed")
    result = p.copy()
    result[eligible] = ((1 - WEIGHT) * p[eligible]
                        + WEIGHT * p[eligible].sum(1)[:, None]
                        * (expert[eligible] / expert_sum[:, None]))
    audit.require(np.isfinite(result).all() and (result >= 0).all(), "Invalid allocation scores")
    audit.require(np.array_equal(result[~eligible], p[~eligible]), "Empty-history scores changed")
    error = float(np.max(np.abs(result.sum(1) - p.sum(1))))
    audit.require(error <= 1e-14, "Event sums were not preserved")
    return result, {"max_absolute_event_sum_difference": error,
                    "empty_history_events": int((~eligible).sum()),
                    "empty_history_scores_exactly_unchanged": True,
                    "eligible_events": int(eligible.sum()),
                    "candidate_minimum": float(result.min()), "candidate_maximum": float(result.max()),
                    "candidate_rows_above_one": int((result > 1).sum()), "clipping_performed": False}


def align_indices(source_ids, target_ids):
    audit.require(source_ids.ndim == target_ids.ndim == 1, "Invalid event ID dimensions")
    audit.require(source_ids.dtype.kind in "iu" and target_ids.dtype.kind in "iu", "Invalid event IDs")
    audit.require(len(source_ids) == len(target_ids) == len(np.unique(source_ids)) == len(np.unique(target_ids)),
                  "Duplicate/missing event IDs")
    order = np.argsort(source_ids)
    positions = np.searchsorted(source_ids[order], target_ids)
    audit.require((positions < len(source_ids)).all(), "Different event ID sets")
    index = order[positions]
    audit.require(np.array_equal(source_ids[index], target_ids), "Different event ID sets")
    return index


def allocation_decomposition(y, baseline, candidate):
    positive = y.astype(bool)
    clicked = np.broadcast_to((y.sum(1) > 0)[:, None], y.shape)
    nneg = int((~positive).sum())
    parts = {}
    for name, negatives in (("negative_from_clicked_event", ~positive & clicked),
                             ("negative_from_no_click_event", ~positive & ~clicked)):
        n = int(negatives.sum())
        selected = positive | negatives
        a, b = (audit.auc(y[selected], p[selected]) if n else None for p in (baseline, candidate))
        parts[name] = {"negative_rows": n, "positive_negative_pairs": int(y.sum()) * n,
                       "pair_weight": n / nneg, "v6_auc": a, "candidate_auc": b,
                       "weighted_auc_change": (n / nneg) * (b - a) if n else 0.0}
    for name, p in (("v6", baseline), ("candidate", candidate)):
        rebuilt = sum(v["pair_weight"] * v[name + "_auc"] for v in parts.values() if v["negative_rows"])
        audit.require(abs(rebuilt - audit.auc(y, p)) < 1e-12, "AUC decomposition does not reconcile")
    rows, cols = np.nonzero(y)
    mask = ~y[rows].astype(bool)
    same = []
    for p in (baseline, candidate):
        hit, others = p[rows, cols, None], p[rows]
        same.append(float(((hit > others) + .5 * (hit == others))[mask].mean()))
    return {"negative_event_partitions": parts,
            "same_event_pairs_subset": {"v6_auc": same[0], "candidate_auc": same[1],
                                        "auc_change": same[1] - same[0],
                                        "pair_weight_in_global_auc": 9 / nneg}}


def read_zip(path):
    audit.require(audit.sha256(path) == ZIP_SHA, "Transferred ZIP differs from the reviewed archive")
    with zipfile.ZipFile(path) as archive:
        wanted = {PREFIX + name for name in ATT_FILES} | {PREFIX + "TRANSFER_MANIFEST.json", PREFIX + "SHA256SUMS"}
        audit.require(set(archive.namelist()) == wanted and len(archive.infolist()) == len(wanted),
                      "Unexpected ZIP entries")
        audit.require(archive.testzip() is None, "ZIP CRC failure")
        content = {name: archive.read(PREFIX + name) for name in ATT_FILES}
        transfer = json.loads(archive.read(PREFIX + "TRANSFER_MANIFEST.json"))
        sums = {}
        for line in archive.read(PREFIX + "SHA256SUMS").decode("utf-8").splitlines():
            digest, name = line.split("  ", 1)
            audit.require(name not in sums, "Duplicate checksum entry")
            sums[name] = digest
        audit.require(set(sums) == set(transfer["files"]) == set(ATT_FILES), "Transfer inventory differs")
        for name, data in content.items():
            audit.require(byte_hash(data) == ATT_FILES[name] == transfer["files"][name]["sha256"] == sums[name],
                          f"Transfer checksum mismatch: {name}")
            audit.require(len(data) == transfer["files"][name]["bytes"], "Transfer size mismatch")
    protocol = json.loads(content["outputs/frozen_protocol.json"])
    report = json.loads(content["outputs/ensemble_trial_results.json"])
    audit.require(report["protocol"] == {k: v for k, v in protocol.items() if k not in {"folds", "input_sha256"}},
                  "Original report/protocol disagree")
    audit.require(protocol["attention_expert_weight"] == .15 and protocol["v5_weight"] == .85,
                  "Original integration recipe differs")
    audit.require(protocol["empty_history_rule"] == "retain V5 unchanged", "Original eligibility rule differs")
    audit.require({k: v["cutoff"] for k, v in protocol["folds"].items()} == FOLDS, "Original cutoffs differ")
    audit.require(report["v5_reproduction_passed"] is True and report["supported_for_long_window_check"] is False,
                  "Original trial status differs")
    arrays = {}
    for key, spec in CASES.items():
        with np.load(io.BytesIO(content[spec["attention_file"]]), allow_pickle=False) as saved:
            audit.require(set(saved.files) == {"valid_idx", "labels", "baseline", "candidate", "attention_expert", "attention_raw"},
                          "Unexpected attention array keys")
            arrays[key] = {k: saved[k].copy() for k in saved.files}
    return protocol, report, arrays


def raw_data(path):
    with np.load(path, allow_pickle=False) as saved:
        raw = {k: saved[k] for k in ("u", "t", "y", "h")}
    audit.require(raw["u"].shape == raw["t"].shape == (TRAIN_EVENTS,), "Training event count differs")
    audit.require(raw["y"].shape == (TRAIN_EVENTS, 10) and int(raw["y"].sum()) == TRAIN_CLICKS,
                  "Training labels differ")
    audit.require(np.isin(raw["y"], [0, 1]).all() and (raw["y"].sum(1) <= 1).all(), "Invalid training labels")
    audit.require(raw["h"].shape == (TRAIN_EVENTS, 20), "Training histories differ")
    audit.require(((raw["h"] >= -1) & (raw["h"] < 50000)).all(), "Invalid history IDs")
    audit.require(((raw["t"] >= audit.JAN20) & (raw["t"] < audit.END)).all(), "Training timeline differs")
    return raw


def check_case(key, attention, historical_report, v6_path, raw, check, replay_interval):
    spec = CASES[key]
    old_ids, old_y = attention["valid_idx"], attention["labels"]
    audit.require(len(old_ids) == spec["events"] and old_y.shape == (spec["events"], 10), "Wrong attention cohort")
    audit.require(old_ids.dtype.kind in "iu" and ((old_ids >= 0) & (old_ids < TRAIN_EVENTS)).all(), "Invalid attention indices")
    audit.require(np.array_equal(old_y, raw["y"][old_ids]), "Attention labels do not match training cache")
    old_mask = (raw["h"][old_ids] >= 0).any(1)
    audit.require(int((~old_mask).sum()) == spec["empty_events"], "History eligibility differs from reviewed archive")
    for name in ("baseline", "candidate", "attention_expert", "attention_raw"):
        a = attention[name]
        audit.require(a.shape == old_y.shape and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all(),
                      f"Invalid historical scores: {name}")
    expected_old = attention["baseline"].copy()
    expected_old[old_mask] = .85 * attention["baseline"][old_mask] + .15 * attention["attention_expert"][old_mask]
    difference = float(np.max(np.abs(expected_old - attention["candidate"])))
    audit.require(difference <= 1e-15, "Original 85/15 blend does not replay with actual history masks")
    audit.require(np.array_equal(attention["candidate"][~old_mask], attention["baseline"][~old_mask]),
                  "Original empty-history scores changed")
    record = historical_report[spec["report_key"]]
    reproduced = {}
    for name, field in (("baseline", "baseline_v5_auc"), ("candidate", "candidate_auc"),
                        ("attention_expert", "attention_calibrated_auc"), ("attention_raw", "attention_raw_auc")):
        value = audit.auc(old_y, attention[name])
        audit.require(abs(value - record[field]) < 1e-12, f"Original {field} differs")
        reproduced[field] = value
    audit.require(abs(reproduced["candidate_auc"] - reproduced["baseline_v5_auc"] - record["auc_change"]) < 1e-12,
                  "Original AUC change differs")
    check()
    proof = {"reproduced_historical_metrics": reproduced, "max_original_blend_difference": difference,
             "actual_history_mask_verified": True, "empty_history_events": int((~old_mask).sum())}
    if key == "primary" and replay_interval:
        interval = audit.bootstrap(old_y, attention["baseline"], attention["candidate"],
            {"overall": np.ones(len(old_y), dtype=bool)}, check, resamples=300, progress=False)["overall"]
        audit.require(np.allclose(interval["paired_gain_95_percentile_interval"],
                                 historical_report["primary_paired_gain_95_percentile_interval"], rtol=0, atol=1e-14),
                      "Original paired interval does not reproduce")
        proof["original_primary_300_resample_replay"] = interval
    idx, y, p, _, _ = audit.load_cohort(v6_path, spec, raw)
    aligned = align_indices(old_ids, idx)
    audit.require(np.array_equal(old_y[aligned], y), "Attention/V6 labels differ after alignment")
    with np.load(v6_path, allow_pickle=False) as saved:
        old_v5 = saved["baseline"]
    audit.require(np.allclose(attention["baseline"][aligned], old_v5, rtol=0, atol=1e-15),
                  "Attention V5 baseline differs from the hash-verified V6 trial's V5 baseline")
    proof.update(v6_auc=audit.auc(y, p), event_alignment_verified=True, v5_baselines_match=True)
    return y, p, attention["attention_expert"][aligned], (raw["h"][idx] >= 0).any(1), proof


def timer():
    start = time.monotonic()

    def check():
        if time.monotonic() - start > BUDGET:
            raise TimeoutError("30-minute budget exceeded; stop this recipe and review the log")

    return start, check


def verify_inputs(inputs, check):
    for path, want in inputs.items():
        audit.require(audit.sha256(path) == want, f"Frozen input changed: {path}")
        check()


def prepare(args):
    start, check = timer()
    repo = args.repo.resolve(strict=True)
    work = audit.inside(repo / "artifacts/experiments" / EXPERIMENT, repo)
    audit.require(not work.exists(), f"Preparation already exists; inspect/reuse it: {work}")
    zip_path = audit.inside(args.attention_zip, repo)
    source = audit.inside(Path(__file__), repo)
    helper = source.with_name("event_scale_diagnostic.py")
    audit.require(audit.sha256(helper) == HELPER_SHA, "Original numerical helper changed")
    v6_manifest_path = audit.inside(repo / "artifacts/experiments/v6-event-scale-v1/inputs.json", repo)
    v6_manifest_hash = audit.sha256(v6_manifest_path)
    v6_manifest = json.loads(v6_manifest_path.read_text(encoding="utf-8"))
    audit.require(v6_manifest["experiment"] == "v6-event-scale-v1", "Wrong V6 input manifest")
    audit.require(v6_manifest["script_sha256"] == HELPER_SHA, "V6 manifest refers to a different diagnostic source")
    paths = {key: audit.inside(v6_manifest["inputs"][key]["path"], repo) for key in CASES}
    training = audit.inside(repo / "data/cache/train_events.npz", repo)
    inputs = {str(source): audit.sha256(source), str(helper): HELPER_SHA, str(zip_path): ZIP_SHA,
              str(v6_manifest_path): v6_manifest_hash, str(training): TRAIN_SHA}
    for key, path in paths.items():
        audit.require(v6_manifest["inputs"][key]["sha256"] == CASES[key]["sha256"], "V6 manifest prediction hash differs")
        inputs[str(path)] = CASES[key]["sha256"]
    verify_inputs(inputs, check)
    original, old_report, attention = read_zip(zip_path)
    legacy_hashes = {}
    for name in LEGACY_NAMES:
        original_path = "/home/edwin/projects/techarena-final/src/" + name
        expected = original["input_sha256"][original_path]
        path = audit.inside(repo / "legacy" / name, repo)
        audit.require(audit.sha256(path) == expected, f"Historical legacy source differs: {path}")
        inputs[str(path)] = expected
        legacy_hashes[name] = expected
    old_training = "/home/edwin/projects/techarena/data/cache/train_events.npz"
    audit.require(original["input_sha256"][old_training] == TRAIN_SHA, "Historical training-cache provenance differs")
    raw = raw_data(training)
    proof = {"transfer_checksums_and_zip_verified": True, "legacy_sources_match_frozen_protocol": legacy_hashes,
             "active_training_cache_matches_historical_hash": True,
             "temporal_source_review": {
                 "meta24": "pre-Jan24 attention/initialization and aggregates; Jan24 calibration labels",
                 "primary": "pre-Jan25 attention/initialization and aggregates; Jan24 calibrator",
                 "secondary": "pre-Jan26 attention/initialization and aggregates; Jan25 calibrator from pre-Jan25 features",
                 "limitations": "Source and cached-evidence verification; checkpoints, split caches and neural replay are not verified by this ZIP.",
             }, "historical_cases": {}}
    for key in CASES:
        print(f"Replaying original attention evidence and aligning V6: {key}", flush=True)
        *_, case_proof = check_case(key, attention[key], old_report, paths[key], raw, check, replay_interval=True)
        proof["historical_cases"][key] = case_proof
    verify_inputs(inputs, check)
    check()
    work.mkdir(parents=True)
    provenance_path = work / "provenance.json"
    audit.write_new(provenance_path, proof)
    inputs[str(provenance_path)] = audit.sha256(provenance_path)
    frozen = {"experiment": EXPERIMENT, "repo": str(repo), "protocol": RECIPE,
              "protocol_sha256": audit.canonical_hash(RECIPE), "inputs_sha256": inputs,
              "attention_zip": str(zip_path), "training_cache": str(training),
              "v6_paths": {k: str(v) for k, v in paths.items()}, "provenance": str(provenance_path),
              "setup_elapsed_seconds": round(time.monotonic() - start, 3),
              "new_candidate_scores_computed": False}
    audit.write_new(work / "inputs.json", frozen)
    print(f"Preparation passed; single recipe frozen before candidate scoring: {work / 'inputs.json'}", flush=True)


def run(args):
    start, check = timer()
    path = args.manifest.resolve(strict=True)
    manifest_hash = audit.sha256(path)
    frozen = json.loads(path.read_text(encoding="utf-8"))
    repo = Path(frozen["repo"]).resolve(strict=True)
    audit.inside(path, repo / "artifacts/experiments" / EXPERIMENT)
    output = audit.inside(args.output, repo / "docs/results" / EXPERIMENT)
    audit.require(not output.exists(), f"Result exists; refusing to overwrite: {output}")
    audit.require(frozen["experiment"] == EXPERIMENT and frozen["protocol"] == RECIPE
                  and frozen["protocol_sha256"] == audit.canonical_hash(RECIPE), "Frozen recipe changed")
    inputs = frozen["inputs_sha256"]
    source = Path(__file__).resolve()
    audit.require(inputs.get(str(source)) == audit.sha256(source), "Source was not frozen by prepare")
    audit.require(inputs.get(str(source.with_name("event_scale_diagnostic.py"))) == HELPER_SHA,
                  "Numerical helper was not frozen")
    for p in inputs:
        audit.inside(p, repo)
    audit.require(inputs.get(str(audit.inside(frozen["attention_zip"], repo))) == ZIP_SHA,
                  "Attention ZIP was not frozen as the reviewed input")
    audit.require(inputs.get(str(audit.inside(frozen["training_cache"], repo))) == TRAIN_SHA,
                  "Training cache was not frozen as the historical input")
    audit.require(set(frozen["v6_paths"]) == set(CASES), "Unexpected V6 cohorts")
    for key, p in frozen["v6_paths"].items():
        audit.require(inputs.get(str(audit.inside(p, repo))) == CASES[key]["sha256"],
                      "V6 cohort was not frozen as the historical input")
    verify_inputs(inputs, check)
    _, old_report, attention = read_zip(frozen["attention_zip"])
    raw = raw_data(frozen["training_cache"])
    results = {}
    for key in CASES:
        y, p, e, mask, proof = check_case(key, attention[key], old_report, frozen["v6_paths"][key], raw,
                                        check, replay_interval=False)
        candidate, conservation = allocation(p, e, mask)
        result = audit.metrics(y, p, candidate)
        # Avoid calling the score a normalized probability in the new report.
        result["candidate_auc"] = result.pop("normalized_auc")
        result["event_sum_checks"] = conservation
        result["decomposition"] = allocation_decomposition(y, p, candidate)
        print(json.dumps({key: {k: v for k, v in result.items() if k != "decomposition"}}, indent=2), flush=True)
        result.update(audit.bootstrap(y, p, candidate, {"overall": np.ones(len(y), dtype=bool)}, check,
                                     resamples=RESAMPLES)["overall"])
        result["historical_checks_repeated"] = proof
        results[key] = result
        check()
    decisions = {"primary_interval_entirely_positive": results["primary"]["paired_gain_95_percentile_interval"][0] > 0,
                 "secondary_point_change_nonnegative": results["secondary"]["auc_change"] >= 0}
    verify_inputs(inputs, check)
    audit.require(audit.sha256(path) == manifest_hash, "Frozen input manifest changed during run")
    check()
    report = {"complete": True, "protocol": RECIPE, "protocol_sha256": audit.canonical_hash(RECIPE),
              "manifest_sha256": manifest_hash, "source_sha256": audit.sha256(source),
              "inputs_sha256": inputs, "results": results, "decision_checks": decisions,
              "supports_five_day_review": all(decisions.values()), "automatic_promotion_performed": False,
              "interpretation": "Failure closes this single allocation recipe. A pass supports reviewing the identical recipe's five-day comparison, not a submission. Reused dates; intervals not selection-adjusted.",
              "input_hashes_rechecked_unchanged": True, "training_performed": False,
              "evaluation_cache_read": False, "submission_created": False,
              "runtime": {"python": sys.version, "numpy": np.__version__,
                          "elapsed_seconds": round(time.monotonic() - start, 3)}}
    audit.write_new(output, report)
    print(json.dumps({"decision_checks": decisions, "supports_five_day_review": all(decisions.values()),
                      "report": str(output)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    p = actions.add_parser("prepare")
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--attention-zip", type=Path, required=True)
    r = actions.add_parser("run")
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.action == "prepare" else run)(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TimeoutError, zipfile.BadZipFile) as error:
        raise SystemExit(f"STOPPED: {error}") from error
