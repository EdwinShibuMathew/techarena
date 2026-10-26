"""Replay a promoted V8 candidate from saved V6 assets and its fixed ranker."""
import argparse
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/v8-pairlogit"))
import v8_core as core


def replay_v6(root, device="cuda"):
    """Reproduce all five frozen members and the exact baseline CSV; never fit trees."""
    import torch
    from consolidate import verify_bundle, asset
    from predict_v6 import portable_pipeline
    sys.path.insert(0, str(root / "legacy"))
    from build_submission import neural_pipeline, FIT_FIELDS, TARGET_FIELDS, load_combiner, predict_combiner
    from submission_export import export_submission
    manifest = verify_bundle(root)
    core.require(device in {"cpu", "cuda"}, "Unsupported inference device")
    core.require(device != "cuda" or torch.cuda.is_available(), "CUDA is unavailable")
    torch.set_num_threads(4)
    with np.load(root / "data/cache/train_events.npz", allow_pickle=False) as data:
        training = {k: data[k] for k in FIT_FIELDS}
        core.require(len(data["u"]) == 686824 and np.all(data["t"] < core.FOLDS["jan27"]), "Invalid V6 fitting timeline")
    with np.load(root / "data/cache/evaluation_events.npz", allow_pickle=False) as data:
        core.require(not ({"y", "clicked", "label"} & set(data.files)), "Evaluation contains labels")
        target = {k: data[k] for k in TARGET_FIELDS}
    core.require(len(target["u"]) == len(np.unique(target["u"])) == 52939, "Invalid evaluation cohort")
    core.require(target["i"].shape == (52939, 10), "Invalid candidate shape")
    core.require(np.all((target["t"] >= core.FOLDS["jan27"]) & (target["t"] < 1738540800)), "Invalid evaluation timeline")
    rowids = np.load(root / "data/cache/evaluation_rowids.npy", allow_pickle=False)
    core.require(rowids.shape == (529390,) and rowids.dtype.kind == "U" and len(np.unique(rowids)) == len(rowids), "Invalid row IDs")
    pipeline = portable_pipeline(neural_pipeline)
    members, errors, seed_features, columns = [], {}, None, None
    for record in manifest["members"]:
        print("Replaying protected V6 member", record["key"], flush=True)
        x, names, *_ = pipeline(asset(root, record["checkpoint"]), training, target,
                                root / "data/cache/preference_training.npz", torch.device(device))
        p = predict_combiner(load_combiner(asset(root, record["combiner"])), x, names).reshape(-1, 10)
        with np.load(asset(root, record["reference"]), allow_pickle=False) as data:
            expected = data["combiner_predictions"].reshape(-1, 10)
        error = float(np.max(np.abs(p - expected)))
        core.require(np.allclose(p, expected, rtol=0, atol=1e-15), "Original member replay differs: " + record["key"])
        errors[record["key"]] = error
        if record["key"] == "dim8_seed42":
            seed_features, columns = x, names
        members.append(p)
    p = (2 / 3) * ((members[0] + members[1] + members[2]) / 3) + (1 / 3) * (.5 * members[3] + .5 * members[4])
    temporary_root = root / "artifacts/experiments/v8-pairlogit/replay"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
        baseline = Path(temporary) / "v6.csv"
        export_submission(rowids, p, root / "data/raw/example_submission.csv", baseline, 529390)
        core.require(core.sha256(baseline) == core.V6_SHA, "Five-member V6 CSV replay fingerprint differs")
    return p, members[0], seed_features, columns, rowids, errors


def validate_release(root, config):
    core.require(config.get("protocol_sha256") == core.canonical(core.PROTOCOL), "Unknown V8 recipe")
    core.require(config.get("baseline_sha256") == core.V6_SHA, "V8 comparator differs")
    core.require(config.get("evaluation_rows") == 529390, "Invalid V8 row count")
    core.require(config.get("member_weight") == 2 / 9 and config.get("feature_count") == 37, "V8 composition changed")
    required = {config["model"], config["link"], config["screen_report"], config["stress_report"]}
    core.require(required.issubset(config["files_sha256"]), "Release provenance is incomplete")
    core.verify_files(root, config["files_sha256"])
    core.verify_files(root, config["code_sha256"])
    screen = core.read_json(core.inside(root, config["screen_report"]))
    stress = core.read_json(core.inside(root, config["stress_report"]))
    for report in (screen, stress):
        core.require(report.get("complete") is True and report["protocol_sha256"] == config["protocol_sha256"], "Incomplete or different validation recipe")
    core.require(all(core.screen_gates(screen["results"]).values()), "V8 screen gates failed")
    core.require(all(core.stress_gates(stress["results"]).values()), "V8 stress gates failed")


def compute_candidate(root, config, device):
    from catboost import CatBoostRanker, Pool
    validate_release(root, config)
    baseline, old_member, x, names, rowids, errors = replay_v6(root, device)
    core.require(list(names) == config["feature_names"], "Feature order differs from final fitting")
    link = core.read_json(core.inside(root, config["link"]))
    core.require(link.get("slope") == 1 and np.isfinite(link["bias"]), "Invalid saved probability link")
    model = CatBoostRanker()
    model.load_model(str(core.inside(root, config["model"])))
    core.require(model.tree_count_ == 545 and model.feature_names_ == names, "Saved V8 ranker schema differs")
    raw = model.predict(Pool(x, feature_names=names), thread_count=4)
    candidate = core.replacement(baseline, old_member, core.sigmoid(raw + link["bias"]).reshape(-1, 10))
    return rowids, candidate, errors


def output_path(root, output, protected):
    output = Path(output).resolve()
    core.require(output.is_relative_to((root / "artifacts").resolve()), "Output must be inside ignored artifacts")
    for path in protected:
        path = Path(path)
        core.require(output != path.resolve() and not (output.exists() and path.exists() and os.path.samefile(output, path)), "Output aliases a protected input")
    return output


def export_verified(root, output, rowids, probabilities, expected_sha=None, protected=()):
    """Validate a new file before atomic no-replace publication."""
    sys.path.insert(0, str(root / "legacy"))
    from submission_export import export_submission, validate_csv
    from row_auc_final_utils import verify_csv_probabilities
    output = output_path(root, output, protected)
    if output.exists():
        core.require(expected_sha is not None and core.sha256(output) == expected_sha, "Preserve existing different output")
        verify_csv_probabilities(output, rowids, probabilities)
        validate_csv(output, root / "data/raw/example_submission.csv", 529390)
        return expected_sha
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v8-", dir=output.parent) as temporary:
        staged = Path(temporary) / "candidate.csv"
        export_submission(rowids, probabilities, root / "data/raw/example_submission.csv", staged, 529390)
        validate_csv(staged, root / "data/raw/example_submission.csv", 529390)
        verify_csv_probabilities(staged, rowids, probabilities)
        fingerprint = core.sha256(staged)
        core.require(expected_sha is None or fingerprint == expected_sha, "V8 CSV fingerprint differs")
        os.link(staged, output)
    return fingerprint


def generate(root, output, device="cuda"):
    root = Path(root).resolve()
    config = core.read_json(root / "configs/v8.json")
    core.require(config.get("submission_ready") is True, "V8 is not submission-ready")
    baseline_manifest = core.read_json(root / "artifacts/v6/manifest.json")
    protected = [core.inside(root, path) for path in baseline_manifest["files"]]
    protected += [core.inside(root, path) for path in config["files_sha256"]]
    protected += [root / "configs/v8.json"]
    output = output_path(root, output, protected)
    if output.exists():
        core.require(core.sha256(output) == config["submission_sha256"], "Preserve existing different output")
    rowids, p, errors = compute_candidate(root, config, device)
    fingerprint = export_verified(root, output, rowids, p, config["submission_sha256"], protected)
    validate_release(root, config)
    return {"submission_ready": True, "submission_sha256": fingerprint, "rows": len(rowids), "v6_replay_max_errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/submissions/v8_candidate.csv")
    args = parser.parse_args()
    report = generate(ROOT, args.output, args.device)
    print(report)


if __name__ == "__main__":
    main()
