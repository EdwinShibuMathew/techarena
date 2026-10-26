"""Replay/export a gated V8 enriched Logloss release; never fit a classifier."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/v8-enriched-logloss"))
import enriched_core as core
from predict_v8 import export_verified, output_path, replay_v6


def validate_release(root, config):
    manifest = core.load_manifest(root)
    core.require(config["version"] == "v8_enriched_logloss_seed42" and
                 config["protocol_sha256"] == core.canonical(core.PROTOCOL), "Unknown V8 recipe")
    core.require(config["baseline_sha256"] == core.V6_SHA and config["evaluation_rows"] == 529390
                 and config["evaluation_events"] == 52939, "Invalid baseline/evaluation binding")
    core.require(config["member_weight"] == 2 / 9 and config["feature_count"] == 43
                 and config["feature_names"] == list(core.COLUMNS), "Invalid composition/schema")
    core.require(set(config["code_sha256"]) == set(core.CODE_FILES)
                 and config["code_sha256"] == manifest["code_sha256"], "Incomplete/different source inventory")
    required = dict(manifest["inputs_sha256"])
    for relative in (core.WORK / "inputs.json", core.REPORTS / "preparation.json",
                     core.REPORTS / "screen.json", core.REPORTS / "stress.json",
                     Path(config["model"]), Path(config["fit_report"])):
        required[relative.as_posix()] = core.sha256(core.inside(root, relative))
    core.require(config["files_sha256"] == required, "Incomplete/different release inventory")
    core.verify_files(root, config["files_sha256"])
    core.verify_files(root, config["code_sha256"])
    for stage in ("screen", "stress"):
        core.validate_report(root, manifest, stage)
    fit = core.read_json(core.inside(root, config["fit_report"]))
    core.require(fit["case"] == "final" and fit["model"] == config["model"]
                 and fit["model_sha256"] == config["files_sha256"][config["model"]]
                 and fit["requested_parameters"] == manifest["classifier_parameters"]["final"]
                 and fit["feature_names"] == list(core.COLUMNS), "Final model provenance differs")
    return manifest, fit


def compute_candidate(root, config, device="cuda"):
    from catboost import CatBoostClassifier, Pool
    _, fit = validate_release(root, config)
    baseline, old_member, x, names, rowids, errors = replay_v6(root, device)
    x, names = core.enrich(np.asarray(x, dtype=np.float32), names)
    model = CatBoostClassifier()
    model.load_model(str(core.inside(root, config["model"])))
    core.require(model.tree_count_ == 545 and model.feature_names_ == names
                 and list(model.classes_) == [0, 1], "Saved classifier schema differs")
    resolved = model.get_all_params()
    for key, expected in fit["requested_parameters"].items():
        if key in {"allow_writing_files", "thread_count", "use_best_model"}:
            continue  # Runtime-only settings are not persisted by CatBoost.
        actual = resolved.get(key)
        equal = np.isclose(actual, expected, rtol=0, atol=1e-7) if isinstance(expected, float) else actual == expected
        core.require(bool(equal), f"Saved classifier parameter differs: {key}")
    new = model.predict_proba(Pool(x, feature_names=names), thread_count=4)[:, 1].reshape(-1, 10)
    return rowids, core.replacement(baseline, old_member, new), errors


def publish_ready(root, config, rowids, probabilities, fingerprint, errors, protected):
    """CSV first, evidence second, ready configuration last; interruptions fail closed."""
    validate_release(root, config)
    output = root / "artifacts/submissions/v8_candidate.csv"
    export_verified(root, output, rowids, probabilities, fingerprint, protected)
    ready = {**config, "submission_ready": True}
    release = {"complete": True, "submission_ready": True, "submission_sha256": fingerprint,
        "protocol_sha256": core.canonical(core.PROTOCOL), "rows": len(rowids),
        "v6_member_replay_errors": errors, "independent_csv_replay": True,
        "configuration_canonical_sha256": core.canonical(ready), "competition_upload_performed": False}
    core.write_new(root / core.REPORTS / "release.json", release)
    # A hardlink publishes a fully written JSON file without replacing any prior release.
    pending = root / core.WORK / "ready_config.json"
    core.write_new(pending, ready)
    destination = root / "configs/v8.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    import os
    os.link(pending, destination)


def release_candidate(root, manifest, fitted, rowids, candidate, budget):
    fingerprint = export_verified(root, root / core.WORK / "draft.csv", rowids, candidate)
    fit_report = (Path(fitted["model"]).parent / "fit.json").as_posix()
    files = dict(manifest["inputs_sha256"])
    for relative in (core.WORK / "inputs.json", core.REPORTS / "preparation.json",
                     core.REPORTS / "screen.json", core.REPORTS / "stress.json", Path(fitted["model"]), Path(fit_report)):
        files[relative.as_posix()] = core.sha256(core.inside(root, relative))
    config = {"version": "v8_enriched_logloss_seed42", "submission_ready": False,
        "protocol_sha256": core.canonical(core.PROTOCOL), "baseline_sha256": core.V6_SHA,
        "evaluation_rows": 529390, "evaluation_events": 52939, "member_weight": 2 / 9,
        "feature_count": 43, "feature_names": list(core.COLUMNS), "model": fitted["model"], "fit_report": fit_report,
        "files_sha256": files, "code_sha256": manifest["code_sha256"], "submission_sha256": fingerprint,
        "runtime": manifest["runtime"]}
    core.write_new(root / core.WORK / "candidate_config.json", config)
    replay_ids, replay, errors = compute_candidate(root, config)
    core.require(np.array_equal(rowids, replay_ids) and np.allclose(candidate, replay, rtol=0, atol=1e-15),
                 "Independent candidate replay differs")
    export_verified(root, root / core.WORK / "independent_replay.csv", replay_ids, replay, fingerprint)
    budget.check()
    protected = [core.inside(root, path) for path in files] + [root / "configs/v8.json"]
    publish_ready(root, config, replay_ids, replay, fingerprint, errors, protected)


def generate(root, output, device="cuda"):
    root = Path(root).resolve()
    config = core.read_json(root / "configs/v8.json")
    core.require(config.get("submission_ready") is True, "V8 is not submission-ready")
    validate_release(root, config)
    release = core.read_json(root / core.REPORTS / "release.json")
    core.require(release["configuration_canonical_sha256"] == core.canonical(config)
                 and release["submission_sha256"] == config["submission_sha256"]
                 and release["complete"] is True and release["independent_csv_replay"] is True,
                 "Incomplete/different ready release")
    protected = [core.inside(root, path) for path in config["files_sha256"]] + [root / "configs/v8.json"]
    output = output_path(root, output, protected)
    if output.exists():
        core.require(core.sha256(output) == config["submission_sha256"], "Preserve existing different output")
    rowids, candidate, errors = compute_candidate(root, config, device)
    fingerprint = export_verified(root, output, rowids, candidate, config["submission_sha256"], protected)
    validate_release(root, config)
    return {"submission_ready": True, "rows": len(rowids), "sha256": fingerprint, "v6_replay_errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/submissions/v8_candidate.csv")
    args = parser.parse_args()
    print(json.dumps(generate(ROOT, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
