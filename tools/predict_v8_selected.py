"""Replay a temporally selected and fully gated V8 recipe, without classifier fitting."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/v8-temporal-combiner-search"))
import search_core as core
from predict_v8 import export_verified, output_path, replay_v6


def required_files(root, manifest, fitted):
    files = {**manifest["inputs_sha256"], **fitted["files_sha256"]}
    paths = [core.WORK / "inputs.json", Path(fitted["model"]).parent / "fit.json"]
    paths += [core.REPORTS / (name + ".json") for name in ("preparation", "selection", "selected_recipe", "screen", "stress")]
    for relative in paths:
        files[relative.as_posix()] = core.sha256(core.inside(root, relative))
    return files


def validate_release(root, config):
    manifest = core.load_manifest(root)
    selected = core.selection(root, manifest)
    core.require(config["version"] == "v8_temporal_selected_seed42" and config["protocol_sha256"] == core.canonical(core.PROTOCOL)
                 and config["selected_recipe_sha256"] == core.canonical(selected)
                 and config["variant"] == selected["winner"], "Unknown or different selected recipe")
    names = list(core.COLUMNS if selected["winner"] == "B" else core.BASE_COLUMNS)
    core.require(config["baseline_sha256"] == core.V6_SHA and config["evaluation_rows"] == 529390
                 and config["evaluation_events"] == 52939 and config["member_weight"] == 2 / 9
                 and config["feature_count"] == len(names) and config["feature_names"] == names, "Invalid composition/schema")
    core.require(set(config["code_sha256"]) == set(core.CODE_FILES)
                 and config["code_sha256"] == manifest["code_sha256"], "Incomplete/different source inventory")
    fit_path = core.WORK / "models/final/fit.json"
    core.require(config["fit_report"] == fit_path.as_posix(), "Wrong final fit report")
    fitted = core.read_json(root / fit_path)
    expected_model = (core.WORK / "models/final/classifier.cbm").as_posix()
    expected_correction = (core.WORK / "models/final/correction.json").as_posix() if selected["winner"] == "C" else None
    expected_assets = {expected_model} | ({expected_correction} if expected_correction else set())
    core.require(fitted["case"] == "final" and fitted["variant"] == selected["winner"]
                 and fitted["requested_parameters"] == selected["parameters"]["final"]
                 and fitted["feature_names"] == names and fitted["model"] == config["model"] == expected_model
                 and fitted["correction"] == expected_correction and set(fitted["files_sha256"]) == expected_assets,
                 "Final model provenance differs")
    core.require(config["files_sha256"] == required_files(root, manifest, fitted), "Incomplete/different release inventory")
    core.verify_files(root, config["files_sha256"])
    core.verify_files(root, config["code_sha256"])
    for stage in ("screen", "stress"):
        core.validate_report(root, manifest, stage)
    return manifest, fitted


def compute_candidate(root, config, device="cuda"):
    _, fitted = validate_release(root, config)
    baseline, old, x, names, rowids, errors = replay_v6(root, device)
    core.require(tuple(names) == core.BASE_COLUMNS, "Replayed base schema differs")
    spec = importlib.util.spec_from_file_location("selected_v8_inference_helpers", ROOT / "experiments/v8-temporal-combiner-search/runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    new = runner.load_predictor(root, fitted, np.asarray(x, dtype=np.float32))
    return rowids, core.replacement(baseline, old, new), errors


def release_candidate(root, manifest, fitted, rowids, candidate, budget):
    selected = core.selection(root, manifest)
    fingerprint = export_verified(root, root / core.WORK / "draft.csv", rowids, candidate)
    config = {"version": "v8_temporal_selected_seed42", "submission_ready": False,
        "protocol_sha256": core.canonical(core.PROTOCOL), "selected_recipe_sha256": core.canonical(selected),
        "variant": selected["winner"], "baseline_sha256": core.V6_SHA, "evaluation_rows": 529390,
        "evaluation_events": 52939, "member_weight": 2 / 9, "feature_count": len(fitted["feature_names"]),
        "feature_names": fitted["feature_names"], "model": fitted["model"],
        "fit_report": (Path(fitted["model"]).parent / "fit.json").as_posix(),
        "files_sha256": required_files(root, manifest, fitted), "code_sha256": manifest["code_sha256"],
        "submission_sha256": fingerprint, "runtime": manifest["runtime"]}
    core.write_new(root / core.WORK / "candidate_config.json", config)
    replay_ids, replay, errors = compute_candidate(root, config)
    core.require(np.array_equal(rowids, replay_ids) and np.allclose(candidate, replay, rtol=0, atol=1e-15),
                 "Independent candidate replay differs")
    export_verified(root, root / core.WORK / "independent_replay.csv", replay_ids, replay, fingerprint)
    budget.check()
    protected = [core.inside(root, path) for path in config["files_sha256"]] + [root / "configs/v8.json"]
    publish_ready(root, config, replay_ids, replay, fingerprint, errors, protected)


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
