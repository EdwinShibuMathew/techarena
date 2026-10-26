"""Reload the promoted seed44 neural checkpoint and reused Jan25 combiner; never train."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/v8-seed44-row-auc"))
import seed44_core as core
from predict_v8 import replay_v6, export_verified, output_path


def required_files(root, manifest):
    files = dict(manifest["inputs_sha256"])
    paths = [core.WORK / "inputs.json", core.REPORTS / "preparation.json", core.neural_path("jan27")]
    for stage in ("screen", "stress"):
        path = core.REPORTS / (stage + ".json")
        paths.append(path)
        report = core.read_json(root / path)
        results = report["results"].values() if stage == "screen" else [report["results"]]
        for result in results:
            fitted = result["fit"]
            files.update(fitted["neural_sha256"])
            files[fitted["model"]] = fitted["model_sha256"]
            files[result["predictions"]] = result["predictions_sha256"]
    for fold in core.FOLDS:
        paths.append(core.neural_path(fold).with_suffix(".json"))
    for path in paths:
        files[path.as_posix()] = core.sha256(core.inside(root, path))
    return files


def validate_release(root, config):
    manifest = core.load_manifest(root)
    core.require(config["version"] == "v8_seed44_row_auc" and config["protocol_sha256"] == core.canonical(core.PROTOCOL)
                 and config["member"] == "dim8_seed44" and config["member_weight"] == 2 / 9
                 and config["feature_names"] == list(core.BASE_COLUMNS) and config["feature_count"] == 37
                 and config["evaluation_rows"] == 529390 and config["evaluation_events"] == 52939
                 and config["baseline_sha256"] == core.V6_SHA, "Invalid seed44 recipe/composition")
    core.require(config["neural"] == core.neural_path("jan27").as_posix()
                 and config["combiner"] == (core.WORK / "combiners/secondary.cbm").as_posix()
                 and config["combiner_policy"] == "reuse validated Jan25 combiner without refit", "Final fitting policy differs")
    core.require(set(config["code_sha256"]) == set(core.CODE_FILES)
                 and config["code_sha256"] == manifest["code_sha256"], "Incomplete/different source inventory")
    for stage in ("screen", "stress"):
        core.validate_report(root, manifest, stage)
    core.require(config["files_sha256"] == required_files(root, manifest), "Incomplete/different release inventory")
    core.verify_files(root, config["files_sha256"])
    core.verify_files(root, config["code_sha256"])
    core.validate_neural(root, manifest, "jan27")
    screen = core.read_json(root / core.REPORTS / "screen.json")
    fit = screen["results"]["secondary"]["fit"]
    core.require(fit["model"] == config["combiner"] and fit["model_sha256"] == config["files_sha256"][config["combiner"]], "Reused combiner differs")
    return manifest


def compute_candidate(root, config, device="cuda"):
    import torch
    from catboost import CatBoostClassifier, Pool
    from build_submission import neural_pipeline, FIT_FIELDS, TARGET_FIELDS
    manifest = validate_release(root, config)
    baseline, _, _, _, rowids, errors = replay_v6(root, device)
    with np.load(root / "data/cache/train_events.npz", allow_pickle=False) as data:
        training = {k: data[k] for k in FIT_FIELDS}
    with np.load(root / "data/cache/evaluation_events.npz", allow_pickle=False) as data:
        core.require(not ({"y", "clicked", "label"} & set(data.files)), "Evaluation contains labels")
        target = {k: data[k] for k in TARGET_FIELDS}
    with np.load(root / "artifacts/v6/dim8_seed44/reference_predictions.npz", allow_pickle=False) as data:
        original = data["combiner_predictions"].reshape(-1, 10)
    x, names, *_ = neural_pipeline(root / config["neural"], training, target,
        root / core.WORK / "prefix/jan27.npz", torch.device(device), final_seed=44)
    core.require(tuple(names) == core.BASE_COLUMNS and x.shape == (529390, 37), "Final feature schema differs")
    model = CatBoostClassifier()
    model.load_model(str(core.inside(root, config["combiner"])))
    core.require(model.tree_count_ == 545 and model.feature_names_ == list(core.BASE_COLUMNS), "Reused combiner schema differs")
    resolved = model.get_all_params()
    for key, expected in manifest["parameters"]["secondary"].items():
        if key in {"allow_writing_files", "thread_count", "use_best_model"}:
            continue
        actual = resolved.get(key)
        same = np.isclose(actual, expected, rtol=0, atol=1e-7) if isinstance(expected, float) else actual == expected
        core.require(bool(same), f"Saved combiner parameter differs: {key}")
    new = model.predict_proba(Pool(x, feature_names=names), thread_count=4)[:, 1].reshape(-1, 10)
    return rowids, core.replacement(baseline, original, new), errors


def release_candidate(root, manifest, budget):
    config = {"version": "v8_seed44_row_auc", "submission_ready": False, "protocol_sha256": core.canonical(core.PROTOCOL),
        "member": "dim8_seed44", "member_weight": 2 / 9, "feature_count": 37, "feature_names": list(core.BASE_COLUMNS),
        "evaluation_rows": 529390, "evaluation_events": 52939, "baseline_sha256": core.V6_SHA,
        "neural": core.neural_path("jan27").as_posix(), "combiner": (core.WORK / "combiners/secondary.cbm").as_posix(),
        "combiner_policy": "reuse validated Jan25 combiner without refit", "code_sha256": manifest["code_sha256"],
        "files_sha256": required_files(root, manifest), "runtime": manifest["runtime"]}
    rowids, candidate, errors = compute_candidate(root, config)
    fingerprint = export_verified(root, root / core.WORK / "draft.csv", rowids, candidate)
    config["submission_sha256"] = fingerprint
    core.write_new(root / core.WORK / "candidate_config.json", config)
    replay_ids, replay, replay_errors = compute_candidate(root, config)
    core.require(np.array_equal(rowids, replay_ids) and np.allclose(candidate, replay, rtol=0, atol=1e-15), "Independent seed44 replay differs")
    export_verified(root, root / core.WORK / "independent_replay.csv", replay_ids, replay, fingerprint)
    budget.check()
    protected = [core.inside(root, path) for path in config["files_sha256"]] + [root / "configs/v8.json"]
    publish_ready(root, config, replay_ids, replay, fingerprint, replay_errors, protected)


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
