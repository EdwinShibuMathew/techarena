"""Independent residual-event V8 replay; no fitting or gate bypass."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments/v8-residual-event"))
import residual_core as core
from predict_v8 import replay_v6, export_verified, output_path


def model_inventory(root,manifest,winner):
    report = core.read_json(root/core.REPORTS/"screen.json")
    return {member:{"base":manifest["original_models"][member]["jan27"],
        "adapter":core.adapter_path(member,kind,"jan27").as_posix(),"kind":kind,
        "combiner":report["results"]["secondary"]["member_models"][member]["combiner"]}
        for member,kind in core.COMPOSITIONS[winner].items()}


def required_files(root,manifest,winner):
    files = dict(manifest["inputs_sha256"])
    paths = [core.WORK/"inputs.json",core.REPORTS/"preparation.json",core.REPORTS/"selected.json"]
    for stage in ("development","screen","stress"):
        path = core.REPORTS/(stage+".json"); paths.append(path)
        files.update(core.read_json(root/path)["files_sha256"])
    for member,kind in core.COMPOSITIONS[winner].items():
        path = core.adapter_path(member,kind,"jan27"); paths.extend([path,path.with_suffix(".json")])
    for path in paths:
        files[path.as_posix()] = core.sha256(core.inside(root,path))
    return files


def validate_release(root,config):
    manifest = core.load_manifest(root); winner = core.selection(root,manifest)
    core.require(config["version"] == "v8_residual_event" and config["protocol_sha256"] == core.canonical(core.PROTOCOL)
        and config["composition"] == winner and config["weights"] == core.PROTOCOL["weights"]
        and config["feature_names"] == list(core.BASE_COLUMNS) and config["baseline_sha256"] == core.V6_SHA
        and config["evaluation_rows"] == 529390 and config["evaluation_events"] == 52939, "Invalid residual release recipe")
    core.require(config["code_sha256"] == manifest["code_sha256"] and set(config["code_sha256"]) == set(core.CODE_FILES), "Incomplete source inventory")
    core.require(config["selection_sha256"] == core.sha256(root/core.REPORTS/"selected.json"), "Selection receipt differs")
    for stage in ("screen","stress"):
        core.validate_report(root,manifest,stage)
    core.require(config["models"] == model_inventory(root,manifest,winner), "Final model inventory differs")
    core.require(config["files_sha256"] == required_files(root,manifest,winner), "Incomplete release inventory")
    core.verify_files(root,config["code_sha256"]); core.verify_files(root,config["files_sha256"])
    for member,kind in core.COMPOSITIONS[winner].items():
        core.validate_adapter(root,manifest,member,kind,"jan27")
    return manifest


def compute_candidate(root,config,device="cuda"):
    manifest = validate_release(root,config)
    import torch
    from catboost import CatBoostClassifier,Pool
    from build_submission import FIT_FIELDS,TARGET_FIELDS
    from residual_model import pipeline,load_adapter
    baseline,_,_,_,rowids,errors = replay_v6(root,device)
    with np.load(root/"data/cache/train_events.npz",allow_pickle=False) as d:
        training = {k:d[k] for k in FIT_FIELDS}
    with np.load(root/"data/cache/evaluation_events.npz",allow_pickle=False) as d:
        core.require(not ({"y","clicked","label"}&set(d.files)), "Evaluation labels are forbidden")
        target = {k:d[k] for k in TARGET_FIELDS}
    original,updated = {},{}
    for member in core.MEMBERS:
        with np.load(root/f"artifacts/v6/{member}/reference_predictions.npz",allow_pickle=False) as d:
            original[member] = d["combiner_predictions"].reshape(-1,10)
    for member,paths in config["models"].items():
        adapter = load_adapter(root,manifest,member,paths["kind"],"jan27",device)
        x,names,*_ = pipeline(root/paths["base"],adapter,training,target,root/manifest["prefixes"]["jan27"],torch.device(device))
        core.require(x.shape == (529390,37) and tuple(names) == core.BASE_COLUMNS, "Final feature schema differs")
        model = CatBoostClassifier(); model.load_model(str(core.inside(root,paths["combiner"])))
        core.require(model.tree_count_ == 545 and model.feature_names_ == list(core.BASE_COLUMNS), "Final combiner schema differs")
        resolved = model.get_all_params()
        for key,value in manifest["parameters"][member]["secondary"].items():
            if key in {"allow_writing_files","thread_count","use_best_model"}:
                continue
            actual = resolved.get(key)
            same = np.isclose(actual,value,rtol=0,atol=1e-7) if isinstance(value,float) else actual == value
            core.require(bool(same),f"Final combiner parameter differs: {member}/{key}")
        updated[member] = model.predict_proba(Pool(x,feature_names=names),thread_count=4)[:,1].reshape(-1,10)
    return rowids,core.compose(baseline,original,updated,config["composition"]),errors


def release_candidate(root,manifest,budget):
    winner = core.selection(root,manifest)
    config = {"version":"v8_residual_event","submission_ready":False,"protocol_sha256":core.canonical(core.PROTOCOL),
        "composition":winner,"weights":core.PROTOCOL["weights"],"feature_names":list(core.BASE_COLUMNS),
        "baseline_sha256":core.V6_SHA,"evaluation_rows":529390,"evaluation_events":52939,
        "selection_sha256":core.sha256(root/core.REPORTS/"selected.json"),"models":model_inventory(root,manifest,winner),
        "files_sha256":required_files(root,manifest,winner),"code_sha256":manifest["code_sha256"],"runtime":manifest["runtime"]}
    rowids,p,errors = compute_candidate(root,config)
    fingerprint = export_verified(root,root/core.WORK/"draft.csv",rowids,p)
    config["submission_sha256"] = fingerprint
    core.write_new(root/core.WORK/"candidate_config.json",config)
    replay_ids,replay,errors = compute_candidate(root,config)
    core.require(np.array_equal(rowids,replay_ids) and np.allclose(p,replay,rtol=0,atol=1e-15), "Independent residual replay differs")
    export_verified(root,root/core.WORK/"independent_replay.csv",replay_ids,replay,fingerprint)
    budget.check()
    protected = [core.inside(root,p) for p in config["files_sha256"]]+[root/"configs/v8.json"]
    publish_ready(root,config,replay_ids,replay,fingerprint,errors,protected)


def publish_ready(root,config,rowids,probabilities,fingerprint,errors,protected):
    validate_release(root,config)
    export_verified(root,root/"artifacts/submissions/v8_candidate.csv",rowids,probabilities,fingerprint,protected)
    ready = {**config,"submission_ready":True}
    core.write_new(root/core.REPORTS/"release.json",{"complete":True,"submission_ready":True,"submission_sha256":fingerprint,
        "protocol_sha256":core.canonical(core.PROTOCOL),"rows":len(rowids),"v6_member_replay_errors":errors,
        "independent_csv_replay":True,"configuration_canonical_sha256":core.canonical(ready),"competition_upload_performed":False})
    pending = root/core.WORK/"ready_config.json"; core.write_new(pending,ready)
    destination = root/"configs/v8.json"; destination.parent.mkdir(parents=True,exist_ok=True)
    import os
    os.link(pending,destination)


def generate(root,output,device="cuda"):
    root = Path(root).resolve(); config = core.read_json(root/"configs/v8.json")
    core.require(config.get("submission_ready") is True, "V8 is not submission-ready")
    validate_release(root,config)
    receipt = core.read_json(root/core.REPORTS/"release.json")
    core.require(receipt["configuration_canonical_sha256"] == core.canonical(config)
        and receipt["submission_sha256"] == config["submission_sha256"] and receipt["complete"] is True
        and receipt["independent_csv_replay"] is True, "Incomplete ready release")
    protected = [core.inside(root,p) for p in config["files_sha256"]]+[root/"configs/v8.json"]
    output = output_path(root,output,protected)
    if output.exists():
        core.require(core.sha256(output) == config["submission_sha256"], "Preserve existing different output")
    rowids,p,errors = compute_candidate(root,config,device)
    fingerprint = export_verified(root,output,rowids,p,config["submission_sha256"],protected)
    validate_release(root,config)
    return {"submission_ready":True,"rows":len(rowids),"sha256":fingerprint,"v6_replay_errors":errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts/submissions/v8_candidate.csv")
    args = parser.parse_args(); print(json.dumps(generate(ROOT,args.output,args.device),indent=2))


if __name__ == "__main__":
    main()
