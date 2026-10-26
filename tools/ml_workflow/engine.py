"""Standalone chronological orchestration; metrics and release gates are owned here."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import signal
import threading
import sys
import time

import numpy as np

from consolidate import verify_bundle, write_json
from experiment_runner import PairedBootstrapEvaluator, compute_roc_auc
from .contracts import (ExperimentSpec, CaseSpec, MEMBERS, STAGES, DAY, JAN20, V6_SHA,
    require, identifier, inside, digest, canonical, first_indices, validate_events, context, compose)
from .storage import read, write_new, save_arrays, snapshot, verify_snapshot
from .budget import Budget
from .cache import request as cache_request, get_or_compute

ROOT=Path(__file__).resolve().parents[2]


def locate(root,run_id):
    identifier(run_id)
    paths=[inside(root,p)/run_id for p in ("artifacts/validation","artifacts/experiments")]
    found=[p for p in paths if (p/"manifest.json").is_file()]
    require(len(found)==1,"Run missing or ambiguous")
    require(not found[0].is_symlink(),"Symlink run directory")
    return found[0]


def prepare(root,spec_path,run_id):
    root=Path(root).resolve();identifier(run_id)
    spec=ExperimentSpec.parse(read(spec_path)).value
    if spec["purpose"]=="research":
        verify_bundle(root)
        require(Path(spec_path).resolve().is_relative_to(root/"experiments"),"Research specification must be experiment source")
    work=inside(root,"artifacts/validation" if spec["purpose"]=="synthetic_validation" else "artifacts/experiments")/run_id
    require(not any((inside(root,p)/run_id).exists() for p in ("artifacts/validation","artifacts/experiments")),"Preserve existing run ID")
    for asset in spec["inputs"].values():
        require(digest(inside(root,asset["path"]))==asset["sha256"],"Declared input changed")
    work.mkdir(parents=True)
    source=snapshot(root,work)
    require(spec["adapter"] in source["files"],"Adapter must be committed source")
    if spec["purpose"]=="research":
        relative=Path(spec_path).resolve().relative_to(root).as_posix()
        require(relative in source["files"] and digest(spec_path)==source["files"][relative],"Research specification must be committed")
        carry=read(inside(root,spec["inputs"]["carry"]["path"]))
        require(carry["compute_limit_seconds"]==14400 and 0<=carry["total_compute_seconds"]<=14400,"Invalid carry decision")
        carried=carry["total_compute_seconds"]
        carry_identity=spec["inputs"]["carry"]["sha256"]
    else:
        carried,carry_identity=0.,None
    manifest={"schema_version":1,"run_id":run_id,"spec":spec,"spec_sha256":canonical(spec),"source":source,
        "carried_seconds":carried,"carry_identity":carry_identity,"validation_status":"retrospective dates; not independent or selection-adjusted evidence"}
    write_new(work/"manifest.json",manifest)
    write_new(work/"prepared.json",{"manifest_sha256":digest(work/"manifest.json"),"complete":True})
    load(root,work)
    return {"run_id":run_id,"purpose":spec["purpose"],"source_commit":source["commit"],"prepared":True}


def load(root,work):
    start=time.monotonic();m=read(work/"manifest.json")
    require(read(work/"prepared.json")["manifest_sha256"]==digest(work/"manifest.json"),"Frozen run manifest changed")
    ExperimentSpec.parse(m["spec"])
    require(canonical(m["spec"])==m["spec_sha256"],"Specification changed")
    verify_snapshot(work,m["source"])
    for name,asset in m["spec"]["inputs"].items():
        require(digest(inside(root,asset["path"]))==asset["sha256"],"Input changed: "+name)
    # An old adapter is never silently combined with a changed runtime.
    for name,expected in m["source"]["files"].items():
        if name.startswith("tools/ml_workflow/") or name in {"tools/run_experiment.py","tools/experiment_runner.py","tools/consolidate.py"}:
            actual=ROOT/name
            require(actual.is_file() and digest(actual)==expected,
                "Runtime changed; replay using "+str(work/"source/tools/run_experiment.py")+" --root "+str(root))
    m["hashing_seconds"]=time.monotonic()-start
    return m


def adapter(work,m):
    path=inside(work/"source",m["spec"]["adapter"])
    module_spec=importlib.util.spec_from_file_location("workflow_adapter_"+m["run_id"].replace("-","_"),path)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    require(m["spec"]["purpose"] in module.PURPOSES,"Adapter purpose mismatch")
    return module


def events(root,m,name):
    with np.load(inside(root,m["spec"]["inputs"][name]["path"]),allow_pickle=False) as stored:
        data={k:stored[k] for k in stored.files}
    validate_events(data,name=="train")
    if name=="train":
        require(((data["t"]>=JAN20)&(data["t"]<JAN20+7*DAY)).all(),"Training data outside official period")
        if m["spec"]["purpose"]=="research":
            require(len(data["u"])==686824 and int(data["y"].sum())==432023,"Official training cache differs")
    return data


def baseline(root,m,name,rowids):
    with np.load(inside(root,m["spec"]["inputs"][name]["path"]),allow_pickle=False) as d:
        version=str(d["version"].item())
        require(version==("synthetic" if m["spec"]["purpose"]=="synthetic_validation" else "V6"),"Wrong baseline identity")
        require(np.array_equal(d["member_names"],list(MEMBERS)) and np.array_equal(d["rowids"],rowids),"Baseline event/member order changed")
        require(d["members"].shape==(5,*rowids.shape),"Invalid baseline member dimensions")
        original={name:d["members"][i].astype(np.float64) for i,name in enumerate(MEMBERS)}
        p=compose(original)
        require(np.allclose(p,d["scores"],rtol=0,atol=1e-15),"V6 baseline reconstruction differs")
    return original,p


def _model(root,work,m,module,raw,candidate,member,cutoff):
    import torch
    fit=np.flatnonzero(raw["t"]<cutoff)
    require(len(fit)>0,"Empty neural fitting prefix")
    dimension,seed,pref_epoch=MEMBERS[member]
    metadata={"member":member,"dimension":dimension,"seed":seed,"preference_epoch":pref_epoch,
        "epoch":2 if m["spec"]["purpose"]=="synthetic_validation" else 8,"cutoff":cutoff,
        "maximum_fitting_timestamp":int(raw["t"][fit].max()),"fitting_events":len(fit),
        "fitting_indices_sha256":hashlib.sha256(fit.astype(np.int64).tobytes()).hexdigest(),
        "protocol_sha256":m["spec_sha256"],"purpose":m["spec"]["purpose"]}
    path=work/"models"/candidate/member/str(cutoff)/"model.pt"
    receipt=path.with_suffix(".json")
    if path.exists() or receipt.exists():
        require(receipt.exists() and read(receipt)["sha256"]==digest(path),"Partial or changed model")
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        module.train(context(raw,fit,True),metadata,path,m["spec"]["candidates"][candidate])
    saved=torch.load(path,map_location="cpu",weights_only=True)
    require(saved["metadata"]==metadata,"Model cutoff/epoch/lineage differs")
    require(isinstance(saved["model_state"],dict) and saved["model_state"],"Missing saved model state")
    require(all(not isinstance(v,torch.Tensor) or not v.is_floating_point() or bool(torch.isfinite(v).all()) for v in saved["model_state"].values()),"Nonfinite checkpoint")
    if not receipt.exists():write_new(receipt,{"sha256":digest(path),"metadata":metadata})
    return path


def _features(root,work,m,module,raw,indices,path,cutoff,training_raw=None,use_cache=True):
    training_raw=raw if training_raw is None else training_raw
    fitting=np.flatnonzero(training_raw["t"]<cutoff)
    require(len(indices)>0 and (raw["t"][indices]>=cutoff).all(),"Feature target precedes neural prefix")
    start=time.monotonic()
    prefix,target=context(training_raw,fitting,True),context(raw,indices,False)
    if use_cache:
        identity=cache_request(work/"source",module,path,prefix,target)
        data,receipt=get_or_compute(root,identity,lambda:module.extract(prefix,target,path))
        cache_log=work/"cache-receipts"/(receipt["key"]+".json")
        # Each extraction gets a separate timing record; repeated keys never overwrite evidence.
        write_new(cache_log.with_name(receipt["key"]+"-"+str(time.time_ns())+".json"),receipt)
    else:
        data=module.extract(prefix,target,path)
    require(set(data)=={"X","feature_names"} and data["X"].shape==(len(indices)*10,len(data["feature_names"]))
        and data["X"].dtype==np.float32 and np.isfinite(data["X"]).all(),"Invalid extracted features")
    return data,time.monotonic()-start


def point(y,before,after):
    a,b=compute_roc_auc(y,before),compute_roc_auc(y,after)
    return {"baseline_auc":a,"candidate_auc":b,"auc_diff":b-a}


def _score(root,work,m,module,raw,candidate,case):
    fit=first_indices(raw["u"],raw["t"],case.meta_cutoff,case.meta_cutoff+DAY)
    valid=first_indices(raw["u"],raw["t"],case.target_cutoff,case.target_end)
    require(len(fit) and len(valid) and raw["t"][fit].max()<case.target_cutoff
        and not np.intersect1d(fit,valid).size,"Invalid chronological cohorts")
    if m["spec"]["purpose"]=="research" and case.name in {"primary","secondary","five_day"}:
        require(len(valid)=={"primary":43964,"secondary":35138,"five_day":51488}[case.name],"Historical first-user cohort differs")
    original,before=baseline(root,m,case.baseline,raw["rowids"][valid])
    updated=dict(original);files={};models={};timings={"training":0.,"extraction":0.,"combiner":0.}
    for member in m["spec"]["candidates"][candidate]["members"]:
        start=time.monotonic()
        meta=_model(root,work,m,module,raw,candidate,member,case.meta_cutoff)
        target=_model(root,work,m,module,raw,candidate,member,case.target_cutoff)
        timings["training"]+=time.monotonic()-start
        x,t=_features(root,work,m,module,raw,fit,meta,case.meta_cutoff);timings["extraction"]+=t
        tx,t=_features(root,work,m,module,raw,valid,target,case.target_cutoff);timings["extraction"]+=t
        require(np.array_equal(x["feature_names"],tx["feature_names"]),"Feature schema transfer changed")
        combiner=work/"combiners"/candidate/(case.name+"_"+member+".cbm")
        require(not combiner.exists(),"Combiner already exists; no automatic refit")
        combiner.parent.mkdir(parents=True,exist_ok=True)
        start=time.monotonic();module.fit_combiner(x,raw["y"][fit].copy(),combiner,m["spec"]["seed"])
        pred=module.predict(tx,combiner,raw["rowids"][valid].copy())
        timings["combiner"]+=time.monotonic()-start
        require(np.array_equal(pred["rowids"],raw["rowids"][valid]),"Prediction ordering differs")
        updated[member]=pred["scores"]
        models[member]={"meta":meta.relative_to(work).as_posix(),"target":target.relative_to(work).as_posix(),
            "combiner":combiner.relative_to(work).as_posix(),"feature_names":tx["feature_names"].tolist()}
        for p in (meta,meta.with_suffix(".json"),target,target.with_suffix(".json"),combiner):files[p.relative_to(work).as_posix()]=digest(p)
    after=compose(updated); y=raw["y"][valid]
    path=work/"predictions"/(candidate+"_"+case.name+".npz")
    save_arrays(path,indices=valid,rowids=raw["rowids"][valid],labels=y,baseline=before,candidate=after)
    files[path.relative_to(work).as_posix()]=digest(path)
    result={"case":asdict(case),"events":len(valid),"fit_events":len(fit),"maximum_meta_label_timestamp":int(raw["t"][fit].max()),
        **point(y,before,after),"models":models,"files":files,"timings":timings}
    if case.name in {"primary","secondary","five_day"}:
        start=time.monotonic()
        result.update(asdict(PairedBootstrapEvaluator().evaluate(after,before,y,raw["u"][valid])))
        require(result["resamples"]>=990,"Insufficient paired bootstrap draws")
        result["timings"]["evaluation"]=time.monotonic()-start
    hours=(raw["t"][valid]-case.target_cutoff)/3600
    late=(hours>=48)&(hours<120)
    if case.name=="five_day":
        require(late.any(),"Missing late cohort")
        if m["spec"]["purpose"]=="research":require(int(late.sum())==7596,"Historical late cohort differs")
        result["late"]=point(y[late],before[late],after[late])
    warm=np.isin(raw["u"][valid],raw["u"][raw["t"]<case.target_cutoff])
    masks={"warm":warm,"cold":~warm,"history_empty":(raw["h"][valid]<0).all(1),"late":late}
    result["diagnostics"]={name:{"events":int(mask.sum()),**(point(y[mask],before[mask],after[mask]) if mask.any() and len(np.unique(y[mask]))==2 else {})} for name,mask in masks.items()}
    return result


def _report(work,m,stage,results,passed,extra=None):
    files={k:v for row in results.values() for k,v in row["files"].items()}
    report={"schema_version":1,"stage":stage,"purpose":m["spec"]["purpose"],"complete":True,"passed":bool(passed),
        "submission_ready":False,"manifest_sha256":digest(work/"manifest.json"),"results":results,"files":files,
        "hashing_seconds":m["hashing_seconds"],"validation_status":m["validation_status"],**(extra or {})}
    path=work/(stage+".json");write_new(path,report)
    write_new(work/(stage+".seal.json"),{"sha256":digest(path)})
    return report


def checked(work,stage):
    path=work/(stage+".json")
    require(path.is_file() and read(work/(stage+".seal.json"))["sha256"]==digest(path),"Missing or changed "+stage+" report")
    result=read(path)
    require(result["manifest_sha256"]==digest(work/"manifest.json"),"Report belongs to another run")
    for p,h in result["files"].items():require(digest(inside(work,p))==h,"Recorded artifact changed")
    require(result["complete"] is True,"Incomplete report")
    for row in result["results"].values():
        require(all(np.isfinite(row[k]) for k in ("baseline_auc","candidate_auc","auc_diff"))
            and 0<=row["baseline_auc"]<=1 and 0<=row["candidate_auc"]<=1
            and abs(row["candidate_auc"]-row["baseline_auc"]-row["auc_diff"])<1e-12,"Inconsistent report metrics")
        if stage in {"screen","stress"}:
            require(type(row["resamples"]) is int and 990<=row["resamples"]<=1000 and np.isfinite(row["ci_lower"]) and np.isfinite(row["ci_upper"])
                and -1<=row["ci_lower"]<=row["ci_upper"]<=1,"Invalid confidence evidence")
    if stage=="develop":
        spec=read(work/"manifest.json")["spec"]
        actual=_winner(spec,result["results"])
        require(result.get("winner")==actual,"Development selection changed")
    else:actual=None
    expected=(actual is not None) if stage=="develop" else (
        result["results"]["primary"]["ci_lower"]>0 and result["results"]["secondary"]["auc_diff"]>=0) if stage=="screen" else (
        result["results"]["five_day"]["ci_lower"]>0 and result["results"]["five_day"]["late"]["auc_diff"]>=0)
    require(result["passed"] is True and bool(expected),stage+" gate failed; recipe is terminal")
    return result


def _winner(spec,results):
    keys={candidate+"/"+case["name"] for candidate in spec["candidate_order"] for case in spec["development"]}
    require(set(results)==keys,"Development comparisons incomplete")
    eligible=[]
    for candidate in spec["candidate_order"]:
        gains=[results[candidate+"/"+case["name"]]["auc_diff"] for case in spec["development"]]
        if min(gains)>=0 and np.mean(gains)>0:eligible.append((candidate,min(gains)))
    if not eligible:return None
    best=max(value for _,value in eligible)
    return next(name for name,value in eligible if best-value<=1e-12)


def _private_replay(root,work,m,number):
    """Execute the unmodified V6 CLI from pinned source with fresh output/receipt."""
    bundle=verify_bundle(root)
    fixture=work/"v6-replays"/str(number)
    fixture.mkdir(parents=True,exist_ok=False)
    names=list(read(work/"source/configs/source_manifest.json")["files"])+[
        "configs/v6.json","configs/source_manifest.json","tools/consolidate.py","tools/predict_v6.py"]
    for name in names:
        dest=inside(fixture,name);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(inside(work/"source",name),dest)
    for name in list(bundle["files"])+["artifacts/v6/manifest.json"]:
        dest=inside(fixture,name);dest.parent.mkdir(parents=True,exist_ok=True)
        os.link(inside(root,name),dest)
    subprocess.run([sys.executable,str(fixture/"tools/predict_v6.py"),"--device",m["spec"]["device"]],cwd=fixture,check=True)
    receipt=read(fixture/"local/v6_relocation_check.json")
    require(receipt["passed"] is True and receipt["submission_sha256"]==V6_SHA,"V6 replay failed")
    require(set(receipt["member_max_abs_differences"])==set(MEMBERS)
        and all(0<=v<=1e-15 for v in receipt["member_max_abs_differences"].values()),"Incomplete V6 member replay")
    return receipt


def finalize(root,work,m,module,raw,winner):
    screen=checked(work,"screen");checked(work,"stress")
    evaluation=events(root,m,"evaluation")
    require((evaluation["t"]>=JAN20+7*DAY).all(),"Evaluation precedes final cutoff")
    original,before=baseline(root,m,"baseline_evaluation",evaluation["rowids"])
    research=m["spec"]["purpose"]=="research"
    if research:
        require(len(evaluation["u"])==52939 and len(np.unique(evaluation["u"]))==52939,"Official evaluation dimensions differ")
        actual_ids=np.load(root/"data/cache/evaluation_rowids.npy",allow_pickle=False)
        require(np.array_equal(actual_ids,evaluation["rowids"].ravel()),"Official evaluation row IDs differ")
    paths={}
    for member in m["spec"]["candidates"][winner]["members"]:
        model=_model(root,work,m,module,raw,winner,member,JAN20+7*DAY)
        combiner=inside(work,screen["results"]["secondary"]["models"][member]["combiner"])
        paths[member]={"model":model,"combiner":combiner}
    example=work/"example.csv"
    import csv
    with example.open("x",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream);writer.writerow(["rowid","predicted_probability"])
        writer.writerows((row,0) for row in evaluation["rowids"].ravel())
    sys.path.insert(0,str(work/"source/legacy"))
    from submission_export import export_submission,validate_csv
    from row_auc_final_utils import verify_csv_probabilities
    predictions=[];csv_hashes=[];files={};v6=[]
    for number in (1,2):
        if research:
            v6.append(_private_replay(root,work,m,number))
            for i,member in enumerate(MEMBERS):
                with np.load(root/f"artifacts/v6/{member}/reference_predictions.npz",allow_pickle=False) as saved:
                    require(np.allclose(original[member],saved["combiner_predictions"].reshape(-1,10),rtol=0,atol=1e-15),"Evaluation baseline is not canonical V6")
        updated=dict(original)
        for member,p in paths.items():
            features,_=_features(root,work,m,module,evaluation,np.arange(len(evaluation["u"])),p["model"],JAN20+7*DAY,raw,use_cache=False)
            require(features["feature_names"].tolist()==screen["results"]["secondary"]["models"][member]["feature_names"],"Final schema differs from validated combiner")
            result=module.predict(features,p["combiner"],evaluation["rowids"].copy())
            require(np.array_equal(result["rowids"],evaluation["rowids"]),"Final prediction order differs")
            updated[member]=result["scores"]
            for path in (p["model"],p["model"].with_suffix(".json"),p["combiner"]):files[path.relative_to(work).as_posix()]=digest(path)
        candidate=compose(updated);predictions.append(candidate)
        path=work/("candidate.csv" if number==1 else "independent-replay.csv")
        export_submission(evaluation["rowids"].ravel(),candidate,example,path,candidate.size)
        validate_csv(path,example,candidate.size);verify_csv_probabilities(path,evaluation["rowids"].ravel(),candidate)
        csv_hashes.append(digest(path));files[path.relative_to(work).as_posix()]=digest(path)
    require(csv_hashes[0]==csv_hashes[1] and np.array_equal(predictions[0],predictions[1]),"Independent saved-model replay differs")
    load(root,work)
    report={"schema_version":1,"complete":True,"passed":True,"purpose":m["spec"]["purpose"],"submission_ready":False,
        "manifest_sha256":digest(work/"manifest.json"),"csv_sha256":csv_hashes[0],"rows":predictions[0].size,
        "independent_replay":True,"files":files,"v6_replays":v6,"winner":winner}
    write_new(work/"finalize.json",report)
    write_new(work/"finalize.seal.json",{"sha256":digest(work/"finalize.json")})
    return 0


def verify(root,run_id):
    work=locate(root,run_id);m=load(root,work)
    for stage in ("develop","screen","stress"):checked(work,stage)
    final=read(work/"finalize.json")
    require(read(work/"finalize.seal.json")["sha256"]==digest(work/"finalize.json") and final["complete"] is True
        and final["manifest_sha256"]==digest(work/"manifest.json") and final["independent_replay"] is True,"Incomplete final verification")
    for name,h in final["files"].items():require(digest(inside(work,name))==h,"Final artifact changed")
    for stage in STAGES:
        require(read(work/(stage+".ledger.json"))["exit_code"]==0,"Stage did not complete")
    research=m["spec"]["purpose"]=="research"
    ready=False
    if research and (work/"published.json").exists():
        receipt=read(work/"published.json")
        config=inside(root,"configs/"+m["spec"]["release_name"]+".json")
        output=inside(root,"artifacts/submissions/"+m["spec"]["release_name"]+".csv")
        require(config.is_file() and read(config)==receipt and receipt["manifest_sha256"]==digest(work/"manifest.json")
            and receipt["submission_ready"] is True and digest(output)==final["csv_sha256"],"Publication incomplete or changed")
        ready=True
    return {"run_id":run_id,"purpose":m["spec"]["purpose"],"validation_complete":True,
        "submission_ready":ready,"rows":final["rows"],"csv_sha256":final["csv_sha256"],"source_commit":m["source"]["commit"]}


def publish(root,work,m):
    """Production readiness is published only after successful phase accounting."""
    require(m["spec"]["purpose"]=="research","Synthetic runs cannot publish a submission")
    verify(root,m["run_id"])
    final=read(work/"finalize.json");name=m["spec"]["release_name"]
    output=inside(root,"artifacts/submissions/"+name+".csv");config=inside(root,"configs/"+name+".json")
    require(not config.exists() and not output.exists(),"Preserve existing release")
    output.parent.mkdir(parents=True,exist_ok=True)
    os.link(work/"candidate.csv",output)
    receipt={"schema_version":1,"submission_ready":True,"purpose":"research","run_id":m["run_id"],
        "csv_sha256":final["csv_sha256"],"source_snapshot_sha256":m["source"]["sha256"],"manifest_sha256":digest(work/"manifest.json"),
        "final_report_sha256":digest(work/"finalize.json"),"competition_upload_performed":False}
    write_new(work/"published.json",receipt)
    candidate=work/"ready-config.json";write_new(candidate,receipt);os.link(candidate,config)


def _worker_lease(root,work,m,stage):
    ledger=work/"compute.json" if m["spec"]["purpose"]=="synthetic_validation" else inside(root,"artifacts/experiments/compute-ledger.json")
    claim=read(work/(stage+".attempt.json"));state=read(ledger)
    attempt=state["attempts"][-1]
    require(claim["spec_sha256"]==m["spec_sha256"] and attempt["owner"]==m["run_id"]+"/"+stage
        and attempt["status"]=="running" and attempt["controller_pid"]==os.getppid(),"Worker requires active budget-owning controller")
    return attempt["controller_pid"]


def worker(root,work,stage):
    m=load(root,work);controller=_worker_lease(root,work,m,stage)
    def watch_controller():
        while True:
            if os.getppid()!=controller:
                os.killpg(os.getpgrp(),signal.SIGKILL)
            time.sleep(.1)
    threading.Thread(target=watch_controller,daemon=True).start()
    module=adapter(work,m);raw=events(root,m,"train")
    if stage=="develop":
        results={};eligible=[]
        for candidate in m["spec"]["candidate_order"]:
            gains=[]
            for value in m["spec"]["development"]:
                row=_score(root,work,m,module,raw,candidate,CaseSpec.parse(value))
                results[candidate+"/"+value["name"]]=row;gains.append(row["auc_diff"])
            if min(gains)>=0 and np.mean(gains)>0:eligible.append((candidate,min(gains)))
        winner=_winner(m["spec"],results)
        report=_report(work,m,stage,results,winner is not None,{"winner":winner,"selection_rule":m["spec"]["selection"]})
    else:
        development=checked(work,"develop");winner=development["winner"]
        if stage in {"stress","finalize"}:checked(work,"screen")
        if stage=="finalize":
            checked(work,"stress")
            return finalize(root,work,m,module,raw,winner)
        names=("primary","secondary") if stage=="screen" else ("five_day",)
        results={name:_score(root,work,m,module,raw,winner,CaseSpec.parse(m["spec"]["promotion"][name])) for name in names}
        passed=(results["primary"]["ci_lower"]>0 and results["secondary"]["auc_diff"]>=0) if stage=="screen" else (
            results["five_day"]["ci_lower"]>0 and results["five_day"]["late"]["auc_diff"]>=0)
        report=_report(work,m,stage,results,passed,{"winner":winner})
    return 0 if report["passed"] else 2


def run(root,run_id,stage):
    root=Path(root).resolve();work=locate(root,run_id);m=load(root,work)
    require(stage in STAGES,"Unknown phase")
    for prerequisite in STAGES[:STAGES.index(stage)]:checked(work,prerequisite)
    require(not (work/(stage+".attempt.json")).exists(),"Phase attempted; preserve evidence and use a new reviewed run")
    spec=m["spec"]
    ledger=work/"compute.json" if spec["purpose"]=="synthetic_validation" else inside(root,"artifacts/experiments/compute-ledger.json")
    write_new(work/(stage+".attempt.json"),{"stage":stage,"spec_sha256":m["spec_sha256"],"state":"claimed"})
    with Budget(ledger,run_id+"/"+stage,spec["budget"]["phase_seconds"],spec["budget"]["total_seconds"],m["carried_seconds"],m["carry_identity"]) as budget:
        command=[sys.executable,str(work/"source/tools/run_experiment.py"),stage,"--root",str(root),"--run-id",run_id,"--worker"]
        code=budget.run(command,work/"source")
        require(code in (0,2),"Phase subprocess failed with exit code "+str(code))
    write_new(work/(stage+".ledger.json"),{"exit_code":code,"compute":read(ledger),"source_commit":m["source"]["commit"]})
    if code==0 and stage=="finalize" and spec["purpose"]=="research":publish(root,work,m)
    return code
