"""One Jan21 ensemble-selection campaign, then one frozen composition's promotion."""
import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import robust_core as core

ROOT = core.ROOT
sys.path.insert(0,str(ROOT/"tools"))
spec = importlib.util.spec_from_file_location("robust_seed44_helpers",ROOT/"experiments/v8-seed44-row-auc/runner.py")
seed44 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seed44)
old = seed44.old
from experiment_runner import PairedBootstrapEvaluator, compute_roc_auc
from predict_v6 import portable_pipeline
from predict_v8 import replay_v6

WORK,REPORTS = core.WORK,core.REPORTS
save_npz,runtime = old.save_npz,old.runtime


def parameters(path):
    saved,_ = old.original_parameters(path,list(core.BASE_COLUMNS))
    return core.prior.previous.prior.classifier_parameters(saved)


def features(root,manifest,raw,member,fold,indices,checkpoint,name,budget,device="cuda"):
    import torch
    from build_submission import neural_pipeline,FIT_FIELDS,TARGET_FIELDS
    indices = np.asarray(indices,dtype=np.int64)
    cutoff = core.FOLDS[fold]
    core.require(len(indices)>0 and (raw["t"][indices]>=cutoff).all(),"Feature event before cutoff")
    identity = {"checkpoint_sha256":core.sha256(checkpoint),"cutoff":cutoff,
        "indices_sha256":__import__("hashlib").sha256(indices.tobytes()).hexdigest(),"member":member,
        "inputs_and_code":core.canonical({"inputs":manifest["inputs_sha256"],"code":manifest["code_sha256"]}),"device":device}
    path = root/WORK/"cache"/(name+".npz")
    metadata = path.with_suffix(".json")
    if path.exists() or metadata.exists():
        record = core.read_json(metadata)
        core.require(record["identity"]==identity and core.sha256(path)==record["sha256"],"Stale feature cache")
        with np.load(path,allow_pickle=False) as d:
            core.require(np.array_equal(d["indices"],indices) and tuple(d["feature_names"])==core.BASE_COLUMNS,"Cache index/schema differs")
            return d["X"]
    fitting = np.flatnonzero(raw["t"]<cutoff)
    budget.check()
    x,names,*_ = portable_pipeline(neural_pipeline)(checkpoint,{k:raw[k][fitting] for k in FIT_FIELDS},
        {k:raw[k][indices] for k in TARGET_FIELDS},root/manifest["prefixes"][fold],torch.device(device))
    core.require(tuple(names)==core.BASE_COLUMNS and x.shape==(len(indices)*10,37) and np.isfinite(x).all(),"Feature schema/values differ")
    x = x.astype(np.float32)
    save_npz(path,X=x,indices=indices,feature_names=np.asarray(names))
    core.write_new(metadata,{"identity":identity,"sha256":core.sha256(path)})
    budget.check()
    return x


def fit_combiner(root,manifest,member,case,x,y,target,name,budget):
    from catboost import CatBoostClassifier,Pool
    path = root/WORK/"combiners"/(name+".cbm")
    core.require(not path.exists(),"Preserve existing combiner")
    model = CatBoostClassifier(**manifest["parameters"][member][case])
    budget.check()
    model.fit(Pool(x,label=y.ravel(),feature_names=list(core.BASE_COLUMNS)),verbose=False)
    path.parent.mkdir(parents=True,exist_ok=True)
    model.save_model(str(path))
    prediction = model.predict_proba(Pool(target,feature_names=list(core.BASE_COLUMNS)),thread_count=4)[:,1].reshape(-1,10)
    reloaded = CatBoostClassifier()
    reloaded.load_model(str(path))
    core.require(model.tree_count_==545 and np.allclose(prediction.ravel(),reloaded.predict_proba(target,thread_count=4)[:,1],rtol=0,atol=1e-15),"Combiner replay differs")
    budget.check()
    return prediction,{path.relative_to(root).as_posix():core.sha256(path)}


def prepare(root,source_root):
    import torch
    from build_submission import load_combiner,predict_combiner
    core.require(not (root/WORK).exists(),"Preserve prior robust-ensemble attempt")
    previous_budget = root/core.prior.WORK/"budget.json"
    carry = core.read_json(previous_budget)
    decision_path = root/core.prior.REPORTS/"decision.json"
    decision = core.read_json(decision_path)
    core.require(decision["status"]=="rejected_at_five_day_gate" and carry["used_seconds"]==decision["total_compute_seconds"],"Previous budget/decision differs")
    core.write_new(root/WORK/"budget.json",{"used_seconds":carry["used_seconds"],"carried_seconds":carry["used_seconds"],
        "phases":{},"carry_from":previous_budget.relative_to(root).as_posix(),"carry_sha256":core.sha256(previous_budget)})
    with core.Budget(root/WORK,"prepare") as budget:
        core.require(torch.cuda.is_available(),"Existing CUDA environment required")
        previous = core.prior.load_manifest(root)
        inputs = dict(previous["inputs_sha256"])
        code = {p:core.sha256(root/p) for p in core.CODE_FILES}
        for p in (previous_budget,decision_path,root/core.prior.WORK/"inputs.json",root/core.prior.REPORTS/"preparation.json"):
            inputs[p.relative_to(root).as_posix()] = core.sha256(p)
        def stage(relative,source,expected=None):
            path = core.inside(source_root,source)
            h = core.sha256(path)
            core.require(expected is None or h==expected,f"Reviewed input changed: {source}")
            destination = WORK/"source"/relative
            old.copy_checked(path,root/destination,h)
            inputs[destination.as_posix()] = h
            return destination.as_posix()
        # Reuse the closed seed44 component, never its promotion decision.
        seed44_models = {}
        for fold in ("jan21","jan22","jan24","jan25","jan26"):
            path = core.prior.neural_path(fold)
            core.prior.validate_neural(root,previous,fold)
            core.require(core.sha256(root/path)==decision["training_summary"][fold]["checkpoint_sha256"],"Closed seed44 checkpoint changed")
            seed44_models[fold] = path.as_posix()
            for p in (path,path.with_suffix(".json")):
                inputs[p.as_posix()] = core.sha256(root/p)
        seed44_outputs,seed44_combiners = {},{}
        for case in core.CASE_FOLDS:
            stage_name = "stress" if case=="five_day" else "screen"
            report_path = core.prior.REPORTS/(stage_name+".json")
            report = core.read_json(root/report_path)
            result = report["results"] if case=="five_day" else report["results"][case]
            core.verify_files(root,{result["predictions"]:result["predictions_sha256"],result["fit"]["model"]:result["fit"]["model_sha256"]})
            seed44_outputs[case],seed44_combiners[case] = result["predictions"],result["fit"]["model"]
            for p in (report_path,Path(result["predictions"]),Path(result["fit"]["model"])):
                inputs[p.as_posix()] = core.sha256(root/p)
        long_report_path = root/core.prior.WORK/"source/reports/five_training.json"
        long_report = core.read_json(long_report_path)
        row_report_path = root/"artifacts/experiments/v8-pairlogit/source/reports/five_training.json"
        row_report = core.read_json(row_report_path)
        original_models = {"dim8_seed44":previous["original_models"]}
        original_combiners = {"dim8_seed44":previous["original_combiners"]}
        preferences = {"dim8_seed44":previous["preferences"]}
        reference_paths = {}
        headers = {}
        for member in core.PAIR:
            study = "techarena-dim16-trial" if member.endswith("42") else "techarena-dim16-seed43-trial"
            seed = core.MEMBERS[member]["seed"]
            reports = {}
            for name in ("frozen_protocol","neural_training_results"):
                reports[name] = stage(f"{member}/{name}.json",f"local/archive/{study}/outputs/{name}.json")
            protocol = core.read_json(root/reports["frozen_protocol"])
            training = core.read_json(root/reports["neural_training_results"])
            core.require(protocol["dimension"]==16 and protocol["neural_seed"]==seed and protocol["preference_epochs"]==196,"Historical dim16 protocol differs")
            preferences[member],original_models[member],original_combiners[member],reference_paths[member] = {},{},{},{}
            for fold in ("jan21","jan22","jan24","jan25","jan26"):
                early = fold in {"jan21","jan22"}
                folder = f"local/archive/techarena-v5-long-window/{member}" if early else f"local/archive/{study}"
                for kind,filename in (("preference","preference_best.pt"),("event","event_best.pt")):
                    expected = (long_report["members"][member][fold][filename]["sha256"] if early else
                                core.PREFERENCE_HASHES[member][fold] if kind=="preference" else None)
                    relative = stage(f"{member}/{fold}/{filename}",f"{folder}/{fold}/work/{filename}",expected)
                    (preferences[member] if kind=="preference" else original_models[member])[fold] = relative
                    d = torch.load(root/relative,map_location="cpu",weights_only=True)
                    core.require(d["dimension"]==16 and d["epoch"]==(196 if kind=="preference" else 8),"Historical member epoch/dimension differs")
                    for k,v in (("seed",seed),("cutoff",core.FOLDS[fold])):
                        core.require(k not in d or d[k]==v,"Conflicting historical metadata")
                    if not early:
                        core.require(d["validation_auc"]==training["folds"][fold][kind]["validation_auc"],"Historical training report mismatch")
                    headers[f"{member}/{fold}/{kind}"] = {k:d.get(k) for k in ("dimension","epoch","seed","cutoff","n_users")}
                    core.require(all(bool(torch.isfinite(v).all()) for v in d["model_state"].values() if v.is_floating_point()),"Nonfinite historical weights")
            preferences[member]["jan27"] = f"artifacts/v6/{member}/preference.pt"
            for case,day in (("primary","jan24"),("secondary","jan25"),("five_day","jan21")):
                source = (f"local/archive/techarena-v5-long-window/work/{member}_combiner_jan21.cbm" if case=="five_day" else
                          f"local/archive/{study}/work/combiner_{day}.cbm")
                original_combiners[member][case] = stage(f"{member}/{case}.cbm",source)
                if case!="five_day":
                    filename = "two_day_ensemble_predictions.npz" if case=="primary" else "jan26_ensemble_predictions.npz"
                    reference_paths[member][case] = stage(f"{member}/{filename}",f"local/archive/{study}/work/{filename}",core.PREDICTION_HASHES[member][case])
            core.require(core.sha256(root/original_combiners[member]["secondary"])==core.sha256(root/f"artifacts/v6/{member}/combiner.cbm"),"Final combiner does not match Jan25")
        # The retained dim8 members already use mixed loss. Only Jan21 models are
        # needed for fresh development-reference fitting.
        for member in ("dim8_seed42","dim8_seed43"):
            original_models[member] = {"jan21":stage(f"{member}/jan21/event.pt",
                f"local/archive/techarena-row-auc-five-day/{member}/jan21/work/event_best.pt",
                row_report["members"][member]["jan21"]["sha256"])}
        params = {}
        for member in core.MEMBERS:
            params[member] = {"development":parameters(root/f"artifacts/v6/{member}/combiner.cbm")}
            if member in core.PAIR:
                params[member].update({case:parameters(root/path) for case,path in original_combiners[member].items()})
        manifest = {"protocol":core.PROTOCOL,"protocol_sha256":core.canonical(core.PROTOCOL),"code_sha256":code,
            "inputs_sha256":inputs,"preferences":preferences,"original_models":original_models,"original_combiners":original_combiners,
            "parameters":params,"seed44_parameters":previous["parameters"],"cohorts":previous["cohorts"],"seed44_models":seed44_models,"seed44_outputs":seed44_outputs,
            "seed44_combiners":seed44_combiners,"prefixes":{fold:(core.prior.WORK/"prefix"/(fold+".npz")).as_posix() for fold in core.FOLDS},
            "original_predictions":{member:{} for member in core.PAIR},"runtime":runtime()}
        raw = old.load_raw(root)
        replay_checks = {}
        for member in core.PAIR:
            replay_checks[member] = {}
            for case,(_,fold) in core.CASE_FOLDS.items():
                idx,y,baseline,_,_ = old.audit.load_cohort(root/manifest["cohorts"][case],old.audit.SPECS[case],raw)
                if case=="five_day":
                    with np.load(root/manifest["cohorts"][case],allow_pickle=False) as data:
                        expected = data["candidate_members"][list(core.MEMBERS).index(member)]
                else:
                    with np.load(root/reference_paths[member][case],allow_pickle=False) as data:
                        core.require(np.array_equal(data["valid_idx"],idx) and np.array_equal(data["labels"],y),"Reference cohort differs")
                        expected = data["dim16" if member.endswith("42") else "new_dim16"]
                x = features(root,manifest,raw,member,fold,idx,root/original_models[member][fold],f"verify_{member}_{case}",budget)
                p = predict_combiner(load_combiner(root/original_combiners[member][case]),x,list(core.BASE_COLUMNS)).reshape(-1,10)
                core.require(np.allclose(p,expected,rtol=0,atol=1e-15),"Original dim16 replay differs")
                cache = root/WORK/"cache"/f"original_{member}_{case}.npz"
                save_npz(cache,indices=idx,probabilities=p)
                manifest["original_predictions"][member][case] = cache.relative_to(root).as_posix()
                inputs[cache.relative_to(root).as_posix()] = core.sha256(cache)
                replay_checks[member][case] = float(np.max(np.abs(p-expected)))
        baseline,_,_,_,ids,errors = replay_v6(root,"cuda")
        cache = root/WORK/"cache/baseline_evaluation.npz"
        save_npz(cache,baseline=baseline,rowids=ids)
        manifest["baseline_cache"] = cache.relative_to(root).as_posix()
        inputs[manifest["baseline_cache"]] = core.sha256(cache)
        core.verify_files(root,inputs); core.verify_files(root,code); budget.check()
        core.write_new(root/WORK/"inputs.json",manifest)
        core.write_new(root/REPORTS/"preparation.json",{"complete":True,"manifest_sha256":core.sha256(root/WORK/"inputs.json"),
            "protocol_sha256":core.canonical(core.PROTOCOL),"source_sha256":code,"dim16_replay_errors":replay_checks,
            "v6_replay_errors":errors,"v6_csv_sha256":core.V6_SHA,"headers":headers,"runtime":runtime(),
            "carried_seconds":carry["used_seconds"],"new_neural_training_performed":False})


def train_fold(root,manifest,member,fold,budget):
    path = root/core.neural_path(member,fold)
    if not path.exists():
        budget.run([sys.executable,"-u",str(root/"experiments/v8-robust-ensemble/train_member.py"),member,fold],root)
    core.validate_new(root,manifest,member,fold)


def updated_checkpoint(root,manifest,member,fold):
    return root/(manifest["seed44_models"][fold] if member=="dim8_seed44" and fold!="jan27" else core.neural_path(member,fold))


def diagnostics(raw,idx,y,baseline,candidate,cutoff):
    warm = np.isin(raw["u"][idx],np.unique(raw["u"][raw["t"]<cutoff]))
    history = (raw["h"][idx]>=0).sum(1)
    days = ((raw["t"][idx]-cutoff)//core.DAY).astype(int)
    masks = {"warm":warm,"cold":~warm,"history_0":history==0,"history_1_5":(history>=1)&(history<=5),"history_6_20":history>=6}
    masks.update({f"day_{d}":days==d for d in np.unique(days)})
    results = {}
    for name,mask in masks.items():
        entry = {"events":int(mask.sum())}
        if mask.any() and len(np.unique(y[mask]))==2:
            a,b = compute_roc_auc(y[mask],baseline[mask]),compute_roc_auc(y[mask],candidate[mask])
            entry.update(baseline_auc=a,candidate_auc=b,auc_diff=b-a)
        results[name] = entry
    temporal = core.pair_decomposition(y,baseline,candidate,days)
    core.require(abs(temporal["reconstructed_change"]-(compute_roc_auc(y,candidate)-compute_roc_auc(y,baseline)))<1e-12,"Temporal pair reconstruction differs")
    return {"strata":results,"temporal_pairs":temporal,"used_for_routing":False}


def develop(root):
    manifest = core.load_manifest(root)
    core.require(not (root/REPORTS/"development.json").exists(),"Preserve development result")
    with core.Budget(root/WORK,"development") as budget:
        raw = old.load_raw(root)
        idx = core.first_indices(raw["u"],raw["t"],core.FOLDS["jan21"],core.FOLDS["jan22"])
        y,times = raw["y"][idx],raw["t"][idx]
        original_x,updated_x,files = {},{},{}
        for member in core.PAIR:
            train_fold(root,manifest,member,"jan21",budget)
        for member in core.MEMBERS:
            original_x[member] = features(root,manifest,raw,member,"jan21",idx,root/manifest["original_models"][member]["jan21"],"dev_original_"+member,budget)
        for member in core.COMPOSITIONS["B"]:
            path = updated_checkpoint(root,manifest,member,"jan21")
            updated_x[member] = features(root,manifest,raw,member,"jan21",idx,path,"dev_updated_"+member,budget)
            files[path.relative_to(root).as_posix()] = core.sha256(path)
        results = {key:{"folds":[]} for key in core.COMPOSITIONS}
        for number,(fit,valid) in enumerate(core.development_folds(times)):
            original,updated = {},{}
            for member in core.MEMBERS:
                x = original_x[member]
                original[member],assets = fit_combiner(root,manifest,member,"development",x[np.repeat(fit,10)],y[fit],x[np.repeat(valid,10)],f"development_{number}_original_{member}",budget)
                files.update(assets)
            for member in core.COMPOSITIONS["B"]:
                x = updated_x[member]
                updated[member],assets = fit_combiner(root,manifest,member,"development",x[np.repeat(fit,10)],y[fit],x[np.repeat(valid,10)],f"development_{number}_updated_{member}",budget)
                files.update(assets)
            baseline = core.original_ensemble(original)
            before = compute_roc_auc(y[valid],baseline)
            for key,members in core.COMPOSITIONS.items():
                candidate = core.compose(baseline,original,{m:updated[m] for m in members},key)
                after = compute_roc_auc(y[valid],candidate)
                decomposition = old.allocation.allocation_decomposition(y[valid],baseline,candidate)
                no_click = decomposition["negative_event_partitions"]["negative_from_no_click_event"]["weighted_auc_change"]
                path = root/WORK/f"development_{number}_{key}.npz"
                save_npz(path,fit_idx=idx[fit],indices=idx[valid],labels=y[valid],baseline=baseline,candidate=candidate)
                files[path.relative_to(root).as_posix()] = core.sha256(path)
                results[key]["folds"].append({"fold":number,"baseline_auc":before,"candidate_auc":after,"auc_diff":after-before,
                    "no_click_contribution":no_click,"decomposition":decomposition,"fit_events":int(fit.sum()),"valid_events":int(valid.sum()),
                    "maximum_fit_timestamp":int(times[fit].max()),"minimum_validation_timestamp":int(times[valid].min()),
                    "diagnostics":diagnostics(raw,idx[valid],y[valid],baseline,candidate,core.FOLDS["jan21"])})
        winner = core.choose(results)
        core.load_manifest(root)
        path = root/REPORTS/"development.json"
        core.write_new(path,{"complete":True,"protocol_sha256":core.canonical(core.PROTOCOL),"source_sha256":manifest["code_sha256"],
            "input_manifest_sha256":core.sha256(root/WORK/"inputs.json"),"winner":winner,"results":results,
            "files_sha256":files,"promotion_labels_used_for_selection":False,"runtime":runtime()})
        if winner is not None:
            core.write_new(root/REPORTS/"selected.json",{"composition":winner,"members":list(core.COMPOSITIONS[winner]),
                "protocol_sha256":core.canonical(core.PROTOCOL),"development_sha256":core.sha256(path)})
        print(json.dumps({"winner":winner,"gains":{k:[f["auc_diff"] for f in v["folds"]] for k,v in results.items()},
            "no_click_contributions":{k:[f["no_click_contribution"] for f in v["folds"]] for k,v in results.items()}}),flush=True)
    return winner is not None


def case_arrays(root,manifest,raw,case):
    idx,y,baseline,_,_ = old.audit.load_cohort(root/manifest["cohorts"][case],old.audit.SPECS[case],raw)
    original = {}
    for member in core.PAIR:
        with np.load(root/manifest["original_predictions"][member][case],allow_pickle=False) as d:
            core.require(np.array_equal(d["indices"],idx),"Original member event order differs")
            original[member] = d["probabilities"]
    with np.load(root/manifest["seed44_outputs"][case],allow_pickle=False) as d:
        core.require(np.array_equal(d["indices"],idx) and np.array_equal(d["labels"],y)
            and np.array_equal(d["baseline"],baseline),"Reused seed44 cohort differs")
        original["dim8_seed44"],new44 = d["old_member"],d["new_member"]
    return idx,y,baseline,original,new44


def score_case(root,manifest,raw,case,fit_idx,idx,y,baseline,original,new44,x_fit,x_target,budget):
    composition = core.selection(root,manifest)
    updated,files,fits = {},{},{}
    for member in core.PAIR:
        updated[member],assets = fit_combiner(root,manifest,member,case,x_fit[member],raw["y"][fit_idx],x_target[member],f"{case}_{member}",budget)
        files.update(assets)
        neural = {core.neural_path(member,f).as_posix():core.sha256(root/core.neural_path(member,f)) for f in core.CASE_FOLDS[case]}
        files.update(neural)
        fits[member] = {"combiner":next(iter(assets)),"parameters":manifest["parameters"][member][case],"neural_sha256":neural}
    if composition=="B":
        updated["dim8_seed44"] = new44
        fits["dim8_seed44"] = {"combiner":manifest["seed44_combiners"][case],"reused_from_closed_trial":True,
            "neural_sha256":{manifest["seed44_models"][f]:core.sha256(root/manifest["seed44_models"][f]) for f in core.CASE_FOLDS[case]}}
    candidate = core.compose(baseline,original,updated,composition)
    path = root/WORK/(case+"_predictions.npz")
    arrays = {"indices":idx,"labels":y,"baseline":baseline,"candidate":candidate}
    arrays.update({"new_"+m:p for m,p in updated.items()})
    save_npz(path,**arrays)
    files[path.relative_to(root).as_posix()] = core.sha256(path)
    metrics = asdict(PairedBootstrapEvaluator().evaluate(candidate,baseline,y,raw["u"][idx]))
    metrics.update({"n_bootstrap":1000,"valid_bootstrap_samples":metrics["resamples"],"events":len(idx),"fitting_events":len(fit_idx),
        "member_models":fits,"predictions":path.relative_to(root).as_posix(),"predictions_sha256":core.sha256(path),
        "decomposition":old.allocation.allocation_decomposition(y,baseline,candidate),
        "diagnostics":diagnostics(raw,idx,y,baseline,candidate,core.FOLDS[core.CASE_FOLDS[case][1]])})
    core.prior.previous.prior.validate_metrics(metrics)
    budget.check()
    return metrics,candidate,files


def report(root,manifest,stage,results,checks,files):
    core.load_manifest(root)
    core.write_new(root/REPORTS/(stage+".json"),{"complete":True,"submission_ready":False,
        "protocol_sha256":core.canonical(core.PROTOCOL),"source_sha256":manifest["code_sha256"],
        "input_manifest_sha256":core.sha256(root/WORK/"inputs.json"),"selection_sha256":core.sha256(root/REPORTS/"selected.json"),
        "composition":core.selection(root,manifest),"results":results,"decision_checks":checks,"passed":all(checks.values()),
        "files_sha256":files,"runtime":runtime()})
    print(json.dumps({"stage":stage,"checks":checks,"passed":all(checks.values())}),flush=True)


def screen(root):
    manifest = core.load_manifest(root)
    core.selection(root,manifest)
    core.require(not (root/REPORTS/"screen.json").exists(),"Preserve screen result")
    with core.Budget(root/WORK,"screen") as budget:
        raw = old.load_raw(root)
        idx25,y25,base25,orig25,new44_25 = case_arrays(root,manifest,raw,"primary")
        idx26,y26,base26,orig26,new44_26 = case_arrays(root,manifest,raw,"secondary")
        meta24 = core.first_indices(raw["u"],raw["t"],core.FOLDS["jan24"],core.FOLDS["jan25"])
        day25 = raw["t"][idx25]<core.FOLDS["jan26"]
        meta25 = idx25[day25]
        core.temporal_check(raw,meta24,idx25,core.FOLDS["jan24"],core.FOLDS["jan25"])
        core.temporal_check(raw,meta25,idx26,core.FOLDS["jan25"],core.FOLDS["jan26"])
        x24,x25,x26 = {},{},{}
        for member in core.PAIR:
            for fold in ("jan24","jan25","jan26"):
                train_fold(root,manifest,member,fold,budget)
            for fold,idx,output in (("jan24",meta24,x24),("jan25",idx25,x25),("jan26",idx26,x26)):
                output[member] = features(root,manifest,raw,member,fold,idx,root/core.neural_path(member,fold),f"new_{member}_{fold}",budget)
        results = {}
        results["primary"],_,files = score_case(root,manifest,raw,"primary",meta24,idx25,y25,base25,orig25,new44_25,x24,x25,budget)
        results["secondary"],_,more = score_case(root,manifest,raw,"secondary",meta25,idx26,y26,base26,orig26,new44_26,
            {m:x25[m][np.repeat(day25,10)] for m in core.PAIR},x26,budget)
        files.update(more)
        checks = core.screen_gates(results)
        report(root,manifest,"screen",results,checks,files)
    return all(checks.values())


def stress(root):
    manifest = core.load_manifest(root)
    core.validate_report(root,manifest,"screen")
    core.require(not (root/REPORTS/"stress.json").exists(),"Preserve stress result")
    with core.Budget(root/WORK,"stress") as budget:
        raw = old.load_raw(root)
        idx,y,baseline,original,new44 = case_arrays(root,manifest,raw,"five_day")
        meta = core.first_indices(raw["u"],raw["t"],core.FOLDS["jan21"],core.FOLDS["jan22"])
        core.temporal_check(raw,meta,idx,core.FOLDS["jan21"],core.FOLDS["jan22"])
        x_fit,x_target = {},{}
        for member in core.PAIR:
            train_fold(root,manifest,member,"jan21",budget)
            train_fold(root,manifest,member,"jan22",budget)
            x_fit[member] = features(root,manifest,raw,member,"jan21",meta,root/core.neural_path(member,"jan21"),"dev_updated_"+member,budget)
            x_target[member] = features(root,manifest,raw,member,"jan22",idx,root/core.neural_path(member,"jan22"),"five_target_"+member,budget)
        result,candidate,files = score_case(root,manifest,raw,"five_day",meta,idx,y,baseline,original,new44,x_fit,x_target,budget)
        hours = (raw["t"][idx]-core.FOLDS["jan22"])/3600
        late = (hours>=48)&(hours<120)
        core.require(int(late.sum())==7596,"Late cohort differs")
        a,b = compute_roc_auc(y[late],baseline[late]),compute_roc_auc(y[late],candidate[late])
        result["late_48_120h"] = {"events":int(late.sum()),"baseline_auc":a,"candidate_auc":b,"auc_diff":b-a}
        checks = core.stress_gates(result)
        report(root,manifest,"stress",result,checks,files)
    return all(checks.values())


def finalize(root):
    from predict_v8_candidate import release_candidate
    manifest = core.load_manifest(root)
    composition = core.selection(root,manifest)
    core.validate_report(root,manifest,"screen")
    core.validate_report(root,manifest,"stress")
    core.require(not (root/"configs/v8.json").exists(),"Preserve existing release")
    with core.Budget(root/WORK,"finalize") as budget:
        for member in core.COMPOSITIONS[composition]:
            train_fold(root,manifest,member,"jan27",budget)
    with core.Budget(root/WORK,"verify") as budget:
        release_candidate(root,manifest,budget)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","development","screen","stress","finalize"])
    parser.add_argument("--source-root",type=Path,default=ROOT.parents[2])
    args = parser.parse_args()
    if args.action=="prepare":
        prepare(ROOT,args.source_root.resolve())
    elif not {"development":develop,"screen":screen,"stress":stress,"finalize":finalize}[args.action](ROOT):
        raise SystemExit(2)


if __name__=="__main__":
    main()
