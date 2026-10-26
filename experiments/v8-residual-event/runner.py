"""One bounded three-composition residual experiment and gated release."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import residual_core as core

ROOT, WORK, REPORTS = core.ROOT, core.WORK, core.REPORTS
sys.path.insert(0,str(ROOT/"tools"))
spec = importlib.util.spec_from_file_location("residual_prior_helpers",ROOT/"experiments/v8-robust-ensemble/runner.py")
helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)
old = helpers.old
save_npz, runtime = old.save_npz, old.runtime
from experiment_runner import PairedBootstrapEvaluator, compute_roc_auc


def cohort(root, manifest, raw, case):
    idx,y,baseline,_,_ = old.audit.load_cohort(root/manifest["cohorts"][case],old.audit.SPECS[case],raw)
    with np.load(root/manifest["cohorts"][case],allow_pickle=False) as d:
        original = ({m:d["candidate_members"][n] for n,m in enumerate(core.MEMBERS)} if case == "five_day" else
                    {m:d["new_"+m] for m in ("dim8_seed42","dim8_seed43")})
    if case != "five_day":
        for member in core.prior.PAIR:
            with np.load(root/manifest["previous_predictions"][member][case],allow_pickle=False) as d:
                core.require(np.array_equal(d["indices"],idx), "Original member indices differ")
                original[member] = d["probabilities"]
        with np.load(root/manifest["previous_seed44_predictions"][case],allow_pickle=False) as d:
            core.require(np.array_equal(d["indices"],idx) and np.array_equal(d["baseline"],baseline), "Original seed44 alignment differs")
            original["dim8_seed44"] = d["old_member"]
    core.require(np.allclose(core.original_ensemble(original),baseline,rtol=0,atol=1e-15), "Saved V6 ensemble differs")
    return idx,y,baseline,original


def prepare(root, source_root):
    import torch
    from residual_model import pipeline, ResidualAdapter
    from build_submission import load_combiner,predict_combiner,neural_pipeline,FIT_FIELDS,TARGET_FIELDS
    from predict_v6 import portable_pipeline
    from predict_v8 import replay_v6
    core.require(not (root/WORK).exists(), "Preserve previous residual attempt")
    carry_path = root/core.prior.WORK/"budget.json"
    carry = core.read_json(carry_path)
    core.require(all(v["status"] != "running" for v in carry["phases"].values()), "Previous compute is still active")
    core.write_new(root/WORK/"budget.json",{"used_seconds":carry["used_seconds"],"carried_seconds":carry["used_seconds"],
        "carry_from":carry_path.relative_to(root).as_posix(),"carry_sha256":core.sha256(carry_path),"phases":{}})
    with core.Budget(root/WORK,"prepare") as budget:
        core.require(torch.cuda.is_available(), "Existing CUDA environment required")
        previous = core.prior.load_manifest(root)
        row = core.read_json(root/"artifacts/experiments/v8-pairlogit/inputs.json")
        inputs = dict(previous["inputs_sha256"])
        inputs.update(row["inputs_sha256"]); inputs.update(row["prepared_sha256"])
        code = {p:core.sha256(root/p) for p in core.CODE_FILES}
        original_models = {m:dict(v) for m,v in previous["original_models"].items()}
        original_models["dim8_seed42"] = row["models"]
        original_combiners = {m:dict(v) for m,v in previous["original_combiners"].items()}
        original_combiners["dim8_seed42"] = row["combiners"]
        row_train = core.read_json(root/"artifacts/experiments/v8-pairlogit/source/reports/row_training.json")
        five_train = core.read_json(root/"artifacts/experiments/v8-pairlogit/source/reports/five_training.json")
        row_result = core.read_json(root/"docs/results/techarena-row-auc-trial/ensemble_trial_results.json")
        five_result = core.read_json(root/"docs/results/techarena-row-auc-five-day/paired_results.json")
        def stage(relative, study, expected):
            source = core.inside(source_root,"local/archive/"+study+"/"+relative)
            destination = WORK/"source"/study/relative
            old.copy_checked(source,root/destination,expected)
            inputs[destination.as_posix()] = expected
            return destination.as_posix()
        original_models["dim8_seed43"] = {}
        for fold in core.FOLDS:
            if fold == "jan27":
                continue
            early = fold in {"jan21","jan22"}
            study = "techarena-row-auc-five-day" if early else "techarena-row-auc-trial"
            expected = (five_train if early else row_train)["members"]["dim8_seed43"][fold]["sha256"]
            original_models["dim8_seed43"][fold] = stage(f"dim8_seed43/{fold}/work/event_best.pt",study,expected)
        original_combiners["dim8_seed43"] = {}
        for case,day in (("primary","jan24"),("secondary","jan25"),("five_day","jan21")):
            relative = "work/dim8_seed43_combiner_jan21.cbm" if case == "five_day" else f"work/dim8_seed43_{day}.cbm"
            study = "techarena-row-auc-five-day" if case == "five_day" else "techarena-row-auc-trial"
            original_combiners["dim8_seed43"][case] = stage(relative,study,(five_result if case == "five_day" else row_result)["artifact_sha256"][relative])
        for member in core.MEMBERS:
            original_models[member]["jan27"] = f"artifacts/v6/{member}/event.pt"
        for p in (carry_path,root/core.prior.WORK/"inputs.json",root/core.prior.REPORTS/"preparation.json"):
            inputs[p.relative_to(root).as_posix()] = core.sha256(p)
        # Bind all reused prediction caches and their original provenance reports.
        for stage_name in ("screen","stress"):
            path = root/core.prior.prior.REPORTS/(stage_name+".json")
            inputs[path.relative_to(root).as_posix()] = core.sha256(path)
        for values in previous["original_predictions"].values():
            for path in values.values():
                inputs[path] = core.sha256(root/path)
        for path in previous["seed44_outputs"].values():
            inputs[path] = core.sha256(root/path)
        for paths in original_models.values():
            for path in paths.values():
                core.require(path in inputs, "Unbound original model")
        parameters = {}
        for member in core.MEMBERS:
            parameters[member] = {"development":helpers.parameters(root/f"artifacts/v6/{member}/combiner.cbm")}
            parameters[member].update({case:helpers.parameters(root/original_combiners[member][case]) for case in core.CASE_FOLDS})
            core.require(core.sha256(root/original_combiners[member]["secondary"]) == core.sha256(root/f"artifacts/v6/{member}/combiner.cbm"), "Final combiner policy differs")
        manifest = {"protocol":core.PROTOCOL,"protocol_sha256":core.canonical(core.PROTOCOL),"code_sha256":code,
            "inputs_sha256":inputs,"original_models":original_models,"original_combiners":original_combiners,
            "parameters":parameters,"cohorts":previous["cohorts"],"prefixes":previous["prefixes"],
            "previous_predictions":previous["original_predictions"],"previous_seed44_predictions":previous["seed44_outputs"],"runtime":runtime()}
        raw = old.load_raw(root)
        replay,zero = {},{}
        for case,(_,fold) in core.CASE_FOLDS.items():
            idx,y,baseline,original = cohort(root,manifest,raw,case)
            replay[case],zero[case] = {},{}
            fit = np.flatnonzero(raw["t"] < core.FOLDS[fold])
            training = {k:raw[k][fit] for k in FIT_FIELDS}
            # Independently verify newly staged seed43 assets against saved vectors.
            member = "dim8_seed43"
            x,names,*_ = portable_pipeline(neural_pipeline)(root/original_models[member][fold],training,
                {k:raw[k][idx] for k in TARGET_FIELDS},root/manifest["prefixes"][fold],torch.device("cuda"))
            p = predict_combiner(load_combiner(root/original_combiners[member][case]),x,names).reshape(-1,10)
            core.require(np.allclose(p,original[member],rtol=0,atol=1e-15), "Staged seed43 replay differs")
            replay[case][member] = float(np.max(np.abs(p-original[member])))
            cold = ~np.isin(raw["u"][idx],np.unique(raw["u"][fit]))
            subset = np.unique(np.r_[np.arange(min(8,len(idx))),np.flatnonzero(cold)[:8]])
            for member,info in core.MEMBERS.items():
                torch.manual_seed(core.seed(member,"both"))
                adapter = ResidualAdapter(info["dimension"],"both").cuda().eval()
                x,names,r,p,old_r,old_p,_ = pipeline(root/original_models[member][fold],adapter,training,
                    {k:raw[k][idx[subset]] for k in TARGET_FIELDS},root/manifest["prefixes"][fold],torch.device("cuda"))
                error = max(float(np.max(np.abs(r-old_r))),float(np.max(np.abs(p-old_p))))
                core.require(error <= 1e-15, "Zero adapter changed original probabilities")
                zero[case][member] = error
            budget.check()
        _,_,_,_,_,errors = replay_v6(root,"cuda")
        core.verify_files(root,inputs); core.verify_files(root,code)
        core.write_new(root/WORK/"inputs.json",manifest)
        core.write_new(root/REPORTS/"preparation.json",{"complete":True,"manifest_sha256":core.sha256(root/WORK/"inputs.json"),
            "protocol_sha256":core.canonical(core.PROTOCOL),"source_sha256":code,"newly_staged_replay_errors":replay,
            "zero_adapter_errors":zero,"v6_replay_errors":errors,"v6_csv_sha256":core.V6_SHA,
            "carried_seconds":carry["used_seconds"],"runtime":runtime()})
        print(json.dumps({"prepared":True,"zero_adapter_errors":zero,"v6_sha256":core.V6_SHA}),flush=True)


def train_fold(root,manifest,member,kind,fold,budget):
    path = root/core.adapter_path(member,kind,fold)
    if not path.exists():
        budget.run([sys.executable,"-u",str(root/"experiments/v8-residual-event/train_adapter.py"),member,kind,fold],root)
    core.validate_adapter(root,manifest,member,kind,fold)


def features(root,manifest,raw,member,fold,idx,name,budget,kind=None):
    import torch
    from residual_model import pipeline,load_adapter
    from predict_v6 import portable_pipeline
    from build_submission import neural_pipeline,FIT_FIELDS,TARGET_FIELDS
    idx = np.asarray(idx,dtype=np.int64)
    cutoff = core.FOLDS[fold]
    core.require(len(idx) and (raw["t"][idx]>=cutoff).all(), "Feature target precedes neural cutoff")
    path = root/WORK/"cache"/(name+".npz")
    metadata = path.with_suffix(".json")
    base = root/manifest["original_models"][member][fold]
    identity = {"base_sha256":core.sha256(base),"cutoff":cutoff,"indices_sha256":hashlib.sha256(idx.tobytes()).hexdigest(),
        "kind":kind,"adapter_sha256":None if kind is None else core.sha256(root/core.adapter_path(member,kind,fold)),
        "inputs_and_source_sha256":core.canonical({"inputs":manifest["inputs_sha256"],"code":manifest["code_sha256"]}),"device":"cuda"}
    if path.exists() or metadata.exists():
        record = core.read_json(metadata)
        core.require(record["identity"] == identity and record["sha256"] == core.sha256(path), "Stale residual feature cache")
        with np.load(path,allow_pickle=False) as d:
            core.require(np.array_equal(d["indices"],idx) and tuple(d["feature_names"]) == core.BASE_COLUMNS, "Feature order/schema differs")
            return d["X"]
    fit = np.flatnonzero(raw["t"]<cutoff)
    training,target = {k:raw[k][fit] for k in FIT_FIELDS},{k:raw[k][idx] for k in TARGET_FIELDS}
    budget.check()
    if kind is None:
        x,names,*_ = portable_pipeline(neural_pipeline)(base,training,target,root/manifest["prefixes"][fold],torch.device("cuda"))
    else:
        adapter = load_adapter(root,manifest,member,kind,fold,"cuda")
        x,names,*_ = pipeline(base,adapter,training,target,root/manifest["prefixes"][fold],torch.device("cuda"))
    core.require(tuple(names) == core.BASE_COLUMNS and x.shape == (len(idx)*10,37) and np.isfinite(x).all(), "Invalid residual features")
    x = x.astype(np.float32)
    save_npz(path,X=x,indices=idx,feature_names=np.asarray(names))
    core.write_new(metadata,{"identity":identity,"sha256":core.sha256(path)})
    budget.check()
    return x


def fit_combiner(root,manifest,member,case,x,y,target,name,budget):
    from catboost import CatBoostClassifier,Pool
    path = root/WORK/"combiners"/(name+".cbm")
    core.require(not path.exists(), "Preserve combiner")
    model = CatBoostClassifier(**manifest["parameters"][member][case])
    budget.check(); model.fit(Pool(x,label=y.ravel(),feature_names=list(core.BASE_COLUMNS)),verbose=False)
    path.parent.mkdir(parents=True,exist_ok=True); model.save_model(str(path))
    p = model.predict_proba(Pool(target,feature_names=list(core.BASE_COLUMNS)),thread_count=4)[:,1].reshape(-1,10)
    other = CatBoostClassifier(); other.load_model(str(path))
    core.require(model.tree_count_ == 545 and np.allclose(p.ravel(),other.predict_proba(target,thread_count=4)[:,1],rtol=0,atol=1e-15), "Combiner replay differs")
    budget.check()
    return p,{path.relative_to(root).as_posix():core.sha256(path)}


def metrics(y,baseline,candidate):
    a,b = compute_roc_auc(y,baseline),compute_roc_auc(y,candidate)
    return {"baseline_auc":a,"candidate_auc":b,"auc_diff":b-a}


def development(root):
    manifest = core.load_manifest(root)
    core.require(not (root/REPORTS/"development.json").exists(), "Preserve development result")
    with core.Budget(root/WORK,"development") as budget:
        raw = old.load_raw(root)
        results,files = {k:{"origins":[]} for k in core.COMPOSITIONS},{}
        variants = [("dim8_seed42","event")]+[(m,"both") for m in core.MEMBERS]
        for fold in core.ORIGINS:
            fit,valid = core.development_indices(raw,fold)
            original,updated = {},{}
            for member in core.MEMBERS:
                x = features(root,manifest,raw,member,fold,fit,f"dev_{fold}_fit_original_{member}",budget)
                target = features(root,manifest,raw,member,fold,valid,f"dev_{fold}_valid_original_{member}",budget)
                original[member],assets = fit_combiner(root,manifest,member,"development",x,raw["y"][fit],target,f"dev_{fold}_original_{member}",budget)
                files.update(assets)
            for member,kind in variants:
                train_fold(root,manifest,member,kind,fold,budget)
                path = core.adapter_path(member,kind,fold)
                for p in (path,path.with_suffix(".json")):
                    files[p.as_posix()] = core.sha256(root/p)
                x = features(root,manifest,raw,member,fold,fit,f"dev_{fold}_fit_{member}_{kind}",budget,kind)
                target = features(root,manifest,raw,member,fold,valid,f"dev_{fold}_valid_{member}_{kind}",budget,kind)
                updated[member,kind],assets = fit_combiner(root,manifest,member,"development",x,raw["y"][fit],target,f"dev_{fold}_{member}_{kind}",budget)
                files.update(assets)
            baseline,y = core.original_ensemble(original),raw["y"][valid]
            hours = (raw["t"][valid]-core.FOLDS[fold])/3600
            for key,composition in core.COMPOSITIONS.items():
                candidate = core.compose(baseline,original,{m:updated[m,k] for m,k in composition.items()},key)
                decomposition = old.allocation.allocation_decomposition(y,baseline,candidate)
                late = hours>=48
                bands = {}
                for low,high in ((8,24),(24,48),(48,72),(72,96)):
                    mask = (hours>=low)&(hours<high)
                    if mask.any():
                        bands[f"{low}_{high}h"] = {"events":int(mask.sum()),**metrics(y[mask],baseline[mask],candidate[mask])}
                entry = {"origin":fold,**metrics(y,baseline,candidate),"fit_events":len(fit),"valid_events":len(valid),
                    "maximum_fit_timestamp":int(raw["t"][fit].max()),"minimum_validation_timestamp":int(raw["t"][valid].min()),
                    "no_click_contribution":decomposition["negative_event_partitions"]["negative_from_no_click_event"]["weighted_auc_change"],
                    "decomposition":decomposition,"bands":bands,"late":None if not late.any() else
                    {"events":int(late.sum()),**metrics(y[late],baseline[late],candidate[late])},
                    "diagnostics":helpers.diagnostics(raw,valid,y,baseline,candidate,core.FOLDS[fold])}
                results[key]["origins"].append(entry)
                path = WORK/f"development_{fold}_{key}.npz"
                save_npz(root/path,fit_idx=fit,indices=valid,labels=y,baseline=baseline,candidate=candidate)
                files[path.as_posix()] = core.sha256(root/path)
            print(json.dumps({"origin":fold,"gains":{k:v["origins"][-1]["auc_diff"] for k,v in results.items()}}),flush=True)
        winner = core.choose(results)
        core.load_manifest(root)
        path = root/REPORTS/"development.json"
        core.write_new(path,{"complete":True,"protocol_sha256":core.canonical(core.PROTOCOL),"source_sha256":manifest["code_sha256"],
            "input_manifest_sha256":core.sha256(root/WORK/"inputs.json"),"results":results,"winner":winner,
            "files_sha256":files,"primary_secondary_labels_used_for_selection":False,"five_day_dates_used_for_development":True,
            "validation_status":core.PROTOCOL["validation_status"],"runtime":runtime()})
        if winner is not None:
            core.write_new(root/REPORTS/"selected.json",{"composition":winner,"members":core.COMPOSITIONS[winner],
                "protocol_sha256":core.canonical(core.PROTOCOL),"development_sha256":core.sha256(path)})
        print(json.dumps({"winner":winner}),flush=True)
    return winner is not None


def score_case(root,manifest,raw,case,budget):
    winner = core.selection(root,manifest)
    meta_fold,target_fold = core.CASE_FOLDS[case]
    idx,y,baseline,original = cohort(root,manifest,raw,case)
    fit = core.first_indices(raw["u"],raw["t"],core.FOLDS[meta_fold],core.FOLDS[meta_fold]+core.DAY)
    core.temporal_check(raw,fit,idx,core.FOLDS[meta_fold],core.FOLDS[target_fold])
    updated,files,fits = {},{},{}
    for member,kind in core.COMPOSITIONS[winner].items():
        for fold in (meta_fold,target_fold):
            train_fold(root,manifest,member,kind,fold,budget)
            path = core.adapter_path(member,kind,fold)
            files[path.as_posix()] = core.sha256(root/path)
            files[path.with_suffix(".json").as_posix()] = core.sha256(root/path.with_suffix(".json"))
        x = features(root,manifest,raw,member,meta_fold,fit,f"{case}_fit_{member}_{kind}",budget,kind)
        target = features(root,manifest,raw,member,target_fold,idx,f"{case}_target_{member}_{kind}",budget,kind)
        updated[member],assets = fit_combiner(root,manifest,member,case,x,raw["y"][fit],target,f"{case}_{member}_{kind}",budget)
        files.update(assets)
        fits[member] = {"kind":kind,"combiner":next(iter(assets)),"parameters":manifest["parameters"][member][case],
            "adapters":{fold:core.adapter_path(member,kind,fold).as_posix() for fold in (meta_fold,target_fold)}}
    candidate = core.compose(baseline,original,updated,winner)
    path = WORK/(case+"_predictions.npz")
    save_npz(root/path,indices=idx,labels=y,baseline=baseline,candidate=candidate,**{"new_"+m:p for m,p in updated.items()})
    files[path.as_posix()] = core.sha256(root/path)
    result = asdict(PairedBootstrapEvaluator().evaluate(candidate,baseline,y,raw["u"][idx]))
    result.update(n_bootstrap=1000,valid_bootstrap_samples=result["resamples"],events=len(idx),fitting_events=len(fit),
        predictions=path.as_posix(),predictions_sha256=core.sha256(root/path),member_models=fits,
        decomposition=old.allocation.allocation_decomposition(y,baseline,candidate),
        diagnostics=helpers.diagnostics(raw,idx,y,baseline,candidate,core.FOLDS[target_fold]))
    core.prior.prior.previous.prior.validate_metrics(result)
    if case == "five_day":
        hours = (raw["t"][idx]-core.FOLDS[target_fold])/3600
        late = (hours>=48)&(hours<120)
        core.require(int(late.sum()) == 7596, "Late cohort differs")
        result["late_48_120h"] = {"events":int(late.sum()),**metrics(y[late],baseline[late],candidate[late])}
    budget.check()
    return result,files


def report(root,manifest,stage,results,checks,files):
    core.load_manifest(root)
    core.write_new(root/REPORTS/(stage+".json"),{"complete":True,"submission_ready":False,
        "protocol_sha256":core.canonical(core.PROTOCOL),"source_sha256":manifest["code_sha256"],
        "input_manifest_sha256":core.sha256(root/WORK/"inputs.json"),"selection_sha256":core.sha256(root/REPORTS/"selected.json"),
        "composition":core.selection(root,manifest),"results":results,"decision_checks":checks,"passed":all(checks.values()),
        "files_sha256":files,"runtime":runtime()})
    print(json.dumps({"stage":stage,"checks":checks,"passed":all(checks.values())}),flush=True)


def screen(root):
    manifest = core.load_manifest(root); core.selection(root,manifest)
    core.require(not (root/REPORTS/"screen.json").exists(), "Preserve screen result")
    with core.Budget(root/WORK,"screen") as budget:
        raw = old.load_raw(root); results,files = {},{}
        for case in ("primary","secondary"):
            results[case],more = score_case(root,manifest,raw,case,budget); files.update(more)
        checks = core.screen_gates(results); report(root,manifest,"screen",results,checks,files)
    return all(checks.values())


def stress(root):
    manifest = core.load_manifest(root); core.validate_report(root,manifest,"screen")
    core.require(not (root/REPORTS/"stress.json").exists(), "Preserve stress result")
    with core.Budget(root/WORK,"stress") as budget:
        result,files = score_case(root,manifest,old.load_raw(root),"five_day",budget)
        checks = core.stress_gates(result); report(root,manifest,"stress",result,checks,files)
    return all(checks.values())


def finalize(root):
    from predict_v8_residual import release_candidate
    manifest = core.load_manifest(root); winner = core.selection(root,manifest)
    core.validate_report(root,manifest,"screen"); core.validate_report(root,manifest,"stress")
    core.require(not (root/"configs/v8.json").exists(), "Preserve existing V8 release")
    with core.Budget(root/WORK,"finalize") as budget:
        for member,kind in core.COMPOSITIONS[winner].items():
            train_fold(root,manifest,member,kind,"jan27",budget)
    with core.Budget(root/WORK,"verify") as budget:
        release_candidate(root,manifest,budget)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","development","screen","stress","finalize"])
    parser.add_argument("--source-root",type=Path,default=ROOT.parents[2]); args = parser.parse_args()
    if args.action == "prepare":
        prepare(ROOT,args.source_root.resolve())
    elif not {"development":development,"screen":screen,"stress":stress,"finalize":finalize}[args.action](ROOT):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
