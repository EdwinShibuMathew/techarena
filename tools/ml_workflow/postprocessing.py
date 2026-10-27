"""Schema-v2 ensemble postprocessing using the shared controller and frozen assets."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import fcntl
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
from consolidate import verify_bundle, write_json
from .contracts import (ExperimentSpec, MEMBERS, DAY, JAN20, require, inside, identifier,
                        digest, canonical, first_indices, context, validate_events, compose)
from .storage import read, write_new, save_arrays, snapshot, verify_snapshot
from .budget import Budget
from .cache import request, get_or_compute, arrays_identity

ROOT=Path(__file__).resolve().parents[2]


def inventory(root,m):
    return read(inside(root,m['spec']['inputs']['inventory']['path']))


def prepare(root,spec_path,run_id):
    from . import engine
    root=Path(root).resolve();identifier(run_id)
    spec=ExperimentSpec.parse(read(spec_path)).value
    research=spec['purpose']=='research'
    if research and spec['schema_version']==4:
        from .history_postprocessing import preflight
        preflight(root,spec)
    if research:require(Path(spec_path).resolve().is_relative_to(root/'experiments'),'Commit the research specification')
    work=inside(root,'artifacts/experiments' if research else 'artifacts/validation')/run_id
    require(not any((inside(root,p)/run_id).exists() for p in ('artifacts/experiments','artifacts/validation')),'Preserve existing run ID')
    carry=read(inside(root,spec['inputs']['carry']['path'])) if research else {'compute_limit_seconds':spec['budget']['total_seconds'],'total_compute_seconds':0.}
    require(carry['compute_limit_seconds']==spec['budget']['total_seconds'],'Carry limit differs')
    ledger=inside(root,'artifacts/experiments/compute-ledger.json') if research else work/'compute.json'
    carry_id=spec['inputs']['carry']['sha256'] if research else None
    with Budget(ledger,run_id+'/prepare',spec['budget']['phase_seconds'],spec['budget']['total_seconds'],carry['total_compute_seconds'],carry_id) as budget:
        work.mkdir(parents=True,exist_ok=not research) # validation budget may have created its parent
        write_new(work/'prepare.attempt.json',{'spec_sha256':canonical(spec),'state':'claimed'})
        if research:verify_bundle(root)
        for value in spec['inputs'].values():require(digest(inside(root,value['path']))==value['sha256'],'Declared input changed')
        from .policy import validate_anchor
        validate_anchor(root,spec)
        source=snapshot(root,work)
        require(spec['adapter'] in source['files'],'Adapter must be committed')
        if research:
            relative=Path(spec_path).resolve().relative_to(root).as_posix()
            require(relative in source['files'] and source['files'][relative]==digest(spec_path),'Specification must be committed')
        m={'schema_version':2,'run_id':run_id,'spec':spec,'spec_sha256':canonical(spec),'source':source,
           'carried_seconds':carry['total_compute_seconds'],'carry_identity':carry_id,
           'validation_status':'reused retrospective dates; not independent or selection-adjusted evidence'}
        write_new(work/'preparation.json',m)
        write_new(work/'preparation.seal.json',{'sha256':digest(work/'preparation.json')})
        command=[sys.executable,str(work/'source/tools/run_experiment.py'),'prepare','--root',str(root),'--run-id',run_id,'--worker']
        code=budget.run(command,work/'source');require(code==0,'Input preparation worker failed')
    write_new(work/'prepare.ledger.json',{'exit_code':code,'compute':read(ledger),'source_commit':source['commit']})
    m['generated_inputs']=read(work/'generated-inputs.json')
    for asset in m['generated_inputs'].values():require(digest(inside(root,asset['path']))==asset['sha256'],'Generated input changed')
    write_new(work/'manifest.json',m);write_new(work/'prepared.json',{'manifest_sha256':digest(work/'manifest.json'),'complete':True})
    engine.load(root,work)
    return {'run_id':run_id,'purpose':spec['purpose'],'source_commit':source['commit'],'prepared':True}


def prepare_worker(root,run_id):
    from . import engine
    paths=[inside(root,p)/identifier(run_id) for p in ('artifacts/experiments','artifacts/validation')]
    found=[p for p in paths if (p/'preparation.json').is_file()];require(len(found)==1,'Missing or ambiguous preparation')
    work=found[0];m=read(work/'preparation.json')
    require(read(work/'preparation.seal.json')['sha256']==digest(work/'preparation.json'),'Preparation changed')
    ExperimentSpec.parse(m['spec']);require(canonical(m['spec'])==m['spec_sha256'],'Preparation specification changed')
    verify_snapshot(work,m['source'])
    engine._start_watchdog(engine._worker_lease(root,work,m,'prepare'))
    for asset in m['spec']['inputs'].values():require(digest(inside(root,asset['path']))==asset['sha256'],'Input changed')
    generated={}
    for name in ('train','evaluation'):
        with np.load(inside(root,m['spec']['inputs'][name]['path']),allow_pickle=False) as saved:
            data={k:saved[k] for k in saved.files}
        n=len(data['u'])
        if 'rowids' not in data:
            if name=='train':data['rowids']=np.char.add('t',np.arange(n*10).astype('U7')).reshape(n,10)
            else:data['rowids']=np.load(inside(root,m['spec']['inputs']['evaluation_rowids']['path']),allow_pickle=False).reshape(n,10)
        validate_events(data,name=='train')
        path=work/'inputs'/(name+'.npz');save_arrays(path,**data)
        generated[name]={'path':path.relative_to(root).as_posix(),'sha256':digest(path)}
    write_new(work/'generated-inputs.json',generated)
    if m['spec']['schema_version']==4 and m['spec']['purpose']=='research':
        # Establish exact anchors under the preparation lease, before research.
        from .history_postprocessing import final_anchor
        sys.path.insert(0,str(work/'source/legacy'))
        m['generated_inputs']=generated
        final_anchor(root,work,m,'prepare')
    return 0


def extract(fitting,target,checkpoint,positive,device):
    import torch
    from predict_v6 import portable_pipeline
    from build_submission import neural_pipeline
    x,names,raw,p,*_=portable_pipeline(neural_pipeline)(checkpoint,fitting,target,positive,torch.device(device))
    return {'X':np.asarray(x,dtype=np.float32),'feature_names':np.asarray(names,dtype='U'),'probs':p,'raw_probs':raw}


def features(root,work,m,inv,raw,indices,member,cutoff):
    import torch
    from . import engine
    sys.path.insert(0,str(work/'source/legacy'))
    fold='jan'+str(20+(cutoff-JAN20)//DAY)
    model=inside(root,inv['models'][member][fold]);positive=inside(root,inv['prefixes'][fold])
    state=torch.load(model,map_location='cpu',weights_only=True)
    dimension,seed,_=MEMBERS[member]
    require(state['dimension']==dimension and state['epoch']==8,'Wrong frozen event model')
    for name,value in (('cutoff',cutoff),('seed',seed)):
        require(name not in state or state[name]==value,'Conflicting original checkpoint metadata')
    del state
    with np.load(positive,allow_pickle=False) as saved:
        require(int(saved['cutoff'])==cutoff and (saved['timestamps']<cutoff).all(),'Preference prefix crosses cutoff')
    prefix=context(raw,np.flatnonzero(raw['t']<cutoff),True);target=context(raw,indices,False)
    require((target['t']>=cutoff).all(),'Target precedes frozen prefix')
    frozen=read(work/'source/configs/source_manifest.json')['files']
    files=tuple(frozen)+('tools/predict_v6.py','tools/ml_workflow/postprocessing.py')
    module=SimpleNamespace(extract=extract,EXTRACTOR_FILES=files,FEATURE_NAMES=inv['feature_names'],
                           EXTRACTION_DEVICE=m['spec']['device'],EXTRACTION_PRECISION='float32 neural; float64 probabilities')
    start=time.monotonic();identity=request(work/'source',module,model,prefix,target,cutoff)
    identity['positive_sha256']=digest(positive)
    hashing=time.monotonic()-start
    data,receipt=get_or_compute(root,identity,lambda:extract(prefix,target,model,positive,m['spec']['device']))
    receipt.update(hashing_seconds=hashing,member=member,cutoff=cutoff)
    write_new(work/'cache-receipts'/(receipt['key']+'-'+str(time.time_ns())+'.json'),receipt)
    print('features',member,fold,len(indices),'hit' if receipt['hit'] else 'computed',flush=True)
    return data


def cached_scores(root,work,m,identity,compute):
    """Complete content-addressed OOF entries include their fitted combiner hashes."""
    key=canonical(identity);parent=inside(root,'artifacts/cache/ml-workflow/postprocess-v1');parent.mkdir(parents=True,exist_ok=True)
    folder=inside(parent,key);start=time.monotonic()
    with inside(parent,key+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        hit=folder.exists()
        if not hit:
            with tempfile.TemporaryDirectory(prefix='.oof-',dir=parent) as temporary:
                stage=Path(temporary)/'entry';stage.mkdir()
                values=compute(stage);save_arrays(stage/'scores.npz',**values)
                files={p.name:digest(p) for p in stage.iterdir() if p.is_file()}
                write_new(stage/'metadata.json',{'identity':identity,'files':files,'arrays':arrays_identity(values)})
                stage.rename(folder)
        record=read(folder/'metadata.json');require(record['identity']==identity,'Cross-fit cache identity differs')
        for path,expected in record['files'].items():require(digest(inside(folder,path))==expected,'Cross-fit cache artifact changed')
        with np.load(folder/'scores.npz',allow_pickle=False) as saved:values={k:saved[k] for k in saved.files}
        require(arrays_identity(values)==record['arrays'],'Cross-fit cache arrays changed')
    receipt={'key':key,'hit':hit,'seconds':time.monotonic()-start,'metadata_sha256':digest(folder/'metadata.json'),
             'identity_sha256':canonical(identity),'producer_models':{k:v for k,v in record['files'].items() if k.endswith('.cbm')}}
    path=work/'cache-receipts'/('oof-'+key+'-'+str(time.time_ns())+'.json');write_new(path,receipt)
    return values,path


def crossfit(root,work,m,module,inv,raw,fit,cutoff,parameter_case):
    from catboost import CatBoostClassifier,Pool
    matrices={member:features(root,work,m,inv,raw,fit,member,cutoff) for member in MEMBERS}
    folds=module.user_folds(raw['u'][fit]);require(set(folds)==set(range(5)),'Missing cross-fit fold')
    identity={'schema_version':1,'source':{p:h for p,h in m['source']['files'].items() if p.startswith('tools/ml_workflow/') or p==m['spec']['adapter']},
              'cutoff':cutoff,'users':arrays_identity({'u':raw['u'][fit],'indices':fit,'labels':raw['y'][fit],'folds':folds}),
              'features':{k:arrays_identity(v) for k,v in matrices.items()},'parameters':{k:inv['parameters'][k][parameter_case] for k in MEMBERS},
              'runtime':{name:importlib.metadata.version(name) for name in ('numpy','torch','catboost','scikit-learn')}}
    def compute(folder):
        members={}
        for member,data in matrices.items():
            p=np.empty((len(fit),10),dtype=np.float64)
            def fit_fold(fold):
                train=folds!=fold;held=~train
                require(not np.intersect1d(raw['u'][fit][train],raw['u'][fit][held]).size,'User leakage in cross-fit')
                model=CatBoostClassifier(**deepcopy(inv['parameters'][member][parameter_case]),verbose=False)
                model.fit(Pool(data['X'][np.repeat(train,10)],label=raw['y'][fit][train].ravel(),feature_names=data['feature_names'].tolist()))
                model.save_model(str(folder/(member+'-'+str(fold)+'.cbm')))
                prediction=model.predict_proba(Pool(data['X'][np.repeat(held,10)],feature_names=data['feature_names'].tolist()),thread_count=4)[:,1].reshape(-1,10)
                print('crossfit',member,cutoff,fold,flush=True)
                return held,prediction
            # Fixed per-fit parameters/seeds; at most two independent CPU folds.
            with ThreadPoolExecutor(max_workers=min(2,max(1,(os.cpu_count() or 4)//4))) as pool:
                for held,prediction in pool.map(fit_fold,range(5)):p[held]=prediction
            members[member]=p
        return {'scores':compose(members),'folds':folds,'indices':fit,'users':raw['u'][fit]}
    values,receipt=cached_scores(root,work,m,identity,compute)
    require(np.array_equal(values['indices'],fit) and np.array_equal(values['folds'],folds),'Cross-fit ordering changed')
    return values['scores'],receipt


def reference_scores(root,work,m,inv,raw,fit,valid,case):
    from catboost import CatBoostClassifier,Pool
    members={};files={}
    saved_case=case.name if case.name in ('primary','secondary','five_day') else 'five_day' if case.meta_cutoff==JAN20+DAY and case.target_cutoff==JAN20+2*DAY else None
    for member in MEMBERS:
        target=features(root,work,m,inv,raw,valid,member,case.target_cutoff)
        if saved_case:
            path=inside(root,inv['combiners'][member][saved_case]);model=CatBoostClassifier();model.load_model(str(path))
        else:
            meta=features(root,work,m,inv,raw,fit,member,case.meta_cutoff)
            path=work/'reference-combiners'/(case.name+'-'+member+'.cbm');path.parent.mkdir(parents=True,exist_ok=True)
            v3=m['spec']['schema_version'] in (3,4)
            receipt=path.with_suffix('.json')
            expected={'fit':arrays_identity({'X':meta['X'],'indices':fit,'labels':raw['y'][fit]}),
                      'parameters':inv['parameters'][member]['development'],'schema':meta['feature_names'].tolist()}
            if path.exists():
                require(v3 and receipt.is_file(),'Do not refit a reference combiner')
                record=read(receipt)
                require(record['identity']==expected and record['sha256']==digest(path),'Reference combiner changed')
                model=CatBoostClassifier();model.load_model(str(path))
            else:
                require(not receipt.exists(),'Partial reference combiner')
                model=CatBoostClassifier(**deepcopy(inv['parameters'][member]['development']),verbose=False)
                model.fit(Pool(meta['X'],label=raw['y'][fit].ravel(),feature_names=meta['feature_names'].tolist()))
                model.save_model(str(path))
                if v3:write_new(receipt,{'identity':expected,'sha256':digest(path)})
            files[path.relative_to(work).as_posix()]=digest(path)
            if v3:files[receipt.relative_to(work).as_posix()]=digest(receipt)
        members[member]=model.predict_proba(Pool(target['X'],feature_names=target['feature_names'].tolist()),thread_count=4)[:,1].reshape(-1,10)
    before=compose(members)
    if saved_case:
        with np.load(inside(root,inv['cohorts'][saved_case]),allow_pickle=False) as saved:
            positions=np.flatnonzero(np.isin(saved['valid_idx'],valid))
            require(np.array_equal(saved['valid_idx'][positions],valid),'Historical cohort order differs')
            require(np.array_equal(saved['labels'][positions],raw['y'][valid]),'Historical labels differ')
            require(np.allclose(before,saved['candidate'][positions],rtol=0,atol=1e-15),'Historical V6 comparator did not reproduce')
    return before,files


def score(root,work,m,module,raw,candidate,case):
    from . import engine
    from .policy import is_trial, V7_SHA
    trial=is_trial(m['spec'])
    start=time.monotonic();inv=inventory(root,m)
    fit=first_indices(raw['u'],raw['t'],case.meta_cutoff,case.meta_cutoff+DAY)
    valid=first_indices(raw['u'],raw['t'],case.target_cutoff,case.target_end)
    require(len(fit) and len(valid) and raw['t'][fit].max()<case.target_cutoff,'Invalid calibration timeline')
    if m['spec']['purpose']=='research' and case.name in ('primary','secondary','five_day'):
        require(len(valid)=={'primary':43964,'secondary':35138,'five_day':51488}[case.name],'Official first-user cohort differs')
    parameter_case=case.name if case.name in ('primary','secondary','five_day') else 'five_day' if case.meta_cutoff==JAN20+DAY else 'development'
    scores,receipt=crossfit(root,work,m,module,inv,raw,fit,case.meta_cutoff,parameter_case)
    metadata={'spec_sha256':m['spec_sha256'],'meta_cutoff':case.meta_cutoff,'fitting_events':len(fit),
              'maximum_label_timestamp':int(raw['t'][fit].max()),'oof_scores_sha256':canonical(arrays_identity({'scores':scores,'labels':raw['y'][fit],'users':raw['u'][fit]}))}
    v3=m['spec']['schema_version']==3
    params=deepcopy(m['spec']['candidates'][candidate])
    if v3:metadata.update(candidate=candidate,parameters_sha256=canonical(params))
    path=work/'calibrators'/candidate/(canonical(metadata)+'.json');path.parent.mkdir(parents=True,exist_ok=True)
    fit_start=time.monotonic()
    if path.exists():saved=module.load_calibrator(path,metadata,deepcopy(params)) if v3 else module.load_calibrator(path,metadata)
    else:saved=module.fit_calibrator(scores.copy(),raw['y'][fit].copy(),deepcopy(metadata),path,deepcopy(params)) if v3 else module.fit_calibrator(scores,raw['y'][fit].copy(),deepcopy(metadata),path)
    require(saved['metadata']==metadata,'Adapter modified expected calibration metadata')
    before,files=reference_scores(root,work,m,inv,raw,fit,valid,case)
    if v3:
        require(saved['parameters']==params,'Adapter modified expected candidate parameters')
        norm_scores=scores
        if trial:
            from predict_v7 import calibrate_v7_probabilities
            norm_scores=calibrate_v7_probabilities(scores,m['spec']['anchor_gamma'])
        logmass=np.log(np.maximum(norm_scores.sum(1),params['config']['log_floor']))
        expected_norm={'log_mass_mean':float(logmass.mean()),'log_mass_scale':max(float(logmass.std()),params['config']['scale_floor'])}
        require(saved['normalization']==expected_norm,'Calibration normalization differs from fitting scores')
    after=module.predict_calibrator(before.copy() if v3 else before,deepcopy(saved) if v3 else saved['coefficients']);y=raw['y'][valid]
    require(after.shape==before.shape and np.isfinite(after).all() and ((after>=0)&(after<=1)).all(),'Invalid postprocessor output')
    if not v3:require(np.allclose(after.sum(1),before.sum(1),rtol=0,atol=1e-14),'Postprocessor changed event mass')
    else:
        for i in range(10):
            for j in range(i):require(np.array_equal(np.sign(after[:,i]-after[:,j]),np.sign(before[:,i]-before[:,j])),'Postprocessor changed within-event order or ties')
    pred=work/'predictions'/(candidate+'-'+case.name+'.npz')
    save_arrays(pred,indices=valid,rowids=raw['rowids'][valid],labels=y,baseline=before,candidate=after)
    for p in (path,receipt,pred):files[p.relative_to(work).as_posix()]=digest(p)
    result={'case':asdict(case),'events':len(valid),'fit_events':len(fit),'maximum_meta_label_timestamp':int(raw['t'][fit].max()),
            **engine.point(y,before,after),'models':{'calibrator':path.relative_to(work).as_posix()},'files':files,
            'coefficients':saved['coefficients'],'objective':saved['objective'],'identity_objective':saved['identity_objective'],
            'timings':{'total':time.monotonic()-start,'calibration_and_reference':time.monotonic()-fit_start},
            'maximum_event_mass_change':float(np.abs(after.sum(1)-before.sum(1)).max())}
    if case.name in ('primary','secondary','five_day'):
        result.update(asdict(engine.PairedBootstrapEvaluator().evaluate(after,before,y,raw['u'][valid])))
        require(result['resamples']>=990,'Insufficient paired draws')
    if trial or case.name in ('primary','secondary','five_day'):
        from predict_v7 import calibrate_v7_probabilities
        old_v7=calibrate_v7_probabilities(before,1.12)
        result['v7_diagnostic']={**engine.point(y,before,old_v7),
                                 'candidate_minus_v7_auc':result['candidate_auc']-engine.compute_roc_auc(y,old_v7),
                                 'used_for_selection':False}
    if trial:
        result['v7_comparison']={**engine.point(y,old_v7,after),'anchor_sha256':V7_SHA}
        if case.name in ('primary','secondary','five_day'):
            result['v7_comparison'].update(asdict(engine.PairedBootstrapEvaluator().evaluate(after,old_v7,y,raw['u'][valid])))
            require(result['v7_comparison']['resamples']>=990,'Insufficient paired V7 draws')
    hours=(raw['t'][valid]-case.target_cutoff)/3600;late=(hours>=48)&(hours<120)
    if case.name=='five_day':
        require(int(late.sum())==7596 if m['spec']['purpose']=='research' else late.any(),'Missing late cohort')
        result['late']=engine.point(y[late],before[late],after[late])
    if trial and late.any():result['v7_comparison']['late']=engine.point(y[late],old_v7[late],after[late])
    warm=np.isin(raw['u'][valid],raw['u'][raw['t']<case.target_cutoff])
    masks={'warm':warm,'cold':~warm,'history_empty':(raw['h'][valid]<0).all(1),'late':late}
    result['diagnostics']={k:{'events':int(v.sum()),**(engine.point(y[v],before[v],after[v]) if v.any() and len(np.unique(y[v]))==2 else {})} for k,v in masks.items()}
    print('result',candidate,case.name,result['auc_diff'],'vs_V7',result.get('v7_comparison',{}).get('auc_diff'),saved['coefficients'],flush=True)
    return result


def final_probabilities(root,work,m,module,calibrator,number):
    if m['spec']['schema_version']==4:
        from .history_postprocessing import final_probabilities as history_final
        return history_final(root,work,m,module,calibrator,number)
    from . import engine
    from predict_v7 import load_aligned_probabilities
    v3=m['spec']['schema_version']==3
    research=m['spec']['purpose']=='research'
    if research:engine._private_replay(root,work,m,number)
    evaluation=engine.events(root,m,'evaluation')
    ids=evaluation['rowids'].ravel()
    if not research:
        _,fresh=engine.baseline(root,m,'baseline_evaluation',evaluation['rowids'])
        saved=module.load_calibrator(calibrator)
        return ids,module.predict_calibrator(fresh,deepcopy(saved) if v3 else saved['coefficients'])
    require(len(evaluation['u'])==52939 and len(np.unique(evaluation['u']))==52939,'Official evaluation users differ')
    require(np.array_equal(ids,np.load(root/'data/cache/evaluation_rowids.npy',allow_pickle=False)),'Official row ordering differs')
    require((evaluation['t']>=JAN20+7*DAY).all(),'Final inference timeline differs')
    original={}
    for member in MEMBERS:
        with np.load(root/f'artifacts/v6/{member}/reference_predictions.npz',allow_pickle=False) as saved:
            original[member]=saved['combiner_predictions'].reshape(-1,10)
    before=compose(original)
    fresh=load_aligned_probabilities(work/'v6-replays'/str(number)/'artifacts/reproduced_v6.csv',ids).reshape(-1,10)
    require(np.allclose(fresh,before,rtol=0,atol=1e-15),'Fresh V6 differs from final references')
    from .policy import is_trial
    if is_trial(m['spec']):
        from predict_v7 import calibrate_v7_probabilities
        from submission_export import export_submission,validate_csv
        from row_auc_final_utils import verify_csv_probabilities
        anchor=calibrate_v7_probabilities(fresh,m['spec']['anchor_gamma'])
        anchor_path=work/'v6-replays'/str(number)/'v7_anchor.csv'
        example=inside(root,m['spec']['inputs']['example']['path'])
        export_submission(ids,anchor,example,anchor_path,529390)
        validate_csv(anchor_path,example,529390);verify_csv_probabilities(anchor_path,ids,anchor)
        require(digest(anchor_path)==m['spec']['anchor_sha256'],'Fresh V7 anchor fingerprint differs')
    saved=module.load_calibrator(calibrator)
    require(saved['metadata']['spec_sha256']==m['spec_sha256'] and saved['metadata']['meta_cutoff']==JAN20+5*DAY
            and saved['metadata']['maximum_label_timestamp']<JAN20+6*DAY,'Final calibrator must reuse validated Jan25 fitting')
    if v3:
        expected=m['spec']['candidates'][saved['metadata']['candidate']]
        require(saved['parameters']==expected and saved['metadata']['parameters_sha256']==canonical(expected),'Final candidate parameters differ')
    return ids,module.predict_calibrator(fresh,deepcopy(saved) if v3 else saved['coefficients'])


def finalize(root,work,m,module,raw,winner):
    from . import engine
    sys.path.insert(0,str(work/'source/legacy'))
    from submission_export import export_submission,validate_csv
    from row_auc_final_utils import verify_csv_probabilities
    screen=engine.checked(work,'screen');engine.checked(work,'stress')
    path=inside(work,screen['results']['secondary']['models']['calibrator'])
    outputs=[];predictions=[]
    for number in (1,2):
        ids,p=final_probabilities(root,work,m,module,path,number)
        output=work/('candidate.csv' if number==1 else 'independent-replay.csv')
        example=inside(root,m['spec']['inputs']['example']['path'])
        export_submission(ids,p,example,output,len(ids))
        validate_csv(output,example,len(ids));verify_csv_probabilities(output,ids,p)
        outputs.append(output);predictions.append(p)
    require(np.array_equal(predictions[0],predictions[1]) and digest(outputs[0])==digest(outputs[1]),'Independent candidate replay differs')
    engine.load(root,work)
    report={'schema_version':2,'purpose':m['spec']['purpose'],'complete':True,'passed':True,'submission_ready':False,
            'manifest_sha256':digest(work/'manifest.json'),'independent_replay':True,'winner':winner,'rows':len(ids),
            'csv_sha256':digest(outputs[0]),'files':{p.relative_to(work).as_posix():digest(p) for p in (*outputs,path)}}
    from .policy import is_trial, is_dual
    if (is_trial(m['spec']) or is_dual(m['spec'])) and m['spec']['purpose']=='research':
        for number in (1,2):
            for relative in ('v7_anchor.csv','local/v6_relocation_check.json'):
                artifact=work/'v6-replays'/str(number)/relative
                report['files'][artifact.relative_to(work).as_posix()]=digest(artifact)
    if m['spec']['schema_version']==4:
        graph_arrays=[]
        for number in (1,2):
            path=work/'graphs'/('final-'+str(number)+'.npz')
            with np.load(path,allow_pickle=False) as saved:
                graph_arrays.append(arrays_identity({k:saved[k] for k in saved.files}))
            for artifact in (path,path.with_suffix('.json')):
                report['files'][artifact.relative_to(work).as_posix()]=digest(artifact)
        require(graph_arrays[0]==graph_arrays[1],'Independent final graphs differ')
    write_new(work/'finalize.json',report);write_new(work/'finalize.seal.json',{'sha256':digest(work/'finalize.json')})
    return 0


def replay_release(root, output, release_name="v7_1"):
    """Use maintained accounting/publication around the frozen scientific code."""
    from postprocessor_replay import replay_release as maintained_replay
    return maintained_replay(root, output, release_name)


def replay_worker(root, output, token, release_name="v7_1"):
    """Old direct workers cannot satisfy the staged publication contract."""
    raise ValueError('Use the maintained replay bridge with an active controller lease')
