"""Import explicitly declared immutable assets; scientific work starts in budgeted prepare."""
import argparse
from copy import deepcopy
from pathlib import Path
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from ml_workflow.contracts import DAY,JAN20,V6_SHA,digest,inside,require
from ml_workflow.storage import read,write_new
from adapter import CONFIG


def stage(root,asset_root,baseline_root):
    source=asset_root/'artifacts/experiments/v8-residual-event/inputs.json'
    expected=read(root/'docs/results/v8-residual-event/preparation.json')['manifest_sha256']
    require(digest(source)==expected,'Reused manifest differs from archived producer evidence')
    old=read(source)
    decision=root/'docs/results/v8-residual-event/decision.json';carry=read(decision)
    require(read(asset_root/'artifacts/experiments/v8-residual-event/budget.json')['used_seconds']==carry['total_compute_seconds'],'Carry journal differs')
    for base in (asset_root,baseline_root):
        require(not (base/'artifacts/experiments/compute-ledger.json').exists(),'An authoritative shared ledger exists; continue that workspace instead of resetting it')
    bundle=read(baseline_root/'artifacts/v6/manifest.json')
    declared={p:r['sha256'] for p,r in bundle['files'].items()}
    sources={p:baseline_root for p in declared}
    declared['artifacts/v6/manifest.json']=digest(baseline_root/'artifacts/v6/manifest.json');sources['artifacts/v6/manifest.json']=baseline_root
    wanted={p for d in old['original_models'].values() for p in d.values()}|{p for d in old['original_combiners'].values() for p in d.values()}|set(old['prefixes'].values())|set(old['cohorts'].values())
    for path in wanted:
        expected=old['inputs_sha256'][path]
        if path in declared:require(declared[path]==expected,'Conflicting asset identities')
        else:declared[path]=expected;sources[path]=asset_root
    for relative in sorted(declared):
        dest=inside(root,relative);original=inside(sources[relative],relative)
        require(original.is_file(),'Missing declared asset: '+relative)
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():require(digest(dest)==declared[relative],'Preserve different staged asset')
        else:os.link(original,dest)
    folder=root/'artifacts/inputs/v7.1-entropy-sharpening-v1';folder.mkdir(parents=True,exist_ok=True)
    inv={'producer_revision':'3a902173cf111dcb748140a31293425ae66b2142','producer_manifest_sha256':digest(source),
         'models':old['original_models'],'combiners':old['original_combiners'],'prefixes':old['prefixes'],
         'parameters':old['parameters'],'cohorts':old['cohorts'],'feature_names':old['protocol']['features']}
    inv_path=folder/'inventory.json';write_new(inv_path,inv)
    inputs={f'asset_{i:03}':{'path':p,'sha256':h} for i,(p,h) in enumerate(sorted(declared.items()))}
    for role,path in (('train','data/cache/train_events.npz'),('evaluation','data/cache/evaluation_events.npz'),
                      ('evaluation_rowids','data/cache/evaluation_rowids.npy'),('example','data/raw/example_submission.csv')):
        inputs[role]={'path':path,'sha256':declared[path]}
    inputs['inventory']={'path':inv_path.relative_to(root).as_posix(),'sha256':digest(inv_path)}
    inputs['carry']={'path':decision.relative_to(root).as_posix(),'sha256':digest(decision)}
    def case(name,meta,target,end):return {'name':name,'meta_cutoff':JAN20+meta*DAY,'target_cutoff':JAN20+target*DAY,'target_end':JAN20+end*DAY,'baseline':'baseline_'+name}
    spec={'schema_version':2,'mode':'ensemble_postprocessor','purpose':'research','release_name':'v7_1','device':'cuda',
          'hypothesis':'Two-parameter entropy-conditioned allocation transfers more robustly than fixed sharpening while retaining V6 event score mass',
          'comparator':'V6','baseline_sha256':V6_SHA,'adapter':'experiments/v7.1-entropy-sharpening-v1/adapter.py','seed':2026,
          'candidates':{'entropy':{'kind':'ensemble_postprocessor','config':deepcopy(CONFIG)}},'candidate_order':['entropy'],
          'selection':'maximin_nonnegative_origins','bootstrap':{'resamples':1000,'seed':2026,'minimum_valid':990},
          'development':[case('jan22_long',1,2,5),case('jan24_forward',2,4,5)],
          'promotion':{name:case(name,a,b,7) for name,a,b in (('primary',4,5),('secondary',5,6),('five_day',1,2))},
          'inputs':inputs,'budget':{'phase_seconds':1200,'total_seconds':14400}}
    destination=root/'experiments/v7.1-entropy-sharpening-v1/experiment.json';write_new(destination,spec)
    print('Imported',len(declared),'explicit assets; input hashes will be verified under the preparation budget.')
    print(destination)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root',type=Path,required=True);parser.add_argument('--baseline-root',type=Path,required=True)
    args=parser.parse_args();stage(ROOT,args.asset_root.resolve(),args.baseline_root.resolve())
