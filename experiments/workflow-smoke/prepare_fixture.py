"""Generate disposable labeled/unlabeled data and an explicitly synthetic spec."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools"))
from ml_workflow.contracts import DAY,JAN20,V6_SHA,MEMBERS,first_indices,compose,digest,identifier
from ml_workflow.storage import write_new,save_arrays


def create(root,fixture_id="workflow-smoke"):
    identifier(fixture_id)
    folder=root/"artifacts/validation-inputs"/fixture_id
    folder.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(2026)
    def make(start,hours,labeled):
        n=hours;t=start+np.arange(n)*3600
        items=np.stack([rng.permutation(10) for _ in range(n)])
        data={"u":np.arange(n)+(0 if labeled else 1000),"i":items,"t":t,
            "tb":((t-JAN20)//(6*3600))%4,"dt":(((t-JAN20)//DAY)>=5).astype(int),
            "h":np.full((n,20),-1,dtype=np.int64),"ht":np.zeros((n,20),dtype=np.int64),
            "rowids":np.array([[f"{'train' if labeled else 'eval'}-{i}-{j}" for j in range(10)] for i in range(n)])}
        data["h"][::2,0]=1;data["ht"][::2,0]=t[::2]-60
        if labeled:data["y"]=((items==9)&(data["tb"][:,None]>0)).astype(np.int8)
        return data
    train=make(JAN20,7*24,True);evaluation=make(JAN20+7*DAY,48,False)
    inputs={}
    def store(name,**arrays):
        path=folder/(name+".npz");save_arrays(path,**arrays)
        inputs[name]={"path":path.relative_to(root).as_posix(),"sha256":digest(path)}
    store("train",**train);store("evaluation",**evaluation)
    development=[{"name":"development_early","meta_cutoff":JAN20+DAY,"target_cutoff":JAN20+2*DAY,
                  "target_end":JAN20+5*DAY,"baseline":"baseline_development"}]
    promotion={name:{"name":name,"meta_cutoff":JAN20+a*DAY,"target_cutoff":JAN20+b*DAY,"target_end":JAN20+7*DAY,"baseline":"baseline_"+name}
        for name,a,b in (("primary",4,5),("secondary",5,6),("five_day",1,2))}
    for case in development+list(promotion.values()):
        idx=first_indices(train["u"],train["t"],case["target_cutoff"],case["target_end"])
        p={member:np.full((len(idx),10),.05) for member in MEMBERS}
        store(case["baseline"],version=np.array("synthetic"),member_names=np.array(list(MEMBERS)),
              rowids=train["rowids"][idx],members=np.stack(list(p.values())),scores=compose(p))
    p={member:np.full(evaluation["i"].shape,.05) for member in MEMBERS}
    store("baseline_evaluation",version=np.array("synthetic"),member_names=np.array(list(MEMBERS)),
          rowids=evaluation["rowids"],members=np.stack(list(p.values())),scores=compose(p))
    spec={"schema_version":1,"purpose":"synthetic_validation","hypothesis":"Verify orchestration and chronological contracts on generated data",
        "comparator":"V6","baseline_sha256":V6_SHA,"adapter":"experiments/workflow-smoke/adapter.py","seed":2026,
        "candidates":{"fixture":{"members":["dim8_seed42","dim16_seed42"]}},"candidate_order":["fixture"],"selection":"maximin_nonnegative_origins",
        "bootstrap":{"resamples":1000,"seed":2026,"minimum_valid":990},"development":development,"promotion":promotion,
        "inputs":inputs,"budget":{"phase_seconds":120,"total_seconds":600}}
    path=folder/"experiment.json";write_new(path,spec);return path


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--fixture-id",default="workflow-smoke")
    args=parser.parse_args();print(create(ROOT,args.fixture_id))
