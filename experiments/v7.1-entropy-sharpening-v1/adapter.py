"""Two-parameter, event-mass-preserving allocation; no neural fitting."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np

PURPOSES={"research","synthetic_validation"}
CONFIG={"formula":"exp(a+b*(normalized_entropy-0.5))","bounds":[-.25,.25],
        "bce_weight":.5,"pair_weight":.5,"l2":.01,"negative_rows_per_positive":16,
        "seed":2026,"folds":5,"maxiter":200,"ftol":1e-12,"gtol":1e-8,
        "bisection_steps":40,"probability_clipping":False,"routing_features":["entropy"]}


def components(probabilities):
    p=np.asarray(probabilities,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=10 or not len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError("Expected finite ten-candidate probabilities in [0,1]")
    mass=p.sum(1,keepdims=True)
    q=np.divide(p,mass,out=np.full_like(p,.1),where=mass>0)
    logq=np.log(q,where=q>0,out=np.full_like(q,-np.inf))
    entropy=-(q*np.where(q>0,logq,0)).sum(1)/np.log(10)
    limit=np.full(len(p),np.exp(.375))
    constrained=(mass[:,0]>1)&(p.max(1)<1)
    if constrained.any():
        lq=logq[constrained];s=mass[constrained]
        low=np.ones(len(lq));high=limit[constrained].copy()
        for _ in range(CONFIG["bisection_steps"]):
            middle=(low+high)/2
            logits=middle[:,None]*lq;z=np.exp(logits-logits.max(1,keepdims=True))
            feasible=(s*z/z.sum(1,keepdims=True)).max(1)<=1
            low=np.where(feasible,middle,low);high=np.where(feasible,high,middle)
        limit[constrained]=low
    limit[(mass[:,0]>1)&(p.max(1)==1)]=1
    return p,mass,logq,entropy,limit


def predict_calibrator(probabilities,coefficients):
    p,mass,logq,entropy,limit=components(probabilities)
    coefficients=np.asarray(coefficients,dtype=np.float64)
    if coefficients.shape!=(2,) or not np.isfinite(coefficients).all() or (np.abs(coefficients)>.25).any():
        raise ValueError("Invalid calibrator coefficients")
    if np.array_equal(coefficients,np.zeros(2)):return p.copy()
    gamma=np.minimum(np.exp(coefficients[0]+coefficients[1]*(entropy-.5)),limit)
    logits=gamma[:,None]*logq
    z=np.exp(logits-logits.max(1,keepdims=True));result=mass*z/z.sum(1,keepdims=True)
    result[gamma==1]=p[gamma==1]
    if not np.isfinite(result).all() or ((result<0)|(result>1)).any():raise ValueError("Invalid calibrated probabilities; no clipping")
    if not np.allclose(result.sum(1),mass[:,0],rtol=0,atol=1e-14):raise ValueError("Event mass changed")
    return result


def user_folds(users):
    users=np.asarray(users)
    unique=np.unique(users);order=np.random.default_rng(CONFIG["seed"]).permutation(len(unique))
    assigned=np.empty(len(unique),dtype=np.int8);assigned[order]=np.arange(len(unique))%CONFIG["folds"]
    return assigned[np.searchsorted(unique,users)]


def fit_calibrator(probabilities,labels,metadata,path):
    import torch
    from scipy.optimize import minimize
    torch.set_num_threads(4)
    p,mass,logq,entropy,limit=components(probabilities)
    y=np.asarray(labels)
    if y.shape!=p.shape or not np.isin(y,[0,1]).all() or (y.sum(1)>1).any():raise ValueError("Invalid calibration labels")
    positive=np.flatnonzero(y.ravel()==1);negative=np.flatnonzero(y.ravel()==0)
    if not len(positive) or not len(negative):raise ValueError("Both row classes required")
    sampled=negative[np.random.default_rng(CONFIG["seed"]).integers(len(negative),size=(len(positive),16))]
    s,lq,h,cap,target=[torch.tensor(a,dtype=torch.float64) for a in (mass,np.where(np.isfinite(logq),logq,0),entropy,limit,y)]
    mask=torch.tensor(np.isfinite(logq))
    pos=torch.tensor(positive);neg=torch.tensor(sampled)
    def objective(values):
        theta=torch.tensor(values,dtype=torch.float64,requires_grad=True)
        gamma=torch.minimum(torch.exp(theta[0]+theta[1]*(h-.5)),cap)
        pred=s*torch.softmax((gamma[:,None]*lq).masked_fill(~mask,-torch.inf),1)
        # Stabilize logarithms only, never exported probabilities.
        safe=pred.clamp(1e-15,1-1e-15)
        bce=-(target*safe.log()+(1-target)*torch.log1p(-safe)).mean()
        odds=(safe.log()-torch.log1p(-safe)).ravel()
        pair=torch.nn.functional.softplus(odds[neg]-odds[pos,None]).mean()
        loss=.5*bce+.5*pair+.01*theta.square().sum()
        loss.backward()
        return float(loss.detach()),theta.grad.detach().numpy().copy()
    identity_loss=objective(np.zeros(2))[0]
    result=minimize(objective,np.zeros(2),jac=True,method="L-BFGS-B",bounds=[(-.25,.25)]*2,
                    options={"maxiter":200,"ftol":1e-12,"gtol":1e-8,"maxls":50})
    if not result.success or not np.isfinite(result.fun):raise ValueError("Calibrator did not converge: "+str(result.message))
    predict_calibrator(p,result.x)
    record={"schema_version":1,"metadata":deepcopy(metadata),"config":deepcopy(CONFIG),"coefficients":result.x.tolist(),
            "objective":float(result.fun),"identity_objective":identity_loss,"iterations":int(result.nit),"converged":True}
    from ml_workflow.storage import write_new
    write_new(path,record)
    return record


def load_calibrator(path,metadata=None):
    saved=json.loads(Path(path).read_text())
    if saved["config"]!=CONFIG or saved["converged"] is not True:raise ValueError("Calibrator contract changed")
    if metadata is not None and saved["metadata"]!=metadata:raise ValueError("Calibrator lineage changed")
    coefficients=np.asarray(saved["coefficients"],dtype=np.float64)
    if coefficients.shape!=(2,) or not np.isfinite(coefficients).all() or (np.abs(coefficients)>.25).any():raise ValueError("Invalid saved coefficients")
    return saved
