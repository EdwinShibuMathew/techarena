"""Synthetic-only adapter: real fitting/reload, no private data or production output."""
from pathlib import Path
import sys
import numpy as np
import torch
from catboost import CatBoostClassifier,Pool

sys.path.insert(0,str(Path(__file__).resolve().parent))
from smoke_model import SmokeModel
from smoke_extract import extract,inputs,FEATURE_NAMES

PURPOSES={"synthetic_validation"}
EXTRACTOR_FILES=("experiments/workflow-smoke/smoke_extract.py","experiments/workflow-smoke/smoke_model.py")


def train(fitting,metadata,path,params):
    if metadata["purpose"]!="synthetic_validation":raise ValueError("Fixture cannot train a research model")
    torch.manual_seed(metadata["seed"]);torch.set_num_threads(2)
    model=SmokeModel(metadata["dimension"])
    optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    x=torch.as_tensor(inputs(fitting));y=torch.as_tensor(fitting["y"].ravel(),dtype=torch.float32)
    for _ in range(metadata["epoch"]):
        optimizer.zero_grad()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x),y)
        loss.backward();optimizer.step()
    with Path(path).open("xb") as stream:torch.save({"metadata":metadata,"model_state":model.state_dict()},stream)


def fit_combiner(features,labels,path,seed):
    model=CatBoostClassifier(iterations=24,depth=3,learning_rate=.3,loss_function="Logloss",random_seed=seed,
                            thread_count=2,allow_writing_files=False,verbose=False)
    model.fit(Pool(features["X"],label=labels.ravel(),feature_names=features["feature_names"].tolist()))
    model.save_model(str(path))


def predict(features,path,rowids):
    model=CatBoostClassifier();model.load_model(str(path))
    p=model.predict_proba(Pool(features["X"],feature_names=features["feature_names"].tolist()),thread_count=2)[:,1]
    return {"rowids":rowids.copy(),"scores":p.reshape(-1,10)}
