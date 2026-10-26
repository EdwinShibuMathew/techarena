"""Deterministic fixture extraction, independent of the training recipe."""
import numpy as np
import torch
from smoke_model import load

FEATURE_NAMES=np.array([f"feature_{i}" for i in range(37)])


def inputs(data):
    shape=data["i"].shape
    return np.stack([data["i"]/9,np.broadcast_to(data["tb"][:,None]/3,shape),
                     np.broadcast_to(data["dt"][:,None],shape)],axis=2).reshape(-1,3).astype(np.float32)


def extract(training,target,model_path):
    if "y" in target:raise ValueError("Target labels leaked to extractor")
    model=load(model_path)
    x=np.zeros((len(target["u"])*10,37),dtype=np.float32)
    with torch.inference_mode():x[:,0]=model(torch.as_tensor(inputs(target))).sigmoid().numpy()
    x[:,1:4]=inputs(target)
    x[:,4]=float(training["y"].mean())
    return {"X":x,"feature_names":FEATURE_NAMES.copy()}
