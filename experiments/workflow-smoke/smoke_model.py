"""Small CPU fixture; never a competition model."""
import torch
from torch import nn


class SmokeModel(nn.Module):
    def __init__(self,dimension):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(3,dimension),nn.Tanh(),nn.Linear(dimension,1))

    def forward(self,x):return self.network(x).squeeze(-1)


def load(path):
    saved=torch.load(path,map_location="cpu",weights_only=True)
    if saved["metadata"]["purpose"]!="synthetic_validation":raise ValueError("Synthetic model purpose mismatch")
    model=SmokeModel(saved["metadata"]["dimension"])
    model.load_state_dict(saved["model_state"])
    return model.eval()
