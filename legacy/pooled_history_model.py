"""Small history correction on a frozen dim8 seed42 event model.

Run the GPU correctness check: python src/pooled_history_model.py --self-test
Training/evaluation are orchestrated by run_history_trial.py.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_final_models import PreferenceModel, EventModel, include_no_click
from submission_export import sha256

CONFIG = {
    "architecture": "pooled_history_correction_v1",
    "dimension": 8, "hidden_units": 32,
    "pools": ["mean", "recency_weighted_mean", "last"],
    "recency_decay_hours": 24.0, "maximum_age_feature_hours": 720.0,
    "trainable_history_embeddings": True, "base_model_frozen": True,
    "epochs": 8, "batch_size": 2048, "learning_rate": 0.001,
    "history_anchor_regularization": 0.02, "correction_regularization": 0.01,
    "seed": 2026, "early_stopping": False,
}


class HistoryCorrection(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.n_items, dimension = base.preference.item.weight.shape
        assert dimension == CONFIG["dimension"]
        original = base.preference.item.weight.detach().clone()
        anchor = torch.cat([original, original.new_zeros(1, dimension)])
        self.register_buffer("history_anchor", anchor)
        self.history_items = nn.Embedding(self.n_items + 1, dimension, padding_idx=self.n_items)
        with torch.no_grad():
            self.history_items.weight.copy_(anchor)
        # Three elementwise candidate/history interactions plus length, age, repeat count.
        self.correction = nn.Sequential(nn.Linear(3 * dimension + 3, CONFIG["hidden_units"]),
                                        nn.SiLU(), nn.Linear(CONFIG["hidden_units"], 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, users, items, context, history, age_hours):
        with torch.no_grad():
            base_scores = self.base(users, items, context)
            candidate_vectors = self.base.preference.item(items)
        present = history >= 0
        safe = torch.where(present, history, self.n_items)
        vectors = self.history_items(safe)
        length = present.sum(dim=1)
        mean = (vectors * present[:, :, None]).sum(dim=1) / length.clamp(min=1)[:, None]
        weights = torch.exp(-age_hours / CONFIG["recency_decay_hours"]) * present
        recent = (vectors * weights[:, :, None]).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-8)[:, None]
        positions = torch.arange(history.shape[1], device=history.device)[None, :]
        last_index = torch.where(present, positions, -1).max(dim=1).values.clamp(min=0)
        last = vectors[torch.arange(len(users), device=users.device), last_index]
        last = last * (length > 0)[:, None]
        last_age = torch.where(present, age_hours, float("inf")).min(dim=1).values
        last_age = last_age.clamp(max=CONFIG["maximum_age_feature_hours"])
        repeats = ((items[:, :, None] == history[:, None, :]) & present[:, None, :]).sum(dim=2)
        scalar = torch.stack([
            torch.log1p(length.float()) / math.log(21),
            torch.log1p(last_age) / math.log(721),
        ], dim=1)[:, None, :].expand(-1, items.shape[1], -1)
        interactions = [candidate_vectors * profile[:, None, :] for profile in [mean, recent, last]]
        features = torch.cat(interactions + [scalar, (torch.log1p(repeats.float()) / math.log(21))[:, :, None]], dim=2)
        delta = self.correction(features).squeeze(-1) * (length > 0)[:, None]
        return base_scores + delta, delta

    def anchor_loss(self, history):
        present = history >= 0
        safe = torch.where(present, history, self.n_items)
        difference = (self.history_items(safe) - self.history_anchor[safe]).square().sum(dim=-1)
        return (difference * present).sum() / present.sum().clamp(min=1)


def load_base(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["dimension"] == 8 and checkpoint["epoch"] == 8
    preference = PreferenceModel(checkpoint["n_users"], checkpoint["n_items"], 8,
                                 torch.ones(checkpoint["n_users"], dtype=torch.bool))
    base = EventModel(preference, checkpoint["n_users"], checkpoint["n_items"])
    base.load_state_dict(checkpoint["model_state"])
    return base, checkpoint


def pack(target, device, n_users, n_items):
    present = target["h"] >= 0
    assert target["i"].shape == (len(target["u"]), 10)
    assert target["h"].shape == target["ht"].shape == (len(target["u"]), 20)
    assert ((target["u"] >= 0) & (target["u"] < n_users)).all()
    assert ((target["i"] >= 0) & (target["i"] < n_items)).all()
    assert ((target["h"] == -1) | (present & (target["h"] < n_items))).all()
    assert (target["ht"][present] < np.broadcast_to(target["t"][:, None], present.shape)[present]).all()
    age = np.where(present, (target["t"][:, None] - target["ht"]) / 3600, 0).astype(np.float32)
    values = [target["u"], target["i"], target["tb"].astype(np.int64) + 4 * target["dt"].astype(np.int64), target["h"]]
    result = [torch.tensor(value, dtype=torch.long, device=device) for value in values]
    return result + [torch.tensor(age, dtype=torch.float32, device=device)]


def train_model(base_path, fitting, labels, cutoff, destination, device):
    start = time.perf_counter()
    assert (fitting["t"] < cutoff).all()
    assert np.isin(labels, [0, 1]).all() and (labels.sum(axis=1) <= 1).all()
    torch.manual_seed(CONFIG["seed"])
    base, metadata = load_base(base_path)
    model = HistoryCorrection(base).to(device)
    before = {key: value.detach().cpu().clone() for key, value in model.base.state_dict().items()}
    inputs = pack(fitting, device, metadata["n_users"], metadata["n_items"])
    targets = torch.tensor(np.where(labels.sum(axis=1) > 0, labels.argmax(axis=1) + 1, 0),
                           dtype=torch.long, device=device)
    with torch.inference_mode():
        sample = [value[:4096] for value in inputs]
        scores, delta = model(*sample)
        assert torch.count_nonzero(delta).item() == 0
        assert torch.equal(scores, model.base(*sample[:3])), "Zero correction did not reproduce base scores"
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=CONFIG["learning_rate"])
    records = []
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        model.base.eval()
        order = torch.randperm(len(targets), device=device)
        total = torch.zeros(3, device=device)
        for begin in range(0, len(order), CONFIG["batch_size"]):
            index = order[begin:begin + CONFIG["batch_size"]]
            batch = [value[index] for value in inputs]
            optimizer.zero_grad()
            scores, delta = model(*batch)
            choice = nn.functional.cross_entropy(include_no_click(scores), targets[index])
            anchor = model.anchor_loss(batch[3])
            penalty = delta.square().mean()
            loss = choice + CONFIG["history_anchor_regularization"] * anchor + CONFIG["correction_regularization"] * penalty
            loss.backward()
            optimizer.step()
            total += torch.stack([choice.detach(), anchor.detach(), penalty.detach()]) * len(index)
        means = (total / len(targets)).cpu().tolist()
        assert np.isfinite(means).all()
        record = {"epoch": epoch, "choice_loss": means[0], "history_anchor_loss": means[1],
                  "correction_mean_square": means[2], "elapsed_seconds": round(time.perf_counter() - start, 1)}
        records.append(record)
        print(json.dumps(record), flush=True)
    assert all(parameter.grad is None for parameter in model.base.parameters())
    assert all(torch.equal(before[key], value.detach().cpu()) for key, value in model.base.state_dict().items())
    checkpoint = {
        "kind": CONFIG["architecture"], "config": CONFIG.copy(), "epoch": CONFIG["epochs"],
        "cutoff": cutoff, "base_checkpoint": str(base_path), "base_sha256": sha256(base_path),
        "n_users": metadata["n_users"], "n_items": metadata["n_items"],
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "training_events": len(targets), "base_unchanged": True,
    }
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(destination)
    result = {"checkpoint": str(destination), "fitting_cutoff": cutoff,
              "training_events": len(targets), "base_unchanged": True,
              "trainable_parameters": sum(p.numel() for p in trainable), "history": records,
              "runtime_seconds": round(time.perf_counter() - start, 2)}
    destination.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def predict_model(path, target, device, cutoff):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["kind"] == CONFIG["architecture"] and checkpoint["config"] == CONFIG
    assert checkpoint["epoch"] == CONFIG["epochs"] and checkpoint["cutoff"] == cutoff
    assert (target["t"] >= cutoff).all()
    base_path = Path(checkpoint["base_checkpoint"])
    assert sha256(base_path) == checkpoint["base_sha256"]
    base, metadata = load_base(base_path)
    model = HistoryCorrection(base)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    inputs = pack(target, device, metadata["n_users"], metadata["n_items"])
    parts = []
    with torch.inference_mode():
        for begin in range(0, len(target["u"]), 4096):
            batch = [value[begin:begin + 4096] for value in inputs]
            scores, delta = model(*batch)
            empty = (batch[3] >= 0).sum(dim=1) == 0
            assert not empty.any() or torch.count_nonzero(delta[empty]).item() == 0
            parts.append(include_no_click(scores).softmax(dim=1)[:, 1:].cpu().numpy())
    probabilities = np.concatenate(parts)
    assert probabilities.shape == target["i"].shape
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    del model, checkpoint, inputs
    torch.cuda.empty_cache()
    return probabilities


def self_test():
    assert torch.cuda.is_available(), "Activate techarena with working CUDA."
    torch.set_num_threads(4)
    torch.manual_seed(CONFIG["seed"])
    base = EventModel(PreferenceModel(6, 15, 8, torch.ones(6, dtype=torch.bool)), 6, 15)
    model = HistoryCorrection(base).cuda()
    now = np.full(4, 1737763200, dtype=np.int64)
    h, ht = np.full((4, 20), -1), np.zeros((4, 20), dtype=np.int64)
    h[0, :3], ht[0, :3] = [1, 2, 3], now[0] - np.array([10800, 7200, 3600])
    h[2, :2], ht[2, :2] = [3, 3], now[2] - np.array([7200, 3600])
    h[3, :2], ht[3, :2] = [4, 5], now[3] - np.array([7200, 3600])
    target = {"u": np.arange(4), "i": np.tile(np.arange(10), (4, 1)), "t": now,
              "tb": np.zeros(4), "dt": np.zeros(4), "h": h, "ht": ht}
    batch = pack(target, torch.device("cuda"), 6, 15)
    frozen = {name: tensor.detach().clone() for name, tensor in model.base.state_dict().items()}
    scores, delta = model(*batch)
    assert torch.equal(scores, model.base(*batch[:3])) and torch.count_nonzero(delta).item() == 0
    assert scores.shape == (4, 10)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=CONFIG["learning_rate"])
    for _ in range(3):
        optimizer.zero_grad()
        scores, delta = model(*batch)
        loss = nn.functional.cross_entropy(include_no_click(scores), torch.tensor([2, 0, 4, 6], device="cuda"))
        loss = loss + CONFIG["history_anchor_regularization"] * model.anchor_loss(batch[3])
        loss.backward()
        optimizer.step()
    assert all(torch.equal(frozen[name], tensor) for name, tensor in model.base.state_dict().items())
    assert all(p.grad is None for p in model.base.parameters())
    assert torch.count_nonzero(model.history_items.weight.grad).item() > 0
    assert torch.count_nonzero(model.history_items.weight[-1]).item() == 0
    scores, delta = model(*batch)
    assert torch.count_nonzero(delta[1]).item() == 0
    probabilities = include_no_click(scores).softmax(1)[:, 1:]
    assert torch.isfinite(probabilities).all() and (probabilities.sum(1) <= 1 + 1e-6).all()
    bad = dict(target, ht=ht.copy())
    bad["ht"][0, 0] = now[0]
    try:
        pack(bad, torch.device("cuda"), 6, 15)
    except AssertionError:
        pass
    else:
        raise AssertionError("Non-past history was accepted")
    print("POOLED HISTORY GPU SELF-TEST PASSED: zero-initialized identity; frozen base; trainable history; neutral empty history; padding; causal timestamps; event probabilities.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    self_test()
