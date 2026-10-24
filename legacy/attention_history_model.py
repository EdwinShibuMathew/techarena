"""Standalone candidate-conditioned history model, not a V5 correction.

An attention-inspired experiment, not a reproduction of the published DIN model.
Uses no user-ID embedding or V5 prediction as a model input.
"""

import math
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_final_models import include_no_click
from submission_export import sha256

CONFIG = {
    "architecture": "candidate_attention_history_v1",
    "dimension": 16, "hidden_units": 32, "seed": 2027,
    "initialization": "same-cutoff dim16 seed42 preference item vectors",
    "pretraining_epochs": 16, "pretraining_learning_rate": 0.002,
    "negative_items": 31, "event_epochs": 8, "event_learning_rate": 0.001,
    "batch_size": 2048, "anchor_regularization": 0.01,
    "neural_early_stopping": False,
}


class AttentionHistory(nn.Module):
    def __init__(self, vectors, bias):
        super().__init__()
        n_items, d = vectors.shape
        assert d == CONFIG["dimension"] and bias.shape == (n_items,)
        self.n_items = n_items
        anchor = torch.cat([vectors.detach().clone(), vectors.new_zeros(1, d)])
        self.register_buffer("anchor", anchor)
        self.items = nn.Embedding.from_pretrained(anchor.clone(), freeze=False, padding_idx=n_items)
        self.item_bias = nn.Embedding.from_pretrained(bias[:, None].detach().clone(), freeze=False)
        self.item_context = nn.Embedding(n_items, 8)
        nn.init.zeros_(self.item_context.weight)
        self.query = nn.Linear(d, d, bias=False)
        self.key = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.query.weight)
        nn.init.eye_(self.key.weight)
        self.log_decay = nn.Parameter(torch.tensor(-2.0))
        self.global_bias = nn.Parameter(torch.tensor(0.0))
        self.network = nn.Sequential(nn.Linear(2 * d + 3, CONFIG["hidden_units"]), nn.SiLU(), nn.Linear(CONFIG["hidden_units"], 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, candidates, history, age, context):
        present = history >= 0
        length = present.sum(1)
        h = self.items(torch.where(present, history, self.n_items)) * present[:, :, None]
        q = self.items(candidates)
        attention_scores = torch.einsum("bkd,bhd->bkh", self.query(q), self.key(h)) / math.sqrt(q.shape[-1])
        attention_scores = attention_scores - nn.functional.softplus(self.log_decay) * torch.log1p(age[:, None, :] / 24)
        # Finite masking and a second mask keep an entirely empty history safe.
        weights = attention_scores.masked_fill(~present[:, None, :], -1e4).softmax(-1)
        weights = weights * present[:, None, :]
        weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-8)
        attended = torch.einsum("bkh,bhd->bkd", weights, h)
        mean = h.sum(1) / length.clamp(min=1)[:, None]
        interactions = q * attended
        repeats = ((candidates[:, :, None] == history[:, None, :]) & present[:, None, :]).sum(-1)
        last_age = torch.where(present, age, 720.0).min(1).values.clamp(max=720)
        scalars = torch.stack([torch.log1p(length.float()) / math.log(21),
                               torch.log1p(last_age) / math.log(721)], -1)
        features = torch.cat([interactions, q * mean[:, None, :],
                              scalars[:, None, :].expand(-1, candidates.shape[1], -1),
                              torch.log1p(repeats.float())[:, :, None] / math.log(21)], -1)
        score = interactions.sum(-1) / math.sqrt(q.shape[-1]) + self.network(features).squeeze(-1)
        context_effect = self.item_context(candidates).gather(2, context[:, None, None].expand(-1, candidates.shape[1], 1)).squeeze(-1)
        return score + self.item_bias(candidates).squeeze(-1) + context_effect + self.global_bias

    def anchor_loss(self, candidates, history):
        present = history >= 0
        safe = torch.where(present, history, self.n_items)
        h = (self.items(safe) - self.anchor[safe]).square().sum(-1)
        q = (self.items(candidates) - self.anchor[candidates]).square().sum(-1)
        return q.mean() + (h * present).sum() / present.sum().clamp(min=1)


def pack(data, device, n_items):
    h, ht, t = data["h"], data["ht"], data["t"]
    assert h.shape == ht.shape == (len(t), 20)
    assert data["i"].shape == (len(t), 10)
    assert np.isin(data["tb"], [0, 1, 2, 3]).all() and np.isin(data["dt"], [0, 1]).all()
    present = h >= 0
    assert ((h == -1) | ((h >= 0) & (h < n_items))).all()
    assert ((data["i"] >= 0) & (data["i"] < n_items)).all()
    assert (ht[present] < np.broadcast_to(t[:, None], h.shape)[present]).all(), "History must precede its event"
    age = np.where(present, (t[:, None] - ht) / 3600, 0).clip(0, 720).astype(np.float32)
    return [torch.as_tensor(data["i"], dtype=torch.long, device=device),
            torch.as_tensor(h, dtype=torch.long, device=device),
            torch.as_tensor(age, dtype=torch.float32, device=device),
            torch.as_tensor(data["tb"].astype(np.int64) + 4 * data["dt"].astype(np.int64), device=device)]


def initialize(checkpoint_path, cutoff):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert state["dimension"] == 16 and state["epoch"] == 196
    # Earlier experimental checkpoint files did not always include cutoff metadata.
    if "cutoff" in state:
        assert state["cutoff"] == cutoff
    weights = state["model_state"]
    vectors, bias = weights["item.weight"].float(), weights["item_bias.weight"][:, 0].float()
    assert vectors.shape == (state["n_items"], 16)
    return AttentionHistory(vectors, bias)


def train(checkpoint_path, data, labels, cutoff, destination, device):
    assert not destination.exists()
    assert (data["t"] < cutoff).all()
    assert labels.shape == data["i"].shape and np.isin(labels, [0, 1]).all()
    assert (labels.sum(1) <= 1).all()
    start = time.perf_counter()
    torch.manual_seed(CONFIG["seed"])
    model = initialize(checkpoint_path, cutoff).to(device)
    inputs = pack(data, device, model.n_items)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    positive = torch.nonzero((y.sum(1) == 1) & (inputs[1] >= 0).any(1), as_tuple=True)[0]
    assert len(positive) > 0
    records = []
    for stage, epochs, lr in [("sampled_click", CONFIG["pretraining_epochs"], CONFIG["pretraining_learning_rate"]),
                              ("event", CONFIG["event_epochs"], CONFIG["event_learning_rate"])]:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        population = positive if stage == "sampled_click" else torch.arange(len(y), device=device)
        if stage == "event":
            # Calibrate only on fitting labels before learning event likelihood.
            model.eval()
            with torch.inference_mode():
                all_scores = torch.cat([model(*[v[b:b + 4096] for v in inputs]) for b in range(0, len(y), 4096)])
                actual = y.sum(1).float().mean()
                lower, upper = -30.0, 30.0
                for _ in range(45):
                    mid = (lower + upper) / 2
                    predicted = torch.sigmoid(torch.logsumexp(all_scores, 1) + mid).mean()
                    if predicted > actual:
                        upper = mid
                    else:
                        lower = mid
                model.global_bias.add_((lower + upper) / 2)
            del all_scores
        model.train()
        for epoch in range(1, epochs + 1):
            order = population[torch.randperm(len(population), device=device)]
            total = 0.0
            for begin in range(0, len(order), CONFIG["batch_size"]):
                index = order[begin:begin + CONFIG["batch_size"]]
                candidates, h, age, context = [value[index] for value in inputs]
                if stage == "sampled_click":
                    clicked = candidates.gather(1, y[index].argmax(1)[:, None])
                    negative = torch.randint(model.n_items, (len(index), CONFIG["negative_items"]), device=device)
                    # Avoid treating the actual target as a sampled negative.
                    while (negative == clicked).any():
                        mask = negative == clicked
                        negative[mask] = torch.randint(model.n_items, (int(mask.sum()),), device=device)
                    candidates = torch.cat([clicked, negative], 1)
                    target = torch.zeros(len(index), dtype=torch.long, device=device)
                    scores = model(candidates, h, age, context)
                else:
                    target = torch.where(y[index].sum(1) > 0, y[index].argmax(1) + 1, 0)
                    scores = include_no_click(model(candidates, h, age, context))
                loss = nn.functional.cross_entropy(scores, target) + CONFIG["anchor_regularization"] * model.anchor_loss(candidates, h)
                assert torch.isfinite(loss), "Nonfinite training loss"
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total += float(loss.detach()) * len(index)
            record = {"stage": stage, "epoch": epoch, "training_loss": total / len(population),
                      "elapsed_seconds": round(time.perf_counter() - start, 1)}
            records.append(record)
            if epoch % 4 == 0 or epoch == 1:
                print(record, flush=True)
    result = {"config": CONFIG.copy(), "cutoff": cutoff, "training_events": len(y),
              "pretraining_click_events": len(positive), "initialization_sha256": sha256(checkpoint_path),
              "n_items": model.n_items, "model_state": model.cpu().state_dict(), "records": records}
    temp = destination.with_suffix(".pt.tmp")
    torch.save(result, temp)
    temp.replace(destination)
    del model, inputs, y, optimizer
    torch.cuda.empty_cache()


def predict(path, target, cutoff, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["config"] == CONFIG and state["cutoff"] == cutoff
    assert (target["t"] >= cutoff).all()
    model = AttentionHistory(torch.zeros(state["n_items"], CONFIG["dimension"]), torch.zeros(state["n_items"]))
    model.load_state_dict(state["model_state"])
    model.to(device).eval()
    inputs = pack(target, device, model.n_items)
    parts = []
    with torch.inference_mode():
        for begin in range(0, len(target["t"]), 2048):
            parts.append(include_no_click(model(*[v[begin:begin + 2048] for v in inputs])).softmax(1)[:, 1:].cpu().numpy())
    result = np.concatenate(parts).astype(np.float64)
    assert np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()
    del model, state, inputs
    torch.cuda.empty_cache()
    return result


def self_test():
    assert torch.cuda.is_available(), "Activate techarena with working CUDA."
    torch.manual_seed(CONFIG["seed"])
    model = AttentionHistory(torch.randn(30, 16), torch.zeros(30)).cuda()
    now = np.full(3, 1737763200, dtype=np.int64)
    h = np.full((3, 20), -1, dtype=np.int64)
    ht = np.zeros_like(h)
    h[0, :3], ht[0, :3] = [1, 2, 3], now[0] - np.array([3600, 7200, 10800])
    h[2, :2], ht[2, :2] = [4, 4], now[2] - np.array([3600, 7200])
    data = {"i": np.tile(np.arange(10), (3, 1)), "h": h, "ht": ht, "t": now,
            "tb": np.zeros(3, dtype=np.int64), "dt": np.zeros(3, dtype=np.int64)}
    inputs = pack(data, torch.device("cuda"), 30)
    scores = model(*inputs)
    assert scores.shape == (3, 10) and torch.isfinite(scores).all()
    permutation = torch.tensor([3, 0, 8, 1, 2, 9, 6, 5, 7, 4], device="cuda")
    permuted = model(inputs[0][:, permutation], *inputs[1:])
    torch.testing.assert_close(permuted, scores[:, permutation])
    # Padding position cannot contribute, including for entirely empty histories.
    with torch.no_grad():
        model.items.weight[-1].fill_(37)
    # Forward explicitly masks padding, so its contents must not matter.
    altered = model(*inputs)
    torch.testing.assert_close(altered, scores)
    with torch.no_grad():
        model.items.weight[-1].zero_()
    loss = nn.functional.cross_entropy(include_no_click(scores), torch.tensor([2, 0, 5], device="cuda"))
    loss.backward()
    assert model.query.weight.grad.abs().sum() > 0
    assert model.items.weight.grad[-1].abs().sum() == 0
    bad = dict(data, ht=ht.copy())
    bad["ht"][0, 0] = now[0]
    try:
        pack(bad, torch.device("cuda"), 30)
    except AssertionError:
        pass
    else:
        raise AssertionError("Equal-time target leaked into history")
    # Exercise both real training stages and checkpoint loading on a tiny dataset.
    # Keep self-test files inside this separate experiment's parent workspace.
    with tempfile.TemporaryDirectory(prefix="attention_selftest_", dir=Path(__file__).resolve().parents[1]) as temp:
        folder = Path(temp)
        initial = folder / "initial.pt"
        torch.save({"dimension": 16, "epoch": 196, "n_items": 30,
                    "model_state": {"item.weight": torch.randn(30, 16), "item_bias.weight": torch.zeros(30, 1)}}, initial)
        labels = np.zeros((3, 10), dtype=np.int8)
        labels[0, 1] = labels[2, 4] = 1
        destination = folder / "attention.pt"
        train(initial, data, labels, int(now[0]) + 1, destination, torch.device("cuda"))
        future = dict(data, t=now + 2)
        probabilities = predict(destination, future, int(now[0]) + 1, torch.device("cuda"))
        assert probabilities.shape == (3, 10) and (probabilities.sum(1) <= 1 + 1e-6).all()
    print("Attention model self-test passed: candidate order, padding, gradients, causal history, both training stages and checkpoint reload.", flush=True)
