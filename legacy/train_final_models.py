"""Train Edwin's fixed TechArena neural models on all seven training days.

Run in WSL from the techarena-final workspace:
    python src/train_final_models.py --seed 42 --check-only
    python src/train_final_models.py --seed 42
    python src/train_final_models.py --seed 43

If preference training finished but event training was interrupted:
    python src/train_final_models.py --seed 42 --resume-events

This script reads training examples only. It does not calculate validation AUC,
fit a combiner, or create a competition submission.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


class PreferenceModel(nn.Module):
    def __init__(self, n_users, n_items, dimension, active_users):
        super().__init__()

        self.user = nn.Embedding(n_users, dimension)
        self.item = nn.Embedding(n_items, dimension)
        self.item_bias = nn.Embedding(n_items, 1)
        self.register_buffer("active_users", active_users)

        nn.init.normal_(self.user.weight, std=0.1)
        nn.init.normal_(self.item.weight, std=0.1)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users, items):
        user_vectors = self.user(users)

        # Users without training clicks get a neutral user vector.
        user_vectors = (
            user_vectors * self.active_users[users, None]
        )

        item_vectors = self.item(items)

        return (
            user_vectors[:, None, :] * item_vectors
        ).sum(dim=-1) + self.item_bias(items).squeeze(-1)


class EventModel(nn.Module):
    def __init__(self, preference, n_users, n_items):
        super().__init__()
        self.preference = preference

        self.user_bias = nn.Embedding(n_users, 1)
        self.user_context = nn.Embedding(n_users, 8)
        self.item_context = nn.Embedding(n_items, 8)
        self.global_bias = nn.Parameter(torch.tensor(0.0))

        for layer in [
            self.user_bias,
            self.user_context,
            self.item_context,
        ]:
            nn.init.zeros_(layer.weight)

    def forward(self, users, items, context):
        scores = self.preference(users, items)

        item_effect = self.item_context(items).gather(
            2,
            context[:, None, None].expand(-1, items.shape[1], 1),
        ).squeeze(-1)

        user_effect = self.user_context(users).gather(
            1, context[:, None]
        )

        return (
            scores
            + self.user_bias(users)
            + self.global_bias
            + item_effect
            + user_effect
        )


def include_no_click(scores):
    # Outcome zero is "no click"; outcomes 1–10 are candidates.
    return torch.cat(
        [torch.zeros_like(scores[:, :1]), scores], dim=1
    )


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, model, optimizer, protocol, stage, epoch):
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "n_users": protocol["n_users"],
        "n_items": protocol["n_items"],
        "dimension": protocol["dimension"],
        "seed": protocol["seed"],
        "cutoff": protocol["cutoff"],
        "stage": stage,
        "epoch": epoch,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_training_data(cache):
    with np.load(cache / "preference_training.npz") as saved:
        positive_users = saved["users"]
        positive_items = saved["items"]
        positive_times = saved["timestamps"]
        n_users = int(saved["n_users"])
        n_items = int(saved["n_items"])
        cutoff = int(saved["cutoff"])

    assert cutoff == 1737936000
    assert n_users == 54456 and n_items == 50000
    assert positive_users.ndim == positive_items.ndim == 1
    assert positive_users.shape == positive_items.shape == positive_times.shape
    assert len(positive_users) == 1009707
    assert (positive_times < cutoff).all()
    assert ((positive_users >= 0) & (positive_users < n_users)).all()
    assert ((positive_items >= 0) & (positive_items < n_items)).all()

    with np.load(cache / "train_events.npz") as saved:
        users = saved["u"]
        items = saved["i"]
        times = saved["t"]
        labels = saved["y"]
        time_buckets = saved["tb"].astype(np.int64)
        day_types = saved["dt"].astype(np.int64)

    assert len(users) == 686824
    assert items.shape == labels.shape == (len(users), 10)
    assert (times < cutoff).all()
    assert np.isin(labels, [0, 1]).all()
    assert (labels.sum(axis=1) <= 1).all()
    assert int(labels.sum()) == 432023
    assert ((users >= 0) & (users < n_users)).all()
    assert ((items >= 0) & (items < n_items)).all()
    assert ((time_buckets >= 0) & (time_buckets <= 3)).all()
    assert np.isin(day_types, [0, 1]).all()

    targets = np.where(
        labels.sum(axis=1) > 0,
        labels.argmax(axis=1) + 1,
        0,
    )
    event_data = (users, items, time_buckets + 4 * day_types, targets)
    return positive_users, positive_items, event_data, n_users, n_items, cutoff


def train_preferences(positive_users, positive_items, protocol, path, log_path):
    start = time.perf_counter()
    torch.manual_seed(protocol["seed"])
    device = torch.device("cuda")
    n_users, n_items = protocol["n_users"], protocol["n_items"]
    active_users = torch.tensor(
        np.bincount(positive_users, minlength=n_users) > 0
    )
    users = torch.tensor(positive_users, dtype=torch.long, device=device)
    items = torch.tensor(positive_items, dtype=torch.long, device=device)

    model = PreferenceModel(
        n_users, n_items, protocol["dimension"], active_users
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.012)
    records = []
    batch_size = 4096
    negative_count = 31

    for epoch in range(1, 197):
        model.train()
        order = torch.randperm(len(users), device=device)
        loss_total = torch.zeros((), device=device)

        for start_idx in range(0, len(users), batch_size):
            batch = order[start_idx:start_idx + batch_size]
            batch_users = users[batch]
            clicked_items = items[batch]
            sampled_items = torch.randint(
                n_items, (len(batch), negative_count), device=device
            )
            candidates = torch.cat(
                [clicked_items[:, None], sampled_items], dim=1
            )
            targets = torch.zeros(len(batch), dtype=torch.long, device=device)

            optimizer.zero_grad()
            scores = model(batch_users, candidates)
            choice_loss = nn.functional.cross_entropy(scores, targets)
            regularization = 0.02 * (
                model.user(batch_users).square().sum(-1).mean()
                + model.item(candidates).square().sum(-1).mean()
            )
            loss = choice_loss + regularization
            loss.backward()
            optimizer.step()
            loss_total += loss.detach() * len(batch)

        mean_loss = float(loss_total.item() / len(users))
        if not math.isfinite(mean_loss):
            raise RuntimeError(f"Non-finite preference loss at epoch {epoch}")
        record = {
            "stage": "preference",
            "seed": protocol["seed"],
            "epoch": epoch,
            "training_loss": mean_loss,
            "elapsed_seconds": round(time.perf_counter() - start, 1),
        }
        records.append(record)
        if epoch == 1 or epoch % 20 == 0 or epoch == 196:
            print(json.dumps(record), flush=True)
            write_json(log_path, {"protocol": protocol, "history": records})

    save_checkpoint(path, model, optimizer, protocol, "preference", 196)
    return {"epoch": 196, "training_loss": mean_loss, "checkpoint": str(path)}


def train_events(event_data, protocol, preference_path, path, log_path):
    start = time.perf_counter()
    torch.manual_seed(protocol["seed"])
    device = torch.device("cuda")
    n_users, n_items = protocol["n_users"], protocol["n_items"]

    checkpoint = torch.load(
        preference_path, map_location=device, weights_only=True
    )
    for key in ["seed", "n_users", "n_items", "dimension", "cutoff"]:
        assert checkpoint[key] == protocol[key], key
    assert checkpoint["stage"] == "preference" and checkpoint["epoch"] == 196

    # Recreate the models after resetting the seed, as in the tested event script.
    preference = PreferenceModel(
        n_users, n_items, protocol["dimension"],
        torch.ones(n_users, dtype=torch.bool),
    ).to(device)
    preference.load_state_dict(checkpoint["model_state"])
    del checkpoint
    model = EventModel(preference, n_users, n_items).to(device)

    train_inputs = [
        torch.tensor(values, dtype=torch.long, device=device)
        for values in event_data[:3]
    ]
    train_targets = torch.tensor(
        event_data[3], dtype=torch.long, device=device
    )
    n_events = len(train_targets)
    anchor_user = preference.user.weight.detach().clone()
    anchor_item = preference.item.weight.detach().clone()

    # Set the global offset using the training event click rate only.
    with torch.inference_mode():
        log_totals = []
        for st in range(0, n_events, 4096):
            batch_inputs = [x[st:st + 4096] for x in train_inputs]
            log_totals.append(model(*batch_inputs).logsumexp(dim=1))
        log_totals = torch.cat(log_totals)
        click_rate = (train_targets > 0).float().mean()
        low, high = -40.0, 40.0
        for _ in range(30):
            midpoint = (low + high) / 2
            estimated_rate = torch.sigmoid(log_totals + midpoint).mean()
            if estimated_rate > click_rate:
                high = midpoint
            else:
                low = midpoint
        model.global_bias.fill_((low + high) / 2)
    del log_totals

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)
    records = []
    batch_size = 2048

    for epoch in range(1, 9):
        model.train()
        order = torch.randperm(n_events, device=device)
        loss_total = torch.zeros((), device=device)
        for st in range(0, len(order), batch_size):
            q = order[st:st + batch_size]
            batch_inputs = [x[q] for x in train_inputs]
            users, items, _ = batch_inputs
            optimizer.zero_grad()
            scores = model(*batch_inputs)
            loss = nn.functional.cross_entropy(
                include_no_click(scores), train_targets[q]
            )
            loss += 0.1 * (
                (preference.user(users) - anchor_user[users])
                .square().sum(-1).mean()
                + (preference.item(items) - anchor_item[items])
                .square().sum(-1).mean()
            )
            loss += 0.05 * (
                model.user_bias(users).square().mean()
                + model.user_context(users).square().sum(-1).mean()
                + model.item_context(items).square().sum(-1).mean()
            )
            loss.backward()
            optimizer.step()
            loss_total += loss.detach() * len(q)

        mean_loss = float(loss_total.item() / n_events)
        if not math.isfinite(mean_loss):
            raise RuntimeError(f"Non-finite event loss at epoch {epoch}")
        record = {
            "stage": "event",
            "seed": protocol["seed"],
            "epoch": epoch,
            "training_loss": mean_loss,
            "elapsed_seconds": round(time.perf_counter() - start, 1),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        write_json(log_path, {"protocol": protocol, "history": records})

    save_checkpoint(path, model, optimizer, protocol, "event", 8)
    return {"epoch": 8, "training_loss": mean_loss, "checkpoint": str(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=[42, 43], required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume-events", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cache = root / "data/cache"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Activate the techarena environment.")
    torch.set_num_threads(4)
    positive_users, positive_items, event_data, n_users, n_items, cutoff = (
        load_training_data(cache)
    )
    protocol = {
        "seed": args.seed,
        "dimension": 8,
        "preference_epochs": 196,
        "event_epochs": 8,
        "preference_learning_rate": 0.012,
        "event_learning_rate": 0.0015,
        "preference_batch_size": 4096,
        "event_batch_size": 2048,
        "sampled_comparisons_per_click": 31,
        "preference_regularization": 0.02,
        "event_anchor_regularization": 0.1,
        "event_effect_regularization": 0.05,
        "n_users": n_users,
        "n_items": n_items,
        "cutoff": cutoff,
        "training_events": len(event_data[0]),
        "click_examples": len(positive_users),
        "evaluation_labels_used": False,
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "gpu": torch.cuda.get_device_name(0),
        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    print(json.dumps({"input_check": "passed", **protocol}, indent=2), flush=True)
    if args.check_only:
        return

    folder = root / "work" / f"seed{args.seed}"
    output = root / "outputs"
    preference_path = folder / "preference_final.pt"
    event_path = folder / "event_final.pt"
    if event_path.exists():
        raise FileExistsError(f"Completed event checkpoint already exists: {event_path}")
    if args.resume_events:
        if not preference_path.is_file():
            raise FileNotFoundError(preference_path)
    elif preference_path.exists():
        raise FileExistsError(
            f"Preference checkpoint already exists: {preference_path}. "
            "Use --resume-events to run only the event stage."
        )

    folder.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    write_json(folder / "training_protocol.json", protocol)
    start = time.perf_counter()
    if args.resume_events:
        preference_result = {"epoch": 196, "checkpoint": str(preference_path)}
    else:
        preference_result = train_preferences(
            positive_users, positive_items, protocol, preference_path,
            output / f"seed{args.seed}_preference_metrics.json",
        )

    # The preference stage has returned, releasing its model and optimizer.
    torch.cuda.empty_cache()
    event_result = train_events(
        event_data, protocol, preference_path, event_path,
        output / f"seed{args.seed}_event_metrics.json",
    )
    result = {
        "seed": args.seed,
        "training_events": protocol["training_events"],
        "click_examples": protocol["click_examples"],
        "preference": preference_result,
        "event": event_result,
        "runtime_seconds": round(time.perf_counter() - start, 1),
    }
    write_json(output / f"final_seed{args.seed}_metrics.json", result)
    print("\nFINAL TRAINING RESULTS", flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
