"""Member-checked mixed event training from verified original preference checkpoints."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np
import robust_core as core


def train(preference_path, arrays, cutoff, output, member, device="cuda"):
    import torch
    from train_final_models import PreferenceModel, EventModel
    from row_auc_objective import mixed_row_auc_loss
    core.require(not output.exists(), "Preserve existing trained checkpoint")
    checkpoint = torch.load(preference_path, map_location="cpu", weights_only=True)
    expected = core.MEMBERS[member]
    seed, dimension = expected["seed"], expected["dimension"]
    core.require(checkpoint["dimension"] == dimension and checkpoint["epoch"] == expected["preference_epoch"], "Wrong preference epoch/dimension")
    if dimension == 8:
        core.require(checkpoint["fine_tuning_sampling_seed"] == 1044, "Wrong seed44 preference")
    for key, value in (("seed", seed), ("cutoff", cutoff)):
        core.require(key not in checkpoint or checkpoint[key] == value, "Conflicting preference metadata")
    n_users, n_items = checkpoint["n_users"], checkpoint["n_items"]
    users, items, times, bucket, day, labels = (np.asarray(arrays[k]) for k in ("u", "i", "t", "tb", "dt", "y"))
    core.require(len(users) > 0 and users.shape == times.shape == bucket.shape == day.shape
                 and items.shape == labels.shape == (len(users), 10), "Invalid fitting shapes")
    core.require((times < cutoff).all() and np.isin(labels, [0, 1]).all() and (labels.sum(1) <= 1).all()
                 and ((users >= 0) & (users < n_users)).all() and ((items >= 0) & (items < n_items)).all()
                 and np.isin(bucket, [0, 1, 2, 3]).all() and np.isin(day, [0, 1]).all(), "Invalid fitting timeline/data")
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    preference = PreferenceModel(n_users, n_items, dimension, torch.ones(n_users, dtype=torch.bool)).to(device)
    preference.load_state_dict(checkpoint["model_state"])
    model = EventModel(preference, n_users, n_items).to(device)
    del checkpoint
    targets = np.where(labels.sum(1) > 0, labels.argmax(1) + 1, 0)
    inputs = [torch.tensor(v, dtype=torch.long, device=device) for v in (users, items, bucket + 4 * day)]
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    anchor_user, anchor_item = preference.user.weight.detach().clone(), preference.item.weight.detach().clone()
    with torch.inference_mode():
        totals = torch.cat([model(*(v[st:st + 4096] for v in inputs)).logsumexp(1)
                            for st in range(0, len(users), 4096)])
        click_rate = (targets > 0).float().mean()
        low, high = -40., 40.
        for _ in range(30):
            middle = (low + high) / 2
            if torch.sigmoid(totals + middle).mean() > click_rate:
                high = middle
            else:
                low = middle
        model.global_bias.fill_((low + high) / 2)
    del totals
    optimizer = torch.optim.Adam(model.parameters(), lr=.0015)
    pair_generator = torch.Generator(device=device).manual_seed(seed + 3000)
    history, start = [], time.monotonic()
    for epoch in range(1, 9):
        model.train()
        order = torch.randperm(len(users), device=device)
        loss_sum = torch.zeros((), device=device)
        for st in range(0, len(order), 2048):
            q = order[st:st + 2048]
            batch_users, batch_items, context = (v[q] for v in inputs)
            optimizer.zero_grad()
            scores = model(batch_users, batch_items, context)
            loss = mixed_row_auc_loss(scores, targets[q], pair_generator)
            loss += .1 * ((preference.user(batch_users) - anchor_user[batch_users]).square().sum(-1).mean()
                          + (preference.item(batch_items) - anchor_item[batch_items]).square().sum(-1).mean())
            loss += .05 * (model.user_bias(batch_users).square().mean()
                           + model.user_context(batch_users).square().sum(-1).mean()
                           + model.item_context(batch_items).square().sum(-1).mean())
            core.require(bool(torch.isfinite(loss)), "Nonfinite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(q)
        record = {"epoch": epoch, "training_loss": float(loss_sum / len(users)), "seconds": time.monotonic() - start}
        history.append(record)
        print({"fold_cutoff": cutoff, **record}, flush=True)
    state = {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "n_users": n_users, "n_items": n_items, "dimension": dimension, "seed": seed, "member": member, "stage": "event", "epoch": 8,
        "cutoff": cutoff, "event_objective": core.CONFIG["objective"], "pair_sampling_seed": seed + 3000,
        "protocol_sha256": core.canonical(core.PROTOCOL), "preference_sha256": core.sha256(preference_path),
        "fitting_events": len(users), "fitting_indices_sha256": arrays["indices_sha256"],
        "maximum_fitting_timestamp": int(times.max())}
    core.require(all(bool(torch.isfinite(v).all()) for v in state["model_state"].values() if v.is_floating_point()), "Nonfinite checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with temporary.open("xb") as stream:
        torch.save(state, stream)
    os.link(temporary, output)
    temporary.unlink()
    core.write_new(output.with_suffix(".json"), {"history": history, "epoch": 8, "seed": seed,
        "cutoff": cutoff, "model_sha256": core.sha256(output), "preference_sha256": core.sha256(preference_path),
        "validation_labels_used": False})
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("member", choices=list(core.COMPOSITIONS["B"]))
    parser.add_argument("fold", choices=list(core.FOLDS))
    args = parser.parse_args()
    root = core.ROOT
    manifest = core.load_manifest(root)
    cutoff = core.FOLDS[args.fold]
    budget = core.read_json(root/core.WORK/"budget.json")
    phase = "development" if args.fold == "jan21" else "screen" if args.fold in {"jan24","jan25","jan26"} else "stress" if args.fold == "jan22" else "finalize"
    core.require(budget["phases"].get(phase,{}).get("status") == "running", "Training requires active staged budget")
    core.require(args.member in core.PAIR or (args.member == "dim8_seed44" and args.fold == "jan27" and core.selection(root,manifest) == "B"), "Seed44 prefix models must be reused")
    with np.load(root / "data/cache/train_events.npz", allow_pickle=False) as data:
        indices = np.flatnonzero(data["t"] < cutoff)
        arrays = {k: data[k][indices] for k in ("u", "i", "t", "tb", "dt", "y")}
    arrays["indices_sha256"] = hashlib.sha256(indices.tobytes()).hexdigest()
    train(root / manifest["preferences"][args.member][args.fold], arrays, cutoff, root / core.neural_path(args.member,args.fold), args.member)


if __name__ == "__main__":
    main()
