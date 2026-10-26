"""Fit eight adapter epochs; never update original V6 parameters or read targets."""
import argparse
import hashlib
import os
import time

import numpy as np
import torch

import residual_core as core
from residual_model import load_base, pack, representations, ResidualAdapter
from row_auc_objective import mixed_row_auc_loss


def fit(base_path, arrays, member, kind, cutoff, output, device="cuda"):
    core.require(not output.exists(), "Preserve existing adapter")
    core.require(len(arrays["u"]) and (arrays["t"] < cutoff).all(), "Adapter fitting crosses cutoff")
    y = arrays["y"]
    core.require(y.shape == (len(arrays["u"]),10) and np.isin(y,[0,1]).all() and (y.sum(1)<=1).all(), "Invalid fitting labels")
    torch.set_num_threads(4)
    base, saved = load_base(base_path,device)
    core.require(saved["dimension"] == core.MEMBERS[member]["dimension"] and saved["epoch"] == 8, "Wrong base model")
    original = {k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
    initial = core.seed(member,kind)
    torch.manual_seed(initial)
    adapter = ResidualAdapter(saved["dimension"],kind).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(),lr=.001,weight_decay=.0001)
    shuffle = torch.Generator(device=device).manual_seed(initial+1000)
    pairs = torch.Generator(device=device).manual_seed(initial+3000)
    packed = pack(arrays,arrays["u"],saved["n_users"],saved["n_items"],device)
    core.require(not bool(packed[-1].any()), "Unexpected cold fitting examples")
    targets = torch.as_tensor(np.where(y.sum(1)>0,y.argmax(1)+1,0),dtype=torch.long,device=device)
    records, start = [],time.monotonic()
    for epoch in range(1,9):
        adapter.train(); total = 0.
        order = torch.randperm(len(targets),generator=shuffle,device=device)
        for st in range(0,len(order),2048):
            idx = order[st:st+2048]
            scores,common,choice = representations(base,tuple(v[idx] for v in packed))
            corrected,delta,*_ = adapter(scores,common,choice)
            loss = mixed_row_auc_loss(corrected,targets[idx],pairs)+.1*delta.square().mean()
            core.require(bool(torch.isfinite(loss)), "Nonfinite adapter training loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(),5.,error_if_nonfinite=True)
            core.require(bool(torch.isfinite(norm)), "Nonfinite gradient")
            optimizer.step(); total += float(loss.detach())*len(idx)
        record = {"epoch":epoch,"training_loss":total/len(targets),"seconds":time.monotonic()-start}
        records.append(record)
        print({"member":member,"kind":kind,"cutoff":cutoff,**record},flush=True)
    core.require(all(torch.equal(original[k],v.detach().cpu()) for k,v in base.state_dict().items())
        and all(p.grad is None for p in base.parameters()), "Frozen V6 weights changed")
    checkpoint = {"member":member,"kind":kind,"dimension":saved["dimension"],"cutoff":cutoff,"epoch":8,
        "seed":initial,"training":core.TRAINING,"base_sha256":core.sha256(base_path),
        "protocol_sha256":core.canonical(core.PROTOCOL),"base_weights_unchanged":True,
        "maximum_fitting_timestamp":int(arrays["t"].max()),"fitting_events":len(targets),
        "fitting_indices_sha256":arrays["indices_sha256"],
        "adapter_state":{k:v.detach().cpu() for k,v in adapter.state_dict().items()}}
    core.require(all(bool(torch.isfinite(v).all()) for v in checkpoint["adapter_state"].values()), "Nonfinite adapter checkpoint")
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with temporary.open("xb") as stream:
        torch.save(checkpoint,stream)
    os.link(temporary,output); temporary.unlink()
    core.write_new(output.with_suffix(".json"),{"history":records,"model_sha256":core.sha256(output),
        "base_sha256":checkpoint["base_sha256"],"validation_labels_used":False,"base_weights_unchanged":True})
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("member",choices=list(core.MEMBERS)); parser.add_argument("kind",choices=["event","both"])
    parser.add_argument("fold",choices=list(core.FOLDS)); args = parser.parse_args()
    manifest = core.load_manifest(core.ROOT)
    budget = core.read_json(core.ROOT/core.WORK/"budget.json")
    phase = "development" if args.fold in core.ORIGINS else "screen" if args.fold in {"jan25","jan26"} else "finalize"
    core.require(budget["phases"].get(phase,{}).get("status") == "running", "Training requires active phase")
    if args.fold not in core.ORIGINS:
        chosen = core.selection(core.ROOT,manifest)
        core.require(core.COMPOSITIONS[chosen].get(args.member) == args.kind, "Unselected adapter")
    cutoff = core.FOLDS[args.fold]
    with np.load(core.ROOT/"data/cache/train_events.npz",allow_pickle=False) as d:
        indices = np.flatnonzero(d["t"] < cutoff)
        arrays = {k:d[k][indices] for k in ("u","i","t","tb","dt","y","h","ht")}
    arrays["indices_sha256"] = hashlib.sha256(indices.tobytes()).hexdigest()
    fit(core.ROOT/manifest["original_models"][args.member][args.fold],arrays,args.member,args.kind,cutoff,
        core.ROOT/core.adapter_path(args.member,args.kind,args.fold))


if __name__ == "__main__":
    main()
