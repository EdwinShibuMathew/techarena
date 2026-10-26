"""Small event/choice heads over a frozen original V6 event scorer."""
import math

import numpy as np
import torch
from torch import nn

import residual_core as core
from train_final_models import PreferenceModel, EventModel, include_no_click


def load_base(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = EventModel(PreferenceModel(saved["n_users"], saved["n_items"], saved["dimension"],
                       torch.ones(saved["n_users"], dtype=torch.bool)), saved["n_users"], saved["n_items"])
    model.load_state_dict(saved["model_state"])
    model.requires_grad_(False)
    return model.to(device).eval(), saved


def pack(data, training_users, n_users, n_items, device):
    u, i, h, ht, t, tb, dt = (np.asarray(data[k]) for k in ("u", "i", "h", "ht", "t", "tb", "dt"))
    core.require(len(u) and i.shape == (len(u), 10) and h.shape == ht.shape == (len(u), 20)
        and u.shape == t.shape == tb.shape == dt.shape, "Invalid residual input shapes")
    core.require(((u >= 0) & (u < n_users)).all() and ((i >= 0) & (i < n_items)).all()
        and ((h == -1) | ((h >= 0) & (h < n_items))).all()
        and np.isin(tb, [0,1,2,3]).all() and np.isin(dt, [0,1]).all(), "Invalid residual IDs/context")
    present = h >= 0
    core.require((ht[present] < np.broadcast_to(t[:,None], h.shape)[present]).all(), "History crosses event timestamp")
    age = np.where(present, (t[:,None]-ht)/3600, 0).clip(0, 720).astype(np.float32)
    cold = np.bincount(training_users, minlength=n_users)[u] == 0
    values = (u, i, tb + 4*dt, h, age, tb, cold)
    types = (torch.long, torch.long, torch.long, torch.long, torch.float32, torch.long, torch.bool)
    return tuple(torch.as_tensor(v, dtype=d, device=device) for v, d in zip(values, types))


@torch.no_grad()
def representations(base, batch, override=None):
    users, items, context, history, age, bucket, cold = batch
    pref = base.preference
    q = pref.item(items)
    uv = pref.user(users) * pref.active_users[users,None]
    if override is None:
        scores = base(users, items, context)
    else:
        # Same arithmetic/order as PreferenceModel + EventModel; replace vectors
        # before multiplying, rather than adding a numerically different delta.
        mask = cold & (history >= 0).any(1)
        uv = torch.where(mask[:,None], override, uv)
        scores = (uv[:,None,:]*q).sum(-1) + pref.item_bias(items).squeeze(-1)
        item_effect = base.item_context(items).gather(2, context[:,None,None].expand(-1,10,1)).squeeze(-1)
        user_effect = base.user_context(users).gather(1, context[:,None])
        scores = scores + base.user_bias(users) + base.global_bias + item_effect + user_effect
    present = history >= 0
    hv = pref.item(history.clamp(min=0)) * present[:,:,None]
    length = present.sum(1)
    mean_h = hv.sum(1)/length.clamp(min=1)[:,None]
    logits = torch.einsum("bkd,bhd->bkh", q, hv)/math.sqrt(q.shape[-1]) - torch.log1p(age[:,None,:]/24)
    weights = logits.masked_fill(~present[:,None,:], -1e4).softmax(-1) * present[:,None,:]
    weights = weights/weights.sum(-1,keepdim=True).clamp(min=1e-8)
    attended = torch.einsum("bkh,bhd->bkd", weights, hv)
    last_age = torch.where(present, age, 720.).min(1).values
    common = torch.cat([uv, q.mean(1), mean_h,
        torch.stack([scores.mean(1), scores.max(1).values, scores.logsumexp(1)],1)/10,
        length[:,None]/20, torch.log1p(last_age[:,None])/math.log(721),
        torch.stack([(present & (age < h)).sum(1) for h in (6,24,72)],1)/20,
        nn.functional.one_hot(bucket,4).float(), cold[:,None].float()],1)
    repeats = ((items[:,:,None] == history[:,None,:]) & present[:,None,:]).sum(-1)
    choice = torch.cat([q, attended, q*attended,
        (torch.log1p(repeats.float())/math.log(21))[:,:,None],
        ((scores-scores.mean(1,keepdim=True))/10)[:,:,None]],2)
    return scores, common, choice


class ResidualAdapter(nn.Module):
    def __init__(self, dimension, kind):
        super().__init__()
        core.require(dimension in (8,16) and kind in {"event","both"}, "Unknown residual architecture")
        self.kind, self.dimension = kind, dimension
        self.event = nn.Sequential(nn.Linear(3*dimension+13,32), nn.SiLU(), nn.Linear(32,1))
        self.choice = nn.Sequential(nn.Linear(3*dimension+2,32), nn.SiLU(), nn.Linear(32,1)) if kind == "both" else None
        mask = torch.ones(3*dimension+13)
        mask[-1] = 0  # No cold fitting examples: do not extrapolate this feature.
        self.register_buffer("event_input_mask", mask)
        for head in (self.event, self.choice):
            if head is not None:
                nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)

    def forward(self, scores, common, choice):
        event = .5*torch.tanh(self.event(common*self.event_input_mask))
        relative = torch.zeros_like(scores)
        if self.choice is not None:
            relative = .5*torch.tanh(self.choice(choice).squeeze(-1))
            relative = relative-relative.mean(1,keepdim=True)
        correction = event+relative
        return scores+correction, correction, event, relative


def load_adapter(root, manifest, member, kind, fold, device):
    saved = core.validate_adapter(root, manifest, member, kind, fold)
    adapter = ResidualAdapter(saved["dimension"],kind).to(device)
    adapter.load_state_dict(saved["adapter_state"])
    core.require(adapter.event_input_mask[-1] == 0 and bool((adapter.event_input_mask[:-1] == 1).all()), "Invalid support mask")
    return adapter.eval()


@torch.inference_mode()
def corrected_probabilities(base, adapter, packed, estimated=None):
    values = []
    for start in range(0, len(packed[0]), 4096):
        batch = tuple(v[start:start+4096] for v in packed)
        override = None if estimated is None else estimated[start:start+4096]
        scores, common, choice = representations(base,batch,override)
        corrected, *_ = adapter(scores,common,choice)
        values.append(include_no_click(corrected).softmax(1)[:,1:].cpu().numpy().astype(np.float64))
    return np.concatenate(values)


def pipeline(base_path, adapter, training, target, positive, device):
    """Retain original feature formulas and raw-before-fallback convention."""
    from predict_v6 import portable_pipeline
    from build_submission import neural_pipeline, adjust_exposure
    x, names, old_raw, old_p, encoder, report = portable_pipeline(neural_pipeline)(base_path,training,target,positive,device)
    core.require(tuple(names) == core.BASE_COLUMNS, "Original schema differs")
    base, saved = load_base(base_path,device)
    packed = pack(target,training["u"],saved["n_users"],saved["n_items"],device)
    raw = corrected_probabilities(base,adapter,packed)
    exposure = np.bincount(training["i"].ravel(),minlength=saved["n_items"])
    probabilities = adjust_exposure(raw,target["i"],exposure)
    cold = packed[-1].cpu().numpy() & (target["h"] >= 0).any(1)
    if cold.any():
        core.require(len(np.unique(target["u"][cold])) == int(cold.sum()), "Cold target users repeat")
        h = target["h"][cold]; present = h >= 0
        vectors = saved["model_state"]["preference.item.weight"].numpy()
        profiles = (vectors[h.clip(min=0)]*present[:,:,None]).sum(1)/present.sum(1,keepdims=True)
        estimated = torch.zeros((len(h),saved["dimension"]),device=device)
        estimated.copy_(torch.as_tensor(encoder.predict(profiles),dtype=torch.float32,device=device))
        subset = tuple(v[torch.as_tensor(cold,device=device)] for v in packed)
        corrected_cold = corrected_probabilities(base,adapter,subset,estimated)
        probabilities[cold] = adjust_exposure(corrected_cold,target["i"][cold],exposure)
    updated = {"pipeline_probability":probabilities, "raw_event_probability":raw,
        "log_probability":np.log(np.maximum(probabilities,1e-8)),
        "event_click_probability":np.broadcast_to(probabilities.sum(1)[:,None],probabilities.shape),
        "event_max_probability":np.broadcast_to(probabilities.max(1)[:,None],probabilities.shape),
        "relative_probability":probabilities-probabilities.mean(1,keepdims=True)}
    for name, values in updated.items():
        x[:,names.index(name)] = values.ravel()
    core.require(np.isfinite(x).all(), "Nonfinite residual features")
    del base, saved
    torch.cuda.empty_cache()
    return x,names,raw,probabilities,old_raw,old_p,report
