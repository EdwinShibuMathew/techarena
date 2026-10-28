"""Small deterministic CPU residual with event-balanced stale-view sampling."""
from pathlib import Path
import os
import tempfile
import numpy as np
import torch
from torch import nn
from ml_workflow.contracts import canonical, require
from v76_protocol import SPEC, PENALTIES


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(13, 16, dtype=torch.float64)
        self.output = nn.Linear(16, 1, dtype=torch.float64)
        nn.init.xavier_uniform_(self.hidden.weight, gain=1.)
        nn.init.zeros_(self.hidden.bias); nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, features):
        return .25 * torch.tanh(self.output(torch.tanh(self.hidden(features))).squeeze(-1))


def validate_views(values):
    X, p, indices = values['X'], values['anchor'], values['indices']
    require(X.shape == (len(indices), 10, 13) and p.shape == (len(indices), 10)
            and values['active'].shape == (len(indices),) and values['active'].dtype == np.bool_
            and np.isfinite(X).all() and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid residual views')


def normalize_fit(values):
    validate_views(values)
    unique, inverse, counts = np.unique(values['indices'], return_inverse=True, return_counts=True)
    weights = 1. / counts[inverse]
    X = values['X'].astype(np.float64)
    mean = (X * weights[:, None, None]).sum((0, 1)) / (10 * len(unique))
    var = (((X - mean) ** 2) * weights[:, None, None]).sum((0, 1)) / (10 * len(unique))
    return {'mean': mean.tolist(), 'scale': np.maximum(np.sqrt(var), 1e-6).tolist()}


def standardized(X, normalization):
    mean, scale = [np.asarray(normalization[k], dtype=np.float64) for k in ('mean', 'scale')]
    require(mean.shape == scale.shape == (13,) and np.isfinite(mean).all() and np.isfinite(scale).all()
            and (scale >= 1e-6).all(), 'Invalid fitting-only normalization')
    return np.clip((np.asarray(X, dtype=np.float64) - mean) / scale, -5, 5)


def logits(p):
    # Endpoint stabilization is loss-only. Inference returns exact input endpoints.
    safe = torch.where((p > 0) & (p < 1), p, torch.full_like(p, .5))
    z = torch.log(safe) - torch.log1p(-safe)
    return torch.where(p == 0, torch.full_like(z, np.log(1e-15)),
                       torch.where(p == 1, torch.full_like(z, -np.log(1e-15)), z))


def objective(model, X, anchor, active, labels, penalty, generator=None, sampled=None):
    delta = model(X) * active[:, None] * ((anchor > 0) & (anchor < 1))
    base = logits(anchor); z = base + delta
    bce = nn.functional.binary_cross_entropy_with_logits(z, labels)
    positive = torch.nonzero(labels.ravel() == 1).ravel(); negative = torch.nonzero(labels.ravel() == 0).ravel()
    require(len(negative) > 0, 'No negative rows')
    if len(positive):
        if sampled is None:
            sampled = negative[torch.randint(len(negative), (len(positive), 16), generator=generator)]
        pair = nn.functional.softplus(z.ravel()[sampled] - z.ravel()[positive, None]).mean()
    else: pair = z.sum() * 0
    divergence = (nn.functional.softplus(z) - nn.functional.softplus(base) - anchor * delta).mean()
    loss = .5 * bce + .5 * pair + penalty * divergence
    require(torch.isfinite(loss).item(), 'Nonfinite residual objective')
    return loss


def fit(values, labels, penalty, metadata, path):
    require(penalty in PENALTIES and metadata['purpose'] in ('research', 'synthetic_validation'), 'Unknown fit contract')
    validate_views(values)
    unique, inverse = np.unique(values['indices'], return_inverse=True)
    require(labels.shape == (len(unique), 10) and np.isin(labels, [0, 1]).all()
            and (labels.sum(1) <= 1).all(), 'Residual fitting labels differ')
    normalization = normalize_fit(values)
    counts = np.bincount(inverse, minlength=len(unique))
    groups = np.split(np.argsort(inverse, kind='stable'), np.cumsum(counts)[:-1])
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.manual_seed(2026)
    model = Residual(); optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    X = torch.from_numpy(standardized(values['X'], normalization))
    anchor = torch.tensor(values['anchor'], dtype=torch.float64)
    active = torch.from_numpy(values['active']); y = torch.tensor(labels, dtype=torch.float64)
    order = np.random.default_rng(2026); views = np.random.default_rng(6026)
    pairs = torch.Generator().manual_seed(5026); trace = []
    for epoch in range(32):
        permuted = order.permutation(len(unique)); total = 0.
        for start in range(0, len(unique), 2048):
            selected = permuted[start:start + 2048]
            chosen = np.array([groups[e][views.integers(len(groups[e]))] for e in selected])
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, X[chosen], anchor[chosen], active[chosen], y[selected], penalty, pairs)
            loss.backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()), 'Nonfinite residual gradient')
            optimizer.step(); total += float(loss.detach()) * len(selected)
        trace.append(total / len(unique))
    require(all(torch.isfinite(p).all().item() for p in model.parameters()), 'Nonfinite residual state')
    state = {'schema_version': 1, 'spec_sha256': canonical(SPEC), 'penalty': penalty, 'metadata': metadata,
             'normalization': normalization, 'model_state': model.state_dict(), 'epochs': 32,
             'fit_events': len(unique), 'fit_views': len(inverse), 'objective_trace': trace}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix='.residual-') as tmp:
        stage = Path(tmp) / 'model.pt'; torch.save(state, stage); os.link(stage, path)
    return state


def load(path, metadata=None, penalty=None):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    require(saved['schema_version'] == 1 and saved['spec_sha256'] == canonical(SPEC)
            and saved['penalty'] in PENALTIES and saved['epochs'] == 32
            and saved['metadata']['purpose'] in ('research', 'synthetic_validation')
            and (metadata is None or saved['metadata'] == metadata)
            and (penalty is None or saved['penalty'] == penalty), 'Residual model provenance differs')
    model = Residual(); model.load_state_dict(saved['model_state']); model.eval()
    require(all(p.dtype == torch.float64 and torch.isfinite(p).all().item() for p in model.parameters()), 'Invalid residual weights')
    standardized(np.zeros((1, 10, 13)), saved['normalization'])
    return model, saved


def predict(values, model, saved):
    validate_views(values); output = []
    torch.set_num_threads(4)
    with torch.no_grad():
        for start in range(0, len(values['indices']), 2048):
            stop = start + 2048; p = torch.tensor(values['anchor'][start:stop], dtype=torch.float64)
            X = torch.from_numpy(standardized(values['X'][start:stop], saved['normalization']))
            delta = model(X) * torch.from_numpy(values['active'][start:stop])[:, None]
            interior = (p > 0) & (p < 1)
            q = torch.where(interior & (delta != 0), torch.sigmoid(logits(p) + delta), p)
            output.append(q.numpy())
    result = np.concatenate(output)
    require(np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all(), 'Invalid residual predictions; clipping prohibited')
    return result
