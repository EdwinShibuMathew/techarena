"""Deterministic bounded linear heads, fit on matched out-of-fold predictions."""
import numpy as np
import torch
from scipy.optimize import minimize
from ml_workflow.contracts import canonical, require
from ml_workflow.storage import read, write_new
from v77_protocol import SPEC, CANDIDATES


def validate(data):
    n = len(data['indices'])
    require(n > 0 and data['event'].shape == (n, 11) and data['row'].shape == (n, 10, 13)
            and data['anchor'].shape == (n, 10) and all(np.isfinite(data[k]).all() for k in ('event', 'row', 'anchor'))
            and ((data['anchor'] >= 0) & (data['anchor'] <= 1)).all(), 'Invalid head inputs')


def normalization(data, weights):
    validate(data)
    require(weights.shape == (len(data['indices']),) and np.isfinite(weights).all() and (weights > 0).all(), 'Invalid fitting weights')
    result = {}
    for key, axes, denominator in [('event', (0,), weights.sum()), ('row', (0, 1), weights.sum() * 10)]:
        X = np.asarray(data[key], dtype=np.float64); w = weights.reshape((-1,) + (1,) * (X.ndim - 1))
        mean = (X * w).sum(axis=axes) / denominator
        scale = np.maximum(np.sqrt(((X - mean) ** 2 * w).sum(axis=axes) / denominator), 1e-6)
        result[key] = {'mean': mean.tolist(), 'scale': scale.tolist()}
    return result


def standardized(data, norm):
    result = {}
    for key, width in [('event', 11), ('row', 13)]:
        mean, scale = [np.asarray(norm[key][k], dtype=np.float64) for k in ('mean', 'scale')]
        require(mean.shape == scale.shape == (width,) and np.isfinite(mean).all() and np.isfinite(scale).all()
                and (scale >= 1e-6).all(), 'Invalid fitting-only normalization')
        result[key] = np.clip((data[key] - mean) / scale, -5, 5)
    return result


def dimension(candidate):
    require(candidate in CANDIDATES, 'Unknown candidate')
    return 1 + CANDIDATES[candidate]['event'] + (13 if CANDIDATES[candidate]['row'] else 0)


def correction(theta, event, row, candidate):
    width = CANDIDATES[candidate]['event']
    event_shift = .25 * torch.tanh(theta[0] + event[:, :width] @ theta[1:width + 1])
    result = event_shift[:, None].expand(-1, 10)
    if CANDIDATES[candidate]['row']:
        relative = .125 * torch.tanh(row @ theta[width + 1:])
        result = result + relative - relative.mean(1, keepdim=True)
    return result


def base_logits(p):
    safe = torch.where((p > 0) & (p < 1), p, torch.full_like(p, .5))
    z = torch.log(safe) - torch.log1p(-safe)
    return torch.where(p == 0, torch.full_like(p, np.log(1e-15)), torch.where(p == 1, torch.full_like(p, -np.log(1e-15)), z))


def fit(data, labels, weights, candidate, metadata, path):
    require(metadata['purpose'] in ('research', 'synthetic_validation'), 'Unknown model purpose')
    validate(data)
    require(labels.shape == data['anchor'].shape and np.isin(labels, [0, 1]).all() and (labels.sum(1) <= 1).all(), 'Invalid labels')
    norm = normalization(data, weights); X = standardized(data, norm)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    event, row, p, y = [torch.tensor(v, dtype=torch.float64) for v in (X['event'], X['row'], data['anchor'], labels)]
    w = torch.tensor(weights, dtype=torch.float64); base = base_logits(p); interior = (p > 0) & (p < 1)
    positive = np.flatnonzero(labels.ravel() == 1); negative = np.flatnonzero(labels.ravel() == 0)
    require(len(positive) and len(negative), 'Both fitting row classes required')
    row_weights = np.repeat(weights, 10)
    probabilities = row_weights[negative]; probabilities = probabilities / probabilities.sum()
    sampled = np.random.default_rng(5026).choice(negative, (len(positive), 16), replace=True, p=probabilities)
    pos, neg = torch.tensor(positive), torch.tensor(sampled)
    pw = torch.tensor(row_weights[positive], dtype=torch.float64)

    def objective(values):
        theta = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        z = base + correction(theta, event, row, candidate) * interior
        bce = (torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction='none') * w[:, None]).sum() / (10 * w.sum())
        pair = (torch.nn.functional.softplus(z.ravel()[neg] - z.ravel()[pos, None]).mean(1) * pw).sum() / pw.sum()
        loss = .5 * bce + .5 * pair + .01 * theta.square().sum()
        loss.backward()
        require(torch.isfinite(loss).item() and torch.isfinite(theta.grad).all().item(), 'Nonfinite objective/gradient')
        return float(loss.detach()), theta.grad.numpy().copy()
    initial = np.zeros(dimension(candidate)); identity_objective = objective(initial)[0]
    result = minimize(
        objective, initial, jac=True, method='L-BFGS-B', options={k: v for k, v in SPEC['optimizer'].items() if k != 'method'})
    require(result.success and np.isfinite(result.fun), 'Head optimizer did not converge: ' + str(result.message))
    _, clustered = np.unique(data['indices'], return_inverse=True)
    cluster_weights = np.bincount(clustered, weights=weights)
    state = {'schema_version': 1, 'protocol_sha256': canonical(SPEC), 'candidate': candidate,
             'coefficients': result.x.tolist(), 'normalization': norm, 'metadata': metadata,
             'converged': True, 'iterations': int(result.nit), 'objective': float(result.fun),
             'identity_objective': identity_objective, 'fit_views': len(weights),
             'unique_events': len(cluster_weights), 'effective_views': float(weights.sum() ** 2 / np.square(weights).sum()),
             'effective_events': float(cluster_weights.sum() ** 2 / np.square(cluster_weights).sum())}
    validate_state(state); write_new(path, state)
    return state


def validate_state(state, metadata=None):
    require(state['schema_version'] == 1 and state['protocol_sha256'] == canonical(SPEC)
            and state['candidate'] in CANDIDATES and state['converged'] is True
            and state['metadata']['purpose'] in ('research', 'synthetic_validation')
            and (metadata is None or state['metadata'] == metadata), 'Head provenance differs')
    theta = np.asarray(state['coefficients'], dtype=np.float64)
    require(theta.shape == (dimension(state['candidate']),) and np.isfinite(theta).all()
            and np.isfinite([state['objective'], state['identity_objective']]).all(), 'Invalid head state')
    standardized({'event': np.zeros((1, 11)), 'row': np.zeros((1, 10, 13))}, state['normalization'])
    return state


def load(path, metadata=None): return validate_state(read(path), metadata)


def predict(data, state):
    validate(data); validate_state(state); norm = standardized(data, state['normalization']); output = []
    theta = torch.tensor(state['coefficients'], dtype=torch.float64)
    torch.set_num_threads(4)
    with torch.no_grad():
        for start in range(0, len(data['indices']), 4096):
            stop = start + 4096
            p = torch.tensor(data['anchor'][start:stop], dtype=torch.float64)
            delta = correction(theta, torch.from_numpy(norm['event'][start:stop]), torch.from_numpy(norm['row'][start:stop]), state['candidate'])
            q = torch.where((p > 0) & (p < 1) & (delta != 0), torch.sigmoid(base_logits(p) + delta), p)
            output.append(q.numpy())
    result = np.concatenate(output)
    require(np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all(), 'Invalid candidate probabilities')
    return result
