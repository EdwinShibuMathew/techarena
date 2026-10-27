"""Frozen-V7, mass-preserving correction from causal sparse item neighbors."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

PURPOSES = {'research', 'synthetic_validation'}
CONFIG = {
    'neighbors': 64, 'minimum_shared_users': 3, 'shrinkage': 20.,
    'block_items': 256, 'batch_events': 2048, 'item_limit': 50000,
    'scale_floor': 1e-6, 'feature_bound': 5., 'alpha_bounds': [0., .25],
    'root_steps': 64, 'mass_tolerance': 1e-14, 'bce_weight': .5,
    'pair_weight': .5, 'l2': .01, 'negative_rows_per_positive': 16,
    'seed': 2026, 'folds': 5, 'maxiter': 200, 'ftol': 1e-12,
    'gtol': 1e-8, 'maxls': 50, 'loss_epsilon': 1e-15,
}
CANDIDATES = {
    'history_' + str(hours) + 'h': {
        'kind': 'history_postprocessor', 'half_life_hours': hours, 'config': deepcopy(CONFIG),
    } for hours in (24, 72)
}


def user_folds(users):
    unique, inverse = np.unique(users, return_inverse=True)
    order = np.random.default_rng(CONFIG['seed']).permutation(len(unique))
    assigned = np.empty(len(unique), dtype=np.int8)
    assigned[order] = np.arange(len(unique)) % CONFIG['folds']
    return assigned[inverse]


def build_graph(prefix, cutoff):
    """Each user contributes one binary interaction per distinct clicked item.

    Encoded IDs are opaque keys: their magnitude is never a model feature.
    Sparse multiplication is blocked; no dense item-by-item matrix is allocated.
    """
    from scipy.sparse import csr_matrix
    from ml_workflow.contracts import require, validate_events

    validate_events(prefix, True)
    require(type(cutoff) is int and (prefix['t'] < cutoff).all(), 'Graph prefix crosses cutoff')
    history = prefix['h'] >= 0
    clicked = prefix['y'] == 1
    users = np.concatenate([np.broadcast_to(prefix['u'][:, None], history.shape)[history],
                            np.broadcast_to(prefix['u'][:, None], clicked.shape)[clicked]])
    items = np.concatenate([prefix['h'][history], prefix['i'][clicked]]).astype(np.int64)
    unique_items, columns = np.unique(items, return_inverse=True)
    unique_users, rows = np.unique(users, return_inverse=True)
    binary = csr_matrix((np.ones(len(items), dtype=np.int32), (rows, columns)),
                        shape=(len(unique_users), len(unique_items)))
    binary.sum_duplicates()
    binary.data[:] = 1
    counts = np.asarray(binary.sum(axis=0)).ravel().astype(np.int64)
    keys, weights = [], []
    for start in range(0, len(unique_items), CONFIG['block_items']):
        shared = (binary[:, start:start + CONFIG['block_items']].T @ binary).tocsr()
        for offset in range(shared.shape[0]):
            row = start + offset
            indices = shared.indices[shared.indptr[offset]:shared.indptr[offset + 1]]
            c = shared.data[shared.indptr[offset]:shared.indptr[offset + 1]].astype(np.float64)
            keep = (indices != row) & (c >= CONFIG['minimum_shared_users'])
            indices, c = indices[keep], c[keep]
            similarity = c / np.sqrt(counts[row] * counts[indices]) * c / (c + CONFIG['shrinkage'])
            # Deterministic ties use the serialized opaque ID encoding.
            chosen = np.lexsort((unique_items[indices], -similarity))[:CONFIG['neighbors']]
            keys.extend((unique_items[row] * CONFIG['item_limit'] + unique_items[indices[chosen]]).tolist())
            weights.extend(similarity[chosen].tolist())
    order = np.argsort(keys, kind='stable')
    graph = {'keys': np.asarray(keys, dtype=np.int64)[order],
             'weights': np.asarray(weights, dtype=np.float64)[order],
             'items': unique_items.astype(np.int64), 'counts': counts,
             'cutoff': np.asarray(cutoff, dtype=np.int64)}
    validate_graph(graph)
    return graph


def validate_graph(graph):
    from ml_workflow.contracts import require
    require(set(graph) == {'keys', 'weights', 'items', 'counts', 'cutoff'}, 'Invalid graph fields')
    k, w, items, counts = [graph[name] for name in ('keys', 'weights', 'items', 'counts')]
    require(k.dtype == np.int64 and w.dtype == np.float64 and k.ndim == w.ndim == 1
            and len(k) == len(w) and (np.diff(k) > 0).all(), 'Invalid graph edges')
    require(items.dtype == counts.dtype == np.int64 and items.ndim == counts.ndim == 1
            and len(items) == len(counts) and (np.diff(items) > 0).all()
            and ((items >= 0) & (items < CONFIG['item_limit'])).all() and (counts > 0).all(), 'Invalid graph inventory')
    require(graph['cutoff'].dtype == np.int64 and graph['cutoff'].shape == (), 'Invalid graph cutoff')
    left, right = k // CONFIG['item_limit'], k % CONFIG['item_limit']
    require(np.isfinite(w).all() and ((w > 0) & (w <= 1)).all()
            and np.isin(left, items).all() and np.isin(right, items).all()
            and (left != right).all(), 'Invalid graph similarities')
    require(not len(left) or np.unique(left, return_counts=True)[1].max() <= CONFIG['neighbors'], 'Too many graph neighbors')
    return graph


def affinities(graph, target, half_life_hours):
    from ml_workflow.contracts import require, validate_events
    validate_graph(graph)
    validate_events(target, False)
    require(half_life_hours in (24, 72), 'Unknown history half-life')
    require((target['t'] >= int(graph['cutoff'])).all(), 'Graph target precedes cutoff')
    result = np.zeros(target['i'].shape, dtype=np.float64)
    if not len(graph['keys']):
        return result
    for start in range(0, len(result), CONFIG['batch_events']):
        end = start + CONFIG['batch_events']
        history = target['h'][start:end]
        present = history >= 0
        age = (target['t'][start:end, None] - target['ht'][start:end]) / 3600
        recency = np.exp2(-np.where(present, age, 0) / half_life_hours) * present
        keys = target['i'][start:end, :, None].astype(np.int64) * CONFIG['item_limit'] + history[:, None, :].clip(min=0)
        positions = np.searchsorted(graph['keys'], keys)
        bounded = positions.clip(max=len(graph['keys']) - 1)
        found = (positions < len(graph['keys'])) & (graph['keys'][bounded] == keys) & present[:, None, :]
        values = np.where(found, graph['weights'][bounded], 0.)
        result[start:end] = (values * recency[:, None, :]).sum(2) / np.maximum(recency.sum(1)[:, None], 1e-300)
    return result


def normalization(features):
    centered = np.asarray(features, dtype=np.float64) - np.asarray(features, dtype=np.float64).mean(1, keepdims=True)
    return {'rms': max(float(np.sqrt(np.mean(centered ** 2))), CONFIG['scale_floor'])}


def standardized(features, norm):
    from ml_workflow.contracts import require
    features = np.asarray(features, dtype=np.float64)
    require(features.ndim == 2 and features.shape[1] == 10 and len(features)
            and np.isfinite(features).all(), 'Invalid graph features')
    require(set(norm) == {'rms'} and np.isfinite(norm['rms']) and norm['rms'] >= CONFIG['scale_floor'], 'Invalid graph normalization')
    return np.clip((features - features.mean(1, keepdims=True)) / norm['rms'],
                   -CONFIG['feature_bound'], CONFIG['feature_bound'])


def tilt(anchor, features, alpha, derivative=False):
    """Solve the common logit offset, with an implicit derivative for fitting."""
    from scipy.special import expit
    from ml_workflow.contracts import require
    p = np.asarray(anchor, dtype=np.float64)
    z = np.asarray(features, dtype=np.float64)
    require(p.ndim == 2 and p.shape[1] == 10 and len(p) and p.shape == z.shape
            and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            and np.isfinite(z).all() and np.isfinite(alpha) and 0 <= alpha <= .25, 'Invalid residual inputs')
    interior = (p > 0) & (p < 1)
    unchanged = (z == z[:, :1]).all(1)
    if alpha == 0 or not np.any(z):
        q = p.copy()
    else:
        safe = np.where(interior, p, .5)
        base = np.log(safe) - np.log1p(-safe)
        delta = alpha * z
        low = -delta.max(1, keepdims=True)
        high = -delta.min(1, keepdims=True)
        mass = p.sum(1, keepdims=True)
        for _ in range(CONFIG['root_steps']):
            middle = (low + high) / 2
            q = np.where(interior, expit(base + delta + middle), p)
            over = q.sum(1, keepdims=True) > mass
            high = np.where(over, middle, high)
            low = np.where(over, low, middle)
        q = np.where(interior, expit(base + delta + (low + high) / 2), p)
    # A common feature offset carries no relative information. Preserve exact
    # fallback scores even when other events in this batch have graph evidence.
    q[unchanged] = p[unchanged]
    require(np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all()
            and np.allclose(q.sum(1), p.sum(1), rtol=0, atol=CONFIG['mass_tolerance']), 'Residual changed event mass or bounds')
    if not derivative:
        return q
    variance = q * (1 - q) * interior
    weighted = (variance * z).sum(1, keepdims=True) / np.maximum(variance.sum(1, keepdims=True), 1e-300)
    jacobian = variance * (z - weighted)
    jacobian[unchanged] = 0
    return q, jacobian


def objective(alpha, anchor, features, labels, positive, sampled):
    from scipy.special import expit
    q, jac = tilt(anchor, features, alpha, derivative=True)
    eps = CONFIG['loss_epsilon']
    safe = np.clip(q, eps, 1 - eps)  # Loss stabilization only; never export this.
    varying = (q > eps) & (q < 1 - eps)
    bce = -(labels * np.log(safe) + (1 - labels) * np.log1p(-safe)).mean()
    grad_bce = (jac * varying * (safe - labels) / (safe * (1 - safe))).mean()
    odds = (np.log(safe) - np.log1p(-safe)).ravel()
    grad_odds = (jac * varying / (safe * (1 - safe))).ravel()
    diff = odds[sampled] - odds[positive, None]
    pair = np.logaddexp(0, diff).mean()
    grad_pair = (expit(diff) * (grad_odds[sampled] - grad_odds[positive, None])).mean()
    return (.5 * bce + .5 * pair + CONFIG['l2'] * alpha ** 2,
            .5 * grad_bce + .5 * grad_pair + 2 * CONFIG['l2'] * alpha)


def validate_state(saved, metadata=None, parameters=None):
    from ml_workflow.contracts import require
    require(saved.get('schema_version') == 3 and saved.get('converged') is True, 'Invalid history calibrator')
    require(saved.get('parameters') in CANDIDATES.values()
            and (parameters is None or saved['parameters'] == parameters), 'Candidate parameters changed')
    require(metadata is None or saved.get('metadata') == metadata, 'Calibrator lineage changed')
    require(type(saved.get('alpha')) is float and np.isfinite(saved['alpha'])
            and 0 <= saved['alpha'] <= .25, 'Invalid residual coefficient')
    standardized(np.zeros((1, 10)), saved['normalization'])
    require(np.isfinite([saved['objective'], saved['identity_objective']]).all()
            and saved['objective'] <= saved['identity_objective'] + 1e-12, 'Invalid fitted objective')
    return saved


def fit_calibrator(anchor, features, labels, metadata, path, parameters):
    from scipy.optimize import minimize
    from ml_workflow.contracts import require
    from ml_workflow.storage import write_new
    require(parameters in CANDIDATES.values(), 'Unknown frozen candidate')
    labels = np.asarray(labels)
    require(labels.shape == np.asarray(anchor).shape == np.asarray(features).shape
            and np.isin(labels, [0, 1]).all() and (labels.sum(1) <= 1).all(), 'Invalid fitting labels')
    norm = normalization(features)
    features = standardized(features, norm)
    positive, negative = [np.flatnonzero(labels.ravel() == value) for value in (1, 0)]
    require(len(positive) and len(negative), 'Both row classes required')
    sampled = negative[np.random.default_rng(CONFIG['seed']).integers(len(negative), size=(len(positive), 16))]
    def evaluate(x):
        value, gradient = objective(float(x[0]), anchor, features, labels, positive, sampled)
        return float(value), np.asarray([gradient])
    identity = evaluate([0.])[0]
    fitted = minimize(evaluate, [0.], jac=True, method='L-BFGS-B', bounds=[CONFIG['alpha_bounds']],
                      options={k: CONFIG[k] for k in ('maxiter', 'ftol', 'gtol', 'maxls')})
    require(fitted.success and np.isfinite(fitted.fun), 'History calibrator did not converge: ' + str(fitted.message))
    saved = {'schema_version': 3, 'metadata': deepcopy(metadata), 'parameters': deepcopy(parameters),
             'alpha': float(fitted.x[0]), 'normalization': norm, 'objective': float(fitted.fun),
             'identity_objective': identity, 'iterations': int(fitted.nit), 'converged': True}
    validate_state(saved, metadata, parameters)
    write_new(Path(path), saved)
    return saved


def load_calibrator(path, metadata=None, parameters=None):
    return validate_state(json.loads(Path(path).read_text()), metadata, parameters)


def predict_calibrator(anchor, features, saved):
    validate_state(saved)
    return tilt(anchor, standardized(features, saved['normalization']), saved['alpha'])
