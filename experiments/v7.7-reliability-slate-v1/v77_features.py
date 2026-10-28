"""Label-safe first-return populations and permutation-equivariant slate features."""
import numpy as np
from ml_workflow.contracts import first_indices, require, DAY
from v77_protocol import WEIGHTS


def cohort(raw, prefix, meta):
    require(prefix <= meta, 'Future training prefix')
    ix = first_indices(raw['u'], raw['t'], prefix, meta + DAY)
    return ix[raw['t'][ix] >= meta]


def balanced_weights(indices, strata):
    require(len(indices) and strata.shape == (len(indices), 2), 'Invalid fitting strata')
    _, code, counts = np.unique(strata, axis=0, return_inverse=True, return_counts=True)
    w = 1. / counts[code]
    _, events = np.unique(indices, return_inverse=True)
    w /= np.bincount(events, weights=w).max()
    require(np.isfinite(w).all() and (w > 0).all() and np.bincount(events, weights=w).max() <= 1 + 1e-12, 'Duplicate event weight inflated')
    return w


def geometry(vectors):
    require(len(vectors) == 5 and all(v.ndim == 3 and v.shape[:2] == vectors[0].shape[:2] and v.shape[1] == 10
                                   and np.isfinite(v).all() for v in vectors), 'Invalid candidate vectors')
    result = np.zeros((len(vectors[0]), 10, 10), dtype=np.float64)
    for weight, values in zip(WEIGHTS, vectors):
        values = np.asarray(values, dtype=np.float64)
        normalized = values / np.maximum(np.linalg.norm(values, axis=2, keepdims=True), 1e-8)
        result += weight * np.einsum('bjd,bkd->bjk', normalized, normalized, optimize=True)
    return result


def rank_credit(p):
    return (((p[:, :, None] > p[:, None, :]) + .5 * (p[:, :, None] == p[:, None, :])).sum(2) - .5) / 9


def user_context(fitting, target, ix, prefix):
    n_users = int(max(fitting['u'].max(), target['u'][ix].max())) + 1
    mask = fitting['t'] < prefix; users = fitting['u'][mask]
    count = np.bincount(users, minlength=n_users)
    clicks = np.bincount(users, weights=fitting['y'][mask].sum(1), minlength=n_users)
    latest = np.full(n_users, -1, dtype=np.int64); np.maximum.at(latest, users, fitting['t'][mask])
    u = target['u'][ix]; present = target['h'][ix] >= 0; length = present.sum(1)
    age = np.where(present, (target['t'][ix, None] - target['ht'][ix]) / 3600, 720.)
    gap = np.where(count[u] > 0, (target['t'][ix] - latest[u]) / 3600, 720.)
    require(mask.any(), 'User context requires a nonempty mature fitting prefix')
    global_ctr = float(fitting['y'][mask].mean())
    result = np.column_stack([np.log1p(count[u]), (clicks[u] + 50 * global_ctr) / (10 * count[u] + 50),
        np.log1p(clicks[u]), np.log1p(np.minimum(gap, 720)), length / 20,
        np.log1p(np.minimum(age.min(1), 720)), count[u] == 0]).astype(np.float64)
    require(np.isfinite(result).all() and (gap >= 0).all(), 'Invalid causal user context')
    return result


def master(members, anchor, context, cosine):
    n = len(anchor)
    require(members.shape == (5, n, 10) and anchor.shape == (n, 10) and context.shape == (n, 7)
            and cosine.shape == (n, 10, 10) and np.isfinite(members).all() and np.isfinite(anchor).all()
            and ((members >= 0) & (members <= 1)).all() and ((anchor >= 0) & (anchor <= 1)).all(), 'Invalid feature schema')
    mass = anchor.sum(1, keepdims=True); q = np.divide(anchor, mass, out=np.full_like(anchor, .1), where=mass > 0)
    entropy = -(q * np.log(np.maximum(q, 1e-300))).sum(1) / np.log(10)
    p = np.clip(members, 1e-6, 1 - 1e-6); logits = np.log(p) - np.log1p(-p)
    mean_logits = np.einsum('m,mnk->nk', WEIGHTS, logits)
    rms = np.sqrt(np.einsum('m,mnk->nk', WEIGHTS, (logits - mean_logits) ** 2).mean(1))
    event = np.column_stack([np.log1p(mass[:, 0]), entropy, q.max(1), rms, context])
    safe = np.clip(anchor, 1e-6, 1 - 1e-6); z0 = np.log(safe) - np.log1p(-safe)
    rank0 = rank_credit(anchor)
    diagonal = np.eye(10, dtype=bool)[None]
    off = np.where(diagonal, 0., cosine)
    weighted = (off * q[:, None, :]).sum(2)
    weighted = np.divide(weighted, 1 - q, out=np.zeros_like(weighted), where=(1 - q) > 0)
    row = np.stack([*[logits[m] - z0 for m in range(5)],
                    *[rank_credit(members[m]) - rank0 for m in range(5)], off.sum(2) / 9,
                    np.where(diagonal, -np.inf, cosine).max(2), weighted], axis=2)
    row -= row.mean(1, keepdims=True)
    require(event.shape == (n, 11) and row.shape == (n, 10, 13) and np.isfinite(event).all() and np.isfinite(row).all(), 'Nonfinite master features')
    return event, row
