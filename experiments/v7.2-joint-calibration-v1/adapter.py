"""Bounded joint calibration of event-wide confidence and candidate scores."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

PURPOSES = {"research", "synthetic_validation"}
CONFIG = {"formula": "logit(p)+(a+b*(H-.5))*centered_log_q+c*(H-.5)+d*standardized_log_mass",
          "bounds": [-.25, .25], "bce_weight": .5, "pair_weight": .5, "l2": .01,
          "negative_rows_per_positive": 16, "seed": 2026, "folds": 5,
          "maxiter": 200, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50,
          "log_floor": 1e-12, "scale_floor": 1e-6, "probability_clipping": False,
          "routing_features": ["entropy", "log_mass", "conditional_scores"]}
CANDIDATES = {
    name: {"kind": "ensemble_postprocessor", "config": deepcopy(CONFIG), "active": active}
    for name, active in (("event", [2, 3]), ("joint", [0, 2, 3]), ("entropy_joint", [0, 1, 2, 3]))
}


def user_folds(users):
    users = np.asarray(users)
    unique = np.unique(users)
    order = np.random.default_rng(CONFIG["seed"]).permutation(len(unique))
    assigned = np.empty(len(unique), dtype=np.int8)
    assigned[order] = np.arange(len(unique)) % CONFIG["folds"]
    return assigned[np.searchsorted(unique, users)]


def components(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 10 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected finite ten-candidate probabilities in [0,1]")
    mass = p.sum(1, keepdims=True)
    q = np.divide(p, mass, out=np.full_like(p, .1), where=mass > 0)
    logq = np.log(np.maximum(q, CONFIG["log_floor"]))
    r = logq - logq.mean(1, keepdims=True)
    h = -(q * logq).sum(1) / np.log(10)
    logmass = np.log(np.maximum(mass[:, 0], CONFIG["log_floor"]))
    interior = (p > 0) & (p < 1)
    safe = np.where(interior, p, .5)
    odds = np.log(safe) - np.log1p(-safe)
    return p, odds, r, h, logmass, interior


def validate_state(saved, metadata=None, parameters=None):
    if saved.get("schema_version") != 2 or saved.get("converged") is not True:
        raise ValueError("Invalid calibration state version/convergence")
    params = saved.get("parameters")
    if params not in CANDIDATES.values() or (parameters is not None and params != parameters):
        raise ValueError("Candidate parameters changed")
    if metadata is not None and saved.get("metadata") != metadata:
        raise ValueError("Calibrator lineage changed")
    theta = np.asarray(saved["coefficients"], dtype=np.float64)
    if theta.shape != (4,) or not np.isfinite(theta).all() or (np.abs(theta) > .25).any():
        raise ValueError("Invalid calibrator coefficients")
    if any(theta[i] != 0 for i in set(range(4)) - set(params["active"])):
        raise ValueError("Inactive coefficient changed")
    normalization = saved["normalization"]
    if (set(normalization) != {"log_mass_mean", "log_mass_scale"}
            or not np.isfinite(list(normalization.values())).all()
            or normalization["log_mass_scale"] < CONFIG["scale_floor"]):
        raise ValueError("Invalid saved normalization")
    if not np.isfinite([saved["objective"], saved["identity_objective"]]).all():
        raise ValueError("Invalid fitting objectives")
    return saved


def predict_calibrator(probabilities, saved):
    from scipy.special import expit
    validate_state(saved)
    p, odds, r, h, logmass, interior = components(probabilities)
    a, b, c, d = saved["coefficients"]
    if not np.any(saved["coefficients"]):
        return p.copy()
    norm = saved["normalization"]
    logmass = (logmass - norm["log_mass_mean"]) / norm["log_mass_scale"]
    delta = (a + b * (h - .5))[:, None] * r + (c * (h - .5) + d * logmass)[:, None]
    result = np.where(interior, expit(odds + delta), p)
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError("Invalid calibrated probabilities; no output clipping")
    return result


def fit_calibrator(probabilities, labels, metadata, path, parameters):
    import torch
    from scipy.optimize import minimize
    from ml_workflow.storage import write_new
    if parameters not in CANDIDATES.values():
        raise ValueError("Unknown frozen candidate")
    torch.set_num_threads(4)
    p, odds, r, h, logmass, interior = components(probabilities)
    y = np.asarray(labels)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all() or (y.sum(1) > 1).any():
        raise ValueError("Invalid calibration labels")
    positive = np.flatnonzero(y.ravel() == 1)
    negative = np.flatnonzero(y.ravel() == 0)
    if not len(positive) or not len(negative):
        raise ValueError("Both row classes required")
    normalization = {"log_mass_mean": float(logmass.mean()),
                     "log_mass_scale": max(float(logmass.std()), CONFIG["scale_floor"])}
    lm = (logmass - normalization["log_mass_mean"]) / normalization["log_mass_scale"]
    # Stable loss-only endpoint logits. Inference preserves the exact endpoints.
    odds = np.where(p == 0, np.log(1e-15), np.where(p == 1, -np.log(1e-15), odds))
    base, rank, entropy, scale, target = [torch.tensor(v, dtype=torch.float64) for v in (odds, r, h, lm, y)]
    varying = torch.tensor(interior)
    sampled = negative[np.random.default_rng(CONFIG["seed"]).integers(len(negative), size=(len(positive), 16))]
    pos, neg = torch.tensor(positive), torch.tensor(sampled)
    active = parameters["active"]

    def objective(values):
        theta = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        full = torch.zeros(4, dtype=torch.float64).scatter(0, torch.tensor(active), theta)
        a, b, c, d = full.unbind()
        delta = (a + b * (entropy - .5))[:, None] * rank + (c * (entropy - .5) + d * scale)[:, None]
        z = base + torch.where(varying, delta, torch.zeros_like(delta))
        bce = torch.nn.functional.binary_cross_entropy_with_logits(z, target)
        flat = z.ravel()
        pair = torch.nn.functional.softplus(flat[neg] - flat[pos, None]).mean()
        loss = .5 * bce + .5 * pair + .01 * theta.square().sum()
        loss.backward()
        return float(loss.detach()), theta.grad.detach().numpy().copy()

    identity = objective(np.zeros(len(active)))[0]
    result = minimize(objective, np.zeros(len(active)), jac=True, method="L-BFGS-B",
                      bounds=[(-.25, .25)] * len(active), options={key: CONFIG[key] for key in ("maxiter", "ftol", "gtol", "maxls")})
    if not result.success or not np.isfinite(result.fun):
        raise ValueError("Calibrator did not converge: " + str(result.message))
    coefficients = np.zeros(4)
    coefficients[active] = result.x
    record = {"schema_version": 2, "metadata": deepcopy(metadata), "parameters": deepcopy(parameters),
              "normalization": normalization, "coefficients": coefficients.tolist(), "objective": float(result.fun),
              "identity_objective": identity, "iterations": int(result.nit), "converged": True}
    validate_state(record, metadata, parameters)
    predict_calibrator(p, record)
    write_new(Path(path), record)
    return record


def load_calibrator(path, metadata=None, parameters=None):
    return validate_state(json.loads(Path(path).read_text()), metadata, parameters)
