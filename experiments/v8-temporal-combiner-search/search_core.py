"""Frozen temporal selection, bounded logit correction, and experiment accounting."""
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/v8-enriched-logloss"))
import enriched_core as prior

require, sha256, canonical = prior.require, prior.sha256, prior.canonical
read_json, write_new, inside, verify_files = prior.read_json, prior.write_new, prior.inside, prior.verify_files
FOLDS, DAY, V6_SHA = prior.FOLDS, prior.DAY, prior.V6_SHA
BASE_COLUMNS, COLUMNS = prior.BASE_COLUMNS, prior.COLUMNS
first_indices, temporal_check, replacement = prior.first_indices, prior.temporal_check, prior.replacement
enrich, cache_identity = prior.enrich, prior.cache_identity
screen_gates, stress_gates = prior.screen_gates, prior.stress_gates
WORK = Path("artifacts/experiments/v8-temporal-combiner-search")
REPORTS = Path("docs/results/v8-temporal-combiner-search")
LIMITS = {"prepare": 600, "selection": 1800, "screen": 1800, "stress": 2700, "finalize": 2700, "verify": 1800}
OFFSET_NAMES = ("margin_mean", "margin_std", "margin_logsumexp", "softmax_entropy", "log_history_length",
                "user_ctr", "unseen_user", "time_bucket_1", "time_bucket_2", "time_bucket_3", "day_type")
PROTOCOL = {
    "experiment": "v8-temporal-combiner-search", "hypothesis": "Temporal selection of regularized or separately calibrated seed42 combiners improves V6 row AUC",
    "comparator": prior.PROTOCOL["comparator"], "member": "dim8_seed42", "weight": 2 / 9,
    "candidates": {"A": {"features": 37, "depth": 4, "l2_leaf_reg": 50},
                   "B": {"features": 43, "depth": 4, "l2_leaf_reg": 50},
                   "C": {"features": 37, "depth": 6, "l2_leaf_reg": 20, "correction": "event-shared raw-margin offset"}},
    "training": {"loss_function": "Logloss", "iterations": 545, "learning_rate": .05, "random_seed": 42,
                 "thread_count": 4, "row_weights": "unit", "task_type": "CPU", "early_stopping": False,
                 "other_parameters": "compatible settings copied from hash-bound original combiner"},
    "development": {"day": "Jan24", "prefix": "pre-Jan24", "folds_hours": [[0, 8, 16], [0, 16, 24]],
        "reference": "original 37-feature depth6/L2=20 recipe freshly fitted on each complete training block",
        "selection": "both fold gains >= 0 and mean > 0; maximum mean; ties within 1e-12 prefer A then B then C",
        "no_fallback_after_screen": True},
    "correction": {"base_fraction": .75, "timestamp_ties": "entirely in later calibration block",
        "features": list(OFFSET_NAMES), "normalization": "calibration-event means/std; std below 1e-6 replaced by 1",
        "objective": "mean row Logloss + 0.001*sum(weights**2)/2; unpenalized intercept",
        "solver": "L-BFGS-B", "maxiter": 200, "gtol": 1e-8, "ftol": 1e-12,
        "nonconvergence": "reject candidate; no retry", "prediction": "sigmoid(base_raw_margin + shared_event_offset)"},
    "composition": prior.PROTOCOL["composition"], "cohorts": prior.PROTOCOL["cohorts"],
    "bootstrap": prior.PROTOCOL["bootstrap"], "gates": prior.PROTOCOL["gates"], "replay": prior.PROTOCOL["replay"],
    "phase_limits_seconds": LIMITS, "total_compute_seconds": 14400,
    "carry": "all prior PairLogit and enriched Logloss compute", "neural_training": False,
    "validation_status": prior.PROTOCOL["validation_status"], "competition_upload": False,
    "stop": "one development search, one frozen winner; no qualifying winner or failed gate ends experiment",
}
CODE_FILES = prior.CODE_FILES + ("experiments/v8-temporal-combiner-search/search_core.py",
    "experiments/v8-temporal-combiner-search/runner.py", "tools/predict_v8_selected.py")


class Budget(prior.old.Budget):
    def __enter__(self):
        self.state = read_json(self.path)
        require(self.phase not in self.state["phases"], "Preserve prior attempt; retries prohibited")
        self.limit = min(LIMITS[self.phase], 14400 - self.state["used_seconds"])
        require(self.limit > 0, "Total compute budget exhausted")
        self.start = time.monotonic()
        self.state["phases"][self.phase] = {"status": "running", "limit_seconds": self.limit}
        self.persist()
        self.timer = threading.Timer(self.limit, self.expire)
        self.timer.daemon = True
        self.timer.start()
        return self


def parameters(saved, variant):
    require(variant in {"A", "B", "C", "reference"}, "Unknown candidate")
    result = prior.classifier_parameters(saved)
    if variant in {"A", "B"}:
        result.update(depth=4, l2_leaf_reg=50)
    return result


def matrix(x, variant):
    x = np.asarray(x)
    require(x.ndim == 2 and x.shape[1] == 37 and len(x) > 0 and len(x) % 10 == 0
            and x.dtype == np.float32 and np.isfinite(x).all(), "Invalid source features")
    if variant == "B":
        return prior.enrich(x, BASE_COLUMNS)
    require(variant in {"A", "C", "reference"}, "Unknown candidate")
    return x, list(BASE_COLUMNS)


def chronological_split(times):
    times = np.asarray(times)
    require(times.ndim == 1 and len(times) >= 4 and np.isfinite(times).all(), "Invalid calibration timeline")
    ordered = np.sort(times, kind="stable")
    cutoff = ordered[int(.75 * len(ordered))]
    fit, calibration = times < cutoff, times >= cutoff
    require(fit.any() and calibration.any() and times[fit].max() < times[calibration].min(), "Empty or overlapping calibration split")
    return fit, calibration, int(cutoff)


def development_folds(times):
    times = np.asarray(times)
    start = FOLDS["jan24"]
    require(times.ndim == 1 and len(times) and ((times >= start) & (times < start + DAY)).all(), "Development events must be Jan24 only")
    folds = []
    for middle, end in ((8, 16), (16, 24)):
        train = times < start + middle * 3600
        valid = (times >= start + middle * 3600) & (times < start + end * 3600)
        require(train.any() and valid.any() and times[train].max() < times[valid].min(), "Invalid development separation")
        folds.append((train, valid))
    return folds


def choose(results):
    require(set(results) == {"A", "B", "C"}, "Incomplete development candidates")
    eligible = []
    for variant in ("A", "B", "C"):
        result = results[variant]
        if result.get("status") == "rejected_nonconvergence":
            continue
        require(result.get("status") == "complete" and len(result["folds"]) == 2, "Incomplete development folds")
        gains = []
        for number, fold in enumerate(result["folds"]):
            require(fold["fold"] == number, "Development fold identity differs")
            require(all(np.isfinite(fold[k]) for k in ("reference_auc", "candidate_auc", "auc_diff")), "Nonfinite development metrics")
            require(abs(fold["candidate_auc"] - fold["reference_auc"] - fold["auc_diff"]) < 1e-12, "Inconsistent development gain")
            gains.append(fold["auc_diff"])
        if min(gains) >= 0 and np.mean(gains) > 0:
            eligible.append((variant, float(np.mean(gains))))
    if not eligible:
        return None
    best = max(value for _, value in eligible)
    return next(variant for variant, value in eligible if best - value <= 1e-12)


def event_features(raw, x):
    raw = np.asarray(raw, dtype=np.float64).reshape(-1, 10)
    require(x.shape == (raw.size, 37) and np.isfinite(raw).all(), "Invalid correction feature inputs")
    shifted = raw - raw.max(1, keepdims=True)
    exp = np.exp(shifted)
    q = exp / exp.sum(1, keepdims=True)
    logq = shifted - np.log(exp.sum(1, keepdims=True))
    context = []
    for name in ("history_length", "user_ctr", "unseen_user", "time_bucket", "day_type"):
        values = x[:, BASE_COLUMNS.index(name)].reshape(-1, 10)
        require(np.isfinite(values).all() and (values == values[:, :1]).all(), "Event context varies across candidates")
        context.append(values[:, 0])
    history, ctr, unseen, bucket, daytype = context
    require((history >= 0).all() and ((ctr >= 0) & (ctr <= 1)).all() and np.isin(unseen, [0, 1]).all()
            and np.isin(bucket, [0, 1, 2, 3]).all() and np.isin(daytype, [0, 1]).all(), "Invalid event context")
    result = np.column_stack((raw.mean(1), raw.std(1), raw.max(1) + np.log(exp.sum(1)),
        -(q * logq).sum(1), np.log1p(history), ctr, unseen, bucket == 1, bucket == 2, bucket == 3, daytype))
    require(result.shape == (len(raw), 11) and np.isfinite(result).all(), "Invalid event features")
    return result


def correction_loss(theta, raw, normalized, labels):
    offsets = normalized @ theta[:-1] + theta[-1]
    z = raw + offsets[:, None]
    p = prior.old.sigmoid(z)
    residual = p - labels
    loss = np.mean(np.logaddexp(0, z) - labels * z) + .001 * np.dot(theta[:-1], theta[:-1]) / 2
    event_gradient = residual.mean(1)
    gradient = np.r_[normalized.T @ event_gradient / len(raw) + .001 * theta[:-1], event_gradient.mean()]
    return float(loss), gradient


class NonConvergence(ValueError):
    pass


def fit_correction(raw, x, labels):
    from scipy.optimize import minimize
    raw = np.asarray(raw, dtype=np.float64).reshape(-1, 10)
    y = np.asarray(labels)
    require(y.shape == raw.shape and np.isin(y, [0, 1]).all() and len(np.unique(y)) == 2
            and (y.sum(1) <= 1).all(), "Invalid calibration labels")
    values = event_features(raw, x)
    mean, std = values.mean(0), values.std(0)
    scale = np.where(std < 1e-6, 1., std)
    normalized = (values - mean) / scale
    result = minimize(correction_loss, np.zeros(12), args=(raw, normalized, y), jac=True,
                      method="L-BFGS-B", options={"maxiter": 200, "gtol": 1e-8, "ftol": 1e-12})
    if not result.success or not np.isfinite(result.x).all():
        raise NonConvergence(f"Correction optimizer did not converge: {result.message}")
    return {"feature_names": list(OFFSET_NAMES), "mean": mean.tolist(), "scale": scale.tolist(),
            "theta": result.x.tolist(), "iterations": int(result.nit), "objective": float(result.fun),
            "success": True, "solver": "L-BFGS-B", "calibration_events": len(raw)}


def apply_correction(raw, x, saved):
    require(saved["feature_names"] == list(OFFSET_NAMES) and saved["success"] is True, "Invalid correction schema/state")
    mean, scale, theta = (np.asarray(saved[k], dtype=float) for k in ("mean", "scale", "theta"))
    require(mean.shape == scale.shape == (11,) and theta.shape == (12,)
            and all(np.isfinite(v).all() for v in (mean, scale, theta)) and (scale > 0).all(), "Invalid correction coefficients")
    raw = np.asarray(raw, dtype=float).reshape(-1, 10)
    features = event_features(raw, x)
    offsets = ((features - mean) / scale) @ theta[:-1] + theta[-1]
    return prior.old.sigmoid(raw + offsets[:, None])


def load_manifest(root):
    path = root / WORK / "inputs.json"
    manifest = read_json(path)
    preparation = read_json(root / REPORTS / "preparation.json")
    require(preparation["complete"] is True and preparation["manifest_sha256"] == sha256(path), "Frozen manifest changed")
    require(manifest["protocol"] == PROTOCOL and manifest["protocol_sha256"] == canonical(PROTOCOL), "Recipe changed")
    require(set(manifest["code_sha256"]) == set(CODE_FILES), "Incomplete source inventory")
    verify_files(root, manifest["code_sha256"])
    verify_files(root, manifest["inputs_sha256"])
    prior.load_manifest(root)
    return manifest


def selection(root, manifest):
    path = root / REPORTS / "selection.json"
    report = read_json(path)
    require(report["complete"] is True and report["protocol_sha256"] == canonical(PROTOCOL)
            and report["source_sha256"] == manifest["code_sha256"]
            and report["input_manifest_sha256"] == sha256(root / WORK / "inputs.json"), "Selection provenance differs")
    winner = choose(report["results"])
    require(winner is not None and report["winner"] == winner, "No qualifying winner; search is closed")
    for variant, result in report["results"].items():
        for fold in result["folds"]:
            for field, identity in (("fit", variant), ("reference_fit", "reference")):
                fit = fold[field]
                require(fit["variant"] == identity and fit["requested_parameters"] == manifest["parameters"]["primary"][identity],
                        "Development model recipe differs")
                verify_files(root, fit["files_sha256"])
            verify_files(root, {fold["predictions"]: fold["predictions_sha256"]})
    frozen = read_json(root / REPORTS / "selected_recipe.json")
    expected = {"winner": winner, "protocol_sha256": canonical(PROTOCOL), "selection_sha256": sha256(path),
                "parameters": {case: manifest["parameters"][case][winner] for case in manifest["parameters"]}}
    require(frozen == expected, "Selected recipe changed after development")
    return frozen


def validate_report(root, manifest, stage):
    selected = selection(root, manifest)
    report = read_json(root / REPORTS / (stage + ".json"))
    require(report["complete"] is True and report["protocol_sha256"] == canonical(PROTOCOL)
            and report["source_sha256"] == manifest["code_sha256"]
            and report["input_manifest_sha256"] == sha256(root / WORK / "inputs.json")
            and report["selected_recipe_sha256"] == canonical(selected), "Evidence provenance differs")
    checks = prior.screen_gates(report["results"]) if stage == "screen" else prior.stress_gates(report["results"])
    require(all(checks.values()) and report["passed"] is True, f"{stage} gate failed: selected recipe is closed")
    results = report["results"].items() if stage == "screen" else [("five_day", report["results"])]
    for case, result in results:
        fitted = result["fit"]
        require(fitted["case"] == case and fitted["variant"] == selected["winner"]
                and fitted["requested_parameters"] == selected["parameters"][case], "Fitted recipe differs")
        verify_files(root, fitted["files_sha256"])
        verify_files(root, {result["predictions"]: result["predictions_sha256"]})
    return report
