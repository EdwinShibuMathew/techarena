"""Frozen successor recipe, checked relative features, and release provenance."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/v8-pairlogit"))
import v8_core as old

require, sha256, canonical = old.require, old.sha256, old.canonical
read_json, write_new, inside, verify_files = old.read_json, old.write_new, old.inside, old.verify_files
DAY, FOLDS, V6_SHA = old.DAY, old.FOLDS, old.V6_SHA
Budget, first_indices, temporal_check = old.Budget, old.first_indices, old.temporal_check
replacement, cache_identity = old.replacement, old.cache_identity
WORK = Path("artifacts/experiments/v8-enriched-logloss")
REPORTS = Path("docs/results/v8-enriched-logloss")
BASE_COLUMNS = (
    "pipeline_probability", "raw_event_probability", "log_probability", "event_click_probability",
    "event_max_probability", "relative_probability", "item_ctr", "user_ctr", "log_item_displays",
    "log_user_events", "time_bucket", "day_type", "history_length", "history_unique_items",
    "log_last_click_age", "clicks_last_6h", "clicks_last_24h", "clicks_last_72h", "repeat_count",
    "repeat_recency", "log_last_repeat_age", "history_dot_mean", "history_dot_max", "history_dot_recent",
    "history_dot_last", "history_dot_max_nonrepeat", "history_cosine_mean", "history_cosine_max",
    "encoded_history_affinity", "user_affinity", "item_vector_norm", "user_vector_norm", "item_bias",
    "user_bias", "unseen_user", "item_time_effect", "user_time_effect",
)
EXTRA_COLUMNS = ("prob_ratio_to_event_max", "prob_diff_from_event_mean", "prob_zscore_in_event",
                 "candidate_event_rank", "candidate_log_odds_event", "history_recency_skew")
COLUMNS = BASE_COLUMNS + EXTRA_COLUMNS
PROTOCOL = {
    "experiment": "v8-enriched-logloss", "hypothesis": "Within-event relative features improve the seed42 Logloss contribution to global row AUC",
    "comparator": old.PROTOCOL["comparator"], "member": "dim8_seed42", "weight": 2 / 9,
    "feature_names": list(COLUMNS), "feature_count": 43,
    "features": {"source": "named pipeline_probability column; contiguous ten-candidate events",
                 "max_floor": 1e-8, "std_floor": 1e-6, "log_odds_clip": [1e-7, 1 - 1e-7],
                 "rank": "descending zero-based average rank, identical probabilities tie",
                 "history": "clicks_last_6h / (clicks_last_72h + 1)", "dtype": "float32"},
    "training": {**old.PROTOCOL["training"], "loss_function": "Logloss", "row_weights": "unit; all fitting rows"},
    "composition": old.PROTOCOL["composition"], "link": "direct positive-class predict_proba; no calibration",
    "cohorts": old.PROTOCOL["cohorts"], "bootstrap": old.PROTOCOL["bootstrap"], "gates": old.PROTOCOL["gates"],
    "replay": old.PROTOCOL["replay"], "phase_limits_seconds": old.PHASE_LIMITS,
    "total_compute_seconds": 14400, "carry": "all previous V8 PairLogit and diagnostic compute",
    "stop": "One fixed successor; failed gate closes recipe; no sweeps or automatic retries",
    "validation_status": old.PROTOCOL["validation_status"], "neural_training": False,
    "early_stopping": False, "competition_upload": False,
}
CODE_FILES = old.CODE_FILES + (
    "experiments/v8-enriched-logloss/enriched_core.py",
    "experiments/v8-enriched-logloss/runner.py", "tools/predict_v8_enriched.py",
)


def enrich(x, names):
    """Append fixed event features without changing the original 37 columns."""
    x = np.asarray(x)
    require(tuple(names) == BASE_COLUMNS, "Original feature schema/order differs")
    require(x.ndim == 2 and x.shape[1] == 37 and len(x) > 0 and len(x) % 10 == 0
            and x.dtype == np.float32 and np.isfinite(x).all(), "Invalid original feature matrix")
    p = x[:, BASE_COLUMNS.index("pipeline_probability")].reshape(-1, 10)
    require(((p >= 0) & (p <= 1)).all(), "Invalid pipeline probabilities")
    recent = x[:, BASE_COLUMNS.index("clicks_last_6h")]
    older = x[:, BASE_COLUMNS.index("clicks_last_72h")]
    require(((recent >= 0) & (older >= recent)).all(), "Invalid history count relationship")
    mean, std = p.mean(1, keepdims=True), p.std(1, keepdims=True)
    # For each candidate count strictly better candidates and average tied positions.
    greater = (p[:, None, :] > p[:, :, None]).sum(2)
    equal = (p[:, None, :] == p[:, :, None]).sum(2)
    rank = greater + (equal - 1) / 2
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    extras = np.column_stack((
        (p / np.maximum(p.max(1, keepdims=True), 1e-8)).ravel(),
        (p - mean).ravel(), ((p - mean) / np.maximum(std, 1e-6)).ravel(), rank.ravel(),
        np.log(clipped / (1 - clipped)).ravel(), recent / (older + 1),
    )).astype(np.float32)
    result = np.column_stack((x, extras))
    require(result.shape == (len(x), 43) and np.isfinite(result).all(), "Invalid enriched feature matrix")
    return result, list(COLUMNS)


def classifier_parameters(saved):
    result = old.ranker_parameters(saved)
    result["loss_function"] = "Logloss"
    return result


def validate_metrics(result, bootstrap=True):
    keys = ["baseline_auc", "candidate_auc", "auc_diff"] + (["ci_lower", "ci_upper"] if bootstrap else [])
    require(all(np.isfinite(result[k]) for k in keys), "Non-finite gate evidence")
    require(0 <= result["baseline_auc"] <= 1 and 0 <= result["candidate_auc"] <= 1, "Invalid AUC")
    require(abs(result["candidate_auc"] - result["baseline_auc"] - result["auc_diff"]) < 1e-12,
            "Inconsistent AUC difference")
    if bootstrap:
        require(result["ci_lower"] <= result["ci_upper"], "Invalid confidence interval")
        require(result["n_bootstrap"] == 1000 and 990 <= result["valid_bootstrap_samples"] <= 1000,
                "Incomplete bootstrap evidence")


def screen_gates(results):
    for value in results.values():
        validate_metrics(value)
    require(set(results) == {"primary", "secondary"}, "Incomplete screen cohorts")
    return old.screen_gates(results)


def stress_gates(results):
    validate_metrics(results)
    validate_metrics(results["late_48_120h"], bootstrap=False)
    return old.stress_gates(results)


def load_manifest(root):
    path = root / WORK / "inputs.json"
    manifest = read_json(path)
    preparation = read_json(root / REPORTS / "preparation.json")
    require(preparation.get("complete") is True and preparation["manifest_sha256"] == sha256(path), "Frozen manifest changed")
    require(manifest["protocol"] == PROTOCOL and manifest["protocol_sha256"] == canonical(PROTOCOL), "Recipe changed")
    require(set(manifest["code_sha256"]) == set(CODE_FILES), "Incomplete source inventory")
    verify_files(root, manifest["code_sha256"])
    verify_files(root, manifest["inputs_sha256"])
    old.load_manifest(root)
    return manifest


def validate_report(root, manifest, stage):
    report = read_json(root / REPORTS / (stage + ".json"))
    require(report.get("complete") is True and report["protocol_sha256"] == canonical(PROTOCOL), "Incomplete or different evidence")
    require(report["source_sha256"] == manifest["code_sha256"] and
            report["input_manifest_sha256"] == sha256(root / WORK / "inputs.json"), "Evidence provenance differs")
    checks = screen_gates(report["results"]) if stage == "screen" else stress_gates(report["results"])
    require(all(checks.values()) and report.get("passed") is True, f"{stage} gate failed: fixed recipe is closed")
    results = report["results"].items() if stage == "screen" else [("five_day", report["results"])]
    for expected_case, result in results:
        fit = result["fit"]
        case = fit["case"]
        require(case == expected_case, "Wrong fitted cohort")
        require(fit["requested_parameters"] == manifest["classifier_parameters"][case], "Fitted recipe differs")
        verify_files(root, {fit["model"]: fit["model_sha256"]})
        verify_files(root, {result["predictions"]: result["predictions_sha256"]})
    return report
