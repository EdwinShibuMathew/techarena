"""Frozen V8 recipe, provenance, temporal selection and compute accounting."""
from contextlib import AbstractContextManager
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/v6-pairlogit-v1-api-fix"))
from pairlogit_core import intercept, ranker_parameters, replacement, sample_pairs, sigmoid

DAY = 86400
JAN20 = 1737331200
FOLDS = {f"jan{day}": JAN20 + (day - 20) * DAY for day in (21, 22, 24, 25, 26, 27)}
V6_SHA = "c386d545b3746e05dd515020c6185afda06cefdbcdf7d48df022454762fb378c"
PHASE_LIMITS = {"prepare": 1800, "screen": 3600, "stress": 3600, "finalize": 3600, "verify": 1800}
PROTOCOL = {
    "experiment": "v8-pairlogit", "hypothesis": "Global sampled PairLogit on the existing seed42 combiner features improves global row AUC over unchanged V6",
    "comparator": "V6 candidate arrays; baseline arrays in the historical archives are V5",
    "member": "dim8_seed42", "weight": 2 / 9, "feature_count": 37,
    "training": {"loss_function": "PairLogit", "iterations": 545, "depth": 6, "learning_rate": .05,
                 "l2_leaf_reg": 20, "random_seed": 42, "thread_count": 4, "task_type": "CPU",
                 "other_parameters": "compatible settings copied from the corresponding hash-bound original combiner"},
    "pairs": {"negatives_per_positive": 16, "seed": 3042, "pool": "all negative rows of fitting-day first-user events, including no-click events",
              "groups": "one group for the entire fitting day", "replacement": False, "pair_weights": 1},
    "link": {"slope": 1, "intercept": "mean(sigmoid(raw_train+b)) = fitting-day row CTR", "bisections": 80},
    "composition": "V6 + (2/9)*(new_seed42-original_seed42); no clipping, sharpening or event normalization",
    "cohorts": {"primary": ["jan24", "jan25", "Jan25-26 first user"],
                "secondary": ["Jan25 subset of primary target features", "jan26", "Jan26 first user"],
                "five_day": ["jan21", "jan22", "Jan22-26 first user"],
                "final": ["Jan26 first-user features from pre-Jan26 model", "jan27", "unlabeled evaluation"]},
    "bootstrap": {"resamples": 1000, "seed": 2026, "unit": "whole user", "interval": "95% percentile", "minimum_valid_fraction": .99},
    "gates": ["primary CI lower > 0", "secondary AUC change >= 0", "five-day CI lower > 0", "five-day late 48-120h AUC change >= 0"],
    "replay": {"rtol": 0, "atol": 1e-15}, "phase_limits_seconds": PHASE_LIMITS, "total_compute_seconds": 14400,
    "stop": "A failed gate ends this recipe; no hyperparameter, weight, pair-count, link, seed or routing sweep",
    "validation_status": "reused development dates, not independent holdout or selection-adjusted evidence",
    "neural_training": False, "early_stopping": False, "competition_upload": False,
}
CODE_FILES = (
    "experiments/v8-pairlogit/v8_core.py", "experiments/v8-pairlogit/runner.py", "tools/predict_v8.py",
    "experiments/v6-pairlogit-v1-api-fix/pairlogit_core.py",
    "experiments/v6-pairlogit-v1-api-fix/conditional_choice_trial.py",
    "experiments/v6-pairlogit-v1-api-fix/event_scale_diagnostic.py",
    "experiments/v6-pairlogit-v1-api-fix/attention_allocation_test.py",
    "tools/experiment_runner.py", "tools/predict_v6.py", "tools/predict_v7.py", "tools/consolidate.py",
    "configs/v6.json", "configs/source_manifest.json",
    "docs/results/techarena-row-auc-trial/ensemble_trial_results.json",
    "docs/results/techarena-row-auc-five-day/paired_results.json",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def inside(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, f"Unsafe relative path: {relative}")
    path = (root / relative).resolve()
    require(path.is_relative_to(root), f"Path escapes root: {relative}")
    return path


def first_indices(users, times, begin, end):
    users, times = np.asarray(users), np.asarray(times)
    require(users.ndim == times.ndim == 1 and users.shape == times.shape, "Invalid temporal vectors")
    eligible = np.flatnonzero((times >= begin) & (times < end))
    ordered = eligible[np.argsort(times[eligible], kind="stable")]
    _, positions = np.unique(users[ordered], return_index=True)
    return np.sort(ordered[positions])


def temporal_check(raw, fit_idx, target_idx, fit_day, target_cutoff):
    fit_idx, target_idx = np.asarray(fit_idx), np.asarray(target_idx)
    require(fit_day < target_cutoff, "Fitting labels must precede the target model cutoff")
    require(np.array_equal(np.sort(fit_idx), first_indices(raw["u"], raw["t"], fit_day, fit_day + DAY)), "Meta cohort differs from first user per fitting day")
    require(np.all(raw["t"][fit_idx] < target_cutoff), "Meta labels cross target cutoff")
    require(np.all(raw["t"][target_idx] >= target_cutoff), "Target event precedes target cutoff")
    require(np.intersect1d(fit_idx, target_idx).size == 0, "Fitting and target events overlap")
    require(len(np.unique(raw["u"][target_idx])) == len(target_idx), "Target users repeat")


def cache_identity(checkpoint_hash, cutoff, indices, inputs_hash, columns):
    indices = np.asarray(indices, dtype=np.int64)
    return {"checkpoint_sha256": checkpoint_hash, "cutoff": int(cutoff), "events": len(indices),
            "event_index_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
            "inputs_sha256": inputs_hash, "feature_names": list(columns), "dtype": "float32"}


def verify_files(root, hashes):
    for relative, expected in hashes.items():
        require(sha256(inside(root, relative)) == expected, f"Frozen input changed: {relative}")


def manifest_line_endings_match(path, expected):
    """Only LF/CRLF serialization may differ; all other manifest bytes stay bound."""
    content = Path(path).read_bytes()
    lf = content.replace(b"\r\n", b"\n")
    return expected in {hashlib.sha256(value).hexdigest() for value in (content, lf, lf.replace(b"\n", b"\r\n"))}


def verify_v6_bundle(root):
    """Verify the original bundle with an explicit JSON line-ending adapter."""
    manifest = read_json(root / "artifacts/v6/manifest.json")
    config = read_json(root / "configs/v6.json")
    require(manifest["config_sha256"] == sha256(root / "configs/v6.json")
            and config["submission_sha256"] == manifest["expected_submission_sha256"] == V6_SHA, "V6 bundle configuration differs")
    source = root / "configs/source_manifest.json"
    require(manifest_line_endings_match(source, manifest["source_manifest_sha256"]), "Source manifest has a content difference beyond LF/CRLF")
    verify_files(root, read_json(source)["files"])
    require([m["key"] for m in manifest["members"]] == config["members"], "V6 member order differs")
    for relative, info in manifest["files"].items():
        path = inside(root, relative)
        require(path.stat().st_size == info["bytes"] and sha256(path) == info["sha256"], f"V6 asset differs: {relative}")
    for member in manifest["members"]:
        for field in ("checkpoint", "preference", "combiner", "reference"):
            require(member[field] in manifest["files"], "Unverified V6 member asset")
    print(f"Verified {len(manifest['files'])} V6 assets and exact historical source bytes (JSON LF/CRLF adapter).", flush=True)
    return manifest


def load_manifest(root):
    path = root / "artifacts/experiments/v8-pairlogit/inputs.json"
    manifest = read_json(path)
    preparation = read_json(root / "docs/results/v8-pairlogit/preparation.json")
    require(preparation.get("complete") is True and preparation["manifest_sha256"] == sha256(path), "Frozen input manifest changed after preparation")
    require(manifest["protocol"] == PROTOCOL and manifest["protocol_sha256"] == canonical(PROTOCOL), "Recipe changed")
    require(set(manifest["code_sha256"]) == set(CODE_FILES), "Validation source inventory differs")
    verify_files(root, manifest["code_sha256"])
    verify_files(root, manifest["inputs_sha256"])
    verify_files(root, manifest["prepared_sha256"])
    return manifest


def screen_gates(results):
    return {"primary_ci_lower_positive": results["primary"]["ci_lower"] > 0,
            "secondary_change_nonnegative": results["secondary"]["auc_diff"] >= 0}


def stress_gates(results):
    return {"five_day_ci_lower_positive": results["ci_lower"] > 0,
            "late_change_nonnegative": results["late_48_120h"]["auc_diff"] >= 0}


class Budget(AbstractContextManager):
    """Persistent phase accounting and a hard process timeout, excluding human waits."""
    def __init__(self, work, phase):
        self.path = Path(work) / "budget.json"
        self.phase = phase
        self.lock = threading.Lock()

    def persist(self):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.path)

    def __enter__(self):
        carry_path = self.path.parent.parent / "v8-pairlogit-budget-carry.json"
        carry = read_json(carry_path) if carry_path.exists() else {"used_seconds": 0., "prior_engineering_attempts": []}
        self.state = read_json(self.path) if self.path.exists() else {**carry, "phases": {}}
        require(self.phase not in self.state["phases"], "Preserve prior phase attempt; automatic retries are prohibited")
        remaining = PROTOCOL["total_compute_seconds"] - self.state["used_seconds"]
        prior_phase_seconds = sum(attempt.get("phases", {}).get(self.phase, {}).get("seconds", 0.)
                                  for attempt in self.state.get("prior_engineering_attempts", []))
        self.limit = min(PHASE_LIMITS[self.phase] - prior_phase_seconds, remaining)
        require(self.limit > 0, "Total compute budget exhausted")
        self.start = time.monotonic()
        self.state["phases"][self.phase] = {"status": "running", "limit_seconds": self.limit}
        self.persist()
        self.timer = threading.Timer(self.limit, self.expire)
        self.timer.daemon = True
        self.timer.start()
        return self

    def check(self):
        if time.monotonic() - self.start >= self.limit:
            raise TimeoutError(f"{self.phase} compute budget exhausted")

    def expire(self):
        with self.lock:
            self.finish("timeout", "Hard phase/global compute limit reached")
        os._exit(124)

    def finish(self, status, error=None):
        elapsed = time.monotonic() - self.start
        self.state["used_seconds"] += elapsed
        self.state["phases"][self.phase] = {"status": status, "seconds": elapsed, "error": error}
        self.persist()

    def __exit__(self, error_type, error, traceback):
        self.timer.cancel()
        with self.lock:
            self.finish("complete" if error is None else "failed", None if error is None else str(error))
        return False
