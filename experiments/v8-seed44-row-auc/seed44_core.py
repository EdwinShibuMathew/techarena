"""One fixed seed44 neural objective extension; provenance, gates and hard budget."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/v8-temporal-combiner-search"))
import search_core as previous
sys.path.insert(0, str(ROOT / "legacy"))
from row_auc_objective import CONFIG

require, sha256, canonical = previous.require, previous.sha256, previous.canonical
read_json, write_new, inside, verify_files = previous.read_json, previous.write_new, previous.inside, previous.verify_files
FOLDS, DAY, V6_SHA, BASE_COLUMNS = previous.FOLDS, previous.DAY, previous.V6_SHA, previous.BASE_COLUMNS
first_indices, temporal_check, replacement = previous.first_indices, previous.temporal_check, previous.replacement
screen_gates, stress_gates = previous.screen_gates, previous.stress_gates
WORK = Path("artifacts/experiments/v8-seed44-row-auc")
REPORTS = Path("docs/results/v8-seed44-row-auc")
LIMITS = {"prepare": 600, "screen": 3600, "stress": 3600, "finalize": 1800, "verify": 1800}
CASE_FOLDS = {"primary": ("jan24", "jan25"), "secondary": ("jan25", "jan26"), "five_day": ("jan21", "jan22")}
PROTOCOL = {
    "experiment": "v8-seed44-row-auc", "hypothesis": "The mixed event/global row-ranking objective that improved seeds42/43 also improves seed44 versus unchanged V6",
    "member": "dim8_seed44", "weight": 2 / 9, "dimension": 8, "seed": 44,
    "preference": "reuse verified epoch260 checkpoints read-only; no preference retraining",
    "event_objective": CONFIG, "event_epochs": 8, "event_batch_size": 2048,
    "optimizer": "fresh Adam defaults, learning rate 0.0015", "pair_sampling_seed": 3044,
    "initialization": "original architecture, RNG reset and fitting-only global offset",
    "regularization": {"anchor": .1, "effects": .05}, "early_stopping": False,
    "inference": "original eleven-outcome softmax, exposure smoothing20/power0.5, ridge history fallback and 37 features",
    "combiner": {"loss_function": "Logloss", "iterations": 545, "depth": 6, "learning_rate": .05,
                 "l2_leaf_reg": 20, "random_seed": 42, "thread_count": 4, "row_weights": "unit",
                 "other_parameters": "compatible settings from corresponding hash-bound seed44 original combiner"},
    "composition": "V6 + (2/9)*(new_seed44-original_seed44); all other members unchanged; no clipping or calibration",
    "cohorts": {"primary": ["pre-Jan24 meta model, Jan24 combiner", "pre-Jan25 target model, first-user Jan25-26"],
                "secondary": ["pre-Jan25 meta model, Jan25 combiner", "pre-Jan26 target model, first-user Jan26"],
                "five_day": ["pre-Jan21 meta model, Jan21 combiner", "pre-Jan22 target model, first-user Jan22-26"],
                "final": ["reuse validated Jan25 combiner without refit", "new pre-Jan27 event model, unlabeled evaluation"]},
    "bootstrap": previous.PROTOCOL["bootstrap"], "gates": previous.PROTOCOL["gates"],
    "replay": previous.PROTOCOL["replay"], "validation_status": previous.PROTOCOL["validation_status"],
    "phase_limits_seconds": LIMITS, "total_compute_seconds": 14400,
    "stop": "one seed/objective; failed integrity, metric gate or timeout closes recipe; no automatic retries",
    "competition_upload": False,
}
CODE_FILES = previous.CODE_FILES + ("legacy/row_auc_objective.py", "legacy/train_final_models.py",
    "experiments/v8-seed44-row-auc/seed44_core.py", "experiments/v8-seed44-row-auc/train_seed44.py",
    "experiments/v8-seed44-row-auc/runner.py", "tools/predict_v8_seed44.py")

# Short legacy checkpoints lack seed/cutoff annotations; reviewed file hashes are
# sealed here and reconciled against the original per-fold training report.
SHORT_HASHES = {
    "jan24": ("135922e7e9313895efc8f65bbc2cc4b9f6ef411b37d8d78f1beb11135d0a1522", "c3ae2fb3ccaa5109b7a1b4161e3004f031f5b1b5824dcc6e691c924f0d1dbe78"),
    "jan25": ("a45fce68388c3b2d4e9b055c01b94dbcdf41838ca28b1ebd3a0fe49a3cb6e3a4", "a7defcb3ef46c46c0360f6fcb319c935ee0497e4ca9632fa8b1e0ad599102725"),
    "jan26": ("c5d923581157453fc20b12e473de42421179f710f604e94e968b0dc5681b3511", "3befb361a97634e6497922674ce09433287c1d0c563f01ec3f178b309f4fb4a3"),
}
COMBINER_HASHES = {"primary": "3ac1129a96ac27c2f4d85999bc83e66759219bb68bc8159248a85d05b8914d90",
                  "secondary": "ceed77f5188bbfdcc6189c3fe16d2e0ea5d399c3dca23eaf1f6faa260d7c5ece",
                  "five_day": "52a0dcad49e2859df6416d3040d6ac8fb6cf5624a99fb12e45e8d5fcf580db64"}


class Budget(previous.Budget):
    """Time the entire stage and terminate a training process group on expiry."""
    child = None

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

    def terminate_child(self):
        child = self.child
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)

    def run(self, command, cwd):
        self.check()
        self.child = subprocess.Popen(command, cwd=cwd, start_new_session=True)
        try:
            code = self.child.wait(timeout=max(.001, self.limit - (time.monotonic() - self.start)))
            require(code == 0, f"Training subprocess failed with exit code {code}")
        except BaseException:
            self.terminate_child()
            raise
        finally:
            self.child = None
        self.check()

    def expire(self):
        self.terminate_child()
        with self.lock:
            self.finish("timeout", "Hard stage/global limit; training process group terminated")
        os._exit(124)

    def __exit__(self, error_type, error, traceback):
        if error is not None:
            self.terminate_child()
        return super().__exit__(error_type, error, traceback)


def neural_path(fold):
    require(fold in FOLDS, "Unknown model prefix")
    return WORK / "models" / fold / "event.pt"


def load_manifest(root):
    path = root / WORK / "inputs.json"
    manifest = read_json(path)
    preparation = read_json(root / REPORTS / "preparation.json")
    require(preparation["complete"] is True and preparation["manifest_sha256"] == sha256(path), "Frozen manifest changed")
    require(manifest["protocol"] == PROTOCOL and manifest["protocol_sha256"] == canonical(PROTOCOL), "Recipe changed")
    require(set(manifest["code_sha256"]) == set(CODE_FILES), "Incomplete source inventory")
    verify_files(root, manifest["code_sha256"])
    verify_files(root, manifest["inputs_sha256"])
    previous.load_manifest(root)
    return manifest


def validate_neural(root, manifest, fold):
    import torch
    path = root / neural_path(fold)
    model = torch.load(path, map_location="cpu", weights_only=True)
    require(model["epoch"] == 8 and model["dimension"] == 8 and model["seed"] == 44
            and model["stage"] == "event" and model["cutoff"] == FOLDS[fold]
            and model["event_objective"] == CONFIG["objective"] and model["pair_sampling_seed"] == 3044
            and model["maximum_fitting_timestamp"] < FOLDS[fold] and model["fitting_events"] > 0
            and model["protocol_sha256"] == canonical(PROTOCOL)
            and model["preference_sha256"] == manifest["inputs_sha256"][manifest["preferences"][fold]], "New neural checkpoint recipe differs")
    require(all(bool(torch.isfinite(v).all()) for v in model["model_state"].values() if v.is_floating_point()), "Nonfinite neural weights")
    return model


def validate_report(root, manifest, stage):
    report = read_json(root / REPORTS / (stage + ".json"))
    require(report["complete"] is True and report["protocol_sha256"] == canonical(PROTOCOL)
            and report["source_sha256"] == manifest["code_sha256"]
            and report["input_manifest_sha256"] == sha256(root / WORK / "inputs.json"), "Evidence provenance differs")
    checks = screen_gates(report["results"]) if stage == "screen" else stress_gates(report["results"])
    require(all(checks.values()) and report["passed"] is True, f"{stage} gate failed: seed44 recipe is closed")
    results = report["results"].items() if stage == "screen" else [("five_day", report["results"])]
    for case, result in results:
        fit = result["fit"]
        expected = {(neural_path(fold)).as_posix() for fold in CASE_FOLDS[case]}
        require(fit["case"] == case and fit["parameters"] == manifest["parameters"][case]
                and set(fit["neural_sha256"]) == expected, "Fitted recipe differs")
        verify_files(root, fit["neural_sha256"])
        verify_files(root, {fit["model"]: fit["model_sha256"], result["predictions"]: result["predictions_sha256"]})
        for fold in CASE_FOLDS[case]:
            validate_neural(root, manifest, fold)
    return report
