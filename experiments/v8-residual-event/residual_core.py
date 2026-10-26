"""Frozen residual-event campaign, selection, provenance and promotion gates."""
from pathlib import Path
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/v8-robust-ensemble"))
import robust_core as prior

require, sha256, canonical = prior.require, prior.sha256, prior.canonical
read_json, write_new, inside, verify_files = prior.read_json, prior.write_new, prior.inside, prior.verify_files
DAY, FOLDS, V6_SHA, BASE_COLUMNS = prior.DAY, prior.FOLDS, prior.V6_SHA, prior.BASE_COLUMNS
MEMBERS, CASE_FOLDS = prior.MEMBERS, prior.CASE_FOLDS
first_indices, temporal_check = prior.first_indices, prior.temporal_check
screen_gates, stress_gates = prior.screen_gates, prior.stress_gates
original_ensemble, pair_decomposition = prior.original_ensemble, prior.pair_decomposition
WORK, REPORTS = Path("artifacts/experiments/v8-residual-event"), Path("docs/results/v8-residual-event")
ORIGINS = ("jan21", "jan22", "jan24")
COMPOSITIONS = {"A": {"dim8_seed42": "event"}, "B": {"dim8_seed42": "both"},
                "C": {member: "both" for member in MEMBERS}}
LIMITS = {"prepare": 600, "development": 6000, "screen": 1800, "stress": 1800, "finalize": 2400, "verify": 600}
TRAINING = {"epochs": 8, "batch_size": 2048, "optimizer": "AdamW", "learning_rate": .001,
            "weight_decay": .0001, "gradient_norm_limit": 5., "correction_regularization": .1,
            "objective": prior.CONFIG, "initialization_seed": "10000+member seed for event; 20000+member seed for both",
            "shuffle_seed": "initialization seed+1000", "pair_seed": "initialization seed+3000"}
PROTOCOL = {
    "experiment": "v8-residual-event", "hypothesis": "A jointly trained, bounded event/choice residual improves global row ranking while preserving the frozen V6 scorer",
    "comparator": "unchanged V6; historical baseline arrays in short archives are V5, candidate arrays are V6",
    "compositions": COMPOSITIONS, "weights": {m: v["weight"] for m, v in MEMBERS.items()},
    "architecture": {"base": "same-prefix original V6 event checkpoint, all parameters frozen; no preference training",
        "hidden": 32, "activation": "SiLU", "output": "zero initialized, .5*tanh; candidate head event-centered",
        "event_inputs": "effective user, mean candidate, mean history vectors; baseline score mean/max/logsumexp divided by10; history length/20; log1p(last age clipped720h)/log721; counts<6/24/72h divided by20; four time-bucket indicators; cold status",
        "choice_inputs": "candidate, attended history, elementwise product; log1p(repeat count)/log21; centered baseline score/10",
        "attention": "softmax(dot(candidate,history)/sqrt(d)-log1p(age/24)); mask padding and renormalize; empty history gives zero",
        "cold_feature": "zero mask for the cold-status feature, which is constant zero in prefix fitting; prevents unsupported extrapolation",
        "day_type": "base retains original eight contexts; adapter shares parameters across day types"},
    "training": TRAINING, "inference": prior.PROTOCOL["inference"], "features": list(BASE_COLUMNS),
    "combiner": prior.PROTOCOL["combiner"], "development": {
        "origins": list(ORIGINS), "fit": "first user in origin's first8h, fresh original and changed-member combiners",
        "validation": "first user after origin+8h, before Jan25; neural models fixed at origin",
        "bands_hours": [[8,24],[24,48],[48,72],[72,96]],
        "eligibility": "all origin aggregate gains >=0; mean gain>0; mean available >=48h gain>=0; mean weighted no-click-negative contribution>=0",
        "selection": "largest minimum origin gain; ties within1e-12 prefer A,B,C; no promotion fallback"},
    "cohorts": prior.PROTOCOL["cohorts"], "bootstrap": prior.PROTOCOL["bootstrap"],
    "gates": prior.PROTOCOL["gates"], "replay": prior.PROTOCOL["replay"],
    "final": "same-prefix pre-Jan27 adapters; reuse each selected member's validated Jan25 combiner",
    "validation_status": "reused retrospective development dates; not independent confirmation or selection-adjusted confidence",
    "phase_limits_seconds": LIMITS, "total_compute_seconds": 14400, "competition_upload": False,
    "stop": "no eligible composition, failed integrity/gate, or timeout closes this recipe; no automatic retries or fallback",
}
CODE_FILES = prior.CODE_FILES + ("experiments/v8-residual-event/residual_core.py",
    "experiments/v8-residual-event/residual_model.py", "experiments/v8-residual-event/train_adapter.py",
    "experiments/v8-residual-event/runner.py", "tools/predict_v8_residual.py")


class Budget(prior.Budget):
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


def adapter_path(member, kind, fold):
    require(member in MEMBERS and kind in {"event", "both"} and fold in FOLDS, "Unknown adapter")
    require(kind == "both" or member == "dim8_seed42", "Undeclared event-only member")
    return WORK / "models" / member / kind / fold / "adapter.pt"


def seed(member, kind):
    return (10000 if kind == "event" else 20000) + MEMBERS[member]["seed"]


def compose(baseline, original, updated, composition):
    require(composition in COMPOSITIONS and set(updated) == set(COMPOSITIONS[composition]), "Wrong composition")
    require(set(original) == set(MEMBERS), "Incomplete original ensemble")
    values = {m: np.asarray(updated.get(m, original[m]), dtype=np.float64) for m in MEMBERS}
    p = original_ensemble(values)
    require(p.shape == np.asarray(baseline).shape and np.isfinite(p).all()
            and ((p >= 0) & (p <= 1)).all(), "Invalid candidate")
    require(np.allclose(original_ensemble(original), baseline, rtol=0, atol=1e-15), "Original V6 ensemble differs")
    return p


def development_indices(raw, fold):
    require(fold in ORIGINS, "Unknown origin")
    start, end = FOLDS[fold], FOLDS["jan25"]
    fit = first_indices(raw["u"], raw["t"], start, start + 8*3600)
    valid = first_indices(raw["u"], raw["t"], start + 8*3600, end)
    require(len(fit) and len(valid) and raw["t"][fit].max() < raw["t"][valid].min(), "Invalid forward origin")
    require(not np.intersect1d(fit, valid).size, "Development events overlap")
    return fit, valid


def choose(results):
    require(set(results) == set(COMPOSITIONS), "Incomplete compositions")
    eligible = []
    for key in COMPOSITIONS:
        rows = results[key]["origins"]
        require([r["origin"] for r in rows] == list(ORIGINS), "Incomplete development origins")
        gains, late, no_click = [], [], []
        for row in rows:
            require(all(np.isfinite(row[k]) for k in ("baseline_auc", "candidate_auc", "auc_diff", "no_click_contribution")), "Nonfinite development metrics")
            require(abs(row["candidate_auc"]-row["baseline_auc"]-row["auc_diff"]) < 1e-12, "Inconsistent development AUC")
            gains.append(row["auc_diff"]); no_click.append(row["no_click_contribution"])
            if row["origin"] != "jan24":
                require(row.get("late") is not None and np.isfinite(row["late"]["auc_diff"]), "Missing late development evidence")
                late.append(row["late"]["auc_diff"])
        if min(gains) >= 0 and np.mean(gains) > 0 and np.mean(late) >= 0 and np.mean(no_click) >= 0:
            eligible.append((key, min(gains)))
    if not eligible:
        return None
    best = max(score for _, score in eligible)
    return next(key for key, score in eligible if best-score <= 1e-12)


def load_manifest(root):
    path = root / WORK / "inputs.json"
    m = read_json(path)
    p = read_json(root / REPORTS / "preparation.json")
    require(p["complete"] is True and p["manifest_sha256"] == sha256(path), "Preparation changed")
    require(m["protocol"] == PROTOCOL and m["protocol_sha256"] == canonical(PROTOCOL), "Residual recipe changed")
    require(set(m["code_sha256"]) == set(CODE_FILES), "Incomplete source inventory")
    verify_files(root, m["code_sha256"]); verify_files(root, m["inputs_sha256"])
    prior.load_manifest(root)
    return m


def validate_adapter(root, manifest, member, kind, fold):
    import torch
    saved = torch.load(root / adapter_path(member, kind, fold), map_location="cpu", weights_only=True)
    require(saved["member"] == member and saved["kind"] == kind and saved["cutoff"] == FOLDS[fold]
        and saved["dimension"] == MEMBERS[member]["dimension"] and saved["epoch"] == 8
        and saved["base_sha256"] == manifest["inputs_sha256"][manifest["original_models"][member][fold]]
        and saved["protocol_sha256"] == canonical(PROTOCOL) and saved["training"] == TRAINING
        and saved["seed"] == seed(member, kind) and saved["maximum_fitting_timestamp"] < FOLDS[fold]
        and saved["base_weights_unchanged"] is True, "Adapter provenance/recipe differs")
    require(all(bool(torch.isfinite(v).all()) for v in saved["adapter_state"].values()), "Nonfinite adapter")
    return saved


def selection(root, manifest):
    path = root / REPORTS / "development.json"
    report = read_json(path)
    require(report["complete"] is True and report["protocol_sha256"] == canonical(PROTOCOL)
        and report["source_sha256"] == manifest["code_sha256"]
        and report["input_manifest_sha256"] == sha256(root / WORK / "inputs.json"), "Invalid development evidence")
    verify_files(root, report["files_sha256"])
    winner = choose(report["results"])
    require(winner is not None and report["winner"] == winner, "No eligible residual composition")
    receipt = read_json(root / REPORTS / "selected.json")
    require(receipt == {"composition": winner, "members": COMPOSITIONS[winner],
        "protocol_sha256": canonical(PROTOCOL), "development_sha256": sha256(path)}, "Selection receipt differs")
    return winner


def validate_report(root, manifest, stage):
    winner = selection(root, manifest)
    report = read_json(root / REPORTS / (stage + ".json"))
    require(report["complete"] is True and report["protocol_sha256"] == canonical(PROTOCOL)
        and report["source_sha256"] == manifest["code_sha256"]
        and report["input_manifest_sha256"] == sha256(root / WORK / "inputs.json")
        and report["selection_sha256"] == sha256(root / REPORTS / "selected.json")
        and report["composition"] == winner, "Invalid promotion evidence")
    checks = screen_gates(report["results"]) if stage == "screen" else stress_gates(report["results"])
    require(report["decision_checks"] == checks and all(checks.values()) and report["passed"] is True,
            f"{stage} gate failed: residual recipe closed")
    verify_files(root, report["files_sha256"])
    cases = report["results"].items() if stage == "screen" else [("five_day",report["results"])]
    for case,result in cases:
        require(set(result["member_models"]) == set(COMPOSITIONS[winner]), "Incomplete fitted members")
        require(result["predictions_sha256"] == report["files_sha256"][result["predictions"]], "Unbound promotion predictions")
        for member,kind in COMPOSITIONS[winner].items():
            fit = result["member_models"][member]
            require(fit["kind"] == kind and fit["parameters"] == manifest["parameters"][member][case]
                and fit["combiner"] in report["files_sha256"], "Combiner recipe differs")
            require(fit["adapters"] == {f:adapter_path(member,kind,f).as_posix() for f in CASE_FOLDS[case]}, "Wrong prefix adapters")
            for fold,path in fit["adapters"].items():
                require(path in report["files_sha256"], "Unbound prefix adapter")
                validate_adapter(root,manifest,member,kind,fold)
    return report
