"""Jan21-only full-ensemble selection and immutable member-aware release evidence."""
from pathlib import Path
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/v8-seed44-row-auc"))
import seed44_core as prior

require, sha256, canonical = prior.require, prior.sha256, prior.canonical
read_json, write_new, inside, verify_files = prior.read_json, prior.write_new, prior.inside, prior.verify_files
FOLDS, DAY, V6_SHA, BASE_COLUMNS = prior.FOLDS, prior.DAY, prior.V6_SHA, prior.BASE_COLUMNS
CONFIG = prior.CONFIG
first_indices, temporal_check = prior.first_indices, prior.temporal_check
screen_gates, stress_gates = prior.screen_gates, prior.stress_gates
WORK = Path("artifacts/experiments/v8-robust-ensemble")
REPORTS = Path("docs/results/v8-robust-ensemble")
LIMITS = {"prepare": 600, "development": 1800, "screen": 3000, "stress": 2400, "finalize": 2700, "verify": 1500}
MEMBERS = {
    "dim8_seed42": {"dimension": 8, "seed": 42, "preference_epoch": 260, "weight": 2/9},
    "dim8_seed43": {"dimension": 8, "seed": 43, "preference_epoch": 260, "weight": 2/9},
    "dim8_seed44": {"dimension": 8, "seed": 44, "preference_epoch": 260, "weight": 2/9},
    "dim16_seed42": {"dimension": 16, "seed": 42, "preference_epoch": 196, "weight": 1/6},
    "dim16_seed43": {"dimension": 16, "seed": 43, "preference_epoch": 196, "weight": 1/6},
}
PAIR = ("dim16_seed42", "dim16_seed43")
COMPOSITIONS = {"A": PAIR, "B": ("dim8_seed44", *PAIR)}
CASE_FOLDS = prior.CASE_FOLDS
PROTOCOL = {
    "experiment": "v8-robust-ensemble", "hypothesis": "Mixed-loss dim16 members provide complementary full-ensemble gains that transfer from sparse early training",
    "members": MEMBERS, "compositions": {k: list(v) for k, v in COMPOSITIONS.items()},
    "development": {"day": "Jan21 only", "neural_cutoff": FOLDS["jan21"], "folds_hours": [[0,8,16],[0,16,24]],
        "reference": "fresh original V6-recipe combiners on each fold's fitting rows, full five-member ensemble",
        "eligibility": "both ensemble gains nonnegative, mean positive, mean no-click-negative contribution nonnegative",
        "selection": "maximize minimum fold ensemble gain; ties within 1e-12 prefer A (fewer changed members)",
        "promotion_labels_used": False},
    "training": {"event_objective": CONFIG, "event_epochs": 8, "batch_size": 2048,
        "optimizer": "fresh Adam lr0.0015", "anchor_regularization": .1, "effect_regularization": .05,
        "pair_seed": "neural seed + 3000", "preference_training": False, "early_stopping": False},
    "seed44": "reuse sealed prefix outputs from closed standalone trial; do not change its failed decision",
    "inference": prior.PROTOCOL["inference"], "combiner": prior.PROTOCOL["combiner"],
    "composition": "V6 + sum(original_member_weight*(updated_member-original_member)); no clipping/calibration/routing",
    "cohorts": prior.PROTOCOL["cohorts"], "final": "train selected full-data members; reuse each validated Jan25 combiner",
    "bootstrap": prior.PROTOCOL["bootstrap"], "gates": prior.PROTOCOL["gates"], "replay": prior.PROTOCOL["replay"],
    "validation_status": prior.PROTOCOL["validation_status"], "phase_limits_seconds": LIMITS,
    "total_compute_seconds": 14400, "competition_upload": False,
    "stop": "no eligible composition or failed integrity/gate/timeout closes this recipe; no fallback or automatic retries",
}
CODE_FILES = prior.CODE_FILES + ("experiments/v8-robust-ensemble/robust_core.py",
    "experiments/v8-robust-ensemble/train_member.py", "experiments/v8-robust-ensemble/runner.py",
    "tools/predict_v8_candidate.py")
PREFERENCE_HASHES = {
    "dim16_seed42": {"jan24": "280d2ccffee98b461270e4652042973a8c632df1da9a71f41add03483cb1fc4b",
        "jan25": "702b7a6f789a62940c41dc490413be26ec8c6e92b90402152cb804335132353d",
        "jan26": "c29c52dd2eddb10b6e6b7927ce02eb1f06ed588fe2875ba36c94939c8719b80d"},
    "dim16_seed43": {"jan24": "52bab7bcc3e2459b6b29b31206beec9fd2ccd214ca2eb2f826e79fde32d44357",
        "jan25": "bdfefccfa396babac92a37bf45ead75a24a5ecad0b9030899730912363036614",
        "jan26": "49f9e1d4329a917235fc2427ea8cf42006b40c22ac7b2ca31a00b3cbe3c58564"},
}
PREDICTION_HASHES = {
    "dim16_seed42": {"primary": "ad6eca36b7dd01b1bb2bbd0bfde4c19061fba940ac01a3671fecb02d0ac6d686",
        "secondary": "96a4f915e64df968d81e90b86d55e334377c46c4ca45b4090b9dcf4c13a59acf"},
    "dim16_seed43": {"primary": "563b68c3239f9e24b691ffcb1393519e53f7d489320de61650efd5e9bf59012b",
        "secondary": "746caee9033582c8c6cffe627071a48c97bb970cfee84c793a06c42c006f5570"},
}


class Budget(prior.Budget):
    def __enter__(self):
        self.state = read_json(self.path)
        require(self.phase not in self.state["phases"], "Preserve prior attempt; retries prohibited")
        self.limit = min(LIMITS[self.phase], 14400-self.state["used_seconds"])
        require(self.limit > 0, "Total compute budget exhausted")
        self.start = time.monotonic()
        self.state["phases"][self.phase] = {"status": "running", "limit_seconds": self.limit}
        self.persist()
        self.timer = threading.Timer(self.limit, self.expire)
        self.timer.daemon = True
        self.timer.start()
        return self


def neural_path(member, fold):
    require(member in COMPOSITIONS["B"] and fold in FOLDS, "Unknown member/prefix")
    return WORK / "models" / member / fold / "event.pt"


def original_ensemble(members):
    require(set(members) == set(MEMBERS), "Incomplete baseline ensemble")
    a,b,c,d,e = [np.asarray(members[k], dtype=np.float64) for k in MEMBERS]
    require(all(v.shape == a.shape and np.isfinite(v).all() for v in (a,b,c,d,e)), "Member shape/values differ")
    return (2/3)*((a+b+c)/3)+(1/3)*(.5*d+.5*e)


def compose(baseline, original, updated, composition):
    require(composition in COMPOSITIONS and set(updated) == set(COMPOSITIONS[composition]), "Wrong composition members")
    result = np.asarray(baseline, dtype=np.float64).copy()
    for member in COMPOSITIONS[composition]:
        before, after = np.asarray(original[member]), np.asarray(updated[member])
        require(before.shape == after.shape == result.shape and np.isfinite(before).all() and np.isfinite(after).all(), "Unaligned member predictions")
        result += MEMBERS[member]["weight"]*(after-before)
    require(np.isfinite(result).all() and ((result>=0)&(result<=1)).all(), "Invalid ensemble probabilities")
    return result


def development_folds(times):
    times = np.asarray(times)
    start = FOLDS["jan21"]
    require(times.ndim == 1 and len(times) and ((times>=start)&(times<start+DAY)).all(), "Development must use Jan21 only")
    result = []
    for middle,end in ((8,16),(16,24)):
        fit = times<start+middle*3600
        valid = (times>=start+middle*3600)&(times<start+end*3600)
        require(fit.any() and valid.any() and times[fit].max()<times[valid].min(), "Invalid forward fold")
        result.append((fit,valid))
    return result


def choose(results):
    require(set(results)==set(COMPOSITIONS), "Incomplete compositions")
    eligible = []
    for key in COMPOSITIONS:
        folds = results[key]["folds"]
        require(len(folds)==2 and [f["fold"] for f in folds]==[0,1], "Incomplete development folds")
        gains, negatives = [], []
        for fold in folds:
            require(all(np.isfinite(fold[k]) for k in ("baseline_auc","candidate_auc","auc_diff","no_click_contribution")), "Nonfinite development evidence")
            require(abs(fold["candidate_auc"]-fold["baseline_auc"]-fold["auc_diff"])<1e-12, "Inconsistent development gain")
            gains.append(fold["auc_diff"]); negatives.append(fold["no_click_contribution"])
        if min(gains)>=0 and np.mean(gains)>0 and np.mean(negatives)>=0:
            eligible.append((key,min(gains)))
    if not eligible:
        return None
    best = max(score for _,score in eligible)
    return next(key for key,score in eligible if best-score<=1e-12)


def pair_decomposition(labels, baseline, candidate, groups):
    """All positive-negative cross-group pairs; exact additive global AUC change."""
    require(np.isin(labels,[0,1]).all(),"Pair labels must be binary")
    groups = np.asarray(groups)
    y = np.asarray(labels).astype(bool)
    require(np.asarray(baseline).shape==np.asarray(candidate).shape==y.shape
        and np.isfinite(baseline).all() and np.isfinite(candidate).all(),"Invalid pair scores")
    require(y.ndim==2 and y.shape[1]==10 and np.asarray(groups).shape==(len(y),), "Invalid pair grouping")
    cells, total = {}, int(y.sum())*int((~y).sum())
    require(total>0, "Pair decomposition needs both classes")
    for positive_group in np.unique(groups):
        positive = y & (groups==positive_group)[:,None]
        for negative_group in np.unique(groups):
            negative = (~y) & (groups==negative_group)[:,None]
            npos,nneg = int(positive.sum()),int(negative.sum())
            if not npos or not nneg:
                continue
            values = []
            for p in (baseline,candidate):
                neg,pos = np.sort(p[negative]),p[positive]
                values.append(float((np.searchsorted(neg,pos,"left")+np.searchsorted(neg,pos,"right")).sum()/(2*npos*nneg)))
            cells[f"{positive_group}->{negative_group}"] = {"pairs": npos*nneg,
                "baseline_auc": values[0],"candidate_auc": values[1],"weighted_auc_change": (values[1]-values[0])*npos*nneg/total}
    return {"cells": cells,"reconstructed_change": sum(v["weighted_auc_change"] for v in cells.values())}


def load_manifest(root):
    path = root/WORK/"inputs.json"
    manifest = read_json(path)
    preparation = read_json(root/REPORTS/"preparation.json")
    require(preparation["complete"] is True and preparation["manifest_sha256"]==sha256(path), "Frozen manifest changed")
    require(manifest["protocol"]==PROTOCOL and manifest["protocol_sha256"]==canonical(PROTOCOL), "Recipe changed")
    require(set(manifest["code_sha256"])==set(CODE_FILES), "Incomplete source inventory")
    verify_files(root,manifest["code_sha256"]); verify_files(root,manifest["inputs_sha256"])
    prior.load_manifest(root)
    return manifest


def validate_new(root,manifest,member,fold):
    import torch
    expected = MEMBERS[member]
    saved = torch.load(root/neural_path(member,fold),map_location="cpu",weights_only=True)
    require(saved["epoch"]==8 and saved["dimension"]==expected["dimension"] and saved["seed"]==expected["seed"]
        and saved["cutoff"]==FOLDS[fold] and saved["stage"]=="event" and saved["member"]==member
        and saved["protocol_sha256"]==canonical(PROTOCOL) and saved["event_objective"]==CONFIG["objective"]
        and saved["pair_sampling_seed"]==expected["seed"]+3000 and saved["maximum_fitting_timestamp"]<FOLDS[fold]
        and saved["preference_sha256"]==manifest["inputs_sha256"][manifest["preferences"][member][fold]], "Neural recipe differs")
    require(all(bool(torch.isfinite(v).all()) for v in saved["model_state"].values() if v.is_floating_point()), "Nonfinite neural checkpoint")
    return saved


def selection(root,manifest):
    path = root/REPORTS/"development.json"
    report = read_json(path)
    require(report["complete"] is True and report["source_sha256"]==manifest["code_sha256"]
        and report["protocol_sha256"]==canonical(PROTOCOL) and report["input_manifest_sha256"]==sha256(root/WORK/"inputs.json"), "Development provenance differs")
    require(report["promotion_labels_used_for_selection"] is False,"Promotion labels used for selection")
    winner = choose(report["results"])
    require(winner is not None and report["winner"]==winner, "No qualifying composition; experiment closed")
    verify_files(root,report["files_sha256"])
    selected = read_json(root/REPORTS/"selected.json")
    require(selected=={"composition": winner,"members": list(COMPOSITIONS[winner]),"protocol_sha256": canonical(PROTOCOL),
        "development_sha256": sha256(path)}, "Selected composition changed")
    return winner


def validate_report(root,manifest,stage):
    winner = selection(root,manifest)
    report = read_json(root/REPORTS/(stage+".json"))
    require(report["complete"] is True and report["source_sha256"]==manifest["code_sha256"]
        and report["protocol_sha256"]==canonical(PROTOCOL) and report["input_manifest_sha256"]==sha256(root/WORK/"inputs.json")
        and report["composition"]==winner and report["selection_sha256"]==sha256(root/REPORTS/"selected.json"), "Evidence provenance differs")
    checks = screen_gates(report["results"]) if stage=="screen" else stress_gates(report["results"])
    require(all(checks.values()) and report["passed"] is True,f"{stage} gate failed: robust ensemble closed")
    verify_files(root,report["files_sha256"])
    results = report["results"].items() if stage=="screen" else [("five_day",report["results"])]
    for case,result in results:
        require(set(result["member_models"])==set(COMPOSITIONS[winner]),"Promotion member inventory differs")
        for member,fit in result["member_models"].items():
            if member in PAIR:
                require(fit["parameters"]==manifest["parameters"][member][case],"Promotion combiner recipe differs")
                expected = {neural_path(member,f).as_posix() for f in CASE_FOLDS[case]}
                require(set(fit["neural_sha256"])==expected,"Promotion neural inventory differs")
                verify_files(root,fit["neural_sha256"])
                for fold in CASE_FOLDS[case]:
                    validate_new(root,manifest,member,fold)
            else:
                require(fit["combiner"]==manifest["seed44_combiners"][case] and fit["reused_from_closed_trial"] is True,"Reused seed44 combiner differs")
    return report
