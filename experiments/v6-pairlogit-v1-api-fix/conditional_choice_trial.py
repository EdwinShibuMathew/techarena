"""Verify, freeze and run one fixed conditional-choice expert against V6.

verify/prepare do not train. run is for Edwin's existing CUDA environment only.
No dependency installation, original-file mutation, evaluation data or export.
"""

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np

import event_scale_diagnostic as audit
import attention_allocation_test as shared


EXPERIMENT = "v6-conditional-choice-v1"
MODEL_FILE = "conditional_choice_model.py"
HELPERS = {"event_scale_diagnostic.py": "0007bc7db9fda6ab689bed23d44b4b06bb0502ff3eee0c0bd3da04f8902293d5",
           "attention_allocation_test.py": "188899687e31a959e8eb805c8d7a186b7d9fa3e3e3934e1dab37640ffa736b13"}
LEGACY = {"run_row_auc_trial.py": "195e16c5d258f1ff47d14542d392ec0016c2b31dee285da7cdf5011376f19281",
          "train_final_models.py": "ccba6e04a614edf954e4424007d80ed658d4ee746f10ec116c256d70cedb8dd3",
          "build_submission.py": "9b32f4d043d9765598245114d288c4b2d89a427e336a551e59c4bbdd6175f14d"}
TRAIN_SHA = "a8fcada9cf0d08f900824594302b2ac45b90254d1780c9d915206ec96cc062b8"
TRAIN_EVENTS, TRAIN_CLICKS, FULL_POSITIVES = 686824, 432023, 1009707
CASES = {"primary": {**audit.SPECS["primary"], "fold": "jan25"},
         "secondary": {**audit.SPECS["secondary"], "fold": "jan26"}}
RECIPE = {
    "experiment": EXPERIMENT, "dimension": 8, "seed": 42,
    "initialization": "hash-bound same-prefix epoch260 seed42 preference checkpoint; no new preference fitting",
    "scorer": "existing user/item dot product, item bias, additive eight-context item effect; event-common terms omitted",
    "objective": "ten-candidate CE mean over clicked events in each full fitting minibatch; no-click-only batches regularize only",
    "epochs": 8, "batch_size": 2048, "optimizer": "fresh Adam defaults", "learning_rate": .0015,
    "shuffle": "dedicated torch.Generator(device) seed42; full prefix stream each epoch",
    "anchor_regularization": .1, "item_context_regularization": .05,
    "regularization_population": "complete fitting minibatch, including no-click events",
    "fallback": "prefix-only clicks, Ridge(alpha=1), replace unseen-with-history user vector only at inference",
    "exposure": {"power": .5, "smoothing": 20, "population": "prefix candidate displays only"},
    "choice": "conditional softmax after exposure correction, computed stably in log space",
    "candidate": "0.85*V6 + 0.15*sum(V6,axis=1)*expert_choice for ALL target events",
    "event_sum_tolerance": {"rtol": 0, "atol": 1e-14},
    "bootstrap": {"resamples": 1000, "seed": 2026, "unit": "whole first event per user, preserving ten candidates",
                  "pairing": "same event multiplicities", "interval": "95% percentile"},
    "gate": "primary paired interval lower > 0 AND secondary global row AUC change >= 0",
    "stop": "45 minutes total run budget; failure closes this recipe; no weight, epoch, seed or routing variants",
    "next_stage": "same fixed recipe's five-day comparison before any full-data promotion",
    "early_stopping": False, "combiner_refit": False, "submission_export": False,
    "status": "reused retrospective development dates; intervals not selection-adjusted",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_raw(path):
    with np.load(path, allow_pickle=False) as saved:
        raw = {k: saved[k] for k in ("u", "i", "t", "tb", "dt", "y", "h", "ht")}
    audit.require(raw["u"].shape == raw["t"].shape == (TRAIN_EVENTS,), "Wrong raw event count")
    audit.require(raw["i"].shape == raw["y"].shape == (TRAIN_EVENTS, 10), "Wrong candidate shapes")
    audit.require(raw["h"].shape == raw["ht"].shape == (TRAIN_EVENTS, 20), "Wrong history shapes")
    audit.require(np.isin(raw["y"], [0, 1]).all() and (raw["y"].sum(1) <= 1).all()
                  and int(raw["y"].sum()) == TRAIN_CLICKS, "Wrong labels")
    audit.require(((raw["t"] >= audit.JAN20) & (raw["t"] < audit.END)).all(), "Wrong raw timeline")
    audit.require(np.isin(raw["tb"], [0, 1, 2, 3]).all() and np.isin(raw["dt"], [0, 1]).all(), "Invalid context")
    audit.require(raw["u"].dtype.kind in "iu" and (raw["u"] >= 0).all(), "Invalid user IDs")
    audit.require(((raw["i"] >= 0) & (raw["i"] < 50000)).all(), "Invalid candidate IDs")
    audit.require(((raw["h"] >= -1) & (raw["h"] < 50000)).all(), "Invalid history IDs")
    present = raw["h"] >= 0
    audit.require((raw["ht"][present] < np.broadcast_to(raw["t"][:, None], present.shape)[present]).all(),
                  "History contains a current/future click")
    return raw


def first_users(raw, cutoff):
    eligible = np.flatnonzero(raw["t"] < cutoff)
    ordered = eligible[np.argsort(raw["t"][eligible], kind="stable")]
    _, where = np.unique(raw["u"][ordered], return_index=True)
    return ordered[np.sort(where)]


def multiset(users, items, times):
    order = np.lexsort((times, items, users))
    return np.column_stack((users[order], items[order], times[order]))


def prefix_examples(raw, full, cutoff, n_users):
    """Preserve full-cache order but independently prove its fitting multiset.

    Old timestamps alone do not permit using a future-only user's history.
    Repeated click examples remain repeated. No deduplication is performed.
    """
    fit_idx = np.flatnonzero(raw["t"] < cutoff)
    audit.require(len(fit_idx) > 0 and cutoff <= int(full["cutoff"]), "Invalid prefix")
    size = max(n_users, int(full["n_users"]), int(raw["u"].max()) + 1)
    seen = np.bincount(raw["u"][fit_idx], minlength=size) > 0
    keep = seen[full["users"]] & (full["timestamps"] < cutoff)
    positive = {k: full[k][keep] for k in ("users", "items", "timestamps")}
    first = first_users(raw, cutoff)
    h, ht = raw["h"][first], raw["ht"][first]
    present = h >= 0
    old_users = np.broadcast_to(raw["u"][first, None], h.shape)[present]
    rows, columns = np.nonzero(raw["y"][fit_idx])
    clicked_rows = fit_idx[rows]
    expected_users = np.r_[old_users, raw["u"][clicked_rows]]
    expected_items = np.r_[h[present], raw["i"][clicked_rows, columns]]
    expected_times = np.r_[ht[present], raw["t"][clicked_rows]]
    audit.require(np.array_equal(multiset(positive["users"], positive["items"], positive["timestamps"]),
                                 multiset(expected_users, expected_items, expected_times)),
                  "Prefix clicks differ from initial histories plus observed prefix clicks")
    audit.require((positive["users"] < n_users).all() and (positive["users"] >= 0).all(), "Positive user exceeds checkpoint table")
    audit.require((positive["timestamps"] < cutoff).all() and seen[positive["users"]].all(), "Prefix fitting boundary violated")
    positive.update(cutoff=np.asarray(cutoff), n_users=np.asarray(n_users), n_items=np.asarray(50000))
    proof = {"cutoff": cutoff, "fitting_events": len(fit_idx), "fitting_users": int(seen.sum()),
             "initial_history_clicks": int(present.sum()), "observed_clicks": len(clicked_rows),
             "positive_examples": len(positive["users"]), "multiset_proof_passed": True,
             "future_only_users_excluded": True, "genuine_repeats_retained": True}
    return positive, proof


def load_full_positive(path):
    with np.load(path, allow_pickle=False) as saved:
        full = {k: saved[k] for k in ("users", "items", "timestamps", "n_users", "n_items", "cutoff")}
    n = len(full["users"])
    audit.require(n == FULL_POSITIVES and full["items"].shape == full["timestamps"].shape == (n,), "Wrong full positive cache")
    audit.require(int(full["cutoff"]) == audit.END and int(full["n_items"]) == 50000, "Full positive cache header differs")
    audit.require(full["users"].dtype.kind in "iu" and full["items"].dtype.kind in "iu", "Invalid preference IDs")
    audit.require(((full["users"] >= 0) & (full["users"] < int(full["n_users"]))).all(), "Invalid positive users")
    audit.require(((full["items"] >= 0) & (full["items"] < 50000)).all(), "Invalid positive items")
    audit.require((full["timestamps"] < audit.END).all(), "Future positive timestamps")
    return full


def load_checkpoint(path, cutoff, positive):
    import torch
    saved = torch.load(path, map_location="cpu", weights_only=True)
    for key, want in {"dimension": 8, "epoch": 260, "fine_tuning_initial_epoch": 196,
                      "fine_tuning_learning_rate": .003, "fine_tuning_sampling_seed": 1042,
                      "n_items": 50000}.items():
        audit.require(saved.get(key) == want, f"Checkpoint {key} differs: {path}")
    for key, want in {"cutoff": cutoff, "seed": 42, "stage": "preference"}.items():
        if key in saved:
            audit.require(saved[key] == want, f"Conflicting checkpoint {key}: {path}")
    state = saved["model_state"]
    audit.require(set(state) == {"user.weight", "item.weight", "item_bias.weight", "active_users"}, "Unexpected preference state")
    n_users = int(saved["n_users"])
    expected = {"user.weight": (n_users, 8), "item.weight": (50000, 8),
                "item_bias.weight": (50000, 1), "active_users": (n_users,)}
    for key, shape in expected.items():
        audit.require(tuple(state[key].shape) == shape, f"Wrong tensor shape: {key}")
        if key != "active_users":
            audit.require(state[key].dtype == torch.float32 and bool(torch.isfinite(state[key]).all()), "Invalid preference tensor")
    audit.require(state["active_users"].dtype == torch.bool, "Invalid active-user mask")
    if positive is not None:
        known = np.bincount(positive["users"], minlength=n_users) > 0
        audit.require(np.array_equal(state["active_users"].numpy(), known), "Checkpoint active users disagree with proved prefix clicks")
    header = {k: saved[k] for k in ("dimension", "epoch", "n_users", "n_items", "fine_tuning_initial_epoch",
                                    "fine_tuning_learning_rate", "fine_tuning_sampling_seed")}
    header["explicit_metadata"] = {k: saved[k] for k in ("seed", "cutoff", "stage") if k in saved}
    header["missing_explicit_metadata"] = [k for k in ("seed", "cutoff", "stage") if k not in saved]
    header["identity_basis"] = "original V6 trial checkpoint hash + fine-tune protocol/source + proved prefix active-user mask"
    return state, header


def verify_hashes(inputs, check=lambda: None):
    for name, expected in inputs.items():
        audit.require(audit.sha256(name) == expected, f"Input changed: {name}")
        check()


def verify(args):
    import torch
    from sklearn.linear_model import Ridge  # availability check only; no fitting
    repo = args.repo.resolve(strict=True)
    work = audit.inside(repo / "artifacts/experiments" / EXPERIMENT, repo)
    audit.require(not work.exists(), f"Verification already exists; inspect it instead of overwriting: {work}")
    source = audit.inside(__file__, repo)
    inputs = {str(source): audit.sha256(source)}
    for name, expected in HELPERS.items():
        inputs[str(source.with_name(name))] = expected
    model_source = source.with_name(MODEL_FILE)
    inputs[str(model_source)] = audit.sha256(model_source)
    for name, expected in LEGACY.items():
        inputs[str(repo / "legacy" / name)] = expected
    raw_path = repo / "data/cache/train_events.npz"
    inputs[str(raw_path)] = TRAIN_SHA
    bundle_path = repo / "artifacts/v6/manifest.json"
    bundle = read_json(bundle_path)
    audit.require(bundle["expected_submission_sha256"] == "c386d545b3746e05dd515020c6185afda06cefdbcdf7d48df022454762fb378c",
                  "Bundle is not the submitted V6")
    inputs[str(bundle_path)] = audit.sha256(bundle_path)
    full_path = repo / "data/cache/preference_training.npz"
    inputs[str(full_path)] = bundle["files"]["data/cache/preference_training.npz"]["sha256"]
    row_trial = repo / "local/archive/techarena-row-auc-trial"
    row_protocol_path = row_trial / "outputs/frozen_protocol.json"
    row_protocol = read_json(row_protocol_path)
    inputs[str(row_protocol_path)] = audit.sha256(row_protocol_path)
    audit.require(row_protocol["updated_members"] == ["dim8_seed42", "dim8_seed43"]
                  and row_protocol["preference_epoch"] == 260, "Unexpected original row trial")
    audit.require(row_protocol["input_sha256"]["/home/edwin/projects/techarena-final/src/run_row_auc_trial.py"]
                  == LEGACY["run_row_auc_trial.py"], "Original row-trial source identity differs")
    fine = repo / "local/archive/techarena-preference-finetune-trial"
    fine_protocol_path = fine / "outputs/frozen_protocol.json"
    fine_protocol = read_json(fine_protocol_path)
    inputs[str(fine_protocol_path)] = audit.sha256(fine_protocol_path)
    audit.require(fine_protocol["preference_total_epochs"] == 260
                  and fine_protocol["fine_tuning_sampling_seeds"] == [1042, 1043], "Unexpected original preference schedule")
    v6_paths, cp_paths, prefix_sources = {}, {}, {}
    for key, spec in CASES.items():
        fold = spec["fold"]
        cp = audit.inside(fine / f"dim8_seed42/{fold}/work/preference_best.pt", repo)
        old = f"/home/edwin/projects/techarena-preference-finetune-trial/dim8_seed42/{fold}/work/preference_best.pt"
        inputs[str(cp)] = row_protocol["input_sha256"][old]
        rel = f"dim8_seed42/{fold}/src/train_preferences.py"
        pref_source = fine / rel
        inputs[str(pref_source)] = fine_protocol["source_sha256"][rel]
        text = pref_source.read_text(encoding="utf-8")
        ast.parse(text)
        audit.require("torch.manual_seed(42)" in text and "torch.manual_seed(1042)" in text
                      and f"split_{fold}.npz" in text and "max_epochs = 260" in text,
                      f"Fold/seed trainer source proof differs: {fold}")
        v6 = row_trial / "work" / spec["filename"]
        inputs[str(v6)] = spec["sha256"]
        cp_paths[key], v6_paths[key], prefix_sources[key] = str(cp), str(v6), str(pref_source)
    for path in inputs:
        audit.inside(path, repo)
    verify_hashes(inputs)
    raw, full = load_raw(raw_path), load_full_positive(full_path)
    # Independently prove that the active full cache is the expected raw-data
    # multiset before filtering it, without restoring archived prefix caches.
    _, full_proof = prefix_examples(raw, full, audit.END, int(full["n_users"]))
    results, positives = {}, {}
    for key, spec in CASES.items():
        state, header = load_checkpoint(cp_paths[key], spec["begin"], None)
        audit.require(int(raw["u"].max()) < header["n_users"] <= 54456, "Checkpoint user table cannot cover target users")
        positive, proof = prefix_examples(raw, full, spec["begin"], header["n_users"])
        _, checked = load_checkpoint(cp_paths[key], spec["begin"], positive)
        audit.load_cohort(v6_paths[key], spec, raw)
        idx = np.load(v6_paths[key], allow_pickle=False)
        with idx:
            target_idx = idx["valid_idx"]
        fit_counts = np.bincount(raw["u"][raw["t"] < spec["begin"]], minlength=header["n_users"])
        cold = fit_counts[raw["u"][target_idx]] == 0
        proof.update(target_events=len(target_idx), unseen_target_events=int(cold.sum()),
                     unseen_with_history=int((cold & (raw["h"][target_idx] >= 0).any(1)).sum()))
        results[key] = {"checkpoint": checked, "checkpoint_sha256": inputs[cp_paths[key]],
                        "preference_source": prefix_sources[key], "prefix": proof}
        positives[key] = positive
        print(json.dumps({key: results[key]}, indent=2), flush=True)
        del state
    verify_hashes(inputs)
    work.mkdir(parents=True)
    generated = {}
    for key, positive in positives.items():
        path = work / f"{key}_prefix_clicks.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **positive)
        generated[key] = str(path)
        inputs[str(path)] = audit.sha256(path)
    report = {"verified": True, "experiment": EXPERIMENT, "repo": str(repo), "inputs_sha256": inputs,
              "training_cache": str(raw_path), "full_positive_cache": str(full_path), "full_multiset_proof": full_proof,
              "checkpoints": cp_paths, "prefix_clicks": generated, "v6_paths": v6_paths, "cases": results,
              "runtime": {"python": sys.version, "torch": torch.__version__, "cuda_available": torch.cuda.is_available()},
              "training_performed": False, "original_artifacts_modified": False,
              "metadata_limit": "Absent seed/cutoff fields are reported, never fabricated; original hash/source evidence supplies identity."}
    audit.write_new(work / "verification.json", report)
    print(f"Verification passed: {work / 'verification.json'}. No training performed.", flush=True)


def prepare(args):
    repo = args.repo.resolve(strict=True)
    work = audit.inside(repo / "artifacts/experiments" / EXPERIMENT, repo)
    verification_path = work / "verification.json"
    report = read_json(verification_path)
    audit.require(report["verified"] is True and report["repo"] == str(repo), "Invalid verification report")
    inputs = dict(report["inputs_sha256"])
    inputs[str(verification_path)] = audit.sha256(verification_path)
    audit.require(inputs.get(str(Path(__file__).resolve())) == audit.sha256(__file__), "Source changed since verification")
    verify_hashes(inputs)
    audit.write_new(work / "inputs.json", {"experiment": EXPERIMENT, "repo": str(repo), "protocol": RECIPE,
        "protocol_sha256": audit.canonical_hash(RECIPE), "inputs_sha256": inputs,
        "verification": str(verification_path), "checkpoints": report["checkpoints"],
        "prefix_clicks": report["prefix_clicks"], "v6_paths": report["v6_paths"],
        "training_cache": report["training_cache"], "training_performed": False})
    print(f"Fixed recipe prepared: {work / 'inputs.json'}. No training performed.", flush=True)


def run(args):
    # This command is deliberately separate from verification/preparation.
    import torch
    from conditional_choice_model import fit, predict
    start = time.monotonic()

    def check():
        if time.monotonic() - start > 2700:
            raise TimeoutError("45-minute total run budget exceeded; preserve the log and stop")

    audit.require(torch.cuda.is_available(), "CUDA unavailable; use the existing techarena environment, do not substitute CPU training")
    torch.set_num_threads(4)
    path = args.manifest.resolve(strict=True)
    manifest_sha = audit.sha256(path)
    frozen = read_json(path)
    repo = Path(frozen["repo"]).resolve(strict=True)
    work = audit.inside(repo / "artifacts/experiments" / EXPERIMENT, repo)
    audit.inside(path, work)
    output = audit.inside(repo / "docs/results" / EXPERIMENT / "conditional_choice.json", repo)
    audit.require(not output.exists(), f"Refusing to overwrite: {output}")
    audit.require(frozen["experiment"] == EXPERIMENT and frozen["protocol"] == RECIPE
                  and frozen["protocol_sha256"] == audit.canonical_hash(RECIPE), "Frozen recipe changed")
    inputs = frozen["inputs_sha256"]
    for name in inputs:
        audit.inside(name, repo)
    source = Path(__file__).resolve()
    audit.require(inputs.get(str(source)) == audit.sha256(source), "Running source not frozen")
    for name in list(HELPERS) + [MODEL_FILE]:
        p = source.with_name(name)
        audit.require(inputs.get(str(p)) == audit.sha256(p), f"Running dependency not frozen: {name}")
    for name in ("training_cache", "verification"):
        audit.require(str(audit.inside(frozen[name], repo)) in inputs, f"Unfrozen {name}")
    audit.require(inputs[frozen["training_cache"]] == TRAIN_SHA, "Wrong training cache hash")
    verified = read_json(frozen["verification"])
    audit.require(verified["verified"] is True and verified["repo"] == str(repo), "Verification identity changed")
    for group in ("checkpoints", "prefix_clicks", "v6_paths", "training_cache"):
        audit.require(frozen[group] == verified[group], f"Prepared {group} differs from verified inputs")
    audit.require(set(frozen["checkpoints"]) == set(frozen["prefix_clicks"]) == set(frozen["v6_paths"]) == set(CASES), "Wrong cases")
    for key, spec in CASES.items():
        for group in ("checkpoints", "prefix_clicks", "v6_paths"):
            audit.require(str(audit.inside(frozen[group][key], repo)) in inputs, f"Unfrozen {group}/{key}")
        audit.require(inputs[frozen["v6_paths"][key]] == spec["sha256"], "Wrong V6 comparator")
    verify_hashes(inputs, check)
    attempt = work / "run"
    audit.require(not attempt.exists(), f"Run already started; preserve and inspect existing artifacts: {attempt}")
    raw = load_raw(frozen["training_cache"])
    attempt.mkdir()
    new_artifacts, results = {}, {}
    for key, spec in CASES.items():
        with np.load(frozen["prefix_clicks"][key], allow_pickle=False) as saved:
            positive = {k: saved[k] for k in saved.files}
        audit.require(int(positive["cutoff"]) == spec["begin"]
                      and (positive["timestamps"] < spec["begin"]).all(), "Prefix click cutoff changed")
        state, header = load_checkpoint(frozen["checkpoints"][key], spec["begin"], positive)
        audit.require(int(positive["n_users"]) == header["n_users"] and int(positive["n_items"]) == 50000,
                      "Prefix click dimensions differ from the checkpoint")
        idx, y, baseline, _, _ = audit.load_cohort(frozen["v6_paths"][key], spec, raw)
        fit_idx = np.flatnonzero(raw["t"] < spec["begin"])
        print(f"Training fixed conditional-choice expert: {key}; {len(fit_idx)} prefix events", flush=True)
        model, records = fit(state, raw, fit_idx, torch.device("cuda"), check)
        check()
        model_path = attempt / f"{key}_choice.pt"
        checkpoint = {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                      "dimension": 8, "seed": 42, "epoch": 8, "cutoff": spec["begin"],
                      "protocol_sha256": frozen["protocol_sha256"], "input_manifest_sha256": manifest_sha,
                      "preference_sha256": inputs[frozen["checkpoints"][key]], "training_records": records}
        with model_path.open("xb") as stream:
            torch.save(checkpoint, stream)
        new_artifacts[str(model_path)] = audit.sha256(model_path)
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        expert, fallback, encoder = predict(model, raw, fit_idx, idx, positive, torch.device("cuda"), check)
        audit.require(all(torch.equal(before[k], v.detach().cpu()) for k, v in model.state_dict().items()),
                      "Inference changed model state")
        candidate, conservation = shared.allocation(baseline, expert, np.ones(len(idx), dtype=bool))
        # The reused allocation helper names its mask after the old attention
        # history rule. This new recipe allocates ALL events, so report that
        # policy explicitly rather than imply that empty histories are frozen.
        conservation["allocation_events"] = conservation.pop("eligible_events")
        conservation["events_excluded_from_allocation"] = conservation.pop("empty_history_events")
        conservation.pop("empty_history_scores_exactly_unchanged")
        conservation["actual_empty_history_events"] = int((raw["h"][idx] < 0).all(1).sum())
        conservation["allocation_policy"] = "all target events, including empty histories; existing cold fallback policy"
        predictions_path = attempt / f"{key}_predictions.npz"
        with predictions_path.open("xb") as stream:
            np.savez_compressed(stream, valid_idx=idx, labels=y, baseline=baseline, candidate=candidate,
                                expert_choice=expert, protocol_sha256=np.asarray(frozen["protocol_sha256"]),
                                coefficients=encoder["coefficients"], intercept=encoder["intercept"])
        new_artifacts[str(predictions_path)] = audit.sha256(predictions_path)
        metrics = audit.metrics(y, baseline, candidate)
        metrics["candidate_auc"] = metrics.pop("normalized_auc")
        metrics["event_sum_checks"] = conservation
        metrics["fallback"] = fallback
        metrics["decomposition"] = shared.allocation_decomposition(y, baseline, candidate)
        print(json.dumps({key: {k: v for k, v in metrics.items() if k != "decomposition"}}, indent=2), flush=True)
        metrics.update(audit.bootstrap(y, baseline, candidate, {"overall": np.ones(len(idx), dtype=bool)},
                                       check, resamples=1000)["overall"])
        metrics["training_records"] = records
        results[key] = metrics
        del model, state, checkpoint, before, expert, candidate, baseline
        torch.cuda.empty_cache()
        check()
    gate = {"primary_interval_entirely_positive": results["primary"]["paired_gain_95_percentile_interval"][0] > 0,
            "secondary_point_change_nonnegative": results["secondary"]["auc_change"] >= 0}
    verify_hashes(inputs, check)
    verify_hashes(new_artifacts, check)
    audit.require(audit.sha256(path) == manifest_sha, "Input manifest changed during run")
    check()
    report = {"complete": True, "protocol": RECIPE, "protocol_sha256": frozen["protocol_sha256"],
              "manifest_sha256": manifest_sha, "source_sha256": audit.sha256(source),
              "inputs_sha256": inputs, "new_artifacts_sha256": new_artifacts,
              "results": results, "decision_checks": gate, "supports_five_day_review": all(gate.values()),
              "input_hashes_rechecked_unchanged": True, "original_artifacts_modified": False,
              "evaluation_data_read": False, "submission_created": False, "automatic_promotion_performed": False,
              "interpretation": "Failure closes this fixed recipe; a pass requires identical-recipe five-day review, never automatic promotion.",
              "runtime": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                          "gpu": torch.cuda.get_device_name(0), "elapsed_seconds": round(time.monotonic() - start, 3)}}
    audit.write_new(output, report)
    print(json.dumps({"decision_checks": gate, "supports_five_day_review": all(gate.values()),
                      "report": str(output)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("verify", "prepare"):
        commands.add_parser(action).add_argument("--repo", required=True, type=Path)
    commands.add_parser("run").add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    {"verify": verify, "prepare": prepare, "run": run}[args.action](args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TimeoutError) as error:
        raise SystemExit(f"STOPPED: {error}") from error
