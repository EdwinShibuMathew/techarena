"""Train two frozen full-data row-ranking event models and export a candidate.

In techarena-final: self-test, setup, train, build.
All new files live in techarena-row-auc-final. V5 is read-only; no upload.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from row_auc_objective import CONFIG
from row_auc_final_utils import (SEEDS, FULL_CUTOFF, TRAIN_EVENTS, EVAL_EVENTS, EVAL_ROWS,
    VERSION, adapt_final_event, probabilities, verify_csv_probabilities, save_npz, csv_self_test)
from run_row_auc_trial import read_json, write_json, verify as verify_short, verify_training as verify_short_training
from run_row_auc_five_day import verify as verify_long, verify_training as verify_long_training, verify_artifacts
from run_expanded_combiner_trial import ensemble, WEIGHTS, load_references, read_raw
from submission_export import sha256, export_submission, validate_csv


def old_member_paths(root):
    p = root.parent
    return [p / "techarena-preference-finetune-trial/final/work" / f"seed{seed}/evaluation_predictions.npz" for seed in SEEDS] + [
        p / "techarena-seed44-trial/final/work/seed44/evaluation_predictions.npz",
        p / "techarena-dim16-trial/final/work/seed42/evaluation_predictions.npz",
        p / "techarena-dim16-trial/final/work/seed43/evaluation_predictions.npz"]


def check_v5(root):
    checks_path = root / "outputs/five_model_submission_checks.json"
    report = read_json(checks_path)
    assert report["version"] == "v5" and report["submission_ready"] and report["validation_pipeline_check_passed"]
    assert report["weights"] == WEIGHTS
    csv_path = root / "outputs/techarena_submission_v5.csv"
    assert sha256(csv_path) == report["submission_sha256"], "V5 CSV changed"
    members = []
    for path in old_member_paths(root):
        with np.load(path, allow_pickle=False) as saved:
            members.append(probabilities(saved["combiner_predictions"]))
    rebuilt = ensemble(members)
    with np.load(root / "work/five_model_evaluation.npz", allow_pickle=False) as saved:
        assert np.array_equal(saved["weights"], np.asarray(WEIGHTS))
        np.testing.assert_allclose(rebuilt, probabilities(saved["predictions"]), rtol=0, atol=1e-15)
    rowids = np.load(root / "data/cache/evaluation_rowids.npy", allow_pickle=False)
    assert rowids.shape == (EVAL_ROWS,) and rowids.dtype.kind == "U"
    verify_csv_probabilities(csv_path, rowids, rebuilt)
    example = root.parent / "techarena/data/raw/example_submission.csv"
    validate_csv(csv_path, example, EVAL_ROWS)
    return members, rebuilt, rowids


def verify(final, protocol):
    assert protocol["version"] == VERSION and protocol["event_objective"] == CONFIG
    assert protocol["member_weights"] == WEIGHTS and protocol["updated_seeds"] == SEEDS
    for path, expected in protocol["input_sha256"].items():
        assert sha256(Path(path)) == expected, f"Changed frozen input: {path}"
    verify_artifacts(final, protocol["prepared_sha256"])


def setup(root, final):
    if final.exists():
        verify(final, read_json(final / "outputs/frozen_protocol.json"))
        print("Final row-ranking workspace already prepared; continue with train/build.", flush=True)
        return
    import torch
    from build_submission import load_combiner
    from build_preference_finetuned_submission import check_full_seed
    short, long = root.parent / "techarena-row-auc-trial", root.parent / "techarena-row-auc-five-day"
    sp, lp = [read_json(folder / "outputs/frozen_protocol.json") for folder in [short, long]]
    sr, lr = read_json(short / "outputs/ensemble_trial_results.json"), read_json(long / "outputs/paired_results.json")
    verify_short(short, sp)
    verify_short_training(short, read_json(short / "outputs/training_results.json"))
    verify_artifacts(short, sr["artifact_sha256"])
    verify_long(long, lp)
    verify_long_training(long, read_json(long / "outputs/training_results.json"))
    verify_artifacts(long, lr["artifact_sha256"])
    assert sr["supported_for_five_day_check"] and sr["primary_paired_gain_95_percentile_interval"][0] > 0
    assert sr["jan26_secondary"]["auc_change"] >= 0
    assert lr["supported_for_full_data_review"] and lr["overall_gain_gate_passed"] and lr["late_window_gate_passed"]
    assert lr["five_day_overall"]["paired_gain_95_percentile_interval"][0] > 0 and lr["late_48_120h"]["auc_change"] >= 0
    assert lr["baseline_ensemble_and_original_members_reproduced"] and lr["unchanged_three_members_reused_exactly"]
    check_v5(root)
    inputs = [Path(__file__)] + [root / "src" / name for name in ["row_auc_final_utils.py", "row_auc_objective.py",
        "run_row_auc_trial.py", "run_row_auc_five_day.py", "run_expanded_combiner_trial.py", "expanded_combiner_features.py",
        "prepare_preference_finetune_trial.py", "build_submission.py", "submission_features.py", "train_final_models.py",
        "submission_export.py", "audit_v5.py", "build_preference_finetuned_submission.py"]]
    for folder, protocol, report_name, report in [(short, sp, "ensemble_trial_results.json", sr), (long, lp, "paired_results.json", lr)]:
        inputs += [folder / "outputs/frozen_protocol.json", folder / "outputs" / report_name, folder / "outputs/training_results.json"]
        inputs += [Path(p) for p in protocol["input_sha256"]]
        inputs += [folder / p for p in protocol["prepared_sha256"]]
        inputs += [folder / p for p in report["artifact_sha256"]]
    protected = [root / "outputs/techarena_submission_v5.csv", root / "outputs/five_model_submission_checks.json",
        root / "outputs/five_model_submission_manifest.json", root / "work/five_model_evaluation.npz"] + old_member_paths(root)
    inputs += protected
    inputs += [root / "data/cache" / name for name in ["train_events.npz", "preference_training.npz", "evaluation_events.npz", "evaluation_rowids.npy"]]
    example = root.parent / "techarena/data/raw/example_submission.csv"
    inputs.append(example)
    fine = root.parent / "techarena-preference-finetune-trial/final"
    fine_protocol = read_json(fine / "outputs/frozen_protocol.json")
    assert sha256(fine / "src/train_final_models.py") == fine_protocol["trainer_sha256"]
    inputs += [fine / "outputs/frozen_protocol.json", fine / "src/train_final_models.py"]
    preferences, combiners, prepared, training_inputs = {}, {}, {}, []
    for seed in SEEDS:
        checked = check_full_seed(fine, seed, fine_protocol)
        pref = Path(checked["preference"]["path"])
        inputs += [Path(v["path"]) for v in checked.values()] + [fine / "work" / f"seed{seed}/training_protocol.json"]
        initial = fine_protocol["initial_preference_checkpoints"][f"seed{seed}"]
        inputs.append(Path(initial["path"]))
        protected += [Path(v["path"]) for v in checked.values()]
        preferences[str(seed)] = {"path": str(pref), "sha256": sha256(pref)}
        training_inputs.append(pref)
        source = short / f"dim8_seed{seed}/jan26/src"
        texts = {"train_events.py": adapt_final_event((source / "train_events.py").read_text(encoding="utf-8"), seed),
                 "train_preferences.py": (source / "train_preferences.py").read_text(encoding="utf-8"),
                 "row_auc_objective.py": (source / "row_auc_objective.py").read_text(encoding="utf-8")}
        assert texts["row_auc_objective.py"] == (root / "src/row_auc_objective.py").read_text(encoding="utf-8")
        prepared[seed] = texts
        combiner = short / "work" / f"dim8_seed{seed}_jan25.cbm"
        model = load_combiner(combiner)
        assert model.tree_count_ == 545 and len(model.feature_names_) == 37
        combiners[str(seed)] = {"path": str(combiner), "sha256": sha256(combiner)}
        inputs.append(combiner)
        del model
    # Preserve retained full-data checkpoint files as well as their predictions.
    retained = [(root.parent / "techarena-seed44-trial/final/work/seed44", 44, 8),
                (root.parent / "techarena-dim16-trial/final/work/seed42", 42, 16),
                (root.parent / "techarena-dim16-trial/final/work/seed43", 43, 16)]
    for folder, seed, dimension in retained:
        for stage, epoch in [("preference", 260 if dimension == 8 else 196), ("event", 8)]:
            path = folder / f"{stage}_final.pt"
            saved = torch.load(path, map_location="cpu", weights_only=True)
            assert saved["seed"] == seed and saved["dimension"] == dimension and saved["epoch"] == epoch
            assert saved["stage"] == stage and saved["cutoff"] == FULL_CUTOFF
            assert saved["n_users"] == 54456 and saved["n_items"] == 50000
            inputs.append(path)
            protected.append(path)
            del saved
    with np.load(root / "data/cache/preference_training.npz", allow_pickle=False) as data:
        assert int(data["cutoff"]) == FULL_CUTOFF and len(data["users"]) == 1009707
        assert (data["timestamps"] < FULL_CUTOFF).all() and int(data["n_users"]) == 54456 and int(data["n_items"]) == 50000
    training_inputs.append(root / "data/cache/train_events.npz")
    input_hashes = {str(p.resolve(strict=True)): sha256(p) for p in inputs}
    protocol = {"version": VERSION, "event_objective": CONFIG, "updated_seeds": SEEDS, "member_weights": WEIGHTS,
        "updated_members": ["dim8_seed42", "dim8_seed43"], "unchanged_members": ["dim8_seed44", "dim16_seed42", "dim16_seed43"],
        "preference_epoch": 260, "preference_training_performed": False, "training_cutoff": FULL_CUTOFF,
        "combiner_rule": "reuse the successful short trial's January25 combiners; no refitting or weight search",
        "preference_sources": preferences, "combiner_sources": combiners, "example_path": str(example),
        "short_primary_auc": sr["two_day_primary"]["candidate_row_ranking_auc"],
        "short_secondary_auc": sr["jan26_secondary"]["candidate_row_ranking_auc"],
        "five_day_gain": lr["five_day_overall"]["auc_change"],
        "candidate_csv": "outputs/techarena_submission_v6_candidate.csv", "upload_performed": False,
        "evaluation_labels_used": False,
        "input_sha256": input_hashes, "training_inputs_sha256": {str(p.resolve(strict=True)): sha256(p) for p in training_inputs},
        "v5_protected_sha256": {str(p.resolve(strict=True)): sha256(p) for p in protected}, "prepared_sha256": {}}
    (final / "outputs").mkdir(parents=True)
    (final / "work").mkdir()
    for seed, texts in prepared.items():
        folder = final / f"seed{seed}"
        for part in ["src", "work", "data/cache"]:
            (folder / part).mkdir(parents=True)
        for name, text in texts.items():
            path = folder / "src" / name
            path.write_text(text, encoding="utf-8")
            protocol["prepared_sha256"][str(path.relative_to(final))] = sha256(path)
        (folder / "work/preference_best.pt").symlink_to(Path(preferences[str(seed)]["path"]).resolve(strict=True))
        (folder / "data/cache/train_events.npz").symlink_to((root / "data/cache/train_events.npz").resolve(strict=True))
    write_json(final / "outputs/frozen_protocol.json", protocol)
    verify(final, protocol)
    print("FINAL ROW-RANKING SETUP PASSED", flush=True)
    print(json.dumps({"workspace": str(final), "updated_seeds": SEEDS, "event_epochs": 8,
        "preference_training_performed": False, "combiners_reused": True, "v5_reconstructed_and_protected": True,
        "candidate_csv": str(final / protocol["candidate_csv"])}, indent=2), flush=True)


def check_event(final, protocol, seed):
    import torch
    path = final / f"seed{seed}/work/event_best.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    for key, expected in {"seed": seed, "epoch": 8, "stage": "event", "cutoff": FULL_CUTOFF,
        "dimension": 8, "n_users": 54456, "n_items": 50000, "event_objective": CONFIG["objective"],
        "pair_sampling_seed": seed + 3000, "preference_checkpoint_sha256": protocol["preference_sources"][str(seed)]["sha256"],
        "frozen_protocol_sha256": sha256(final / "outputs/frozen_protocol.json")}.items():
        assert saved[key] == expected, (seed, key)
    assert all(torch.isfinite(v).all() for v in saved["model_state"].values() if v.is_floating_point())
    return {"path": str(path), "sha256": sha256(path), "epoch": 8}


def verify_training(final, protocol, report):
    assert set(report["seeds"]) == {str(seed) for seed in SEEDS}
    for seed in SEEDS:
        actual = check_event(final, protocol, seed)
        assert report["seeds"][str(seed)] == actual, f"Changed final seed{seed} checkpoint"


def train(final, protocol):
    output = final / "outputs/training_results.json"
    if output.exists():
        verify_training(final, protocol, read_json(output))
        print("Both full-data event fits are complete; continue with build.", flush=True)
        return
    start, seeds = time.perf_counter(), {}
    for seed in SEEDS:
        folder = final / f"seed{seed}"
        if not (folder / "work/event_best.pt").exists():
            print(f"\nFull-data row-ranking event fit: seed{seed}, eight epochs", flush=True)
            subprocess.run([sys.executable, str(folder / "src/train_events.py")], cwd=folder, check=True)
        seeds[str(seed)] = check_event(final, protocol, seed)
    verify(final, protocol)
    report = {"seeds": seeds, "training_events": TRAIN_EVENTS, "preference_training_performed": False,
              "runtime_seconds": round(time.perf_counter() - start, 2)}
    write_json(output, report)
    print("\nFINAL ROW-RANKING TRAINING RESULTS", flush=True)
    print(json.dumps(report, indent=2), flush=True)


def validation_replay(root):
    import torch
    from build_submission import TARGET_FIELDS, FIT_FIELDS, neural_pipeline, load_combiner, predict_combiner, check_close
    raw = read_raw(root)
    refs = load_references(root, raw)
    short = root.parent / "techarena-row-auc-trial"
    results = {}
    for case, fold, day, metric_key in [("two_day", "jan25", "jan24", "two_day_primary"),
                                      ("jan26", "jan26", "jan25", "jan26_secondary")]:
        ref = refs[case]
        with np.load(short / "work" / f"{case}_ensemble_predictions.npz", allow_pickle=False) as saved:
            assert np.array_equal(saved["valid_idx"], ref["indices"]) and np.array_equal(saved["labels"], ref["labels"])
            expected = saved["candidate"]
            expected_members = [saved[f"new_dim8_seed{seed}"] for seed in SEEDS]
        members, differences = [], {}
        for number, seed in enumerate(SEEDS):
            folder = short / f"dim8_seed{seed}" / fold
            with np.load(folder / "data/cache" / f"split_{fold}.npz", allow_pickle=False) as data:
                fitting = data["train_idx"]
            cutoff = 1737763200 if fold == "jan25" else 1737849600
            assert np.array_equal(fitting, np.flatnonzero(raw["t"] < cutoff))
            X, names, *_ = neural_pipeline(folder / "work/event_best.pt", {k: raw[k][fitting] for k in FIT_FIELDS},
                {k: raw[k][ref["indices"]] for k in TARGET_FIELDS}, folder / "data/cache/preference_training.npz", torch.device("cuda"))
            model = load_combiner(short / "work" / f"dim8_seed{seed}_{day}.cbm")
            p = predict_combiner(model, X, names).reshape(-1, 10)
            differences[str(seed)] = check_close(p, expected_members[number], f"Row-ranking validation {case}/seed{seed}", rtol=0, atol=1e-15)
            members.append(p)
            del X, model
        candidate = ensemble(members + ref["members"][2:])
        check_close(candidate, expected, f"Row-ranking ensemble {case}", rtol=0, atol=1e-15)
        from audit_v5 import auc
        result_auc = auc(ref["labels"], candidate)
        report = read_json(short / "outputs/ensemble_trial_results.json")
        assert abs(result_auc - report[metric_key]["candidate_row_ranking_auc"]) < 1e-12
        results[case] = {"auc": result_auc, "member_max_abs_differences": differences, "passed": True}
    return raw, results


def build(root, final, protocol):
    output = final / "outputs/candidate_submission_checks.json"
    csv_path = final / protocol["candidate_csv"]
    if output.exists():
        report = read_json(output)
        verify_artifacts(final, report["artifact_sha256"])
        assert sha256(csv_path) == report["submission_sha256"]
        print("Saved candidate complete; displaying its verified checks.", flush=True)
        print("\nROW-RANKING CANDIDATE SUBMISSION CHECKS", flush=True)
        print(output.read_text(encoding="utf-8"), flush=True)
        return
    import torch
    from build_submission import TARGET_FIELDS, FIT_FIELDS, neural_pipeline, load_combiner, predict_combiner
    start = time.perf_counter()
    training_path = final / "outputs/training_results.json"
    training = read_json(training_path)
    verify_training(final, protocol, training)
    old_members, v5, rowids = check_v5(root)
    raw, replay = validation_replay(root)
    with np.load(root / "data/cache/evaluation_events.npz", allow_pickle=False) as saved:
        assert "y" not in saved.files and "clicked" not in saved.files
        target = {k: saved[k] for k in TARGET_FIELDS}
    assert len(target["u"]) == len(np.unique(target["u"])) == EVAL_EVENTS and target["i"].shape == (EVAL_EVENTS, 10)
    assert (target["t"] >= FULL_CUTOFF).all() and (target["t"] < FULL_CUTOFF + 7 * 86400).all()
    protocol_hash = sha256(final / "outputs/frozen_protocol.json")
    query_hashes = {name: sha256(root / "data/cache" / name) for name in ["evaluation_events.npz", "evaluation_rowids.npy"]}
    new_members, prediction_reports, artifacts = [], {}, [training_path]
    artifacts += [final / f"seed{seed}/work/event_best.pt" for seed in SEEDS]
    for seed in SEEDS:
        folder = final / f"seed{seed}/work"
        path = folder / "evaluation_predictions.npz"
        used = folder / "combiner_used.cbm"
        source = Path(protocol["combiner_sources"][str(seed)]["path"])
        expected_model_hash = training["seeds"][str(seed)]["sha256"]
        expected_combiner_hash = protocol["combiner_sources"][str(seed)]["sha256"]
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                assert saved["protocol_sha256"].item() == protocol_hash and saved["event_sha256"].item() == expected_model_hash
                assert saved["combiner_sha256"].item() == expected_combiner_hash
                assert json.loads(saved["query_hashes_json"].item()) == query_hashes
                prediction = probabilities(saved["combiner_predictions"])
                prediction_reports[str(seed)] = json.loads(saved["prediction_report_json"].item())
            assert sha256(used) == expected_combiner_hash
        else:
            model = load_combiner(source)
            print(f"Predicting evaluation with new full-data seed{seed}...", flush=True)
            X, names, raw_p, pipeline_p, encoder, info = neural_pipeline(folder / "event_best.pt",
                {k: raw[k] for k in FIT_FIELDS}, target, root / "data/cache/preference_training.npz", torch.device("cuda"), final_seed=seed)
            prediction = probabilities(predict_combiner(model, X, names))
            if used.exists():
                assert sha256(used) == expected_combiner_hash
            else:
                temporary = used.with_suffix(".cbm.tmp")
                shutil.copyfile(source, temporary)
                assert sha256(temporary) == expected_combiner_hash
                temporary.replace(used)
            save_npz(folder / "history_encoder.npz", coefficients=encoder.coef_, intercept=encoder.intercept_)
            save_npz(path, raw_predictions=raw_p, pipeline_predictions=pipeline_p, combiner_predictions=prediction,
                protocol_sha256=protocol_hash, event_sha256=expected_model_hash, combiner_sha256=expected_combiner_hash,
                query_hashes_json=json.dumps(query_hashes), prediction_report_json=json.dumps(info))
            prediction_reports[str(seed)] = info
            del X, model
            torch.cuda.empty_cache()
        new_members.append(prediction)
        artifacts += [path, used, folder / "history_encoder.npz"]
    candidate_members = new_members + old_members[2:]
    for original, retained in zip(old_members[2:], candidate_members[2:]):
        np.testing.assert_array_equal(original, retained)
    candidate = probabilities(ensemble(candidate_members))
    assert np.ptp(candidate) > 0 and not np.array_equal(candidate, v5), "Candidate is constant or identical to V5"
    pred_path = final / "work/candidate_evaluation.npz"
    save_npz(pred_path, predictions=candidate, weights=np.asarray(WEIGHTS), rowids_sha256=query_hashes["evaluation_rowids.npy"])
    artifacts.append(pred_path)
    verify(final, protocol)
    verify_training(final, protocol, training)
    # If export succeeded before an interrupted report write, validate and reuse
    # the existing candidate; never overwrite an existing competition CSV.
    if csv_path.exists():
        report = validate_csv(csv_path, Path(protocol["example_path"]), EVAL_ROWS)
        verify_csv_probabilities(csv_path, rowids, candidate)
        report.update(submission_ready=True, submission_path=str(csv_path), submission_sha256=sha256(csv_path),
                      submission_bytes=csv_path.stat().st_size)
    else:
        report = export_submission(rowids, candidate, Path(protocol["example_path"]), csv_path, expected_rows=EVAL_ROWS)
        verify_csv_probabilities(csv_path, rowids, candidate)
    verify(final, protocol)
    artifacts.append(csv_path)
    report.update({"version": VERSION, "weights": WEIGHTS, "updated_seeds": SEEDS,
        "validation_pipeline_check_passed": True, "validation_replay": replay,
        "unchanged_three_members_reused_exactly": True, "v5_files_unchanged": True,
        "mean_absolute_change_from_v5": float(np.mean(abs(candidate-v5))),
        "maximum_absolute_change_from_v5": float(np.max(abs(candidate-v5))),
        "upload_performed": False, "evaluation_labels_used": False,
        "runtime_seconds": round(time.perf_counter()-start, 2),
        "artifact_sha256": {str(p.relative_to(final)): sha256(p) for p in artifacts}})
    manifest = {"frozen_protocol_sha256": protocol_hash, "training": training,
        "prediction_reports": prediction_reports, "combiner_sources": protocol["combiner_sources"],
        "reused_predictions_sha256": {str(path): sha256(path) for path in old_member_paths(root)[2:]},
        "v5_protected_sha256": protocol["v5_protected_sha256"], "submission": report}
    write_json(final / "outputs/candidate_submission_manifest.json", manifest)
    report["artifact_sha256"]["outputs/candidate_submission_manifest.json"] = sha256(final / "outputs/candidate_submission_manifest.json")
    write_json(output, report)
    print("\nROW-RANKING CANDIDATE SUBMISSION CHECKS", flush=True)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["self-test", "setup", "train", "build"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.action == "self-test":
        for seed in SEEDS:
            source = root.parent / "techarena-row-auc-trial" / f"dim8_seed{seed}/jan26/src/train_events.py"
            adapt_final_event(source.read_text(encoding="utf-8"), seed)
        csv_self_test()
        print("Final trainer source checks passed: original update block and model classes identical; no validation/evaluation reads.", flush=True)
        return
    final = root.parent / "techarena-row-auc-final"
    if args.action == "setup":
        setup(root, final)
        return
    protocol = read_json(final / "outputs/frozen_protocol.json")
    verify(final, protocol)
    import torch
    assert torch.cuda.is_available(), "Activate techarena with working CUDA."
    torch.set_num_threads(4)
    if args.action == "train":
        train(final, protocol)
    else:
        build(root, final, protocol)


if __name__ == "__main__":
    main()
