"""Verify the supported five-model ensemble and export submission V5.

Run in techarena-final: python src/build_seed44_submission.py
Reuses the four V4 prediction files and predicts with full-data seed44.
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from build_submission import (
    TARGET_FIELDS, FIT_FIELDS, neural_pipeline, load_combiner,
    predict_combiner, check_close,
)
from build_preference_finetuned_submission import read_predictions, blend_four, check_full_seed
from evaluate_seed44_trial import candidate_probabilities
from prepare_seed44_trial import verify_inputs
from submission_export import export_submission, sha256

WEIGHTS = [2 / 9, 2 / 9, 2 / 9, 1 / 6, 1 / 6]


def check_seed44(final, frozen):
    folder = final / "work/seed44"
    protocol = json.loads((folder / "training_protocol.json").read_text(encoding="utf-8"))
    initial = folder / "preference_initial.pt"
    initial_hash = sha256(initial)
    for name, expected in {
        "seed": 44, "dimension": 8, "preference_initial_epochs": 196,
        "preference_extra_epochs": 64, "preference_epochs": 260,
        "preference_initial_learning_rate": 0.012, "preference_learning_rate": 0.003,
        "fine_tuning_sampling_seed": 1044, "adam_moments_reused": True,
        "event_epochs": 8, "event_learning_rate": 0.0015,
        "n_users": 54456, "n_items": 50000, "cutoff": 1737936000,
        "training_events": 686824, "click_examples": 1009707,
        "evaluation_labels_used": False, "trainer_sha256": frozen["trainer_sha256"],
        "initial_preference_checkpoint": str(initial),
        "initial_preference_checkpoint_sha256": initial_hash,
    }.items():
        assert protocol[name] == expected, name
    results = {}
    for name, filename, stage, epoch in [
        ("initial_preference", "preference_initial.pt", "preference", 196),
        ("preference", "preference_final.pt", "preference", 260),
        ("event", "event_final.pt", "event", 8),
    ]:
        path = folder / filename
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        for field, expected in {
            "seed": 44, "dimension": 8, "stage": stage, "epoch": epoch,
            "n_users": 54456, "n_items": 50000, "cutoff": 1737936000,
        }.items():
            assert checkpoint[field] == expected, (path, field)
        if name != "initial_preference":
            assert checkpoint["fine_tuning_initial_epoch"] == 196
            assert checkpoint["fine_tuning_learning_rate"] == 0.003
            assert checkpoint["fine_tuning_sampling_seed"] == 1044
            assert checkpoint["initial_preference_checkpoint_sha256"] == initial_hash
        del checkpoint
        results[name] = {"path": str(path), "sha256": sha256(path), "epoch": epoch}
    return results


def main():
    start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    projects = root.parent
    trial = projects / "techarena-seed44-trial"
    final = trial / "final"
    previous = projects / "techarena-preference-finetune-trial"
    dim16_first = projects / "techarena-dim16-trial"
    dim16_second = projects / "techarena-dim16-seed43-trial"
    output_path = root / "outputs/techarena_submission_v5.csv"
    if output_path.exists():
        raise FileExistsError(f"V5 already exists: {output_path}")
    protocol = json.loads((trial / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    frozen_path = final / "outputs/frozen_protocol.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    report_path = trial / "outputs/ensemble_trial_results.json"
    trial_report = json.loads(report_path.read_text(encoding="utf-8"))
    public_protocol = {k: v for k, v in protocol.items() if not k.endswith("sha256")}
    assert trial_report["protocol"] == public_protocol
    for name, expected in public_protocol.items():
        assert frozen[name] == expected, name
    assert str(report_path) == frozen["validation_report_path"]
    assert sha256(report_path) == frozen["validation_report_sha256"]
    assert sha256(final / "src/train_final_models.py") == frozen["trainer_sha256"]
    for section in ["locked_training_sources", "locked_trial_sources", "locked_baseline_sources"]:
        for name, expected in frozen[section].items():
            assert sha256(Path(name)) == expected, name
    verify_inputs(trial, protocol)
    assert trial_report["supported_for_further_confirmation"] is True
    assert trial_report["primary_paired_gain_95_percentile_interval"][0] > 0
    assert trial_report["jan26_secondary"]["auc_change"] >= 0
    assert frozen["weights"] == WEIGHTS
    assert frozen["neural_seed"] == 44 and frozen["preference_total_epochs"] == 260
    assert frozen["event_epochs"] == 8 and frozen["combiner_trees"] == 545
    checkpoint_report = check_seed44(final, frozen)

    v4_checks_path = root / "outputs/preference_finetuned_submission_checks.json"
    v4_checks = json.loads(v4_checks_path.read_text(encoding="utf-8"))
    assert v4_checks["version"] == "v4" and v4_checks["submission_ready"] is True
    assert v4_checks["validation_pipeline_check_passed"] is True
    assert sha256(root / "outputs/techarena_submission_v4.csv") == v4_checks["submission_sha256"]
    v4_frozen = json.loads((previous / "final/outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    assert sha256(previous / "final/src/train_final_models.py") == v4_frozen["trainer_sha256"]
    reused_checkpoint_checks = {str(seed): check_full_seed(previous / "final", seed, v4_frozen) for seed in [42, 43]}
    member_paths = [
        previous / "final/work/seed42/evaluation_predictions.npz",
        previous / "final/work/seed43/evaluation_predictions.npz",
        dim16_first / "final/work/seed42/evaluation_predictions.npz",
        dim16_first / "final/work/seed43/evaluation_predictions.npz",
    ]
    members = [read_predictions(path) for path in member_paths]
    v4_predictions = read_predictions(root / "work/preference_finetuned_evaluation.npz", "predictions")
    check_close(blend_four(*members), v4_predictions, "Reconstructed V4 evaluation probabilities", rtol=0, atol=1e-15)
    assert torch.cuda.is_available(), "Activate the techarena environment with working CUDA."
    torch.set_num_threads(4)
    device = torch.device("cuda")
    for name in ["train_events.npz", "preference_training.npz"]:
        assert (final / "data/cache" / name).resolve(strict=True) == (root / "data/cache" / name).resolve(strict=True)
    with np.load(root / "data/cache/train_events.npz") as saved:
        raw = {name: saved[name] for name in set(TARGET_FIELDS + FIT_FIELDS)}
    assert len(raw["u"]) == 686824 and (raw["t"] < 1737936000).all()
    combiner_path = trial / "work/combiner_jan25.cbm"
    assert str(combiner_path) == frozen["combiner_source"]["path"]
    assert sha256(combiner_path) == frozen["combiner_source"]["sha256"]
    combiner = load_combiner(combiner_path)

    validation = {}
    for case, filename, fold, cutoff, fitting_day in [
        ("two_day_primary", "two_day", "jan25", 1737763200, "jan24"),
        ("jan26_secondary", "jan26", "jan26", 1737849600, "jan25"),
    ]:
        print(f"Reproducing seed44 validation: {case}...", flush=True)
        with np.load(trial / "work" / f"{filename}_ensemble_predictions.npz") as saved:
            indices, labels = saved["valid_idx"], saved["labels"]
            expected_new, expected_candidate = saved["new_dim8_seed44"], saved["candidate"]
            expected_baseline = saved["baseline"]
        assert np.array_equal(labels, raw["y"][indices])
        assert len(np.unique(raw["u"][indices])) == len(indices)
        assert (raw["t"][indices] >= cutoff).all() and (raw["t"][indices] < 1737936000).all()
        with np.load(previous / "work" / f"{filename}_predictions.npz") as saved:
            assert np.array_equal(saved["valid_idx"], indices)
            first8, second8 = saved["new_dim8_seed42"], saved["new_dim8_seed43"]
            check_close(saved["candidate"], expected_baseline, f"{case} V4 baseline", rtol=0, atol=1e-15)
        with np.load(dim16_first / "work" / f"{filename}_ensemble_predictions.npz") as saved:
            assert np.array_equal(saved["valid_idx"], indices)
            first16 = saved["dim16"]
        with np.load(dim16_second / "work" / f"{filename}_ensemble_predictions.npz") as saved:
            assert np.array_equal(saved["valid_idx"], indices)
            second16 = saved["new_dim16"]
        check_close(blend_four(first8, second8, first16, second16), expected_baseline, f"{case} reconstructed V4", rtol=0, atol=1e-15)
        folder = trial / fold
        with np.load(folder / "data/cache" / f"split_{fold}.npz") as saved:
            fit_idx = saved["train_idx"]
        assert (raw["t"][fit_idx] < cutoff).all()
        model = combiner if fitting_day == "jan25" else load_combiner(trial / "work/combiner_jan24.cbm")
        X, names, _, _, _, _ = neural_pipeline(
            folder / "work/event_best.pt",
            {name: raw[name][fit_idx] for name in FIT_FIELDS},
            {name: raw[name][indices] for name in TARGET_FIELDS},
            folder / "data/cache/preference_training.npz", device,
        )
        new_prediction = predict_combiner(model, X, names).reshape(-1, 10)
        del X, model
        seed_difference = check_close(new_prediction, expected_new, f"{case} seed44 probabilities")
        recreated = candidate_probabilities(first8, second8, new_prediction, 0.5 * first16 + 0.5 * second16)
        ensemble_difference = check_close(recreated, expected_candidate, f"{case} five-model probabilities")
        reproduced_auc = float(roc_auc_score(labels.ravel(), recreated.ravel()))
        expected_auc = trial_report[case]["candidate_five_model_auc"]
        assert abs(reproduced_auc - expected_auc) <= 1e-6
        validation[case] = {"passed": True, "reproduced_auc": reproduced_auc,
                            "saved_auc": expected_auc, "max_seed44_probability_difference": seed_difference,
                            "max_ensemble_probability_difference": ensemble_difference}
        print(json.dumps(validation[case], indent=2), flush=True)

    with np.load(root / "data/cache/evaluation_events.npz") as saved:
        assert "y" not in saved.files
        target = {name: saved[name] for name in TARGET_FIELDS}
    assert len(target["u"]) == len(np.unique(target["u"])) == 52939
    assert target["i"].shape == (52939, 10)
    assert (target["t"] >= 1737936000).all() and (target["t"] < 1738540800).all()
    rowids = np.load(root / "data/cache/evaluation_rowids.npy", allow_pickle=False)
    assert rowids.shape == (529390,)
    print("Predicting evaluation with final seed44...", flush=True)
    folder = final / "work/seed44"
    X, names, raw_p, pipeline_p, encoder, prediction_report = neural_pipeline(
        folder / "event_final.pt", {name: raw[name] for name in FIT_FIELDS},
        target, root / "data/cache/preference_training.npz", device, final_seed=44,
    )
    new_prediction = predict_combiner(combiner, X, names)
    assert new_prediction.shape == (529390,)
    del X
    np.savez_compressed(folder / "evaluation_predictions.npz", raw_predictions=raw_p,
                        pipeline_predictions=pipeline_p, combiner_predictions=new_prediction)
    np.savez_compressed(folder / "history_encoder.npz", coefficients=encoder.coef_, intercept=encoder.intercept_)
    used_combiner = folder / "combiner_used.cbm"
    if used_combiner.exists():
        assert sha256(used_combiner) == sha256(combiner_path)
    else:
        shutil.copyfile(combiner_path, used_combiner)
    ensemble = candidate_probabilities(members[0], members[1], new_prediction, 0.5 * members[2] + 0.5 * members[3])
    assert ensemble.shape == (529390,) and np.ptp(ensemble) > 0
    assert not np.array_equal(ensemble, v4_predictions)
    np.savez_compressed(root / "work/five_model_evaluation.npz", predictions=ensemble, weights=np.array(WEIGHTS))
    verify_inputs(trial, protocol)
    report = export_submission(
        rowids, ensemble, projects / "techarena/data/raw/example_submission.csv",
        output_path, expected_rows=529390,
    )
    report.update({
        "version": "v5", "members": frozen["members"], "weights": WEIGHTS,
        "validation_pipeline_check_passed": True,
        "full_data_seed44_checked": True, "reused_v4_predictions_verified": True,
        "primary_validation_auc": validation["two_day_primary"]["reproduced_auc"],
        "secondary_validation_auc": validation["jan26_secondary"]["reproduced_auc"],
        "runtime_seconds": round(time.perf_counter() - start, 2),
    })
    manifest = {
        "frozen_protocol_sha256": sha256(frozen_path), "validation_verification": validation,
        "seed44_checkpoints": checkpoint_report, "reused_dim8_checkpoint_checks": reused_checkpoint_checks,
        "seed44_prediction_report": prediction_report,
        "reused_prediction_files_sha256": {str(path): sha256(path) for path in member_paths},
        "v4_submission_sha256": v4_checks["submission_sha256"],
        "combiner": frozen["combiner_source"], "script_sha256": sha256(Path(__file__)),
        "submission": report,
    }
    (root / "outputs/five_model_submission_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (root / "outputs/five_model_submission_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nV5 SUBMISSION CHECKS", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
