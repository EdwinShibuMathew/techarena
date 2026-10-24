"""Verify the supported fine-tuning pipeline and export the V4 submission.

Run in techarena-final: python src/build_preference_finetuned_submission.py
Both new full-data seeds must have finished. Creates no new trained models.
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
from prepare_preference_finetune_trial import verify_inputs
from submission_export import export_submission, sha256

WEIGHTS = [1 / 3, 1 / 3, 1 / 6, 1 / 6]


def blend_four(first8, second8, first16, second16):
    members = [np.asarray(p, dtype=np.float64) for p in [first8, second8, first16, second16]]
    assert members[0].size > 0
    assert all(p.shape == members[0].shape for p in members), "Ensemble shapes differ"
    for p in members:
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid member probabilities"
    return (2 / 3) * (0.5 * members[0] + 0.5 * members[1]) + (1 / 3) * (0.5 * members[2] + 0.5 * members[3])


def auc(labels, predictions):
    return float(roc_auc_score(labels.ravel(), predictions.ravel()))


def read_predictions(path, key="combiner_predictions"):
    with np.load(path) as saved:
        result = saved[key].ravel()
    assert result.shape == (529390,), path
    assert np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all(), path
    return result


def check_full_seed(final, seed, frozen):
    folder = final / "work" / f"seed{seed}"
    trained = json.loads((folder / "training_protocol.json").read_text(encoding="utf-8"))
    initial = frozen["initial_preference_checkpoints"][f"seed{seed}"]
    assert sha256(Path(initial["path"])) == initial["sha256"]
    for name, expected in {
        "seed": seed, "dimension": 8, "preference_epochs": 260,
        "preference_initial_epochs": 196, "preference_extra_epochs": 64,
        "preference_learning_rate": 0.003, "fine_tuning_sampling_seed": seed + 1000,
        "event_epochs": 8, "n_users": 54456, "n_items": 50000,
        "cutoff": 1737936000, "training_events": 686824,
        "click_examples": 1009707, "evaluation_labels_used": False,
        "adam_moments_reused": True, "trainer_sha256": frozen["trainer_sha256"],
        "initial_preference_checkpoint": initial["path"],
        "initial_preference_checkpoint_sha256": initial["sha256"],
    }.items():
        assert trained[name] == expected, (seed, name)
    result = {}
    for stage, epoch in [("preference", 260), ("event", 8)]:
        path = folder / f"{stage}_final.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Seed {seed} {stage} training is incomplete: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        for name, expected in {
            "seed": seed, "dimension": 8, "stage": stage, "epoch": epoch,
            "cutoff": 1737936000, "n_users": 54456, "n_items": 50000,
            "fine_tuning_initial_epoch": 196, "fine_tuning_learning_rate": 0.003,
            "fine_tuning_sampling_seed": seed + 1000,
            "initial_preference_checkpoint_sha256": initial["sha256"],
        }.items():
            assert checkpoint[name] == expected, (path, name)
        del checkpoint
        result[stage] = {"path": str(path), "sha256": sha256(path), "epoch": epoch}
    return result


def main():
    start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    projects = root.parent
    trial = projects / "techarena-preference-finetune-trial"
    final = trial / "final"
    dim16_first = projects / "techarena-dim16-trial"
    dim16_second = projects / "techarena-dim16-seed43-trial"
    output_path = root / "outputs/techarena_submission_v4.csv"
    if output_path.exists():
        raise FileExistsError(f"V4 already exists: {output_path}")
    frozen_path = final / "outputs/frozen_protocol.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    protocol = json.loads((trial / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    report_path = trial / "outputs/ensemble_trial_results.json"
    trial_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert str(report_path) == frozen["validation_report_path"]
    assert sha256(report_path) == frozen["validation_report_sha256"]
    assert frozen["trainer_sha256"] == sha256(final / "src/train_final_models.py")
    assert frozen["original_trainer_sha256"] == sha256(root / "src/train_final_models.py")
    public_protocol = {k: v for k, v in protocol.items() if not k.endswith("sha256")}
    assert trial_report["protocol"] == public_protocol
    for name, expected in public_protocol.items():
        assert frozen[name] == expected, name
    assert trial_report["supported_for_further_confirmation"] is True
    assert trial_report["primary_paired_gain_95_percentile_interval"][0] > 0
    assert trial_report["jan26_secondary"]["auc_change"] >= 0
    assert frozen["weights"] == WEIGHTS
    assert frozen["preference_total_epochs"] == 260 and frozen["event_epochs"] == 8
    verify_inputs(trial, protocol)
    # Require both seeds before starting validation or prediction.
    full_checkpoints = {str(seed): check_full_seed(final, seed, frozen) for seed in [42, 43]}
    assert torch.cuda.is_available(), "Activate the techarena environment with working CUDA."
    torch.set_num_threads(4)
    device = torch.device("cuda")
    for name in ["train_events.npz", "preference_training.npz"]:
        assert (final / "data/cache" / name).resolve(strict=True) == (root / "data/cache" / name).resolve(strict=True)
    with np.load(root / "data/cache/train_events.npz") as saved:
        raw = {name: saved[name] for name in set(TARGET_FIELDS + FIT_FIELDS)}
    assert len(raw["u"]) == 686824 and (raw["t"] < 1737936000).all()

    # Verify that the reused dimension-16 files reconstruct the saved V3 exactly.
    dim16_final = dim16_first / "final/work"
    old_member_paths = [
        root / "work/seed42/evaluation_predictions.npz",
        root / "work/seed43/evaluation_predictions.npz",
        dim16_final / "seed42/evaluation_predictions.npz",
        dim16_final / "seed43/evaluation_predictions.npz",
    ]
    old_members = [read_predictions(path) for path in old_member_paths]
    previous_ensemble = read_predictions(root / "work/four_model_evaluation.npz", "predictions")
    check_close(blend_four(*old_members), previous_ensemble, "Reconstructed V3 evaluation predictions", rtol=0, atol=1e-15)

    combiners = {}
    for seed in [42, 43]:
        info = frozen["combiner_sources"][f"seed{seed}"]
        source = trial / "work" / f"dim8_seed{seed}_combiner_jan25.cbm"
        assert str(source) == info["path"] and sha256(source) == info["sha256"]
        combiners[seed] = load_combiner(source)

    validation = {}
    for case, fold, fitting_day, cutoff, metric_key, saved_name in [
        ("two_day_primary", "jan25", "jan24", 1737763200, "two_day_primary", "two_day"),
        ("jan26_secondary", "jan26", "jan25", 1737849600, "jan26_secondary", "jan26"),
    ]:
        print(f"Reproducing fine-tuning validation: {case}...", flush=True)
        with np.load(trial / "work" / f"{saved_name}_predictions.npz") as saved:
            indices, labels = saved["valid_idx"], saved["labels"]
            expected = saved["candidate"]
            expected_seed = {seed: saved[f"new_dim8_seed{seed}"] for seed in [42, 43]}
        assert np.array_equal(labels, raw["y"][indices])
        assert len(np.unique(raw["u"][indices])) == len(indices)
        assert (raw["t"][indices] >= cutoff).all()
        assert (raw["t"][indices] < 1737936000).all()
        reference_name = f"{saved_name}_ensemble_predictions.npz"
        with np.load(dim16_first / "work" / reference_name) as saved:
            assert np.array_equal(saved["valid_idx"], indices)
            first16 = saved["dim16"]
        with np.load(dim16_second / "work" / reference_name) as saved:
            assert np.array_equal(saved["valid_idx"], indices)
            second16 = saved["new_dim16"]
        updated = []
        seed_differences = {}
        for seed in [42, 43]:
            folder = trial / f"dim8_seed{seed}" / fold
            with np.load(folder / "data/cache" / f"split_{fold}.npz") as saved:
                fit_idx = saved["train_idx"]
            assert (raw["t"][fit_idx] < cutoff).all()
            checkpoint = torch.load(folder / "work/preference_best.pt", map_location="cpu", weights_only=True)
            assert checkpoint["dimension"] == 8 and checkpoint["epoch"] == 260
            assert checkpoint["fine_tuning_learning_rate"] == 0.003
            del checkpoint
            model = (combiners[seed] if fitting_day == "jan25" else
                     load_combiner(trial / "work" / f"dim8_seed{seed}_combiner_jan24.cbm"))
            X, names, _, _, _, _ = neural_pipeline(
                folder / "work/event_best.pt",
                {name: raw[name][fit_idx] for name in FIT_FIELDS},
                {name: raw[name][indices] for name in TARGET_FIELDS},
                folder / "data/cache/preference_training.npz", device,
            )
            predictions = predict_combiner(model, X, names).reshape(-1, 10)
            seed_differences[str(seed)] = check_close(predictions, expected_seed[seed], f"{case} seed{seed} predictions")
            updated.append(predictions)
            del X, model
        recreated = blend_four(updated[0], updated[1], first16, second16)
        difference = check_close(recreated, expected, f"{case} ensemble predictions")
        recreated_auc = auc(labels, recreated)
        expected_auc = trial_report[metric_key]["candidate_finetuned_auc"]
        assert abs(recreated_auc - expected_auc) <= 1e-6
        validation[case] = {"passed": True, "reproduced_auc": recreated_auc,
                            "saved_auc": expected_auc, "max_ensemble_probability_difference": difference,
                            "max_seed_probability_differences": seed_differences}
        print(json.dumps(validation[case], indent=2), flush=True)

    with np.load(root / "data/cache/evaluation_events.npz") as saved:
        assert "y" not in saved.files
        target = {name: saved[name] for name in TARGET_FIELDS}
    assert len(target["u"]) == len(np.unique(target["u"])) == 52939
    assert target["i"].shape == (52939, 10)
    assert (target["t"] >= 1737936000).all() and (target["t"] < 1738540800).all()
    rowids = np.load(root / "data/cache/evaluation_rowids.npy", allow_pickle=False)
    assert rowids.shape == (529390,)
    new_members = []
    final_reports = {}
    for seed in [42, 43]:
        print(f"Predicting evaluation with fine-tuned dim8 seed{seed}...", flush=True)
        folder = final / "work" / f"seed{seed}"
        X, names, raw_p, pipeline_p, encoder, prediction_report = neural_pipeline(
            folder / "event_final.pt", {name: raw[name] for name in FIT_FIELDS},
            target, root / "data/cache/preference_training.npz", device, final_seed=seed,
        )
        predictions = predict_combiner(combiners[seed], X, names)
        assert predictions.shape == (529390,)
        new_members.append(predictions)
        np.savez_compressed(folder / "evaluation_predictions.npz", raw_predictions=raw_p,
                            pipeline_predictions=pipeline_p, combiner_predictions=predictions)
        np.savez_compressed(folder / "history_encoder.npz", coefficients=encoder.coef_, intercept=encoder.intercept_)
        source = Path(frozen["combiner_sources"][f"seed{seed}"]["path"])
        destination = folder / "combiner_used.cbm"
        if destination.exists():
            assert sha256(destination) == sha256(source)
        else:
            shutil.copyfile(source, destination)
        final_reports[str(seed)] = prediction_report
        del X, raw_p, pipeline_p, encoder
    ensemble = blend_four(new_members[0], new_members[1], old_members[2], old_members[3])
    assert np.ptp(ensemble) > 0 and not np.array_equal(ensemble, previous_ensemble)
    np.savez_compressed(root / "work/preference_finetuned_evaluation.npz", predictions=ensemble, weights=np.array(WEIGHTS))
    verify_inputs(trial, protocol)
    report = export_submission(
        rowids, ensemble, projects / "techarena/data/raw/example_submission.csv",
        output_path, expected_rows=529390,
    )
    report.update({
        "version": "v4",
        "members": ["finetuned_dim8_seed42", "finetuned_dim8_seed43", "dim16_seed42", "dim16_seed43"],
        "weights": WEIGHTS, "validation_pipeline_check_passed": True,
        "both_full_data_seeds_checked": True,
        "primary_validation_auc": validation["two_day_primary"]["reproduced_auc"],
        "secondary_validation_auc": validation["jan26_secondary"]["reproduced_auc"],
        "runtime_seconds": round(time.perf_counter() - start, 2),
    })
    manifest = {
        "frozen_protocol_sha256": sha256(frozen_path),
        "validation_verification": validation,
        "final_checkpoints": full_checkpoints,
        "full_data_prediction_reports": final_reports,
        "combiner_sources": frozen["combiner_sources"],
        "previous_prediction_files_sha256": {str(path): sha256(path) for path in old_member_paths},
        "script_sha256": sha256(Path(__file__)), "submission": report,
    }
    (root / "outputs/preference_finetuned_submission_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (root / "outputs/preference_finetuned_submission_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nV4 SUBMISSION CHECKS", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
