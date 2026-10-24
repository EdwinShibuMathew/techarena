"""Verify and export the selected equal-weight dim8/dim8/dim16 ensemble.

Run in techarena-final: python src/build_three_model_submission.py
Requires the completed final dim16 seed42 training and existing baseline outputs.
"""

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from submission_export import export_submission, sha256


def main():
    start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    projects = root.parent
    trial = projects / "techarena-dim16-trial"
    final16 = trial / "final"
    output_path = root / "outputs/submission_three_models.csv"
    if output_path.exists():
        raise FileExistsError(f"Three-model submission already exists: {output_path}")
    trial_report = json.loads((trial / "outputs/ensemble_trial_results.json").read_text(encoding="utf-8"))
    assert trial_report["supported_for_further_confirmation"] is True
    assert trial_report["protocol"]["planned_candidate_weights"] == [1 / 3, 1 / 3, 1 / 3]
    protocol = json.loads((final16 / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    assert protocol["dimension"] == 16 and protocol["seed"] == 42
    assert protocol["trainer_sha256"] == sha256(final16 / "src/train_final_models.py")
    trained_protocol = json.loads((final16 / "work/seed42/training_protocol.json").read_text(encoding="utf-8"))
    assert trained_protocol["dimension"] == 16
    assert trained_protocol["seed"] == 42
    assert trained_protocol["trainer_sha256"] == protocol["trainer_sha256"]

    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    sys.path.insert(0, str(trial / "src"))
    from dim16_pipeline import TARGET_FIELDS, FIT_FIELDS, neural_pipeline, load_combiner, predict_combiner, check_close

    with np.load(root / "data/cache/train_events.npz") as saved:
        raw = {name: saved[name] for name in set(TARGET_FIELDS + FIT_FIELDS)}
    assert len(raw["u"]) == 686824
    assert (raw["t"] < 1737936000).all()
    combiner_source = trial / "work/combiner_jan25.cbm"
    combiner = load_combiner(combiner_source)

    print("Reproducing the saved dimension-16 January 26 predictions...", flush=True)
    fold = trial / "jan26"
    with np.load(fold / "data/cache/split_jan26.npz") as saved:
        fit_idx, valid_idx = saved["train_idx"], saved["valid_idx"]
    assert (raw["t"][fit_idx] < 1737849600).all()
    assert (raw["t"][valid_idx] >= 1737849600).all()
    assert (raw["t"][valid_idx] < 1737936000).all()
    X, names, _, _, _, _ = neural_pipeline(
        fold / "work/event_best.pt",
        {name: raw[name][fit_idx] for name in FIT_FIELDS},
        {name: raw[name][valid_idx] for name in TARGET_FIELDS},
        fold / "data/cache/preference_training.npz", device,
    )
    validation16 = predict_combiner(combiner, X, names).reshape(-1, 10)
    del X
    with np.load(trial / "work/jan26_ensemble_predictions.npz") as saved:
        assert np.array_equal(saved["valid_idx"], valid_idx)
        single_difference = check_close(validation16, saved["dim16"], "Dimension-16 validation probabilities")
        recreated = (2 / 3) * saved["baseline"] + (1 / 3) * validation16
        ensemble_difference = check_close(recreated, saved["candidate"], "Three-model validation probabilities")
    reproduced_auc = float(roc_auc_score(raw["y"][valid_idx].ravel(), recreated.ravel()))
    expected_auc = trial_report["jan26_secondary"]["candidate_three_model_auc"]
    assert abs(reproduced_auc - expected_auc) <= 1e-6
    verification = {
        "passed": True,
        "reproduced_three_model_validation_auc": reproduced_auc,
        "saved_three_model_validation_auc": expected_auc,
        "max_dim16_probability_difference": single_difference,
        "max_ensemble_probability_difference": ensemble_difference,
    }
    print(json.dumps(verification, indent=2), flush=True)

    final_folder = final16 / "work/seed42"
    preference = torch.load(final_folder / "preference_final.pt", map_location="cpu", weights_only=True)
    assert preference["dimension"] == 16 and preference["epoch"] == 196
    assert preference["seed"] == 42 and preference["cutoff"] == 1737936000
    del preference
    with np.load(root / "data/cache/evaluation_events.npz") as saved:
        assert "y" not in saved.files
        target = {name: saved[name] for name in TARGET_FIELDS}
    assert len(target["u"]) == 52939
    assert len(np.unique(target["u"])) == 52939
    assert target["i"].shape == (52939, 10)
    assert (target["t"] >= 1737936000).all()
    assert (target["t"] < 1738540800).all()

    print("Predicting evaluation rows with the final dimension-16 model...", flush=True)
    X, names, raw_p, pipeline_p, encoder, final_report = neural_pipeline(
        final_folder / "event_final.pt",
        {name: raw[name] for name in FIT_FIELDS},
        target, root / "data/cache/preference_training.npz", device,
        final_seed=42,
    )
    p16 = predict_combiner(combiner, X, names)
    del X
    np.savez_compressed(final_folder / "evaluation_predictions.npz", raw_predictions=raw_p, pipeline_predictions=pipeline_p, combiner_predictions=p16)
    np.savez_compressed(final_folder / "history_encoder.npz", coefficients=encoder.coef_, intercept=encoder.intercept_)
    used_combiner = final_folder / "combiner_used.cbm"
    if used_combiner.exists():
        assert sha256(used_combiner) == sha256(combiner_source)
    else:
        shutil.copyfile(combiner_source, used_combiner)

    members = []
    for seed in [42, 43]:
        with np.load(root / "work" / f"seed{seed}" / "evaluation_predictions.npz") as saved:
            p = saved["combiner_predictions"].ravel()
        assert p.shape == p16.shape == (529390,)
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        members.append(p)
    baseline = 0.5 * members[0] + 0.5 * members[1]
    with np.load(root / "work/ensemble_evaluation.npz") as saved:
        assert np.array_equal(baseline, saved["predictions"].ravel())
    ensemble = (2 / 3) * baseline + (1 / 3) * p16
    assert np.isfinite(ensemble).all() and np.ptp(ensemble) > 0
    assert ((ensemble >= 0) & (ensemble <= 1)).all()
    rowids = np.load(root / "data/cache/evaluation_rowids.npy", allow_pickle=False)
    np.savez_compressed(root / "work/three_model_evaluation.npz", predictions=ensemble, weights=np.array([1 / 3, 1 / 3, 1 / 3]))

    report = export_submission(
        rowids, ensemble, projects / "techarena/data/raw/example_submission.csv",
        output_path, expected_rows=529390,
    )
    report.update({
        "members": ["dim8_seed42", "dim8_seed43", "dim16_seed42"],
        "weights": [1 / 3, 1 / 3, 1 / 3],
        "validation_pipeline_check_passed": True,
        "runtime_seconds": round(time.perf_counter() - start, 2),
    })
    manifest = {
        "validation_verification": verification,
        "full_data_dim16_prediction": final_report,
        "final_dim16_checkpoint": str(final_folder / "event_final.pt"),
        "final_dim16_checkpoint_sha256": sha256(final_folder / "event_final.pt"),
        "dim16_combiner": str(used_combiner),
        "dim16_combiner_sha256": sha256(used_combiner),
        "dim8_prediction_files_sha256": {
            str(seed): sha256(root / "work" / f"seed{seed}" / "evaluation_predictions.npz") for seed in [42, 43]
        },
        "script_sha256": sha256(Path(__file__)),
        "submission": report,
    }
    (root / "outputs/three_model_submission_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (root / "outputs/three_model_submission_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nTHREE-MODEL SUBMISSION CHECKS", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
