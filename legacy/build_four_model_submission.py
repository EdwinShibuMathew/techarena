"""Export the supported four-model candidate using its frozen weights.

Run in techarena-final: python src/build_four_model_submission.py
Needs completed final dimension-16 seed43 training and the public v2 artifacts.
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
    previous = projects / "techarena-dim16-trial"
    trial = projects / "techarena-dim16-seed43-trial"
    final = previous / "final"
    folder = final / "work/seed43"
    output_path = root / "outputs/submission_four_models.csv"
    if output_path.exists():
        raise FileExistsError(f"Four-model submission already exists: {output_path}")
    weights = [1 / 3, 1 / 3, 1 / 6, 1 / 6]
    trial_report = json.loads((trial / "outputs/ensemble_trial_results.json").read_text(encoding="utf-8"))
    assert trial_report["supported_for_further_confirmation"] is True
    assert trial_report["protocol"]["planned_candidate_weights"] == weights
    protocol = json.loads((final / "outputs/seed43_frozen_protocol.json").read_text(encoding="utf-8"))
    assert protocol["dimension"] == 16 and protocol["seed"] == 43
    assert protocol["ensemble_weights"] == weights
    assert protocol["trainer_sha256"] == sha256(final / "src/train_final_models.py")
    trained_protocol = json.loads((folder / "training_protocol.json").read_text(encoding="utf-8"))
    assert trained_protocol["dimension"] == 16 and trained_protocol["seed"] == 43
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
    assert str(combiner_source) == protocol["combiner_source"]
    combiner = load_combiner(combiner_source)

    print("Reproducing the saved four-model January 26 result...", flush=True)
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
    new_validation = predict_combiner(combiner, X, names).reshape(-1, 10)
    del X
    with np.load(previous / "work/jan26_ensemble_predictions.npz") as saved:
        assert np.array_equal(saved["valid_idx"], valid_idx)
        original_base8 = saved["baseline"]
        original_dim16 = saved["dim16"]
        prior_three_model = saved["candidate"]
    recreated = (2 / 3) * original_base8 + (1 / 3) * (0.5 * original_dim16 + 0.5 * new_validation)
    with np.load(trial / "work/jan26_ensemble_predictions.npz") as saved:
        assert np.array_equal(saved["valid_idx"], valid_idx)
        assert np.array_equal(saved["labels"], raw["y"][valid_idx])
        check_close(saved["baseline"], prior_three_model, "Prior three-model baseline")
        single_difference = check_close(new_validation, saved["new_dim16"], "Seed43 dimension-16 validation probabilities")
        ensemble_difference = check_close(recreated, saved["candidate"], "Four-model validation probabilities")
    reproduced_auc = float(roc_auc_score(raw["y"][valid_idx].ravel(), recreated.ravel()))
    expected_auc = trial_report["jan26_secondary"]["candidate_four_model_auc"]
    assert abs(reproduced_auc - expected_auc) <= 1e-6
    verification = {
        "passed": True,
        "reproduced_four_model_validation_auc": reproduced_auc,
        "saved_four_model_validation_auc": expected_auc,
        "max_seed43_probability_difference": single_difference,
        "max_ensemble_probability_difference": ensemble_difference,
    }
    print(json.dumps(verification, indent=2), flush=True)

    preference = torch.load(folder / "preference_final.pt", map_location="cpu", weights_only=True)
    assert preference["dimension"] == 16 and preference["epoch"] == 196
    assert preference["seed"] == 43 and preference["cutoff"] == 1737936000
    del preference
    with np.load(root / "data/cache/evaluation_events.npz") as saved:
        assert "y" not in saved.files
        target = {name: saved[name] for name in TARGET_FIELDS}
    assert len(target["u"]) == len(np.unique(target["u"])) == 52939
    assert target["i"].shape == (52939, 10)
    assert (target["t"] >= 1737936000).all()
    assert (target["t"] < 1738540800).all()

    print("Predicting evaluation with final dimension-16 seed43...", flush=True)
    X, names, raw_p, pipeline_p, encoder, prediction_report = neural_pipeline(
        folder / "event_final.pt",
        {name: raw[name] for name in FIT_FIELDS},
        target, root / "data/cache/preference_training.npz", device,
        final_seed=43,
    )
    new_dim16 = predict_combiner(combiner, X, names)
    del X
    np.savez_compressed(folder / "evaluation_predictions.npz", raw_predictions=raw_p, pipeline_predictions=pipeline_p, combiner_predictions=new_dim16)
    np.savez_compressed(folder / "history_encoder.npz", coefficients=encoder.coef_, intercept=encoder.intercept_)
    used_combiner = folder / "combiner_used.cbm"
    if used_combiner.exists():
        assert sha256(used_combiner) == sha256(combiner_source)
    else:
        shutil.copyfile(combiner_source, used_combiner)

    member_paths = [
        root / "work/seed42/evaluation_predictions.npz",
        root / "work/seed43/evaluation_predictions.npz",
        final / "work/seed42/evaluation_predictions.npz",
    ]
    members = []
    for path in member_paths:
        with np.load(path) as saved:
            p = saved["combiner_predictions"].ravel()
        assert p.shape == new_dim16.shape == (529390,)
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        members.append(p)
    base8 = 0.5 * members[0] + 0.5 * members[1]
    prior = (2 / 3) * base8 + (1 / 3) * members[2]
    with np.load(root / "work/three_model_evaluation.npz") as saved:
        assert np.array_equal(prior, saved["predictions"].ravel())
    ensemble = (2 / 3) * base8 + (1 / 3) * (0.5 * members[2] + 0.5 * new_dim16)
    assert np.isfinite(ensemble).all() and np.ptp(ensemble) > 0
    assert ((ensemble >= 0) & (ensemble <= 1)).all()
    rowids = np.load(root / "data/cache/evaluation_rowids.npy", allow_pickle=False)
    np.savez_compressed(root / "work/four_model_evaluation.npz", predictions=ensemble, weights=np.array(weights))

    report = export_submission(
        rowids, ensemble, projects / "techarena/data/raw/example_submission.csv",
        output_path, expected_rows=529390,
    )
    report.update({
        "members": ["dim8_seed42", "dim8_seed43", "dim16_seed42", "dim16_seed43"],
        "weights": weights,
        "validation_pipeline_check_passed": True,
        "runtime_seconds": round(time.perf_counter() - start, 2),
    })
    manifest = {
        "validation_verification": verification,
        "full_data_dim16_seed43_prediction": prediction_report,
        "checkpoint": str(folder / "event_final.pt"),
        "checkpoint_sha256": sha256(folder / "event_final.pt"),
        "combiner": str(used_combiner),
        "combiner_sha256": sha256(used_combiner),
        "prior_prediction_files_sha256": {str(path): sha256(path) for path in member_paths},
        "script_sha256": sha256(Path(__file__)),
        "submission": report,
    }
    (root / "outputs/four_model_submission_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (root / "outputs/four_model_submission_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nFOUR-MODEL SUBMISSION CHECKS", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
