"""Verify the saved validation pipeline, then predict and export submission.csv.

Run in WSL: python src/build_submission.py
Optional verification without evaluation prediction: add --verify-only.
Requires train_final_models.py, submission_features.py, submission_export.py.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from train_final_models import PreferenceModel, EventModel, include_no_click
from submission_features import build_features
from submission_export import export_submission, sha256

TARGET_FIELDS = ["u", "i", "t", "tb", "dt", "h", "ht"]
FIT_FIELDS = ["u", "i", "y"]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


@torch.inference_mode()
def predict_events(model, target, device):
    result = np.empty(target["i"].shape, dtype=np.float64)
    for start in range(0, len(target["u"]), 4096):
        stop = start + 4096
        context = target["tb"][start:stop] + 4 * target["dt"][start:stop]
        inputs = [
            torch.tensor(value, dtype=torch.long, device=device)
            for value in [target["u"][start:stop], target["i"][start:stop], context]
        ]
        scores = model(*inputs)
        probabilities = include_no_click(scores).softmax(dim=1)[:, 1:]
        result[start:stop] = probabilities.cpu().numpy()
    return result


def adjust_exposure(original, items, exposure):
    relative_exposure = (exposure[items] + 20) / (exposure.mean() + 20)
    weights = original / relative_exposure ** 0.5
    no_click = np.maximum(1 - original.sum(axis=1, keepdims=True), 1e-8)
    adjusted = weights / (no_click + weights.sum(axis=1, keepdims=True))
    assert np.isfinite(adjusted).all()
    assert ((adjusted >= 0) & (adjusted <= 1)).all()
    assert (adjusted.sum(axis=1) <= 1 + 1e-8).all()
    return adjusted


def neural_pipeline(checkpoint_path, training, target, positive_path, device, final_seed=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    n_users = int(checkpoint["n_users"])
    n_items = int(checkpoint["n_items"])
    dimension = int(checkpoint["dimension"])
    assert dimension == 8 and checkpoint["epoch"] == 8
    if final_seed is not None:
        assert checkpoint["seed"] == final_seed
        assert checkpoint["stage"] == "event"
        assert checkpoint["cutoff"] == 1737936000
        assert n_users == 54456 and n_items == 50000
    state = checkpoint["model_state"]
    assert target["u"].max() < n_users
    assert target["i"].max() < n_items
    assert target["i"].shape == (len(target["u"]), 10)
    assert target["h"].shape == target["ht"].shape == (len(target["u"]), 20)
    assert np.isin(target["dt"], [0, 1]).all()
    assert np.isin(target["tb"], [0, 1, 2, 3]).all()
    present = target["h"] >= 0
    assert ((target["h"] == -1) | ((target["h"] >= 0) & (target["h"] < n_items))).all()
    assert (target["ht"][present] < np.broadcast_to(target["t"][:, None], present.shape)[present]).all()

    preference = PreferenceModel(
        n_users, n_items, dimension, torch.ones(n_users, dtype=torch.bool)
    )
    model = EventModel(preference, n_users, n_items).to(device)
    model.load_state_dict(state)
    model.eval()

    with np.load(positive_path) as saved:
        positive_users = saved["users"]
        positive_items = saved["items"]
    assert positive_users.max() < n_users and positive_items.max() < n_items

    item_vectors = state["preference.item.weight"].detach().numpy()
    user_vectors = state["preference.user.weight"].detach().numpy()
    click_counts = np.bincount(positive_users, minlength=n_users)
    profile_sum = np.zeros((n_users, dimension), dtype="float64")
    for column in range(dimension):
        profile_sum[:, column] = np.bincount(
            positive_users,
            weights=item_vectors[positive_items, column],
            minlength=n_users,
        )
    profile_mean = profile_sum / np.maximum(click_counts[:, None], 1)
    known = click_counts > 0
    encoder = Ridge(alpha=1.0)
    encoder.fit(profile_mean[known], user_vectors[known])

    training_events = np.bincount(training["u"], minlength=n_users)
    exposure = np.bincount(training["i"].ravel(), minlength=n_items)
    raw_probabilities = predict_events(model, target, device)
    probabilities = adjust_exposure(raw_probabilities, target["i"], exposure)

    cold = (training_events[target["u"]] == 0) & (present.sum(axis=1) > 0)
    if cold.any():
        assert len(np.unique(target["u"][cold])) == int(cold.sum())
        cold_history = target["h"][cold]
        cold_present = cold_history >= 0
        history_profiles = (
            item_vectors[cold_history.clip(min=0)] * cold_present[:, :, None]
        ).sum(axis=1) / cold_present.sum(axis=1, keepdims=True)
        estimated_vectors = encoder.predict(history_profiles)
        with torch.inference_mode():
            cold_users = torch.tensor(target["u"][cold], dtype=torch.long, device=device)
            model.preference.user.weight[cold_users] = torch.tensor(
                estimated_vectors, dtype=torch.float32, device=device
            )
            model.preference.active_users[cold_users] = True
        cold_target = {name: values[cold] for name, values in target.items()}
        cold_probabilities = predict_events(model, cold_target, device)
        probabilities[cold] = adjust_exposure(cold_probabilities, target["i"][cold], exposure)

    # CPU state contains the original user vectors. build_features applies the
    # same inferred-user replacement explicitly, as in the supplied builder.
    X, names = build_features(
        training, target, state, encoder.coef_, encoder.intercept_,
        n_users, n_items, probabilities, raw_probabilities,
    )
    report = {
        "events": len(target["u"]),
        "candidate_rows": len(X),
        "feature_count": len(names),
        "encoder_training_users": int(known.sum()),
        "unseen_users": int((training_events[target["u"]] == 0).sum()),
        "unseen_users_with_history_updated": int(cold.sum()),
        "empty_history_events": int((present.sum(axis=1) == 0).sum()),
    }
    del model, preference, checkpoint
    torch.cuda.empty_cache()
    return X, names, raw_probabilities, probabilities, encoder, report


def load_combiner(path):
    model = CatBoostClassifier()
    model.load_model(str(path))
    assert model.tree_count_ == 545
    assert len(model.feature_names_) == 37
    return model


def predict_combiner(model, X, names):
    assert list(names) == model.feature_names_, "Combiner feature order differs."
    predictions = model.predict_proba(Pool(X, feature_names=names), thread_count=4)[:, 1]
    assert np.isfinite(predictions).all()
    assert ((predictions >= 0) & (predictions <= 1)).all()
    return predictions


def check_close(actual, expected, name, rtol=2e-5, atol=2e-6):
    assert actual.shape == expected.shape, f"{name}: different shapes"
    difference = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise RuntimeError(f"{name} failed to reproduce saved values; maximum difference={difference}")
    return difference


def verify_reference(seed, folder, raw, model, device):
    print(f"Verifying saved January 26 pipeline for seed {seed}...", flush=True)
    with np.load(folder / "data/cache/split_jan26.npz") as split:
        fit_idx, target_idx = split["train_idx"], split["valid_idx"]
    assert (raw["t"][fit_idx] < 1737849600).all()
    assert (raw["t"][target_idx] >= 1737849600).all()
    assert (raw["t"][target_idx] < 1737936000).all()
    training = {name: raw[name][fit_idx] for name in FIT_FIELDS}
    target = {name: raw[name][target_idx] for name in TARGET_FIELDS}
    X, names, raw_p, pipeline_p, encoder, report = neural_pipeline(
        folder / "work/event_best.pt", training, target,
        folder / "data/cache/preference_training.npz", device,
    )
    with np.load(folder / "work/event_validation.npz") as saved:
        assert np.array_equal(saved["valid_idx"], target_idx)
        raw_difference = check_close(raw_p, saved["predictions"], "Raw event probabilities")
    with np.load(folder / "work/history_coldstart_validation.npz") as saved:
        assert np.array_equal(saved["valid_idx"], target_idx)
        pipeline_difference = check_close(pipeline_p, saved["predictions"], "Corrected probabilities")
    with np.load(folder / "work/combiner_features.npz") as saved:
        assert names == list(saved["feature_names"])
        feature_difference = check_close(X, saved["X_valid"], "All 37 feature columns")
    predictions = predict_combiner(model, X, names)
    with np.load(folder / "work/combiner_validation.npz") as saved:
        assert np.array_equal(saved["valid_idx"], target_idx)
        reference = saved["combiner_predictions"].ravel()
        prediction_difference = check_close(predictions, reference, "Combiner predictions", rtol=1e-5, atol=2e-6)
    labels = raw["y"][target_idx].ravel()
    auc = float(roc_auc_score(labels, predictions))
    saved_auc = float(roc_auc_score(labels, reference))
    if abs(auc - saved_auc) > 1e-6:
        raise RuntimeError(f"Saved validation AUC was not reproduced for seed {seed}: {auc} versus {saved_auc}")
    return {
        "seed": seed,
        "passed": True,
        "reproduced_saved_validation_auc": auc,
        "saved_validation_auc": saved_auc,
        "max_raw_probability_difference": raw_difference,
        "max_corrected_probability_difference": pipeline_difference,
        "max_feature_difference": feature_difference,
        "max_combiner_probability_difference": prediction_difference,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    projects = root.parent
    output = root / "outputs"
    submission = output / "submission.csv"
    output.mkdir(parents=True, exist_ok=True)
    if submission.exists() and not args.verify_only:
        raise FileExistsError(f"Submission already exists: {submission}")
    assert torch.cuda.is_available(), "Activate the techarena environment with working CUDA."
    torch.set_num_threads(4)
    device = torch.device("cuda")
    start = time.perf_counter()
    folders = {
        42: projects / "techarena-long320",
        43: projects / "techarena-seed43",
    }
    models = {seed: load_combiner(folder / "work/combiner_best.cbm") for seed, folder in folders.items()}
    assert models[42].feature_names_ == models[43].feature_names_

    with np.load(root / "data/cache/train_events.npz") as saved:
        raw = {name: saved[name] for name in set(TARGET_FIELDS + FIT_FIELDS)}
    assert len(raw["u"]) == 686824
    assert (raw["t"] < 1737936000).all()
    reference_checks = [verify_reference(seed, folders[seed], raw, models[seed], device) for seed in [42, 43]]
    write_json(output / "prediction_pipeline_verification.json", {"checks": reference_checks})
    print("Saved validation pipelines reproduced successfully.", flush=True)
    print(json.dumps(reference_checks, indent=2), flush=True)
    if args.verify_only:
        return

    with np.load(root / "data/cache/evaluation_events.npz") as saved:
        assert "y" not in saved.files, "Evaluation cache must contain no label array."
        target = {name: saved[name] for name in TARGET_FIELDS}
    rowids = np.load(root / "data/cache/evaluation_rowids.npy", allow_pickle=False)
    assert len(target["u"]) == 52939
    assert len(np.unique(target["u"])) == 52939
    assert target["i"].shape == (52939, 10)
    assert rowids.shape == (529390,)
    assert (target["t"] >= 1737936000).all()
    assert (target["t"] < 1738540800).all()
    training = {name: raw[name] for name in FIT_FIELDS}
    predictions = []
    seed_reports = []

    for seed in [42, 43]:
        print(f"Predicting evaluation rows with final seed {seed}...", flush=True)
        folder = root / "work" / f"seed{seed}"
        X, names, raw_p, pipeline_p, encoder, report = neural_pipeline(
            folder / "event_final.pt", training, target,
            root / "data/cache/preference_training.npz", device,
            final_seed=seed,
        )
        p = predict_combiner(models[seed], X, names)
        predictions.append(p)
        np.savez_compressed(
            folder / "evaluation_predictions.npz",
            raw_predictions=raw_p,
            pipeline_predictions=pipeline_p,
            combiner_predictions=p,
        )
        np.savez_compressed(folder / "history_encoder.npz", coefficients=encoder.coef_, intercept=encoder.intercept_)
        source_combiner = folders[seed] / "work/combiner_best.cbm"
        final_combiner = folder / "combiner_used.cbm"
        if final_combiner.exists():
            assert sha256(final_combiner) == sha256(source_combiner)
        else:
            shutil.copyfile(source_combiner, final_combiner)
        report.update({
            "seed": seed,
            "event_checkpoint": str(folder / "event_final.pt"),
            "event_checkpoint_sha256": sha256(folder / "event_final.pt"),
            "combiner": str(final_combiner),
            "combiner_sha256": sha256(final_combiner),
        })
        seed_reports.append(report)
        print(json.dumps(report, indent=2), flush=True)
        del X, raw_p, pipeline_p, encoder

    ensemble = 0.5 * predictions[0] + 0.5 * predictions[1]
    assert np.isfinite(ensemble).all() and np.ptp(ensemble) > 0
    np.savez_compressed(root / "work/ensemble_evaluation.npz", predictions=ensemble, weights=np.array([0.5, 0.5]))
    report = export_submission(
        rowids, ensemble,
        projects / "techarena/data/raw/example_submission.csv",
        submission, expected_rows=529390,
    )
    report.update({
        "ensemble_weights": [0.5, 0.5],
        "saved_validation_pipeline_checks_passed": True,
        "runtime_seconds": round(time.perf_counter() - start, 2),
    })
    write_json(output / "submission_checks.json", report)
    write_json(output / "submission_manifest.json", {
        "reference_checks": reference_checks,
        "final_seeds": seed_reports,
        "submission": report,
        "source_sha256": {name: sha256(root / "src" / name) for name in ["build_submission.py", "submission_features.py", "submission_export.py", "train_final_models.py"]},
    })
    print("\nFINAL SUBMISSION CHECKS", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
