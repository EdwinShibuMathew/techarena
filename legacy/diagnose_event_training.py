"""Locate the stage where event24 differs from event8; no model training.

Run in techarena-final: python src/diagnose_event_training.py
Reuses the existing neural checkpoints and tree combiners.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from build_submission import (
    TARGET_FIELDS, FIT_FIELDS, neural_pipeline as event8_pipeline,
    adjust_exposure, load_combiner, predict_combiner, check_close,
)


def auc(labels, predictions):
    return float(roc_auc_score(labels.ravel(), predictions.ravel()))


def main():
    start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    projects = root.parent
    trial = projects / "techarena-event24-trial"
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    sys.path.insert(0, str(trial / "src"))
    from event24_pipeline import neural_pipeline as event24_pipeline

    old_folders = {
        42: {"jan25": projects / "techarena-long320-jan25", "jan26": projects / "techarena-long320"},
        43: {"jan25": projects / "techarena-seed43-jan25", "jan26": projects / "techarena-seed43"},
    }
    old_primary_combiners = {
        42: projects / "techarena-confirm-combiner/work/frozen_combiner.cbm",
        43: projects / "techarena-confirm-seed43-combiner/work/frozen_combiner.cbm",
    }
    with np.load(root / "data/cache/train_events.npz") as saved:
        raw = {name: saved[name] for name in set(TARGET_FIELDS + FIT_FIELDS)}
    records = {}
    stage_keys = ["raw_event_auc", "exposure_adjusted_auc", "history_fallback_auc", "combiner_auc"]

    for case, fold, cutoff, fitting_day in [
        ("two_day_primary", "jan25", 1737763200, "jan24"),
        ("jan26_secondary", "jan26", 1737849600, "jan25"),
    ]:
        saved_name = "two_day_predictions.npz" if fold == "jan25" else "jan26_predictions.npz"
        with np.load(trial / "work" / saved_name) as saved:
            target_idx = saved["valid_idx"]
            labels = saved["labels"]
            expected_new = {seed: saved[f"new_dim8_seed{seed}"] for seed in [42, 43]}
        assert np.array_equal(labels, raw["y"][target_idx])
        records[case] = {}
        for seed in [42, 43]:
            old = old_folders[seed][fold]
            new = trial / f"dim8_seed{seed}" / fold
            with np.load(old / "data/cache" / f"split_{fold}.npz") as saved:
                fit_idx = saved["train_idx"]
            with np.load(new / "data/cache" / f"split_{fold}.npz") as saved:
                assert np.array_equal(fit_idx, saved["train_idx"])
            assert (raw["t"][fit_idx] < cutoff).all()
            assert (raw["t"][target_idx] >= cutoff).all()
            assert (raw["t"][target_idx] < 1737936000).all()
            training = {name: raw[name][fit_idx] for name in FIT_FIELDS}
            target = {name: raw[name][target_idx] for name in TARGET_FIELDS}
            exposure = np.bincount(training["i"].ravel(), minlength=int(raw["i"].max()) + 1)
            old_combiner = old_primary_combiners[seed] if fold == "jan25" else old / "work/combiner_best.cbm"
            new_combiner = trial / "work" / f"dim8_seed{seed}_combiner_{fitting_day}.cbm"
            stages = {}
            for epochs, folder, pipeline, combiner_path in [
                (8, old, event8_pipeline, old_combiner),
                (24, new, event24_pipeline, new_combiner),
            ]:
                print(f"Auditing {case}, seed {seed}, event epochs {epochs}...", flush=True)
                X, names, raw_p, full_p, _, _ = pipeline(
                    folder / "work/event_best.pt", training, target,
                    folder / "data/cache/preference_training.npz", device,
                )
                adjusted_p = adjust_exposure(raw_p, target["i"], exposure)
                model = load_combiner(combiner_path)
                predictions = predict_combiner(model, X, names).reshape(-1, 10)
                if epochs == 24:
                    check_close(predictions, expected_new[seed], "Reproduced event24 combiner probabilities")
                stages[str(epochs)] = {
                    "raw_event_auc": auc(labels, raw_p),
                    "exposure_adjusted_auc": auc(labels, adjusted_p),
                    "history_fallback_auc": auc(labels, full_p),
                    "combiner_auc": auc(labels, predictions),
                    "mean_raw_prediction": float(raw_p.mean()),
                    "mean_adjusted_prediction": float(adjusted_p.mean()),
                    "mean_final_prediction": float(predictions.mean()),
                }
                del X, model, raw_p, full_p, adjusted_p, predictions
            records[case][str(seed)] = {
                "event8": stages["8"],
                "event24": stages["24"],
                "event24_minus_event8": {key: stages["24"][key] - stages["8"][key] for key in stage_keys},
                "actual_click_rate": float(labels.mean()),
            }
    report = {
        "diagnostic": "raw predictions, exposure adjustment, history fallback, and combiner",
        "comparisons": records,
        "runtime_seconds": round(time.perf_counter() - start, 2),
    }
    (root / "outputs/event_training_diagnosis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nEVENT TRAINING STAGE AUDIT", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
