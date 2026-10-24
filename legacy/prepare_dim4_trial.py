"""Prepare/train one fixed dimension-4 experiment in a separate WSL folder.

Run from techarena-final:
    python src/prepare_dim4_trial.py setup
    python src/prepare_dim4_trial.py train
"""

import argparse
import ast
import json
import subprocess
import sys
import time
from pathlib import Path

from prepare_dim16_trial import adapt_scripts, digest


def setup(root, trial):
    if trial.exists():
        raise FileExistsError(f"Already exists: {trial}. After setup, use the train action.")
    projects = root.parent
    source = projects / "techarena-seed43/src"
    preference = (source / "train_preferences.py").read_text(encoding="utf-8")
    events = (source / "train_events.py").read_text(encoding="utf-8")
    pipeline = (root / "src/build_submission.py").read_text(encoding="utf-8")
    guard = 'assert dimension == 8 and checkpoint["epoch"] == 8'
    assert pipeline.count(guard) == 1
    pipeline = pipeline.replace(guard, 'assert dimension == 4 and checkpoint["epoch"] == 8', 1)
    ast.parse(pipeline)
    sources = {
        "jan24": projects / "techarena-confirm-jan24",
        "jan25": projects / "techarena-long320-jan25",
        "jan26": projects / "techarena-seed43",
    }
    prepared = {}
    for fold, folder in sources.items():
        split = f"split_{fold}.npz"
        scripts = adapt_scripts(preference, events, split, seed=42)
        assert scripts["train_preferences.py"].count("dimension = 16") == 1
        scripts["train_preferences.py"] = scripts["train_preferences.py"].replace("dimension = 16", "dimension = 4", 1)
        for text in scripts.values():
            ast.parse(text)
        prepared[fold] = {
            "scripts": scripts,
            "cache": {name: (folder / "data/cache" / name).resolve(strict=True)
                      for name in ["train_events.npz", "preference_training.npz", split]},
        }
    # Check required baseline predictions before creating the trial folder.
    baseline = projects / "techarena-dim16-seed43-trial/work"
    for name in ["two_day_ensemble_predictions.npz", "jan26_ensemble_predictions.npz"]:
        assert (baseline / name).is_file(), baseline / name
    protocol = {
        "dimension": 4, "neural_seed": 42,
        "preference_epochs": 196, "event_epochs": 8,
        "combiner_trees": 545, "combiner_seed": 42,
        "combiner_early_stopping": False,
        "baseline": "public v3: dim8 seeds42/43 plus dim16 seeds42/43",
        "candidate_baseline_weight": 0.8,
        "candidate_dim4_weight": 0.2,
        "planned_candidate_members": ["dim8_seed42", "dim8_seed43", "dim16_seed42", "dim16_seed43", "dim4_seed42"],
        "planned_candidate_weights": [4 / 15, 4 / 15, 2 / 15, 2 / 15, 1 / 5],
        "primary_check": "first event per user across Jan25-26; combiner trained on Jan24",
        "secondary_check": "first event per user on Jan26; combiner trained on Jan25",
        "support_rule": "primary paired-gain interval entirely above zero and secondary AUC change nonnegative",
        "bootstrap_resamples": 300, "bootstrap_seed": 2026,
        "status": "retrospective development experiment; reused validation dates",
        "source_sha256": {},
    }
    for fold, data in prepared.items():
        folder = trial / fold
        for directory in ["src", "data/cache", "work", "outputs"]:
            (folder / directory).mkdir(parents=True, exist_ok=True)
        for name, text in data["scripts"].items():
            path = folder / "src" / name
            path.write_text(text, encoding="utf-8")
            protocol["source_sha256"][str(path.relative_to(trial))] = digest(path)
        for name, original in data["cache"].items():
            (folder / "data/cache" / name).symlink_to(original)
    for directory in ["src", "work", "outputs"]:
        (trial / directory).mkdir()
    helper = trial / "src/dim4_pipeline.py"
    helper.write_text(pipeline, encoding="utf-8")
    protocol["source_sha256"][str(helper.relative_to(trial))] = digest(helper)
    (trial / "outputs/frozen_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({"setup": "passed", "trial_workspace": str(trial),
                      "protocol": {k: v for k, v in protocol.items() if k != "source_sha256"}}, indent=2))


def check_checkpoint(path, epoch):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["dimension"] == 4 and checkpoint["epoch"] == epoch, path
    return {"epoch": epoch, "validation_auc": float(checkpoint["validation_auc"]), "checkpoint": str(path)}


def train(trial):
    start = time.perf_counter()
    protocol = json.loads((trial / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    assert protocol["dimension"] == 4 and protocol["neural_seed"] == 42
    for name, expected in protocol["source_sha256"].items():
        assert digest(trial / name) == expected, f"Source changed: {name}"
    records = {}
    for fold in ["jan24", "jan25", "jan26"]:
        folder = trial / fold
        print(f"\nTraining dimension 4, seed 42, fold {fold}", flush=True)
        records[fold] = {}
        for stage, script, filename, epoch in [
            ("preference", "train_preferences.py", "preference_best.pt", 196),
            ("event", "train_events.py", "event_best.pt", 8),
        ]:
            checkpoint = folder / "work" / filename
            if checkpoint.exists():
                print(f"Using completed {stage} checkpoint for {fold}", flush=True)
            else:
                subprocess.run([sys.executable, str(folder / "src" / script)], cwd=folder, check=True)
            records[fold][stage] = check_checkpoint(checkpoint, epoch)
    report = {"dimension": 4, "seed": 42, "folds": records,
              "runtime_seconds": round(time.perf_counter() - start, 1)}
    (trial / "outputs/neural_training_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nDIMENSION 4 TRAINING RESULTS", flush=True)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "train"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    trial = root.parent / "techarena-dim4-trial"
    if args.action == "setup":
        setup(root, trial)
    else:
        train(trial)


if __name__ == "__main__":
    main()
