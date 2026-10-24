"""Prepare/train a separate, fixed 16-dimensional TechArena experiment.

Run from techarena-final:
    python src/prepare_dim16_trial.py setup
    python src/prepare_dim16_trial.py train
Second seed, after the first experiment passed:
    python src/prepare_dim16_trial.py setup --seed 43
    python src/prepare_dim16_trial.py train --seed 43

Only the selected trial directory is written by these actions.
The existing preference/event scripts supply the training implementation.
"""

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapt_scripts(preference, events, split_name, seed=42):
    assert seed in (42, 43)
    assert preference.count("dimension = 8") == 1
    assert preference.count("max_epochs = 196") == 1
    assert "if epoch != max_epochs:" in preference
    assert "if epoch != 8:" in events
    result = {}
    for name, text in [("train_preferences.py", preference), ("train_events.py", events)]:
        assert text.count("torch.manual_seed(43)") == 1, name
        assert text.count("split_jan26.npz") == 1, name
        text = text.replace("torch.manual_seed(43)", f"torch.manual_seed({seed})", 1)
        text = text.replace("split_jan26.npz", split_name, 1)
        if name == "train_preferences.py":
            text = text.replace("dimension = 8", "dimension = 16", 1)
            guard = "        if epoch != max_epochs:\n            continue"
            assert text.count(guard) == 1
            progress = (
                "        if epoch % 20 == 0 and epoch != max_epochs:\n"
                "            print(json.dumps({\n"
                '                "epoch": epoch,\n'
                '                "training_loss": float(loss_total.item() / len(users)),\n'
                '                "elapsed_seconds": round(time.perf_counter() - start, 1),\n'
                "            }), flush=True)\n\n"
            )
            text = text.replace(guard, progress + guard, 1)
        ast.parse(text)
        result[name] = text
    return result


def setup(root, trial, seed=42):
    if trial.exists():
        raise FileExistsError(f"Trial folder already exists: {trial}. Use the train action after setup.")
    projects = root.parent
    if seed == 43:
        previous = json.loads((projects / "techarena-dim16-trial/outputs/ensemble_trial_results.json").read_text(encoding="utf-8"))
        assert previous["supported_for_further_confirmation"] is True
    source = projects / "techarena-seed43/src"
    preference = (source / "train_preferences.py").read_text(encoding="utf-8")
    events = (source / "train_events.py").read_text(encoding="utf-8")
    pipeline = (root / "src/build_submission.py").read_text(encoding="utf-8")
    assertion = 'assert dimension == 8 and checkpoint["epoch"] == 8'
    assert pipeline.count(assertion) == 1
    pipeline = pipeline.replace(assertion, 'assert dimension == 16 and checkpoint["epoch"] == 8', 1)
    ast.parse(pipeline)
    sources = {
        "jan24": projects / "techarena-confirm-jan24",
        "jan25": projects / "techarena-long320-jan25",
        "jan26": projects / "techarena-seed43",
    }
    prepared = {}
    for fold, folder in sources.items():
        split = f"split_{fold}.npz"
        prepared[fold] = {
            "scripts": adapt_scripts(preference, events, split, seed),
            "cache": {name: (folder / "data/cache" / name).resolve(strict=True) for name in ["train_events.npz", "preference_training.npz", split]},
        }
    protocol = {
        "dimension": 16,
        "neural_seed": seed,
        "preference_epochs": 196,
        "event_epochs": 8,
        "combiner_trees": 545,
        "combiner_seed": 42,
        "combiner_early_stopping": False,
        "primary_check": "first event per user across Jan25-26; combiner trained on Jan24",
        "secondary_check": "first event per user on Jan26; combiner trained on Jan25",
        "planned_candidate_members": (["dim8_seed42", "dim8_seed43", "dim16_seed42"] if seed == 42 else ["dim8_seed42", "dim8_seed43", "dim16_seed42", "dim16_seed43"]),
        "planned_candidate_weights": ([1 / 3, 1 / 3, 1 / 3] if seed == 42 else [1 / 3, 1 / 3, 1 / 6, 1 / 6]),
        "baseline": ("existing equal-weight dim8 seed42/seed43 ensemble" if seed == 42 else "public v2: equal-weight dim8_seed42/dim8_seed43/dim16_seed42 ensemble"),
        "support_rule": "primary paired-gain interval entirely above zero and secondary AUC change nonnegative",
        "bootstrap_resamples": 300,
        "bootstrap_seed": 2026,
        "status": "retrospective development experiment",
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
    (trial / "src").mkdir()
    (trial / "work").mkdir()
    (trial / "outputs").mkdir()
    (trial / "src/dim16_pipeline.py").write_text(pipeline, encoding="utf-8")
    (trial / "outputs/frozen_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({"setup": "passed", "trial_workspace": str(trial), "protocol": {k: v for k, v in protocol.items() if k != "source_sha256"}}, indent=2))


def check_checkpoint(path, epoch):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["dimension"] == 16 and checkpoint["epoch"] == epoch, path
    return {"epoch": epoch, "validation_auc": float(checkpoint["validation_auc"]), "checkpoint": str(path)}


def train(trial):
    start = time.perf_counter()
    protocol = json.loads((trial / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    assert protocol["dimension"] == 16 and protocol["neural_seed"] in (42, 43)
    seed = protocol["neural_seed"]
    for name, expected in protocol["source_sha256"].items():
        assert digest(trial / name) == expected, f"Training source changed: {name}"
    records = {}
    for fold in ["jan24", "jan25", "jan26"]:
        folder = trial / fold
        print(f"\nTraining dimension 16, seed {seed}, fold {fold}", flush=True)
        stages = [
            ("preference", "train_preferences.py", "preference_best.pt", 196),
            ("event", "train_events.py", "event_best.pt", 8),
        ]
        records[fold] = {}
        for stage, script, filename, epoch in stages:
            checkpoint = folder / "work" / filename
            if checkpoint.exists():
                print(f"Using completed {stage} checkpoint for {fold}", flush=True)
            else:
                subprocess.run([sys.executable, str(folder / "src" / script)], cwd=folder, check=True)
            records[fold][stage] = check_checkpoint(checkpoint, epoch)
    report = {"dimension": 16, "seed": seed, "folds": records, "runtime_seconds": round(time.perf_counter() - start, 1)}
    (trial / "outputs/neural_training_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nDIMENSION 16 TRAINING RESULTS", flush=True)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "train"])
    parser.add_argument("--seed", type=int, choices=[42, 43], default=42)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    trial = root.parent / ("techarena-dim16-trial" if args.seed == 42 else "techarena-dim16-seed43-trial")
    if args.action == "setup":
        setup(root, trial, args.seed)
    else:
        train(trial)


if __name__ == "__main__":
    main()
