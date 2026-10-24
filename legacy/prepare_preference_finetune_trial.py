"""Prepare/train a fixed lower-rate preference fine-tuning experiment.

Run in techarena-final:
    python src/prepare_preference_finetune_trial.py setup
    python src/prepare_preference_finetune_trial.py train
The original models are read only; new models live in a separate trial folder.
"""

import argparse
import ast
import json
import subprocess
import sys
import time
from pathlib import Path

from submission_export import sha256

KEYS = ("dim8_seed42", "dim8_seed43")
FOLDS = ("jan24", "jan25", "jan26")


def classes(source):
    return [ast.dump(node, include_attributes=False) for node in ast.parse(source).body if isinstance(node, ast.ClassDef)]


def replace_once(text, old, new):
    assert text.count(old) == 1, f"Expected exactly one occurrence: {old!r}"
    return text.replace(old, new, 1)


def adapt_preference(source, seed, fold):
    text = replace_once(source, "torch.manual_seed(43)", f"torch.manual_seed({seed})")
    text = replace_once(text, "split_jan26.npz", f"split_{fold}.npz")
    assert text.count("dimension = 8") == 1
    assert "if epoch != max_epochs:" in text
    text = replace_once(text, "max_epochs = 196", "max_epochs = 260")
    text = replace_once(text, "for epoch in range(1, max_epochs + 1):", "for epoch in range(197, max_epochs + 1):")
    optimizer_line = "    optimizer = torch.optim.Adam(model.parameters(), lr=0.012)"
    resume = '''    # Start from the already fitted epoch-196 checkpoint; retain Adam moments.
    base_checkpoint = torch.load(
        "work/preference_initial.pt", map_location="cpu", weights_only=True
    )
    assert base_checkpoint["epoch"] == 196 and base_checkpoint["dimension"] == 8
    assert base_checkpoint["n_users"] == n_users and base_checkpoint["n_items"] == n_items
    assert torch.equal(base_checkpoint["model_state"]["active_users"], active_users)
    model.load_state_dict(base_checkpoint["model_state"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    optimizer.load_state_dict(base_checkpoint["optimizer_state"])
    for group in optimizer.param_groups:
        assert group["lr"] == 0.012
        group["lr"] = 0.003
    del base_checkpoint
    # Old checkpoints did not store RNG state. Use a new, declared sampling seed.
    torch.manual_seed(SAMPLING_SEED)
'''.replace("SAMPLING_SEED", str(seed + 1000)).rstrip()
    text = replace_once(text, optimizer_line, resume)
    guard = "        if epoch != max_epochs:\n            continue"
    progress = '''        if epoch % 16 == 0 and epoch != max_epochs:
            print(json.dumps({
                "epoch": epoch,
                "fine_tuning_learning_rate": 0.003,
                "training_loss": float(loss_total.item() / len(users)),
                "elapsed_seconds": round(time.perf_counter() - start, 1),
            }), flush=True)

'''
    text = replace_once(text, guard, progress + guard)
    metadata = '            "validation_auc": auc,\n        }\n\n        torch.save(checkpoint, "work/preference_last.pt")'
    replacement = '''            "validation_auc": auc,
            "fine_tuning_initial_epoch": 196,
            "fine_tuning_learning_rate": 0.003,
            "fine_tuning_sampling_seed": SAMPLING_SEED,
        }

        torch.save(checkpoint, "work/preference_last.pt")'''.replace("SAMPLING_SEED", str(seed + 1000))
    text = replace_once(text, metadata, replacement)
    assert classes(text) == classes(source), "Preference architecture changed"
    ast.parse(text)
    return text


def adapt_events(source, seed, fold):
    text = replace_once(source, "torch.manual_seed(43)", f"torch.manual_seed({seed})")
    text = replace_once(text, "split_jan26.npz", f"split_{fold}.npz")
    assert text.count("for epoch in range(9):") == 1
    assert text.count("if epoch != 8:") == 1
    assert classes(text) == classes(source), "Event architecture changed"
    ast.parse(text)
    return text


def setup(root, trial):
    import torch
    if trial.exists():
        raise FileExistsError(f"Trial exists: {trial}. After setup, use the train action.")
    projects = root.parent
    source = projects / "techarena-seed43/src"
    preference = (source / "train_preferences.py").read_text(encoding="utf-8")
    events = (source / "train_events.py").read_text(encoding="utf-8")
    origins = {
        "dim8_seed42": {"jan24": projects / "techarena-confirm-jan24", "jan25": projects / "techarena-long320-jan25", "jan26": projects / "techarena-long320"},
        "dim8_seed43": {"jan24": projects / "techarena-confirm-seed43-jan24", "jan25": projects / "techarena-seed43-jan25", "jan26": projects / "techarena-seed43"},
    }
    for name in ["techarena-dim16-trial", "techarena-dim16-seed43-trial"]:
        for filename in ["two_day_ensemble_predictions.npz", "jan26_ensemble_predictions.npz"]:
            assert (projects / name / "work" / filename).is_file()
    prepared = {}
    for key in KEYS:
        seed = int(key.rsplit("seed", 1)[1])
        for fold in FOLDS:
            origin = origins[key][fold]
            checkpoint = (origin / "work/preference_best.pt").resolve(strict=True)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert saved["dimension"] == 8 and saved["epoch"] == 196, checkpoint
            assert all(group["lr"] == 0.012 for group in saved["optimizer_state"]["param_groups"])
            del saved
            prepared[(key, fold)] = {
                "scripts": {"train_preferences.py": adapt_preference(preference, seed, fold),
                            "train_events.py": adapt_events(events, seed, fold)},
                "initial": checkpoint,
                "cache": {name: (origin / "data/cache" / name).resolve(strict=True)
                          for name in ["train_events.npz", "preference_training.npz", f"split_{fold}.npz"]},
            }
    protocol = {
        "changed_schedule": "fine-tune dim8 preference checkpoints after epoch196, then refit event models for 8 epochs",
        "dimension": 8, "seeds": [42, 43],
        "preference_initial_epochs": 196, "preference_extra_epochs": 64,
        "preference_total_epochs": 260,
        "preference_initial_learning_rate": 0.012,
        "preference_fine_tuning_learning_rate": 0.003,
        "adam_moments_reused": True,
        "original_rng_state_available": False,
        "fine_tuning_sampling_seeds": [1042, 1043],
        "preference_early_stopping": False,
        "event_epochs": 8, "event_learning_rate": 0.0015,
        "combiner_trees": 545, "combiner_seed": 42, "combiner_early_stopping": False,
        "candidate": "replace the two dim8 models; retain existing dim16 models",
        "weights": [1 / 3, 1 / 3, 1 / 6, 1 / 6],
        "primary_check": "first event per user across Jan25-26; combiner trained on Jan24",
        "secondary_check": "first event per user on Jan26; combiner trained on Jan25",
        "support_rule": "primary paired-gain interval entirely above zero and secondary AUC change nonnegative",
        "bootstrap_resamples": 300, "bootstrap_seed": 2026,
        "status": "retrospective development experiment; reused validation dates",
        "source_sha256": {}, "initial_checkpoint_sha256": {},
    }
    for (key, fold), data in prepared.items():
        folder = trial / key / fold
        for directory in ["src", "data/cache", "work", "outputs"]:
            (folder / directory).mkdir(parents=True, exist_ok=True)
        for name, text in data["scripts"].items():
            path = folder / "src" / name
            path.write_text(text, encoding="utf-8")
            protocol["source_sha256"][str(path.relative_to(trial))] = sha256(path)
        for name, original in data["cache"].items():
            (folder / "data/cache" / name).symlink_to(original)
        initial = folder / "work/preference_initial.pt"
        initial.symlink_to(data["initial"])
        protocol["initial_checkpoint_sha256"][str(initial.relative_to(trial))] = sha256(initial)
    (trial / "work").mkdir()
    (trial / "outputs").mkdir()
    (trial / "outputs/frozen_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({"setup": "passed", "trial_workspace": str(trial),
                      "protocol": {k: v for k, v in protocol.items() if not k.endswith("sha256")}}, indent=2))


def verify_inputs(trial, protocol):
    for section in ["source_sha256", "initial_checkpoint_sha256"]:
        for name, expected in protocol[section].items():
            assert sha256(trial / name) == expected, f"Changed input: {name}"


def train(trial):
    import torch
    start = time.perf_counter()
    protocol = json.loads((trial / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    assert protocol["preference_total_epochs"] == 260 and protocol["event_epochs"] == 8
    verify_inputs(trial, protocol)
    records = {}
    for key in KEYS:
        seed = int(key.rsplit("seed", 1)[1])
        records[key] = {}
        for fold in FOLDS:
            folder = trial / key / fold
            records[key][fold] = {}
            for stage, script, filename, epoch in [
                ("preference", "train_preferences.py", "preference_best.pt", 260),
                ("event", "train_events.py", "event_best.pt", 8),
            ]:
                print(f"\nTraining {stage}: {key}, {fold}", flush=True)
                path = folder / "work" / filename
                if not path.exists():
                    subprocess.run([sys.executable, str(folder / "src" / script)], cwd=folder, check=True)
                saved = torch.load(path, map_location="cpu", weights_only=True)
                assert saved["dimension"] == 8 and saved["epoch"] == epoch
                if stage == "preference":
                    assert saved["fine_tuning_initial_epoch"] == 196
                    assert saved["fine_tuning_learning_rate"] == 0.003
                    assert saved["fine_tuning_sampling_seed"] == seed + 1000
                records[key][fold][stage] = {"epoch": epoch, "validation_auc": float(saved["validation_auc"])}
                del saved
    verify_inputs(trial, protocol)
    report = {"folds": records, "runtime_seconds": round(time.perf_counter() - start, 1)}
    (trial / "outputs/neural_training_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nPREFERENCE FINE-TUNING TRAINING RESULTS", flush=True)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "train"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    trial = root.parent / "techarena-preference-finetune-trial"
    if args.action == "setup":
        setup(root, trial)
    else:
        train(trial)


if __name__ == "__main__":
    main()
