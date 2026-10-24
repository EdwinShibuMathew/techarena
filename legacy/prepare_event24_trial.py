"""Reuse existing preference models and test 24 event-training epochs.

Run in techarena-final:
    python src/prepare_event24_trial.py setup
    python src/prepare_event24_trial.py train
Only the separate techarena-event24-trial workspace is written.
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


def adapt_events(source, seed, fold):
    assert source.count("torch.manual_seed(43)") == 1
    assert source.count("for epoch in range(9):") == 1
    assert source.count("if epoch != 8:") == 1
    assert source.count("split_jan26.npz") == 1
    text = source.replace("torch.manual_seed(43)", f"torch.manual_seed({seed})", 1)
    text = text.replace("for epoch in range(9):", "for epoch in range(25):", 1)
    text = text.replace("if epoch != 8:", "if epoch != 24:", 1)
    text = text.replace("split_jan26.npz", f"split_{fold}.npz", 1)
    guard = "        if epoch != 24:\n            continue"
    assert text.count(guard) == 1
    progress = (
        "        if epoch > 0 and epoch % 4 == 0 and epoch != 24:\n"
        '            print(json.dumps({"epoch": epoch, "elapsed_seconds": round(time.perf_counter() - start, 1)}), flush=True)\n\n'
    )
    text = text.replace(guard, progress + guard, 1)
    before = [ast.dump(n, include_attributes=False) for n in ast.parse(source).body if isinstance(n, ast.ClassDef)]
    after = [ast.dump(n, include_attributes=False) for n in ast.parse(text).body if isinstance(n, ast.ClassDef)]
    assert before == after
    return text


def setup(root, trial):
    import torch
    if trial.exists():
        raise FileExistsError(f"Trial already exists: {trial}. Use the train action after setup.")
    projects = root.parent
    source = projects / "techarena-seed43/src"
    events = (source / "train_events.py").read_text(encoding="utf-8")
    preference = (source / "train_preferences.py").read_text(encoding="utf-8")
    preference_class = next(node for node in ast.parse(preference).body if isinstance(node, ast.ClassDef) and node.name == "PreferenceModel")
    preference = "import torch\nfrom torch import nn\n\n" + ast.get_source_segment(preference, preference_class) + "\n"
    ast.parse(preference)
    pipeline = (root / "src/build_submission.py").read_text(encoding="utf-8")
    assertion = 'assert dimension == 8 and checkpoint["epoch"] == 8'
    assert pipeline.count(assertion) == 1
    pipeline = pipeline.replace(assertion, 'assert dimension == 8 and checkpoint["epoch"] == 24', 1)
    ast.parse(pipeline)
    origins = {
        "dim8_seed42": {"jan24": projects / "techarena-confirm-jan24", "jan25": projects / "techarena-long320-jan25", "jan26": projects / "techarena-long320"},
        "dim8_seed43": {"jan24": projects / "techarena-confirm-seed43-jan24", "jan25": projects / "techarena-seed43-jan25", "jan26": projects / "techarena-seed43"},
    }
    prepared = {}
    for key in KEYS:
        seed = int(key.rsplit("seed", 1)[1])
        for fold in FOLDS:
            origin = origins[key][fold]
            checkpoint = (origin / "work/preference_best.pt").resolve(strict=True)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert saved["dimension"] == 8 and saved["epoch"] == 196
            del saved
            prepared[(key, fold)] = {
                "event_source": adapt_events(events, seed, fold),
                "preference": checkpoint,
                "cache": {name: (origin / "data/cache" / name).resolve(strict=True) for name in ["train_events.npz", "preference_training.npz", f"split_{fold}.npz"]},
            }
    protocol = {
        "changed_setting": "event epochs 8 to 24 for the two dim8 models",
        "dimension": 8, "seeds": [42, 43],
        "preference_epochs": 196, "preference_models_reused": True,
        "event_epochs": 24, "event_learning_rate": 0.0015,
        "event_anchor_regularization": 0.1, "event_effect_regularization": 0.05,
        "combiner_trees": 545, "combiner_seed": 42, "combiner_early_stopping": False,
        "candidate": "replace the two dim8 event models; retain the existing dim16 models",
        "weights": [1 / 3, 1 / 3, 1 / 6, 1 / 6],
        "primary_check": "first event per user across Jan25-26; combiner trained on Jan24",
        "secondary_check": "first event per user on Jan26; combiner trained on Jan25",
        "support_rule": "primary paired-gain interval entirely above zero and secondary AUC change nonnegative",
        "bootstrap_resamples": 300, "bootstrap_seed": 2026,
        "status": "retrospective development experiment",
        "source_sha256": {}, "preference_checkpoint_sha256": {},
    }
    for (key, fold), values in prepared.items():
        folder = trial / key / fold
        for name in ["src", "data/cache", "work", "outputs"]:
            (folder / name).mkdir(parents=True, exist_ok=True)
        scripts = {"train_preferences.py": preference, "train_events.py": values["event_source"]}
        for name, text in scripts.items():
            path = folder / "src" / name
            path.write_text(text, encoding="utf-8")
            protocol["source_sha256"][str(path.relative_to(trial))] = sha256(path)
        for name, original in values["cache"].items():
            (folder / "data/cache" / name).symlink_to(original)
        pref = folder / "work/preference_best.pt"
        pref.symlink_to(values["preference"])
        protocol["preference_checkpoint_sha256"][str(pref.relative_to(trial))] = sha256(pref)
    for name in ["src", "work", "outputs"]:
        (trial / name).mkdir(exist_ok=True)
    helper_path = trial / "src/event24_pipeline.py"
    helper_path.write_text(pipeline, encoding="utf-8")
    protocol["source_sha256"][str(helper_path.relative_to(trial))] = sha256(helper_path)
    (trial / "outputs/frozen_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({"setup": "passed", "trial_workspace": str(trial), "protocol": {k: v for k, v in protocol.items() if not k.endswith("sha256")}}, indent=2))


def train(trial):
    import torch
    start = time.perf_counter()
    protocol = json.loads((trial / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    assert protocol["event_epochs"] == 24 and protocol["dimension"] == 8
    for section in ["source_sha256", "preference_checkpoint_sha256"]:
        for relative, expected in protocol[section].items():
            assert sha256(trial / relative) == expected, relative
    records = {}
    for key in KEYS:
        records[key] = {}
        for fold in FOLDS:
            folder = trial / key / fold
            path = folder / "work/event_best.pt"
            print(f"\nEvent training: {key}, {fold}, 24 epochs", flush=True)
            if not path.exists():
                subprocess.run([sys.executable, str(folder / "src/train_events.py")], cwd=folder, check=True)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            assert saved["dimension"] == 8 and saved["epoch"] == 24
            records[key][fold] = {"epoch": 24, "event_validation_auc": float(saved["validation_auc"])}
            del saved
    report = {"folds": records, "runtime_seconds": round(time.perf_counter() - start, 1)}
    (trial / "outputs/event_training_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nEVENT24 TRAINING RESULTS")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "train"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    trial = root.parent / "techarena-event24-trial"
    if args.action == "setup":
        setup(root, trial)
    else:
        train(trial)


if __name__ == "__main__":
    main()
