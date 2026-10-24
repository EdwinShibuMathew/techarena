"""Prepare/train one new dim8 seed with the V4 training schedule.

Run in techarena-final:
    python src/prepare_seed44_trial.py setup
    python src/prepare_seed44_trial.py train
Only the separate techarena-seed44-trial directory is written.
"""

import argparse
import ast
import json
import subprocess
import sys
import time
from pathlib import Path

from prepare_preference_finetune_trial import adapt_preference, adapt_events, replace_once, classes
from submission_export import sha256


def adapt_initial_preference(source, fold):
    text = replace_once(source, "torch.manual_seed(43)", "torch.manual_seed(44)")
    text = replace_once(text, "split_jan26.npz", f"split_{fold}.npz")
    assert text.count("dimension = 8") == 1 and text.count("max_epochs = 196") == 1
    text = replace_once(text, '"work/preference_best.pt"', '"work/preference_initial.pt"')
    text = replace_once(text, '"work/preference_last.pt"', '"work/preference_initial_last.pt"')
    text = replace_once(text, '"work/preference_validation.npz"', '"work/preference_initial_validation.npz"')
    text = replace_once(text, '"outputs/preference_metrics.json"', '"outputs/preference_initial_metrics.json"')
    guard = "        if epoch != max_epochs:\n            continue"
    progress = '''        if epoch % 20 == 0 and epoch != max_epochs:
            print(json.dumps({
                "stage": "initial_preference", "seed": 44, "epoch": epoch,
                "training_loss": float(loss_total.item() / len(users)),
                "elapsed_seconds": round(time.perf_counter() - start, 1),
            }), flush=True)

'''
    text = replace_once(text, guard, progress + guard)
    assert classes(text) == classes(source)
    ast.parse(text)
    return text


def verify_inputs(trial, protocol):
    for name, expected in protocol["source_sha256"].items():
        assert sha256(trial / name) == expected, f"Changed training source: {name}"
    for name, expected in protocol["baseline_sources_sha256"].items():
        assert sha256(Path(name)) == expected, f"Changed baseline source: {name}"


def setup(root, trial):
    if trial.exists():
        raise FileExistsError(f"Trial already exists: {trial}. After setup, use the train action.")
    projects = root.parent
    baseline = projects / "techarena-preference-finetune-trial"
    baseline_report_path = baseline / "outputs/ensemble_trial_results.json"
    report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    assert report["supported_for_further_confirmation"] is True
    checks_path = root / "outputs/preference_finetuned_submission_checks.json"
    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    assert checks["version"] == "v4" and checks["submission_ready"] is True
    assert checks["validation_pipeline_check_passed"] is True
    assert checks["both_full_data_seeds_checked"] is True
    source = projects / "techarena-seed43/src"
    preference = (source / "train_preferences.py").read_text(encoding="utf-8")
    events = (source / "train_events.py").read_text(encoding="utf-8")
    prepared = {}
    for fold in ["jan24", "jan25", "jan26"]:
        folder = baseline / "dim8_seed42" / fold
        prepared[fold] = {
            "scripts": {
                "train_initial_preferences.py": adapt_initial_preference(preference, fold),
                "train_preferences.py": adapt_preference(preference, 44, fold),
                "train_events.py": adapt_events(events, 44, fold),
            },
            "cache": {name: (folder / "data/cache" / name).resolve(strict=True)
                      for name in ["train_events.npz", "preference_training.npz", f"split_{fold}.npz"]},
        }
    baseline_paths = [baseline_report_path, checks_path]
    for name in ["two_day", "jan26"]:
        baseline_paths.append(baseline / "work" / f"{name}_predictions.npz")
        for folder_name in ["techarena-dim16-trial", "techarena-dim16-seed43-trial"]:
            baseline_paths.append(projects / folder_name / "work" / f"{name}_ensemble_predictions.npz")
    protocol = {
        "changed_setting": "add a third dim8 neural seed using the fixed V4 schedule",
        "dimension": 8, "neural_seed": 44,
        "preference_initial_epochs": 196, "preference_extra_epochs": 64,
        "preference_total_epochs": 260,
        "preference_initial_learning_rate": 0.012,
        "preference_fine_tuning_learning_rate": 0.003,
        "adam_moments_reused": True, "fine_tuning_sampling_seed": 1044,
        "preference_early_stopping": False,
        "event_epochs": 8, "event_learning_rate": 0.0015,
        "combiner_trees": 545, "combiner_seed": 42, "combiner_early_stopping": False,
        "baseline": "public V4: fine-tuned dim8 seeds42/43 plus dim16 seeds42/43",
        "last_reported_public_score": 0.818599,
        "last_reported_rank_snapshot": 21,
        "public_result_date": "2026-10-24",
        "public_result_source": "user-provided leaderboard screenshot",
        "candidate": "average three dim8 members at total weight 2/3; retain two dim16 members at total weight 1/3",
        "members": ["finetuned_dim8_seed42", "finetuned_dim8_seed43", "finetuned_dim8_seed44", "dim16_seed42", "dim16_seed43"],
        "weights": [2 / 9, 2 / 9, 2 / 9, 1 / 6, 1 / 6],
        "primary_check": "first event per user across Jan25-26; combiner trained on Jan24",
        "secondary_check": "first event per user on Jan26; combiner trained on Jan25",
        "support_rule": "primary paired-gain interval entirely above zero and secondary AUC change nonnegative",
        "bootstrap_resamples": 300, "bootstrap_seed": 2026,
        "status": "retrospective development experiment; reused validation dates",
        "source_sha256": {},
        "baseline_sources_sha256": {str(path): sha256(path) for path in baseline_paths},
    }
    for fold, data in prepared.items():
        folder = trial / fold
        for directory in ["src", "data/cache", "work", "outputs"]:
            (folder / directory).mkdir(parents=True, exist_ok=True)
        for name, text in data["scripts"].items():
            ast.parse(text)
            path = folder / "src" / name
            path.write_text(text, encoding="utf-8")
            protocol["source_sha256"][str(path.relative_to(trial))] = sha256(path)
        for name, original in data["cache"].items():
            (folder / "data/cache" / name).symlink_to(original)
    (trial / "work").mkdir()
    (trial / "outputs").mkdir()
    (trial / "outputs/frozen_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({"setup": "passed", "trial_workspace": str(trial),
                      "protocol": {k: v for k, v in protocol.items() if not k.endswith("sha256")}}, indent=2), flush=True)


def train(trial):
    import torch
    start = time.perf_counter()
    protocol = json.loads((trial / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    assert protocol["dimension"] == 8 and protocol["neural_seed"] == 44
    assert protocol["preference_total_epochs"] == 260 and protocol["event_epochs"] == 8
    verify_inputs(trial, protocol)
    records = {}
    for fold in ["jan24", "jan25", "jan26"]:
        folder = trial / fold
        records[fold] = {}
        for stage, script, filename, epoch in [
            ("initial_preference", "train_initial_preferences.py", "preference_initial.pt", 196),
            ("fine_tuned_preference", "train_preferences.py", "preference_best.pt", 260),
            ("event", "train_events.py", "event_best.pt", 8),
        ]:
            print(f"\nSeed44 {fold}: {stage}", flush=True)
            checkpoint = folder / "work" / filename
            if not checkpoint.exists():
                subprocess.run([sys.executable, str(folder / "src" / script)], cwd=folder, check=True)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert saved["dimension"] == 8 and saved["epoch"] == epoch
            if stage == "fine_tuned_preference":
                assert saved["fine_tuning_initial_epoch"] == 196
                assert saved["fine_tuning_learning_rate"] == 0.003
                assert saved["fine_tuning_sampling_seed"] == 1044
            records[fold][stage] = {"epoch": epoch, "validation_auc": float(saved["validation_auc"])}
            del saved
    verify_inputs(trial, protocol)
    report = {"seed": 44, "folds": records, "runtime_seconds": round(time.perf_counter() - start, 1)}
    (trial / "outputs/neural_training_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSEED44 TRAINING RESULTS", flush=True)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "train"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    trial = root.parent / "techarena-seed44-trial"
    if args.action == "setup":
        setup(root, trial)
    else:
        train(trial)


if __name__ == "__main__":
    main()
