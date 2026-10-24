"""Freeze the supported preference schedule for full-data training.

Run from techarena-final: python src/prepare_final_preference_finetune.py
Creates techarena-preference-finetune-trial/final with separate checkpoints.
"""

import ast
import json
from pathlib import Path

from submission_export import sha256


def replace_once(text, old, new):
    assert text.count(old) == 1, f"Expected exactly one occurrence: {old!r}"
    return text.replace(old, new, 1)


def adapt_trainer(source):
    text = replace_once(source, "    for epoch in range(1, 197):", "    for epoch in range(197, 261):")
    text = replace_once(text, "    optimizer = torch.optim.Adam(model.parameters(), lr=0.012)", '''    checkpoint = torch.load(
        protocol["initial_preference_checkpoint"], map_location="cpu", weights_only=True
    )
    for key in ["seed", "n_users", "n_items", "dimension", "cutoff"]:
        assert checkpoint[key] == protocol[key], key
    assert checkpoint["stage"] == "preference" and checkpoint["epoch"] == 196
    assert torch.equal(checkpoint["model_state"]["active_users"], active_users)
    model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    for group in optimizer.param_groups:
        assert group["lr"] == 0.012
        group["lr"] = 0.003
    del checkpoint
    torch.manual_seed(protocol["fine_tuning_sampling_seed"])''')
    text = replace_once(text, "        if epoch == 1 or epoch % 20 == 0 or epoch == 196:", "        if epoch == 197 or epoch % 16 == 0 or epoch == 260:")
    text = replace_once(text, '    save_checkpoint(path, model, optimizer, protocol, "preference", 196)', '    save_checkpoint(path, model, optimizer, protocol, "preference", 260)')
    text = replace_once(text, '    return {"epoch": 196, "training_loss": mean_loss, "checkpoint": str(path)}', '    return {"epoch": 260, "training_loss": mean_loss, "checkpoint": str(path)}')
    text = replace_once(text, '    assert checkpoint["stage"] == "preference" and checkpoint["epoch"] == 196\n\n    # Recreate the models', '    assert checkpoint["stage"] == "preference" and checkpoint["epoch"] == 260\n\n    # Recreate the models')
    text = replace_once(text, '        preference_result = {"epoch": 196, "checkpoint": str(preference_path)}', '        preference_result = {"epoch": 260, "checkpoint": str(preference_path)}')
    text = replace_once(text, '        "stage": stage,\n        "epoch": epoch,\n    }', '''        "stage": stage,
        "epoch": epoch,
        "fine_tuning_initial_epoch": 196,
        "fine_tuning_learning_rate": 0.003,
        "fine_tuning_sampling_seed": protocol["fine_tuning_sampling_seed"],
        "initial_preference_checkpoint_sha256": protocol["initial_preference_checkpoint_sha256"],
    }''')
    text = replace_once(text, '    cache = root / "data/cache"', '''    cache = root / "data/cache"
    frozen = json.loads((root / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    assert frozen["trainer_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert frozen["preference_initial_epochs"] == 196
    assert frozen["preference_extra_epochs"] == 64
    assert frozen["preference_total_epochs"] == 260
    assert frozen["preference_fine_tuning_learning_rate"] == 0.003
    assert frozen["event_epochs"] == 8
    assert frozen["weights"] == [1 / 3, 1 / 3, 1 / 6, 1 / 6]
    initial_info = frozen["initial_preference_checkpoints"][f"seed{args.seed}"]
    initial_path = Path(initial_info["path"])
    assert hashlib.sha256(initial_path.read_bytes()).hexdigest() == initial_info["sha256"]
    report_path = Path(frozen["validation_report_path"])
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == frozen["validation_report_sha256"]''')
    text = replace_once(text, '        "preference_epochs": 196,', '''        "preference_epochs": 260,
        "preference_initial_epochs": 196,
        "preference_extra_epochs": 64,
        "preference_initial_learning_rate": 0.012,
        "fine_tuning_sampling_seed": args.seed + 1000,
        "original_rng_state_available": False,
        "adam_moments_reused": True,
        "initial_preference_checkpoint": str(initial_path),
        "initial_preference_checkpoint_sha256": initial_info["sha256"],''')
    text = replace_once(text, '        "preference_learning_rate": 0.012,', '        "preference_learning_rate": 0.003,')
    text = replace_once(text, 'print("\\nFINAL TRAINING RESULTS", flush=True)', 'print("\\nFINAL PREFERENCE FINE-TUNING RESULTS", flush=True)')
    old_doc = ast.get_docstring(ast.parse(text))
    assert old_doc is not None
    first_node = ast.parse(text).body[0]
    source_doc = ast.get_source_segment(text, first_node)
    text = replace_once(text, source_doc, '''"""Fine-tune the two full-data dim8 preference models, then refit events.

Prepared from the existing full-data trainer. Adds 64 preference epochs at
0.003 with saved Adam moments and sampling seed seed+1000, then fits 8 event
epochs. Reads training data only; evaluation labels are never loaded.
Run this file with --seed 42 or --seed 43; --check-only checks inputs.
"""''')
    before = [ast.dump(n, include_attributes=False) for n in ast.parse(source).body if isinstance(n, ast.ClassDef)]
    after = [ast.dump(n, include_attributes=False) for n in ast.parse(text).body if isinstance(n, ast.ClassDef)]
    assert before == after, "Model architecture changed"
    ast.parse(text)
    return text


def main():
    import torch
    from catboost import CatBoostClassifier
    root = Path(__file__).resolve().parents[1]
    trial = root.parent / "techarena-preference-finetune-trial"
    final = trial / "final"
    if final.exists():
        raise FileExistsError(f"Final workspace already exists: {final}. Use the printed training commands after setup.")
    report_path = trial / "outputs/ensemble_trial_results.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    protocol = json.loads((trial / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    public_protocol = {key: value for key, value in protocol.items() if not key.endswith("sha256")}
    assert report["protocol"] == public_protocol
    assert report["supported_for_further_confirmation"] is True
    assert report["primary_paired_gain_95_percentile_interval"][0] > 0
    assert report["two_day_primary"]["auc_change"] > 0
    assert report["jan26_secondary"]["auc_change"] >= 0
    assert protocol["dimension"] == 8 and protocol["seeds"] == [42, 43]
    assert protocol["preference_initial_epochs"] == 196 and protocol["preference_extra_epochs"] == 64
    assert protocol["preference_total_epochs"] == 260
    assert protocol["preference_fine_tuning_learning_rate"] == 0.003
    assert protocol["fine_tuning_sampling_seeds"] == [1042, 1043]
    assert protocol["event_epochs"] == 8 and protocol["combiner_trees"] == 545
    assert not protocol["preference_early_stopping"] and not protocol["combiner_early_stopping"]
    assert protocol["weights"] == [1 / 3, 1 / 3, 1 / 6, 1 / 6]
    for section in ["source_sha256", "initial_checkpoint_sha256"]:
        for name, expected in protocol[section].items():
            assert sha256(trial / name) == expected, name
    cache_sources = {name: (root / "data/cache" / name).resolve(strict=True)
                     for name in ["train_events.npz", "preference_training.npz"]}
    trainer_source = root / "src/train_final_models.py"
    original_source_hash = sha256(trainer_source)
    trainer = adapt_trainer(trainer_source.read_text(encoding="utf-8"))
    frozen = {key: value for key, value in public_protocol.items()}
    frozen.update({
        "training_cutoff": "2025-01-27T00:00:00+00:00",
        "evaluation_labels_used": False,
        "validation_report_path": str(report_path),
        "validation_report_sha256": sha256(report_path),
        "original_trainer_sha256": original_source_hash,
        "initial_preference_checkpoints": {},
        "combiner_sources": {},
    })
    for seed in [42, 43]:
        folder = root / "work" / f"seed{seed}"
        old_protocol = json.loads((folder / "training_protocol.json").read_text(encoding="utf-8"))
        assert old_protocol["trainer_sha256"] == original_source_hash
        initial = folder / "preference_final.pt"
        checkpoint = torch.load(initial, map_location="cpu", weights_only=True)
        for name, expected in {"seed": seed, "dimension": 8, "epoch": 196,
                               "stage": "preference", "cutoff": 1737936000,
                               "n_users": 54456, "n_items": 50000}.items():
            assert checkpoint[name] == expected, (initial, name)
        assert all(group["lr"] == 0.012 for group in checkpoint["optimizer_state"]["param_groups"])
        del checkpoint
        frozen["initial_preference_checkpoints"][f"seed{seed}"] = {"path": str(initial), "sha256": sha256(initial)}
        combiner = trial / "work" / f"dim8_seed{seed}_combiner_jan25.cbm"
        model = CatBoostClassifier()
        model.load_model(str(combiner))
        assert model.tree_count_ == 545 and len(model.feature_names_) == 37
        frozen["combiner_sources"][f"seed{seed}"] = {"path": str(combiner), "sha256": sha256(combiner)}
    for directory in ["src", "data/cache", "work", "outputs"]:
        (final / directory).mkdir(parents=True, exist_ok=True)
    for name, original in cache_sources.items():
        (final / "data/cache" / name).symlink_to(original)
    trainer_path = final / "src/train_final_models.py"
    trainer_path.write_text(trainer, encoding="utf-8")
    frozen["trainer_sha256"] = sha256(trainer_path)
    (final / "outputs/frozen_protocol.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(json.dumps({"setup": "passed", "final_workspace": str(final), "schedule": {
        "initial_preference_epochs": 196, "extra_preference_epochs": 64,
        "fine_tuning_learning_rate": 0.003, "event_epochs": 8,
        "sampling_seeds": [1042, 1043], "ensemble_weights": protocol["weights"],
    }}, indent=2), flush=True)
    print("\nCheck and training commands:")
    for seed in [42, 43]:
        print(f"python {trainer_path} --seed {seed} --check-only")
        print(f"python {trainer_path} --seed {seed}")


if __name__ == "__main__":
    main()
