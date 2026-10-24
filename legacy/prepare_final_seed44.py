"""Freeze the supported seed44 candidate and prepare full-data training.

Run in techarena-final: python src/prepare_final_seed44.py
Creates techarena-seed44-trial/final; existing V4 models are preserved.
"""

import ast
import json
from pathlib import Path

from prepare_final_preference_finetune import adapt_trainer, replace_once
from prepare_seed44_trial import verify_inputs
from submission_export import sha256


FINAL_MAIN = '''
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    frozen_path = root / "outputs/frozen_protocol.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert source_hash == frozen["trainer_sha256"]
    assert frozen["neural_seed"] == 44 and frozen["dimension"] == 8
    assert frozen["preference_initial_epochs"] == 196
    assert frozen["preference_extra_epochs"] == 64
    assert frozen["preference_total_epochs"] == 260
    assert frozen["preference_initial_learning_rate"] == 0.012
    assert frozen["preference_fine_tuning_learning_rate"] == 0.003
    assert frozen["fine_tuning_sampling_seed"] == 1044
    assert frozen["event_epochs"] == 8
    assert frozen["weights"] == [2 / 9, 2 / 9, 2 / 9, 1 / 6, 1 / 6]
    for section in ["locked_training_sources", "locked_trial_sources", "locked_baseline_sources"]:
        for name, expected in frozen[section].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    report_path = Path(frozen["validation_report_path"])
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == frozen["validation_report_sha256"]
    assert torch.cuda.is_available(), "Activate the techarena environment with working CUDA."
    torch.set_num_threads(4)
    positive_users, positive_items, event_data, n_users, n_items, cutoff = load_training_data(root / "data/cache")
    folder = root / "work/seed44"
    initial_path = folder / "preference_initial.pt"
    preference_path = folder / "preference_final.pt"
    event_path = folder / "event_final.pt"
    protocol = {
        "seed": 44, "dimension": 8, "preference_initial_epochs": 196,
        "preference_extra_epochs": 64, "preference_epochs": 260,
        "preference_initial_learning_rate": 0.012,
        "preference_learning_rate": 0.003, "fine_tuning_sampling_seed": 1044,
        "adam_moments_reused": True, "event_epochs": 8,
        "event_learning_rate": 0.0015, "preference_batch_size": 4096,
        "event_batch_size": 2048, "sampled_comparisons_per_click": 31,
        "preference_regularization": 0.02, "event_anchor_regularization": 0.1,
        "event_effect_regularization": 0.05, "n_users": n_users,
        "n_items": n_items, "cutoff": cutoff, "training_events": len(event_data[0]),
        "click_examples": len(positive_users), "evaluation_labels_used": False,
        "initial_preference_checkpoint": str(initial_path),
        "initial_preference_checkpoint_sha256": None,
        "trainer_sha256": source_hash, "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__), "gpu": torch.cuda.get_device_name(0),
    }
    print(json.dumps({"input_check": "passed", **protocol}, indent=2), flush=True)
    if args.check_only:
        return
    folder.mkdir(parents=True, exist_ok=True)
    output = root / "outputs"
    start = time.perf_counter()

    def check(path, stage, epoch):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        for name in ["seed", "dimension", "n_users", "n_items", "cutoff"]:
            assert checkpoint[name] == protocol[name], (path, name)
        assert checkpoint["stage"] == stage and checkpoint["epoch"] == epoch, path
        if epoch == 260 or stage == "event":
            assert checkpoint["fine_tuning_initial_epoch"] == 196
            assert checkpoint["fine_tuning_learning_rate"] == 0.003
            assert checkpoint["fine_tuning_sampling_seed"] == 1044
            assert checkpoint["initial_preference_checkpoint_sha256"] == protocol["initial_preference_checkpoint_sha256"]

    if initial_path.exists():
        check(initial_path, "preference", 196)
        initial_result = {"epoch": 196, "checkpoint": str(initial_path), "reused_completed_checkpoint": True}
    else:
        assert not preference_path.exists() and not event_path.exists()
        initial_result = train_initial_preferences(
            positive_users, positive_items, protocol, initial_path,
            output / "seed44_initial_preference_metrics.json",
        )
    initial_hash = hashlib.sha256(initial_path.read_bytes()).hexdigest()
    protocol["initial_preference_checkpoint_sha256"] = initial_hash
    protocol_path = folder / "training_protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        for name in ["seed", "trainer_sha256", "initial_preference_checkpoint_sha256"]:
            assert previous[name] == protocol[name], name
    write_json(protocol_path, protocol)
    torch.cuda.empty_cache()
    if preference_path.exists():
        check(preference_path, "preference", 260)
        preference_result = {"epoch": 260, "checkpoint": str(preference_path), "reused_completed_checkpoint": True}
    else:
        assert not event_path.exists()
        preference_result = train_preferences(
            positive_users, positive_items, protocol, preference_path,
            output / "seed44_preference_metrics.json",
        )
    check(preference_path, "preference", 260)
    torch.cuda.empty_cache()
    if event_path.exists():
        check(event_path, "event", 8)
        event_result = {"epoch": 8, "checkpoint": str(event_path), "reused_completed_checkpoint": True}
    else:
        event_result = train_events(
            event_data, protocol, preference_path, event_path,
            output / "seed44_event_metrics.json",
        )
    check(event_path, "event", 8)
    assert hashlib.sha256(initial_path.read_bytes()).hexdigest() == initial_hash
    result = {
        "seed": 44, "training_events": protocol["training_events"],
        "click_examples": protocol["click_examples"], "initial_preference": initial_result,
        "preference": preference_result, "event": event_result,
        "runtime_seconds": round(time.perf_counter() - start, 1),
    }
    write_json(output / "final_seed44_metrics.json", result)
    print("\\nFINAL SEED44 TRAINING RESULTS", flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
'''


def build_trainer(source):
    adapted = adapt_trainer(source)
    original_tree = ast.parse(source)
    adapted_tree = ast.parse(adapted)
    original_functions = {n.name: ast.get_source_segment(source, n) for n in original_tree.body if isinstance(n, ast.FunctionDef)}
    adapted_functions = {n.name: ast.get_source_segment(adapted, n) for n in adapted_tree.body if isinstance(n, ast.FunctionDef)}
    imports = [ast.get_source_segment(source, n) for n in original_tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    models = [ast.get_source_segment(source, n) for n in original_tree.body if isinstance(n, ast.ClassDef)]
    initial = replace_once(original_functions["train_preferences"], "def train_preferences(", "def train_initial_preferences(")
    saver = replace_once(original_functions["save_checkpoint"],
                         '    temporary = path.with_suffix(path.suffix + ".tmp")',
                         '''    if epoch == 260 or stage == "event":
        state.update({
            "fine_tuning_initial_epoch": 196,
            "fine_tuning_learning_rate": 0.003,
            "fine_tuning_sampling_seed": protocol["fine_tuning_sampling_seed"],
            "initial_preference_checkpoint_sha256": protocol["initial_preference_checkpoint_sha256"],
        })
    temporary = path.with_suffix(path.suffix + ".tmp")''')
    parts = [
        '"""Train full-data seed44 with the tested 196+64 preference and 8-event schedule."""',
        *imports, *models, original_functions["include_no_click"],
        original_functions["write_json"], saver, original_functions["load_training_data"],
        initial, adapted_functions["train_preferences"], adapted_functions["train_events"], FINAL_MAIN,
    ]
    text = "\n\n".join(parts) + "\n"
    ast.parse(text)
    return text


def main():
    from catboost import CatBoostClassifier
    root = Path(__file__).resolve().parents[1]
    trial = root.parent / "techarena-seed44-trial"
    final = trial / "final"
    if final.exists():
        raise FileExistsError(f"Already prepared: {final}. Run its trainer after setup.")
    protocol = json.loads((trial / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    report_path = trial / "outputs/ensemble_trial_results.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    public_protocol = {k: v for k, v in protocol.items() if not k.endswith("sha256")}
    assert report["protocol"] == public_protocol
    assert report["supported_for_further_confirmation"] is True
    assert report["primary_paired_gain_95_percentile_interval"][0] > 0
    assert report["two_day_primary"]["auc_change"] > 0 and report["jan26_secondary"]["auc_change"] >= 0
    assert protocol["neural_seed"] == 44 and protocol["dimension"] == 8
    assert protocol["preference_initial_epochs"] == 196 and protocol["preference_extra_epochs"] == 64
    assert protocol["preference_total_epochs"] == 260 and protocol["fine_tuning_sampling_seed"] == 1044
    assert protocol["preference_initial_learning_rate"] == 0.012
    assert protocol["preference_fine_tuning_learning_rate"] == 0.003
    assert protocol["event_epochs"] == 8 and protocol["combiner_trees"] == 545
    assert not protocol["preference_early_stopping"] and not protocol["combiner_early_stopping"]
    assert protocol["weights"] == [2 / 9, 2 / 9, 2 / 9, 1 / 6, 1 / 6]
    verify_inputs(trial, protocol)
    cache = {name: (root / "data/cache" / name).resolve(strict=True)
             for name in ["train_events.npz", "preference_training.npz"]}
    original_trainer = root / "src/train_final_models.py"
    v4 = root.parent / "techarena-preference-finetune-trial/final"
    v4_protocol = json.loads((v4 / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    assert sha256(original_trainer) == v4_protocol["original_trainer_sha256"]
    trainer = build_trainer(original_trainer.read_text(encoding="utf-8"))
    combiner = trial / "work/combiner_jan25.cbm"
    model = CatBoostClassifier()
    model.load_model(str(combiner))
    assert model.tree_count_ == 545 and len(model.feature_names_) == 37
    frozen = dict(public_protocol)
    frozen.update({
        "training_cutoff": "2025-01-27T00:00:00+00:00", "evaluation_labels_used": False,
        "validation_report_path": str(report_path), "validation_report_sha256": sha256(report_path),
        "combiner_source": {"path": str(combiner), "sha256": sha256(combiner)},
        "locked_training_sources": {str(original_trainer): sha256(original_trainer)},
        "locked_trial_sources": {str(trial / name): expected for name, expected in protocol["source_sha256"].items()},
        "locked_baseline_sources": protocol["baseline_sources_sha256"],
    })
    for directory in ["src", "data/cache", "work", "outputs"]:
        (final / directory).mkdir(parents=True, exist_ok=True)
    for name, original in cache.items():
        (final / "data/cache" / name).symlink_to(original)
    trainer_path = final / "src/train_final_models.py"
    trainer_path.write_text(trainer, encoding="utf-8")
    frozen["trainer_sha256"] = sha256(trainer_path)
    (final / "outputs/frozen_protocol.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(json.dumps({"setup": "passed", "final_workspace": str(final),
                      "seed": 44, "preference_epochs": "196 + 64", "event_epochs": 8,
                      "weights": protocol["weights"]}, indent=2), flush=True)
    print("\nCheck and training commands:")
    print(f"python {trainer_path} --check-only")
    print(f"python {trainer_path}")


if __name__ == "__main__":
    main()
