"""Prepare the selected dim16 seed42 model for training on all seven days.

Run in techarena-final: python src/prepare_final_dim16.py
Then run the printed train_final_models.py command in WSL.
"""

import ast
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    trial = root.parent / "techarena-dim16-trial"
    destination = trial / "final"
    if destination.exists():
        raise FileExistsError(f"Final dim16 workspace already exists: {destination}")
    report = json.loads((trial / "outputs/ensemble_trial_results.json").read_text(encoding="utf-8"))
    assert report["supported_for_further_confirmation"] is True
    assert report["primary_paired_gain_95_percentile_interval"][0] > 0
    assert report["jan26_secondary"]["auc_change"] >= 0
    assert report["protocol"]["planned_candidate_weights"] == [1 / 3, 1 / 3, 1 / 3]

    source = (root / "src/train_final_models.py").read_text(encoding="utf-8")
    assert source.count('"dimension": 8,') == 1
    updated = source.replace('"dimension": 8,', '"dimension": 16,', 1)
    before_classes = [ast.dump(node, include_attributes=False) for node in ast.parse(source).body if isinstance(node, ast.ClassDef)]
    after_classes = [ast.dump(node, include_attributes=False) for node in ast.parse(updated).body if isinstance(node, ast.ClassDef)]
    assert before_classes == after_classes

    caches = {
        name: (root / "data/cache" / name).resolve(strict=True)
        for name in [
            "train_events.npz", "preference_training.npz",
            "evaluation_events.npz", "evaluation_rowids.npy", "mappings.pkl",
        ]
    }
    for directory in ["src", "data/cache", "work", "outputs"]:
        (destination / directory).mkdir(parents=True, exist_ok=True)
    trainer = destination / "src/train_final_models.py"
    trainer.write_text(updated, encoding="utf-8")
    for name, original in caches.items():
        (destination / "data/cache" / name).symlink_to(original)
    protocol = {
        "dimension": 16,
        "seed": 42,
        "preference_epochs": 196,
        "event_epochs": 8,
        "training_cutoff": "2025-01-27T00:00:00+00:00",
        "combiner_source": str(trial / "work/combiner_jan25.cbm"),
        "ensemble_members": ["dim8_seed42", "dim8_seed43", "dim16_seed42"],
        "ensemble_weights": [1 / 3, 1 / 3, 1 / 3],
        "trainer_sha256": hashlib.sha256(trainer.read_bytes()).hexdigest(),
    }
    (destination / "outputs/frozen_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({"setup": "passed", "final_dim16_workspace": str(destination), "protocol": protocol}, indent=2))
    print("\nTraining command:")
    print(f"python {trainer} --seed 42")


if __name__ == "__main__":
    main()
