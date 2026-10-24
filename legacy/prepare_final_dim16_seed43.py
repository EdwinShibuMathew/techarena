"""Lock the supported seed43 candidate before training on all seven days.

Run in techarena-final: python src/prepare_final_dim16_seed43.py
Reuses the full-data dimension-16 trainer and verified cached inputs.
"""

import json
from pathlib import Path

from submission_export import sha256


def main():
    root = Path(__file__).resolve().parents[1]
    previous = root.parent / "techarena-dim16-trial"
    trial = root.parent / "techarena-dim16-seed43-trial"
    final = previous / "final"
    report = json.loads((trial / "outputs/ensemble_trial_results.json").read_text(encoding="utf-8"))
    assert report["supported_for_further_confirmation"] is True
    assert report["primary_paired_gain_95_percentile_interval"][0] > 0
    assert report["jan26_secondary"]["auc_change"] >= 0
    weights = [1 / 3, 1 / 3, 1 / 6, 1 / 6]
    assert report["protocol"]["planned_candidate_weights"] == weights
    original = json.loads((final / "outputs/frozen_protocol.json").read_text(encoding="utf-8"))
    assert original["dimension"] == 16
    trainer = final / "src/train_final_models.py"
    assert sha256(trainer) == original["trainer_sha256"]
    assert (final / "work/seed42/event_final.pt").is_file()
    combiner = trial / "work/combiner_jan25.cbm"
    assert combiner.is_file()
    protocol = {
        "dimension": 16,
        "seed": 43,
        "preference_epochs": 196,
        "event_epochs": 8,
        "training_cutoff": "2025-01-27T00:00:00+00:00",
        "combiner_source": str(combiner),
        "ensemble_members": ["dim8_seed42", "dim8_seed43", "dim16_seed42", "dim16_seed43"],
        "ensemble_weights": weights,
        "trainer_sha256": sha256(trainer),
    }
    destination = final / "outputs/seed43_frozen_protocol.json"
    if destination.exists():
        assert json.loads(destination.read_text(encoding="utf-8")) == protocol
    else:
        destination.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({"setup": "passed", "final_dim16_workspace": str(final), "protocol": protocol}, indent=2))
    print("\nTraining command:")
    print(f"python {trainer} --seed 43")


if __name__ == "__main__":
    main()
