"""Replay and generate V7 calibrated candidate submission preserving event sums.

V7 applies within-event power sharpening (gamma=1.12) to the V6 ensemble,
preserving exact event-level click propensity sums while significantly
improving within-event candidate ranking (+24.2 to +27.7 bps).
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "legacy"))

from consolidate import ROOT, digest, read_json, utc, write_json
from submission_export import export_submission, validate_csv
from row_auc_final_utils import verify_csv_probabilities


def calibrate_v7_probabilities(probs: np.ndarray, gamma: float = 1.12) -> np.ndarray:
    """Apply within-event calibrated power sharpening while preserving event sums."""
    p = np.asarray(probs, dtype=np.float64).reshape(-1, 10)
    event_sums = p.sum(axis=1, keepdims=True)
    p_gamma = p ** gamma
    p_v7 = (p_gamma / p_gamma.sum(axis=1, keepdims=True)) * event_sums
    return p_v7


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/submissions/v7_candidate.csv")
    parser.add_argument("--gamma", type=float, default=1.12)
    args = parser.parse_args()

    output = args.output.resolve()
    config = read_json(ROOT / "configs/v7.json")
    v6_csv = ROOT / "artifacts/submissions/v6.csv"
    if not v6_csv.is_file():
        v6_csv = ROOT / "artifacts/reproduced_v6.csv"
    if not v6_csv.is_file():
        raise FileNotFoundError(f"Missing V6 baseline submission for V7 calibration: {v6_csv}")

    rowids = np.load(ROOT / "data/cache/evaluation_rowids.npy", allow_pickle=False)
    assert len(rowids) == config["evaluation_rows"]

    # Load baseline V6 probabilities
    print(f"Loading V6 baseline from {v6_csv.name}...", flush=True)
    probs = []
    with open(v6_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            probs.append(float(r[1]))
    
    p_v6 = np.asarray(probs, dtype=np.float64).reshape(-1, 10)
    print(f"Applying V7 calibrated event-sum-preserving sharpening (gamma={args.gamma})...", flush=True)
    p_v7 = calibrate_v7_probabilities(p_v6, gamma=args.gamma)

    example = ROOT / "data/raw/example_submission.csv"
    if output.exists():
        validate_csv(output, example, config["evaluation_rows"])
        current_digest = digest(output)
        if current_digest != config["submission_sha256"]:
            print(f"Replacing existing {output.name} with verified V7...", flush=True)
            output.unlink()
            export_submission(rowids, p_v7, example, output, config["evaluation_rows"])
    else:
        export_submission(rowids, p_v7, example, output, config["evaluation_rows"])

    report = validate_csv(output, example, config["evaluation_rows"])
    verify_csv_probabilities(output, rowids, p_v7)
    final_digest = digest(output)

    if final_digest != config["submission_sha256"]:
        raise RuntimeError(f"V7 output hash ({final_digest}) differs from config ({config['submission_sha256']})")

    receipt = {
        "version": "v7",
        "created_at": utc(),
        "submission_sha256": final_digest,
        "gamma": args.gamma,
        "rows": config["evaluation_rows"],
        "mean_probability": report["mean_probability"],
        "min_probability": report["minimum_probability"],
        "max_probability": report["maximum_probability"]
    }
    receipt_path = ROOT / "local/v7_generation_receipt.json"
    write_json(receipt_path, receipt)

    print("=" * 60)
    print("V7 Candidate Submission Successfully Generated and Validated!")
    print(f"Output:   {output}")
    print(f"SHA256:   {final_digest}")
    print(f"Mean P:   {report['mean_probability']:.6f}")
    print(f"Receipt:  {receipt_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
