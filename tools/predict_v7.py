"""Verify or generate the fixed V7 candidate from the fingerprinted V6 submission."""

import argparse
import csv
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "legacy"))

from consolidate import digest, read_json, utc, write_json
from submission_export import export_submission, validate_csv
from row_auc_final_utils import verify_csv_probabilities


def calibrate_v7_probabilities(probs: np.ndarray, gamma: float = 1.12) -> np.ndarray:
    """Sharpen allocation, preserving shape, sums and within-event order; never clip."""
    original = np.asarray(probs, dtype=np.float64)
    if original.ndim not in (1, 2) or (original.ndim == 2 and original.shape[1] != 10):
        raise ValueError("Expected flat probabilities or a ten-candidate matrix.")
    if original.size == 0 or original.size % 10:
        raise ValueError("Expected a nonempty whole number of ten-candidate events.")
    if not np.isfinite(original).all() or np.any((original < 0) | (original > 1)):
        raise ValueError("Input probabilities must be finite and in [0, 1].")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be a finite positive number.")
    p = original.reshape(-1, 10)
    event_sums = p.sum(axis=1, keepdims=True)
    p_gamma = p ** gamma
    powered_sums = p_gamma.sum(axis=1, keepdims=True)
    # Preserve reference arithmetic except for underflowed positive events.
    underflow = (powered_sums[:, 0] == 0) & (event_sums[:, 0] > 0)
    if underflow.any():
        scaled = p[underflow] / p[underflow].max(axis=1, keepdims=True)
        p_gamma[underflow] = scaled ** gamma
        powered_sums[underflow] = p_gamma[underflow].sum(axis=1, keepdims=True)
    conditional = np.divide(p_gamma, powered_sums, out=np.zeros_like(p_gamma), where=powered_sums > 0)
    result = conditional * event_sums
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("Sharpening produced scores outside [0, 1]; no clipping is permitted.")
    return result.reshape(original.shape)


def load_aligned_probabilities(path, rowids):
    """Match opaque CSV IDs exactly, independent of serialization order."""
    ids = np.asarray(rowids)
    if ids.ndim != 1 or ids.dtype.kind != "U" or not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("Cached row IDs must be a nonempty vector of unique text identifiers.")
    positions = {rowid: i for i, rowid in enumerate(ids.tolist())}
    seen = set()
    probs = np.empty(len(ids), dtype=np.float64)
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != ["rowid", "predicted_probability"]:
            raise ValueError("Unexpected submission header.")
        for row in reader:
            if len(row) != 2 or row[0] not in positions or row[0] in seen:
                raise ValueError("Submission contains duplicate, unknown or malformed row IDs.")
            value = float(row[1])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Submission probabilities must be finite and in [0, 1].")
            seen.add(row[0])
            probs[positions[row[0]]] = value
    if len(seen) != len(ids):
        raise ValueError("Submission is missing expected row IDs.")
    return probs


def validate_output_path(output, root, inputs):
    output = Path(output).resolve()
    try:
        output.relative_to((root / "artifacts").resolve())
    except ValueError as error:
        raise ValueError("Output must be inside the repository's ignored artifacts directory.") from error
    for path in inputs:
        if output == path.resolve() or (output.exists() and path.exists() and os.path.samefile(output, path)):
            raise ValueError("Output aliases a protected input; refusing to overwrite it.")
    return output


def generate(output, root=ROOT, gamma=None):
    root = Path(root).resolve()
    config = read_json(root / "configs/v7.json")
    baseline_config = read_json(root / "configs/v6.json")
    fixed_gamma = float(config["sharpening_gamma"])
    if gamma is not None and (not np.isfinite(gamma) or gamma != fixed_gamma):
        raise ValueError(f"This fingerprinted recipe requires gamma={fixed_gamma}; no parameter sweep is supported.")
    baselines = [root / "artifacts/submissions/v6.csv", root / "artifacts/reproduced_v6.csv"]
    example = root / "data/raw/example_submission.csv"
    rowid_path = root / "data/cache/evaluation_rowids.npy"
    output = validate_output_path(output, root, [*baselines, example, rowid_path])
    v6_csv = next((path for path in baselines if path.is_file()), None)
    if v6_csv is None:
        raise FileNotFoundError("Missing V6 baseline submission.")
    baseline_hash = digest(v6_csv)
    if baseline_hash != baseline_config["submission_sha256"]:
        raise RuntimeError("V6 input does not match the protected baseline fingerprint.")
    rowids = np.load(rowid_path, allow_pickle=False)
    if rowids.shape != (config["evaluation_rows"],):
        raise ValueError("Evaluation row count differs from the configuration.")
    baseline = load_aligned_probabilities(v6_csv, rowids)
    transformed = calibrate_v7_probabilities(baseline, fixed_gamma)
    expected_hash = config["submission_sha256"]
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    identity = {"version": "v7", "submission_sha256": expected_hash,
                "baseline_sha256": baseline_hash, "gamma": fixed_gamma,
                "rows": config["evaluation_rows"], "evaluation_rowids_sha256": digest(rowid_path)}
    if receipt_path.exists():
        previous = read_json(receipt_path)
        if any(previous.get(key) != value for key, value in identity.items()):
            raise FileExistsError("Existing V7 receipt describes different inputs or outputs.")

    def check(candidate):
        report = validate_csv(candidate, example, config["evaluation_rows"])
        verify_csv_probabilities(candidate, rowids, transformed)
        if digest(candidate) != expected_hash:
            raise RuntimeError("V7 output differs from the configured fingerprint; no output was replaced.")
        return report

    if output.exists():
        if digest(output) != expected_hash:
            raise FileExistsError(f"Refusing to overwrite an existing different file: {output}")
        report = check(output)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".v7-", dir=output.parent) as temporary:
            candidate = Path(temporary) / "candidate.csv"
            export_submission(rowids, transformed, example, candidate, config["evaluation_rows"])
            report = check(candidate)
            if digest(v6_csv) != baseline_hash or digest(rowid_path) != identity["evaluation_rowids_sha256"]:
                raise RuntimeError("Protected inputs changed during generation.")
            # Same-filesystem publication is atomic and cannot replace a racer.
            os.link(candidate, output)
    receipt = {**identity, "created_at": utc(), "min_probability": report["minimum_probability"],
               "max_probability": report["maximum_probability"], "mean_probability": report["mean_probability"]}
    if not receipt_path.exists():
        with tempfile.TemporaryDirectory(prefix=".v7-receipt-", dir=output.parent) as temporary:
            candidate = Path(temporary) / "receipt.json"
            write_json(candidate, receipt)
            os.link(candidate, receipt_path)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/submissions/v7_candidate.csv")
    parser.add_argument("--gamma", type=float, help="Compatibility option; must equal the fixed configured value")
    args = parser.parse_args()
    report = generate(args.output, gamma=args.gamma)
    print(f"Verified V7 candidate: {args.output}\nSHA256: {report['submission_sha256']}")


if __name__ == "__main__":
    main()
