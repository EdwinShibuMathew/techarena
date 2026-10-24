"""Write a competition CSV in exactly the example submission's row order."""

import csv
import hashlib
import math
from itertools import zip_longest
from pathlib import Path

import numpy as np

HEADER = ["rowid", "predicted_probability"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_csv(path, example_path, expected_rows):
    seen = set()
    count = 0
    minimum, maximum = float("inf"), float("-inf")
    total = 0.0
    with Path(path).open(newline="", encoding="utf-8") as output:
        with Path(example_path).open(newline="", encoding="utf-8-sig") as example:
            actual = csv.reader(output)
            reference = csv.reader(example)
            if next(actual, None) != HEADER or next(reference, None) != HEADER:
                raise ValueError("Submission header does not match the example.")
            for row, template in zip_longest(actual, reference):
                if row is None or template is None:
                    raise ValueError("Submission row count differs from the example.")
                if len(row) != 2 or len(template) != 2 or row[0] != template[0]:
                    raise ValueError("Submission row IDs or column count differ from the example.")
                if row[0] in seen:
                    raise ValueError(f"Duplicate submission row ID: {row[0]}")
                seen.add(row[0])
                probability = float(row[1])
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError(f"Invalid probability at row {count + 1}")
                minimum = min(minimum, probability)
                maximum = max(maximum, probability)
                total += probability
                count += 1
    if count != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {count}.")
    return {
        "rows": count,
        "header": HEADER,
        "unique_rowids": len(seen),
        "same_rowids_and_order_as_example": True,
        "all_probabilities_finite_and_in_range": True,
        "minimum_probability": minimum,
        "maximum_probability": maximum,
        "mean_probability": total / count,
    }


def export_submission(rowids, predictions, example_path, output_path, expected_rows):
    rowids = np.asarray(rowids)
    predictions = np.asarray(predictions, dtype=np.float64).ravel()
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"A submission already exists: {output_path}")
    if rowids.ndim != 1 or rowids.dtype.kind != "U":
        raise ValueError("Expected a one-dimensional array of text row IDs.")
    if len(rowids) != expected_rows or len(predictions) != expected_rows:
        raise ValueError("Prediction or row ID count is incorrect.")
    if not np.isfinite(predictions).all() or not ((predictions >= 0) & (predictions <= 1)).all():
        raise ValueError("Predictions must be finite and between zero and one.")

    position = {rowid: index for index, rowid in enumerate(rowids)}
    if len(position) != expected_rows:
        raise ValueError("The cached evaluation row IDs contain duplicates.")
    seen = set()
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Path(example_path).open(newline="", encoding="utf-8-sig") as example:
            reader = csv.reader(example)
            if next(reader, None) != HEADER:
                raise ValueError("Unexpected example submission header.")
            with temporary.open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(HEADER)
                for row in reader:
                    if len(row) != 2 or row[0] not in position or row[0] in seen:
                        raise ValueError("Example contains an unknown or duplicate row ID.")
                    rowid = row[0]
                    seen.add(rowid)
                    writer.writerow([rowid, format(float(predictions[position[rowid]]), ".17g")])
        if len(seen) != expected_rows:
            raise ValueError("Example submission is missing evaluation row IDs.")
        report = validate_csv(temporary, example_path, expected_rows)
        temporary.replace(output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    report.update({
        "submission_ready": True,
        "submission_path": str(output_path),
        "submission_bytes": output_path.stat().st_size,
        "submission_sha256": sha256(output_path),
    })
    return report
