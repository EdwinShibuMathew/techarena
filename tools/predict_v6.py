"""Replay submitted V6 using repository-relative assets, without old study paths."""

import argparse
import inspect
import os
import sys
import tempfile
from pathlib import Path

from consolidate import ROOT, asset, digest, read_json, utc, verify_bundle, write_json, source_manifest_binding


def portable_pipeline(original):
    """Both dimensions share the original formulas; relax only the dimension guard.

    The historical dim16 pipeline was the same dimensional clone. Do not patch
    module globals, alter model parameters, or silently accept different scores.
    """
    text = inspect.getsource(original)
    guard = 'assert dimension == 8 and checkpoint["epoch"] == 8'
    if text.count(guard) != 1:
        raise RuntimeError("Original inference source changed; review the portability adapter.")
    text = text.replace(guard, 'assert dimension in (8, 16) and checkpoint["epoch"] == 8', 1)
    scope = dict(original.__globals__)
    exec(compile(text, "<portable_v6_pipeline>", "exec"), scope)
    return scope[original.__name__]


def export_verified_submission(rowids, probabilities, example, output, expected_rows,
                               expected_sha256, *, verify_inputs):
    """Validate a private candidate before atomic, non-replacing publication.

    Each producer owns its staging directory, including the frozen serializer's
    temporary file. Existing correct output is verified without modification;
    any destination created during publication causes FileExistsError.
    """
    sys.path.insert(0, str(ROOT / "legacy"))
    from submission_export import export_submission, validate_csv
    from row_auc_final_utils import verify_csv_probabilities

    output = Path(output)

    def check(candidate):
        validate_csv(candidate, example, expected_rows)
        verify_csv_probabilities(candidate, rowids, probabilities)
        if digest(candidate) != expected_sha256:
            raise RuntimeError("V6 output differs from the configured fingerprint; no output was published.")

    if output.exists():
        if digest(output) != expected_sha256:
            raise FileExistsError(f"Existing output differs from V6; refusing to overwrite: {output}")
        check(output)
        verify_inputs()
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v6-", dir=output.parent) as temporary:
        candidate = Path(temporary) / "candidate.csv"
        export_submission(rowids, probabilities, example, candidate, expected_rows)
        check(candidate)
        verify_inputs()
        # Both paths are on the same filesystem. Never replace a racing writer.
        os.link(candidate, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/reproduced_v6.csv")
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        relative_output = output.relative_to((ROOT / "artifacts").resolve())
    except ValueError:
        parser.error("Output must be inside the repository's ignored artifacts directory.")
    manifest = verify_bundle()
    config = read_json(ROOT / "configs/v6.json")
    # Only import heavy libraries after byte/provenance verification.
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / "legacy"))
    from build_submission import neural_pipeline, FIT_FIELDS, TARGET_FIELDS, predict_combiner, load_combiner
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Activate the original techarena environment.")
    torch.set_num_threads(4)
    pipeline = portable_pipeline(neural_pipeline)
    cutoff = 1737936000
    with np.load(ROOT / "data/cache/train_events.npz", allow_pickle=False) as saved:
        training = {k: saved[k] for k in FIT_FIELDS}
        assert len(saved["u"]) == 686824 and (saved["t"] < cutoff).all()
        assert int(saved["y"].sum()) == 432023
    with np.load(ROOT / "data/cache/evaluation_events.npz", allow_pickle=False) as saved:
        assert "y" not in saved.files and "clicked" not in saved.files
        target = {k: saved[k] for k in TARGET_FIELDS}
    assert len(target["u"]) == len(np.unique(target["u"])) == config["evaluation_events"]
    assert target["i"].shape == (config["evaluation_events"], 10)
    assert (target["t"] >= cutoff).all() and (target["t"] < 1738540800).all()
    rowids = np.load(ROOT / "data/cache/evaluation_rowids.npy", allow_pickle=False)
    assert rowids.shape == (config["evaluation_rows"],) and rowids.dtype.kind == "U"
    positive = ROOT / "data/cache/preference_training.npz"
    with np.load(positive, allow_pickle=False) as saved:
        assert int(saved["cutoff"]) == cutoff and len(saved["users"]) == 1009707
        assert (saved["timestamps"] < cutoff).all()
    members, differences = [], {}
    for record in manifest["members"]:
        checkpoint_path = asset(ROOT, record["checkpoint"])
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert state["epoch"] == 8 and state["dimension"] == record["dimension"]
        assert state["stage"] == "event" and state["cutoff"] == cutoff
        assert state["n_users"] == 54456 and state["n_items"] == 50000
        seed = int(record["key"].rsplit("seed", 1)[1])
        assert state["seed"] == seed
        assert all(torch.isfinite(v).all() for v in state["model_state"].values() if v.is_floating_point())
        del state
        print(f"Replaying {record['key']} from collected assets...", flush=True)
        X, names, *_ = pipeline(checkpoint_path, training, target, positive, torch.device(args.device))
        p = predict_combiner(load_combiner(asset(ROOT, record["combiner"])), X, names).reshape(-1, 10)
        with np.load(asset(ROOT, record["reference"]), allow_pickle=False) as saved:
            expected = saved["combiner_predictions"].reshape(-1, 10)
        difference = float(np.max(np.abs(p - expected)))
        if not np.allclose(p, expected, rtol=0, atol=1e-15):
            raise RuntimeError(f"{record['key']} did not reproduce; max difference {difference}. Nothing is eligible for archive.")
        differences[record["key"]] = difference
        members.append(p)
        del X
        torch.cuda.empty_cache()
    p = (2 / 3) * ((members[0] + members[1] + members[2]) / 3) + (1 / 3) * (.5 * members[3] + .5 * members[4])
    example = ROOT / "data/raw/example_submission.csv"
    export_verified_submission(rowids, p, example, output, config["evaluation_rows"],
                               config["submission_sha256"], verify_inputs=verify_bundle)
    paths = ["tools/predict_v6.py", "tools/consolidate.py", "configs/v6.json", "configs/source_manifest.json"]
    paths += list(read_json(ROOT / "configs/source_manifest.json")["files"])
    report = {"passed": True, "created_at": utc(), "member_max_abs_differences": differences,
              "source_manifest_binding": source_manifest_binding(ROOT, manifest),
              "bundle_manifest_sha256": digest(ROOT / "artifacts/v6/manifest.json"),
              "submission_sha256": digest(output), "output_relative": "artifacts/" + relative_output.as_posix(),
              "code_sha256": {p: digest(ROOT / p) for p in paths}, "device": args.device,
              "original_folders_modified": False}
    write_json(ROOT / "local/v6_relocation_check.json", report)
    print("V6 portable replay passed; CSV is byte-identical to the submitted candidate.")
    print(f"Receipt: {ROOT / 'local/v6_relocation_check.json'}")


if __name__ == "__main__":
    main()
