"""Source adaptation and CSV checks for the frozen row-ranking final update."""

import ast
import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from prepare_preference_finetune_trial import replace_once, classes
from row_auc_objective import CONFIG
from submission_export import HEADER, sha256

SEEDS = [42, 43]
FULL_CUTOFF, TRAIN_EVENTS, EVAL_EVENTS, EVAL_ROWS = 1737936000, 686824, 52939, 529390
VERSION = "v6_candidate_row_ranking"


def adapt_final_event(source, seed):
    """Preserve the exact tested initialization and eight-epoch update block.

    Remove all validation loading/scoring, take all labeled training events,
    and save a final checkpoint with full-data provenance in a separate folder.
    """
    original = ast.parse(source)
    original_main = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    original_loop = next(n for n in original_main.body if isinstance(n, ast.For))
    original_update = ast.dump(original_loop.body[0], include_attributes=False)
    assert isinstance(original_loop.body[0], ast.If)
    assert source.count("mixed_row_auc_loss(") == 1
    marker = '        model.eval()\n        parts = []\n'
    assert source.count(marker) == 1
    text = source.split(marker, 1)[0]
    # All of the discarded suffix was validation and last-epoch checkpointing.
    text = replace_once(text, "import json\n", "import json\nimport hashlib\n")
    start = f"    torch.manual_seed({seed})"
    guard = f'''    final_root = Path(__file__).resolve().parents[2]
    frozen_path = final_root / "outputs/frozen_protocol.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert frozen["event_objective"] == CONFIG
    assert frozen["version"] == "{VERSION}"
    own_key = str(Path(__file__).resolve().relative_to(final_root))
    assert file_hash(Path(__file__)) == frozen["prepared_sha256"][own_key]
    for name, expected in frozen["training_inputs_sha256"].items():
        assert file_hash(Path(name)) == expected, f"Changed frozen training input: {{name}}"
    for name, expected in frozen["prepared_sha256"].items():
        assert file_hash(final_root / name) == expected, f"Changed prepared source: {{name}}"
    assert not Path("work/event_best.pt").exists(), "Final checkpoint already exists; use the orchestrator to resume."
    torch.manual_seed({seed})'''
    text = replace_once(text, start, guard)
    text = replace_once(text, "from row_auc_objective import mixed_row_auc_loss",
                         "from row_auc_objective import mixed_row_auc_loss, CONFIG")
    text = replace_once(text, '    n_users = checkpoint["n_users"]', f'''    assert checkpoint["seed"] == {seed} and checkpoint["epoch"] == 260
    assert checkpoint["stage"] == "preference" and checkpoint["cutoff"] == {FULL_CUTOFF}
    assert checkpoint["dimension"] == 8 and checkpoint["fine_tuning_sampling_seed"] == {seed + 1000}
    n_users = checkpoint["n_users"]''')
    text = replace_once(text, '    dimension = checkpoint["dimension"]',
                         '    dimension = checkpoint["dimension"]\n    assert n_users == 54456 and n_items == 50000')
    text = replace_once(text, '''    with np.load("data/cache/split_jan26.npz") as split:
        train_idx = split["train_idx"]
        valid_idx = split["valid_idx"]

''', "")
    old_loading = '''    with np.load("data/cache/train_events.npz") as data:
        train_inputs, train_targets, _ = pack(data, train_idx)
        valid_inputs, _, valid_labels = pack(data, valid_idx)

        assert (
            data["t"][train_idx].max()
            < data["t"][valid_idx].min()
        )'''
    new_loading = f'''    with np.load("data/cache/train_events.npz", allow_pickle=False) as data:
        assert len(data["u"]) == {TRAIN_EVENTS}
        assert data["i"].shape == data["y"].shape == ({TRAIN_EVENTS}, 10)
        assert (data["t"] < {FULL_CUTOFF}).all() and (data["t"] >= 1737331200).all()
        assert np.isin(data["y"], [0, 1]).all() and (data["y"].sum(1) <= 1).all()
        assert int(data["y"].sum()) == 432023
        assert ((data["u"] >= 0) & (data["u"] < n_users)).all()
        assert ((data["i"] >= 0) & (data["i"] < n_items)).all()
        assert np.isin(data["tb"], [0, 1, 2, 3]).all() and np.isin(data["dt"], [0, 1]).all()
        train_idx = np.arange({TRAIN_EVENTS})
        train_inputs, train_targets, _ = pack(data, train_idx)'''
    text = replace_once(text, old_loading, new_loading)
    # Serialization and epoch messages are outside the original update block.
    tail = f'''        state = {{
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "n_users": n_users, "n_items": n_items, "dimension": dimension,
            "seed": {seed}, "epoch": 8, "stage": "event", "cutoff": {FULL_CUTOFF},
            "event_objective": CONFIG["objective"], "pair_sampling_seed": {seed + 3000},
            "preference_checkpoint_sha256": frozen["preference_sources"]["{seed}"]["sha256"],
            "frozen_protocol_sha256": file_hash(frozen_path),
        }}
        temporary = Path("work/event_best.pt.tmp")
        torch.save(state, temporary)
        temporary.replace("work/event_best.pt")
        print(json.dumps({{"seed": {seed}, "epoch": 8, "training_events": len(train_idx),
                          "checkpoint": str(Path("work/event_best.pt").resolve()),
                          "runtime_seconds": round(time.perf_counter() - start, 2)}}), flush=True)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
'''
    text += tail
    tree = ast.parse(text)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loop = next(n for n in main.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "epoch")
    assert ast.dump(loop.body[0], include_attributes=False) == original_update, "Tested optimizer update changed"
    old_pack = next(n for n in original_main.body if isinstance(n, ast.FunctionDef) and n.name == "pack")
    new_pack = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "pack")
    assert ast.dump(old_pack, include_attributes=False) == ast.dump(new_pack, include_attributes=False)
    assert classes(text) == classes(source)
    assert "valid_idx" not in text and "valid_labels" not in text and "evaluation_events" not in text
    assert "roc_auc_score(" not in text and "for epoch in range(9):" in text and "if epoch != 8:" in text
    return text


def probabilities(values, expected_rows=EVAL_ROWS):
    p = np.asarray(values, dtype=np.float64).ravel()
    assert p.shape == (expected_rows,) and np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()
    return p


def verify_csv_probabilities(path, rowids, predictions):
    """Compare every CSV probability with its cached value, matched by rowid."""
    assert rowids.ndim == 1 and rowids.dtype.kind == "U"
    predictions = probabilities(predictions, len(rowids))
    positions = {rowid: k for k, rowid in enumerate(rowids)}
    assert len(positions) == len(rowids)
    seen = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        assert next(reader) == HEADER
        for row in reader:
            assert len(row) == 2 and row[0] in positions and row[0] not in seen
            assert float(row[1]) == float(predictions[positions[row[0]]]), "CSV probability differs from cached ensemble"
            seen.add(row[0])
    assert len(seen) == len(rowids)


def save_npz(path, **arrays):
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def csv_self_test():
    from submission_export import export_submission, validate_csv
    rowids = np.array(["r3", "r1", "r2"])
    p = np.array([.09, .7, .004])
    # Scratch files only; exported competition files are untouched.
    with tempfile.TemporaryDirectory(prefix="row_auc_csv_check_", dir=Path(__file__).resolve().parents[1]) as temporary:
        folder = Path(temporary)
        example, output = folder / "example.csv", folder / "candidate.csv"
        with example.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            writer.writerows([[name, .5] for name in ["r1", "r2", "r3"]])
        export_submission(rowids, p, example, output, expected_rows=3)
        verify_csv_probabilities(output, rowids, p)
        validate_csv(output, example, 3)
        try:
            verify_csv_probabilities(output, rowids, p + .01)
        except AssertionError:
            pass
        else:
            raise AssertionError("CSV probability change was not detected")
    print("CSV self-test passed: example row order, exact row-ID/probability alignment and changed-value detection.", flush=True)
