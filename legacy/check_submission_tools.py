"""Local NumPy/CSV checks; actual PyTorch/CatBoost reproduction runs in WSL."""
import ast
import csv
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from submission_features import build_features
from submission_export import export_submission

HERE = Path(__file__).resolve().parent
for filename in ["build_submission.py", "submission_features.py", "submission_export.py"]:
    ast.parse((HERE / filename).read_text(encoding="utf-8"))
print("Submission scripts: syntax passed")

source = Path(r"C:\Users\Edwin\Downloads\techarena_prediction_sources.txt").read_text(encoding="utf-8")
source = source.split("========== build_combiner_features.py ==========", 1)[1].split("========== test_exposure_adjustment.py ==========", 1)[0]
reference_function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "build_fold")

rng = np.random.default_rng(429)
n_fit, n_target, n_users, n_items, dim = 8, 6, 10, 13, 8
cutoff = 1737849600
fit_idx = np.arange(n_fit)
target_idx = np.arange(n_fit, n_fit + n_target)
u = np.concatenate([np.arange(n_fit), [0, 8, 9, 1, 2, 3]]).astype(np.int32)
i = rng.integers(0, n_items, size=(n_fit + n_target, 10), dtype=np.int32)
t = np.concatenate([cutoff - 1000 - np.arange(n_fit), cutoff + np.arange(n_target) * 60]).astype(np.int64)
h = np.full((len(u), 20), -1, dtype=np.int32)
ht = np.zeros(h.shape, dtype=np.int64)
for row, history in [(n_fit + 1, [1, 1, 2]), (n_fit + 3, [4]), (n_fit + 4, list(rng.integers(0, n_items, size=20)))]:
    h[row, :len(history)] = history
    ht[row, :len(history)] = t[row] - 3600 * np.arange(len(history), 0, -1)
i[n_fit + 1, 0] = 1
y = np.zeros(i.shape, dtype=np.int8)
y[:4, 0] = 1
raw = dict(u=u, i=i, t=t, h=h, ht=ht, y=y, tb=rng.integers(0, 4, size=len(u), dtype=np.int8), dt=np.zeros(len(u), dtype=np.int8))


class Tensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def numpy(self):
        return self.value


state = {name: Tensor(rng.normal(size=shape).astype(np.float32)) for name, shape in {
    "preference.item.weight": (n_items, dim),
    "preference.user.weight": (n_users, dim),
    "preference.item_bias.weight": (n_items, 1),
    "user_bias.weight": (n_users, 1),
    "item_context.weight": (n_items, 8),
    "user_context.weight": (n_users, 8),
}.items()}
active = np.ones(n_users, dtype=bool)
active[[3, 8, 9]] = False
state["preference.active_users"] = Tensor(active)
coefficients = rng.normal(size=(dim, dim))
intercept = rng.normal(size=dim)
raw_p = rng.uniform(0.001, 0.05, size=(n_target, 10))
p = raw_p * 0.9


class Saved(dict):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


saved_files = {
    "split.npz": Saved(train_idx=fit_idx, valid_idx=target_idx),
    "history_coldstart_validation.npz": Saved(valid_idx=target_idx, predictions=p),
    "event_validation.npz": Saved(valid_idx=target_idx, predictions=raw_p),
    "history_encoder.npz": Saved(coefficients=coefficients, intercept=intercept),
}


class NumpyProxy:
    def __getattr__(self, name):
        return getattr(np, name)

    def load(self, path):
        return saved_files[Path(path).name]


scope = {
    "np": NumpyProxy(),
    "torch": SimpleNamespace(load=lambda *args, **kwargs: dict(model_state=state, n_users=n_users, n_items=n_items)),
}
exec(compile(ast.Module(body=[reference_function], type_ignores=[]), "original_feature_builder", "exec"), scope)
reference_X, _, _, _, reference_names = scope["build_fold"](Path("/synthetic"), "split.npz", cutoff, raw)
training = {name: raw[name][fit_idx] for name in ["u", "i", "y"]}
target = {name: raw[name][target_idx] for name in ["u", "i", "t", "tb", "dt", "h", "ht"]}
X, names = build_features(training, target, state, coefficients, intercept, n_users, n_items, p, raw_p)
assert names == reference_names and len(names) == 37
np.testing.assert_array_equal(X, reference_X)
assert np.isfinite(X).all()
print("All 37 features exactly match original builder on empty/repeated/cold/inactive/full histories")

with tempfile.TemporaryDirectory(prefix="submission_checks_", dir=HERE) as temporary:
    work = Path(temporary).resolve()
    assert work.is_relative_to(HERE)
    example = work / "example.csv"
    with example.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows([["rowid", "predicted_probability"], ["c", "0.5"], ["a", "0.5"], ["b", "0.5"]])
    rowids = np.array(["a", "b", "c"])
    predictions = np.array([0.2, 0.1, 0.8])
    report = export_submission(rowids, predictions, example, work / "submission.csv", expected_rows=3)
    with (work / "submission.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert [row[0] for row in rows[1:]] == ["c", "a", "b"]
    assert [float(row[1]) for row in rows[1:]] == [0.8, 0.2, 0.1]
    assert report["submission_ready"] and report["unique_rowids"] == 3
    invalid = [
        (np.array(["a", "a", "c"]), predictions),
        (np.array(["a", "b", "unknown"]), predictions),
        (rowids, np.array([0.2, float("nan"), 0.8])),
        (rowids, np.array([0.2, 1.1, 0.8])),
    ]
    for number, (ids, values) in enumerate(invalid):
        path = work / f"invalid{number}.csv"
        try:
            export_submission(ids, values, example, path, expected_rows=3)
        except ValueError:
            assert not path.exists()
        else:
            raise AssertionError("Invalid submission was accepted")
print("CSV export: order/prediction alignment passed; duplicate/unknown IDs, NaN and out-of-range values rejected")
