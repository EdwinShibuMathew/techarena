"""Content addressed feature storage. A verified metadata record commits each entry."""
import fcntl
import hashlib
import inspect
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

from consolidate import write_json
from .contracts import canonical, digest, inside, require
from .storage import read


def arrays_identity(arrays):
    result = {}
    for name, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        require(not value.dtype.hasobject, "Object arrays are forbidden")
        result[name] = {"shape": list(value.shape), "dtype": value.dtype.str,
                        "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    return result


def request(source, module, checkpoint, prefix, target):
    import torch
    names = module.EXTRACTOR_FILES
    actual = Path(inspect.getsourcefile(module.extract)).resolve().relative_to(source.resolve()).as_posix()
    require(actual in names and len(set(names)) == len(names), "Declare the actual extractor and all its helpers")
    files = {name: digest(inside(source, name)) for name in names}
    files.update({name: digest(inside(source, name)) for name in
                  ("tools/ml_workflow/contracts.py", "tools/ml_workflow/cache.py")})
    require(hasattr(module, "FEATURE_NAMES"), "Adapter must declare the ordered feature schema")
    return {"version": 1, "extractor": files, "checkpoint": digest(checkpoint),
            "prefix": arrays_identity(prefix), "target": arrays_identity(target),
            "feature_names": list(module.FEATURE_NAMES), "dtype": "float32",
            "runtime": {"numpy": np.__version__, "torch": torch.__version__,
                        "python": sys.version.split()[0], "device": getattr(module, "EXTRACTION_DEVICE", "cpu"),
                        "precision": getattr(module, "EXTRACTION_PRECISION", "float32")}}


def validate(data, identity):
    require(set(data) == {"X", "feature_names"}, "Invalid cached fields")
    require(data["feature_names"].tolist() == identity["feature_names"], "Feature schema differs")
    require(data["X"].dtype == np.float32 and data["X"].ndim == 2
            and data["X"].shape == (identity["target"]["i"]["shape"][0] * 10, len(identity["feature_names"]))
            and np.isfinite(data["X"]).all(), "Invalid cached features")


def get_or_compute(root, identity, compute):
    start = time.monotonic()
    key = canonical(identity)
    folder = inside(root, "artifacts/cache/ml-workflow/v1/" + key)
    folder.mkdir(parents=True, exist_ok=True)
    with inside(folder, "entry.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        values, metadata = inside(folder, "values.npz"), inside(folder, "metadata.json")
        hit = metadata.exists()
        if hit:
            record = read(metadata)
            require(record["request"] == identity and record["sha256"] == digest(values), "Feature cache integrity failed")
            with np.load(values, allow_pickle=False) as archive:
                data = {k: archive[k] for k in archive.files}
            validate(data, identity)
            require(record["arrays"] == arrays_identity(data), "Feature cache schema changed")
        else:
            data = compute(); validate(data, identity)
            with tempfile.TemporaryDirectory(prefix=".entry-", dir=folder) as tmp:
                staged = Path(tmp) / "values.npz"
                with staged.open("xb") as stream:
                    np.savez_compressed(stream, **data)
                    stream.flush()
                    import os
                    os.fsync(stream.fileno())
                with np.load(staged, allow_pickle=False) as archive:
                    validate({k: archive[k] for k in archive.files}, identity)
                staged.replace(values)
                write_json(metadata, {"request": identity, "sha256": digest(values), "arrays": arrays_identity(data)})
    return data, {"key": key, "hit": hit, "seconds": time.monotonic() - start}
