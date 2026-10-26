"""Combiner Feature Caching Utility for TechArena ChaiScript.

Accelerates ML experimentation by saving and loading precomputed 37-feature
combiner matrices, reducing evaluation preparation from 15-45 minutes to <2 seconds.
"""

from pathlib import Path
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "artifacts/cache"


class CombinerFeatureCache:
    """Manages serialization and instant retrieval of precomputed feature matrices."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, member_key: str, split_name: str) -> Path:
        for value in (member_key, split_name):
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
                raise ValueError("Cache keys must be simple nonempty identifiers.")
        root = self.cache_dir.resolve()
        path = root / f"{member_key}_{split_name}_features.npz"
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Cache entry escapes its directory or is a symlink.")
        return path

    def has_cache(self, member_key: str, split_name: str) -> bool:
        return self._cache_path(member_key, split_name).is_file()

    def save(
        self,
        member_key: str,
        split_name: str,
        X: np.ndarray,
        feature_names: List[str],
        probs: np.ndarray,
        raw_probs: np.ndarray,
        labels: Optional[np.ndarray] = None
    ):
        """Store features as float32 and probabilities in their supplied dtypes.

        Probability rounding can introduce ranking ties, so its precision is
        independent of the deliberate feature-matrix conversion.
        """
        path = self._cache_path(member_key, split_name)
        payload = {
            "X": np.asarray(X, dtype=np.float32),
            "probs": np.asarray(probs),
            "raw_probs": np.asarray(raw_probs),
            "feature_names": np.asarray(feature_names, dtype="U")
        }
        if labels is not None:
            payload["labels"] = np.asarray(labels, dtype=np.int8)
            
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", prefix=".features-", suffix=".npz", dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, **payload)
                stream.flush()
                os.fsync(stream.fileno())
            with np.load(temporary, allow_pickle=False) as stored:
                for name, value in payload.items():
                    if stored[name].dtype != value.dtype or not np.array_equal(stored[name], value):
                        raise ValueError("Cache serialization changed an array.")
            self._cache_path(member_key, split_name)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load(self, member_key: str, split_name: str) -> Dict[str, Any]:
        """Load arrays in their stored dtypes, including older float32 archives.

        Old caches that rounded float64 inputs must be regenerated from the
        originals before precision-sensitive comparisons. Widening a stored
        float32 array cannot restore its original values.
        """
        path = self._cache_path(member_key, split_name)
        if not path.is_file():
            raise FileNotFoundError(f"No feature cache found at {path}")
            
        with np.load(path, allow_pickle=False) as data:
            result = {
                "X": data["X"],
                "probs": data["probs"],
                "raw_probs": data["raw_probs"],
                "feature_names": [str(x) for x in data["feature_names"]]
            }
            if "labels" in data:
                result["labels"] = data["labels"]
        return result
