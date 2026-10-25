"""Combiner Feature Caching Utility for TechArena ChaiScript.

Accelerates ML experimentation by saving and loading precomputed 37-feature
combiner matrices, reducing evaluation preparation from 15-45 minutes to <2 seconds.
"""

from pathlib import Path
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
        return self.cache_dir / f"{member_key}_{split_name}_features.npz"

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
        """Save feature matrix and prediction vectors to compressed .npz archive."""
        path = self._cache_path(member_key, split_name)
        payload = {
            "X": np.asarray(X, dtype=np.float32),
            "probs": np.asarray(probs, dtype=np.float32),
            "raw_probs": np.asarray(raw_probs, dtype=np.float32),
            "feature_names": np.asarray(feature_names, dtype="U")
        }
        if labels is not None:
            payload["labels"] = np.asarray(labels, dtype=np.int8)
            
        np.savez_compressed(path, **payload)

    def load(self, member_key: str, split_name: str) -> Dict[str, Any]:
        """Load feature matrix and prediction vectors from cache."""
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
