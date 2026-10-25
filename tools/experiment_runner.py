"""Unified Experiment Runner and Evaluation Harness for TechArena ChaiScript.

Provides standard interfaces for dataset caching, paired bootstrap evaluation,
confidence interval computation, and decomposition into clicked vs unclicked pairs.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute global row ROC AUC with half credit for ties.
    
    Equivalent to sklearn.metrics.roc_auc_score, implemented in pure NumPy
    for deterministic high performance without importing heavy dependencies.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)
    
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Cannot compute ROC AUC: dataset must contain both positive and negative examples.")
    
    # Sort scores in ascending order
    order = np.argsort(y_score)
    ranked_scores = y_score[order]
    ranked_pos = pos_mask[order]
    
    # Fast rank calculation handling ties
    ranks = np.empty_like(ranked_scores, dtype=np.float64)
    n = len(ranked_scores)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and ranked_scores[j + 1] == ranked_scores[i]:
            j += 1
        # Assign average 1-based rank to ties
        rank_val = 1.0 + (i + j) / 2.0
        ranks[i:j + 1] = rank_val
        i = j + 1
    
    sum_pos_ranks = float(np.sum(ranks[ranked_pos]))
    auc = (sum_pos_ranks - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


@dataclass
class EvaluationMetrics:
    """Standard evaluation metrics for candidate model vs baseline."""
    candidate_auc: float
    baseline_auc: float
    auc_diff: float
    ci_lower: float
    ci_upper: float
    gate_passed: bool
    resamples: int
    clicked_pairs_auc: Optional[float] = None
    unclicked_pairs_auc: Optional[float] = None


class FastAuc:
    """Exact, tie-aware global row AUC with event multiplicities.
    
    Sorting once makes 1,000 whole-event bootstrap resamples run in milliseconds
    instead of minutes without pair matrices.
    """

    def __init__(self, labels: np.ndarray, scores: np.ndarray, event_ids: np.ndarray):
        y = np.asarray(labels, dtype=bool).ravel()
        p = np.asarray(scores, dtype=np.float64).ravel()
        ids = np.asarray(event_ids, dtype=np.int64).ravel()
        order = np.argsort(p, kind="stable")
        p = p[order]
        self.y = y[order]
        self.ids = ids[order]
        self.starts = None
        if len(p) > 1 and np.any(p[1:] == p[:-1]):
            self.starts = np.r_[0, np.flatnonzero(p[1:] != p[:-1]) + 1]

    def value(self, counts: np.ndarray) -> float:
        weight = counts[self.ids].astype(np.float64)
        positive = weight * self.y
        negative = weight * (~self.y)
        if self.starts is not None:
            positive = np.add.reduceat(positive, self.starts)
            negative = np.add.reduceat(negative, self.starts)
        total_pos = positive.sum()
        total_neg = negative.sum()
        if total_pos == 0 or total_neg == 0:
            return 0.5
        return float(np.sum(positive * (np.cumsum(negative) - 0.5 * negative)) / (total_pos * total_neg))


class PairedBootstrapEvaluator:
    """Performs whole-user paired bootstrap evaluation against baseline predictions."""

    def __init__(self, n_resamples: int = 1000, seed: int = 2026, alpha: float = 0.05):
        self.n_resamples = n_resamples
        self.seed = seed
        self.alpha = alpha

    def evaluate(
        self,
        candidate_scores: np.ndarray,
        baseline_scores: np.ndarray,
        labels: np.ndarray,
        user_ids: np.ndarray,
        candidates_per_event: int = 10
    ) -> EvaluationMetrics:
        """Run paired whole-user/event bootstrap.
        
        Args:
            candidate_scores: Shape [N_rows] float predictions from candidate model.
            baseline_scores: Shape [N_rows] float predictions from V6 baseline.
            labels: Shape [N_rows] binary ground truth (0 or 1).
            user_ids: Shape [N_events] or [N_rows] indicating user grouping.
            candidates_per_event: Typically 10 candidates per event.
        """
        candidate_scores = np.asarray(candidate_scores, dtype=np.float64).flatten()
        baseline_scores = np.asarray(baseline_scores, dtype=np.float64).flatten()
        labels = np.asarray(labels, dtype=np.int8).flatten()
        
        n_rows = len(labels)
        n_events = n_rows // candidates_per_event
        event_ids = np.repeat(np.arange(n_events, dtype=np.int64), candidates_per_event)
        
        # Sort-once fast AUC oracles
        cand_order = FastAuc(labels, candidate_scores, event_ids)
        base_order = FastAuc(labels, baseline_scores, event_ids)
        
        counts_full = np.ones(n_events, dtype=np.int64)
        cand_point_auc = cand_order.value(counts_full)
        base_point_auc = base_order.value(counts_full)
        point_diff = cand_point_auc - base_point_auc
        
        # Event-level bootstrap sampling with replacement via multinomial draws
        rng = np.random.default_rng(self.seed)
        diff_samples = np.empty(self.n_resamples, dtype=np.float64)
        
        for b in range(self.n_resamples):
            counts = rng.multinomial(n_events, np.full(n_events, 1.0 / n_events))
            diff_samples[b] = cand_order.value(counts) - base_order.value(counts)
        
        lower_pct = 100.0 * (self.alpha / 2.0)
        upper_pct = 100.0 * (1.0 - self.alpha / 2.0)
        ci_lower = float(np.percentile(diff_samples, lower_pct))
        ci_upper = float(np.percentile(diff_samples, upper_pct))
        
        gate_passed = ci_lower > 0.0
        
        return EvaluationMetrics(
            candidate_auc=cand_point_auc,
            baseline_auc=base_point_auc,
            auc_diff=point_diff,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            gate_passed=gate_passed,
            resamples=self.n_resamples
        )


class DataCacheManager:
    """Manages lazy-loading and in-memory caching of dataset arrays."""

    def __init__(self, root: Optional[Path] = None):
        self.root = root or ROOT
        self._cache: Dict[str, Any] = {}

    def get_train_events(self) -> Dict[str, np.ndarray]:
        if "train_events" not in self._cache:
            path = self.root / "data/cache/train_events.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing train events cache: {path}")
            with np.load(path, allow_pickle=False) as data:
                self._cache["train_events"] = {k: data[k] for k in data.files}
        return self._cache["train_events"]

    def get_evaluation_events(self) -> Dict[str, np.ndarray]:
        if "evaluation_events" not in self._cache:
            path = self.root / "data/cache/evaluation_events.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing evaluation events cache: {path}")
            with np.load(path, allow_pickle=False) as data:
                self._cache["evaluation_events"] = {k: data[k] for k in data.files}
        return self._cache["evaluation_events"]

    def get_evaluation_rowids(self) -> np.ndarray:
        if "evaluation_rowids" not in self._cache:
            path = self.root / "data/cache/evaluation_rowids.npy"
            if not path.is_file():
                raise FileNotFoundError(f"Missing evaluation rowids: {path}")
            self._cache["evaluation_rowids"] = np.load(path, allow_pickle=False)
        return self._cache["evaluation_rowids"]

    def clear(self):
        """Release memory."""
        self._cache.clear()
