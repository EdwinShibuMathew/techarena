"""Within-Event Feature Enrichment for TechArena ChaiScript Combiners.

Provides relative, within-event candidate ranking signals (ratios, differences, z-scores,
and candidate ranks) while maintaining absolute event scale calibration.
"""

from typing import List, Tuple
import numpy as np


ENRICHED_FEATURE_NAMES = [
    "prob_ratio_to_event_max",
    "prob_diff_from_event_mean",
    "prob_zscore_in_event",
    "candidate_event_rank",
    "candidate_log_odds_event",
    "history_recency_skew"
]


def compute_relative_event_features(
    probs: np.ndarray,
    raw_probs: np.ndarray,
    original_X: np.ndarray,
    candidates_per_event: int = 10
) -> Tuple[np.ndarray, List[str]]:
    """Compute 6 enriched within-event relative features.
    
    Args:
        probs: Exposure-adjusted probabilities [N_events, 10] or [N_rows].
        raw_probs: Raw neural probabilities [N_events, 10] or [N_rows].
        original_X: Existing feature matrix [N_rows, 37].
        candidates_per_event: Number of candidates per event (default 10).
        
    Returns:
        new_features: Array of shape [N_rows, 6] float32.
        feature_names: List of 6 feature names.
    """
    n_rows = len(original_X)
    n_events = n_rows // candidates_per_event
    
    p = np.asarray(probs, dtype=np.float32).reshape(n_events, candidates_per_event)
    
    # 1. Ratio to event max
    event_max = np.maximum(p.max(axis=1, keepdims=True), 1e-8)
    ratio_to_max = (p / event_max).flatten()
    
    # 2. Difference from event mean
    event_mean = p.mean(axis=1, keepdims=True)
    diff_from_mean = (p - event_mean).flatten()
    
    # 3. Z-score in event
    event_std = np.maximum(p.std(axis=1, keepdims=True), 1e-6)
    zscore = ((p - event_mean) / event_std).flatten()
    
    # 4. Candidate rank within event (0 = highest predicted probability)
    # argsort descending: argsort(-p) gives ranks
    order = np.argsort(-p, axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    for e in range(n_events):
        ranks[e, order[e]] = np.arange(candidates_per_event, dtype=np.float32)
    candidate_ranks = ranks.flatten()
    
    # 5. Log-odds within event
    clipped_p = np.clip(p, 1e-7, 1.0 - 1e-7)
    log_odds = np.log(clipped_p / (1.0 - clipped_p)).flatten()
    
    # 6. History recency skew: clicks_6h / (clicks_72h + 1.0)
    # In the 37-feature layout: clicks_last_6h is index 15, clicks_last_72h is index 17
    clicks_6h = original_X[:, 15]
    clicks_72h = original_X[:, 17]
    recency_skew = (clicks_6h / (clicks_72h + 1.0)).astype(np.float32)
    
    enriched = np.column_stack([
        ratio_to_max,
        diff_from_mean,
        zscore,
        candidate_ranks,
        log_odds,
        recency_skew
    ]).astype(np.float32)
    
    return enriched, list(ENRICHED_FEATURE_NAMES)


def enrich_feature_matrix(
    X: np.ndarray,
    names: List[str],
    probs: np.ndarray,
    raw_probs: np.ndarray,
    candidates_per_event: int = 10
) -> Tuple[np.ndarray, List[str]]:
    """Combine original 37 features with 6 new enriched features to produce 43 features.
    
    Args:
        X: Shape [N_rows, 37].
        names: List of 37 feature names.
        probs: Probabilities array.
        raw_probs: Raw probabilities array.
        candidates_per_event: Number of candidates per event (default 10).
        
    Returns:
        X_enriched: Shape [N_rows, 43] float32 array.
        names_enriched: List of 43 feature names.
    """
    new_features, new_names = compute_relative_event_features(
        probs, raw_probs, X, candidates_per_event=candidates_per_event
    )
    X_enriched = np.hstack([X, new_features])
    names_enriched = list(names) + new_names
    return X_enriched, names_enriched
