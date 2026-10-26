"""Cold-Start Recency-Weighted History Fallback for TechArena ChaiScript.

Improves user embedding inference for cold users (unseen in training events, 16.4% of population)
by replacing unweighted arithmetic mean of history item embeddings with an exponential recency decay weighting.
"""

from typing import Optional, Tuple
import numpy as np


def compute_recency_weighted_profile(
    history_items: np.ndarray,
    history_ages_hours: np.ndarray,
    item_embeddings: np.ndarray,
    half_life_hours: float = 24.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute recency-weighted mean item vector for each user's history.
    
    Args:
        history_items: Array of shape [N_users, 20] with item IDs (-1 for empty slot).
        history_ages_hours: Array of shape [N_users, 20] with age in hours.
        item_embeddings: Array of shape [N_items, dimension] float item vectors.
        half_life_hours: Decay parameter (default 24.0 hours).
        
    Returns:
        profiles: Shape [N_users, dimension] float32 array of inferred profile vectors.
        has_history: Shape [N_users] boolean mask of users with >= 1 valid history item.
    """
    history_items = np.asarray(history_items)
    history_ages_hours = np.asarray(history_ages_hours, dtype=np.float64)
    item_embeddings = np.asarray(item_embeddings)
    if (history_items.ndim != 2 or history_items.dtype.kind not in "iu"
            or history_ages_hours.shape != history_items.shape or history_items.shape[1] == 0
            or item_embeddings.ndim != 2 or len(item_embeddings) == 0):
        raise ValueError("Expected aligned history matrices, integer item IDs, and a nonempty embedding matrix.")
    if np.any(history_items < -1) or np.any(history_items >= len(item_embeddings)):
        raise ValueError("History item IDs must be -1 or valid embedding indices.")
    if not np.isfinite(half_life_hours) or half_life_hours <= 0:
        raise ValueError("The decay parameter must be finite and positive.")
    if not np.isfinite(item_embeddings).all():
        raise ValueError("Item embeddings must be finite.")
    
    # Valid history mask: item_id >= 0
    valid_mask = (history_items >= 0)
    has_history = (valid_mask.sum(axis=1) > 0)
    if not np.isfinite(history_ages_hours[valid_mask]).all() or np.any(history_ages_hours[valid_mask] < 0):
        raise ValueError("Valid history ages must be finite and nonnegative.")
    
    # Calculate exponential decay weights: exp(-age / half_life)
    # Clipped age between 0 and 720 hours (30 days)
    clipped_ages = np.clip(np.where(valid_mask, history_ages_hours, 0.0), 0.0, 720.0)
    youngest = np.min(np.where(valid_mask, clipped_ages, np.inf), axis=1, keepdims=True)
    youngest = np.where(has_history[:, None], youngest, 0.0)
    # Relative ages preserve the original decay formula without shrinking old profiles.
    with np.errstate(over="ignore"):
        weights = np.exp(-(clipped_ages - youngest) / half_life_hours)
    weights = np.where(valid_mask, weights, 0.0)
    
    weight_sums = weights.sum(axis=1, keepdims=True)
    normalized_weights = np.divide(weights, weight_sums, out=np.zeros_like(weights), where=weight_sums > 0)
    
    # Safe indexing for items
    safe_items = np.where(valid_mask, history_items, 0)
    
    # Lookup embeddings: [N_users, max_history, dimension]
    embedded_history = item_embeddings[safe_items]
    
    # Weighted sum: [N_users, dimension]
    profiles = np.sum(embedded_history * normalized_weights[:, :, np.newaxis], axis=1)
    
    # Zero out rows with no valid history
    profiles = np.where(has_history[:, np.newaxis], profiles, 0.0).astype(np.float32)
    
    return profiles, has_history


class RecencyWeightedRidgeFallback:
    """Estimates cold-user embedding vectors using recency-weighted history profiles."""

    def __init__(self, alpha: float = 1.0, half_life_hours: float = 24.0):
        self.alpha = alpha
        self.half_life_hours = half_life_hours
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None

    def fit(self, profiles_known: np.ndarray, user_embeddings_known: np.ndarray):
        """Fit linear Ridge regression mapping profile -> user embedding on warm users."""
        # Solves: min ||X W + B - Y||^2 + alpha * ||W||^2
        X = np.asarray(profiles_known, dtype=np.float64)
        Y = np.asarray(user_embeddings_known, dtype=np.float64)
        
        n_samples, d = X.shape
        x_mean = X.mean(axis=0)
        y_mean = Y.mean(axis=0)
        
        X_centered = X - x_mean
        Y_centered = Y - y_mean
        
        # Normal equations: (X^T X + alpha * I) W = X^T Y
        A = np.dot(X_centered.T, X_centered) + self.alpha * np.eye(d, dtype=np.float64)
        B = np.dot(X_centered.T, Y_centered)
        
        W = np.linalg.solve(A, B)
        bias = y_mean - np.dot(x_mean, W)
        
        self.coef_ = W.astype(np.float32)
        self.intercept_ = bias.astype(np.float32)

    def predict(self, profiles: np.ndarray) -> np.ndarray:
        """Estimate user embedding from history profile."""
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before predict.")
        X = np.asarray(profiles, dtype=np.float32)
        return (np.dot(X, self.coef_) + self.intercept_).astype(np.float32)
