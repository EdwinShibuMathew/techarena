"""Cold-Start Recency-Weighted History Fallback for TechArena ChaiScript.

Improves user embedding inference for cold users (unseen in training events, 16.4% of population)
by replacing unweighted arithmetic mean of history item embeddings with an exponential recency decay weighting.
"""

from typing import Tuple
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
    n_users, max_history = history_items.shape
    dimension = item_embeddings.shape[1]
    
    # Valid history mask: item_id >= 0
    valid_mask = (history_items >= 0)
    has_history = (valid_mask.sum(axis=1) > 0)
    
    # Calculate exponential decay weights: exp(-age / half_life)
    # Clipped age between 0 and 720 hours (30 days)
    clipped_ages = np.clip(history_ages_hours, 0.0, 720.0)
    weights = np.exp(-clipped_ages / half_life_hours) * valid_mask.astype(np.float64)
    
    weight_sums = weights.sum(axis=1, keepdims=True)
    normalized_weights = weights / np.maximum(weight_sums, 1e-8)
    
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
