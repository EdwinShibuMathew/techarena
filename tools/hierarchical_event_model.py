"""Separate event click scale from conditional candidate choice.

Recomposition blends the original event sum with an estimated event scale.
Positive rescaling retains within-event order; cross-event rankings can change.
These utilities do not establish a measured improvement over V6.
"""

from typing import Dict, Optional, Tuple
import numpy as np


def conditional_item_choice(probs: np.ndarray, candidates_per_event: int = 10) -> np.ndarray:
    """Compute within-event conditional choice distribution: q[e, i] = p[e, i] / sum_j p[e, j].
    
    Args:
        probs: Array of shape [N_events, 10] or [N_rows].
        candidates_per_event: Typically 10.
        
    Returns:
        Conditional distribution summing to 1.0 within each event.
    """
    n_rows = len(probs.flatten())
    n_events = n_rows // candidates_per_event
    p = np.asarray(probs, dtype=np.float64).reshape(n_events, candidates_per_event)
    
    event_sums = np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    q = p / event_sums
    return q.reshape(probs.shape)


def compute_event_features(
    probs: np.ndarray,
    user_ctrs: Optional[np.ndarray] = None,
    time_buckets: Optional[np.ndarray] = None,
    day_types: Optional[np.ndarray] = None,
    candidates_per_event: int = 10
) -> np.ndarray:
    """Extract event-level summary features for predicting event click propensity.
    
    Args:
        probs: Shape [N_events, 10].
        user_ctrs: Optional smoothed user CTR per row [N_rows].
        time_buckets: Optional time bucket [0-3] per row [N_rows].
        day_types: Optional day type [0-1] per row [N_rows].
        candidates_per_event: 10.
        
    Returns:
        Event feature matrix [N_events, n_features].
    """
    n_rows = len(probs.flatten())
    n_events = n_rows // candidates_per_event
    p = np.asarray(probs, dtype=np.float64).reshape(n_events, candidates_per_event)
    
    # 1. Sum of candidate probabilities
    event_sum = p.sum(axis=1, keepdims=True)
    
    # 2. Max candidate probability
    event_max = p.max(axis=1, keepdims=True)
    
    # 3. Min candidate probability
    event_min = p.min(axis=1, keepdims=True)
    
    # 4. Standard deviation of candidate probabilities
    event_std = p.std(axis=1, keepdims=True)
    
    # 5. Shannon entropy of the conditional candidate distribution
    q = p / np.maximum(event_sum, 1e-12)
    clipped_q = np.clip(q, 1e-12, 1.0)
    entropy = -np.sum(clipped_q * np.log(clipped_q), axis=1, keepdims=True)
    
    features = [event_sum, event_max, event_min, event_std, entropy]
    
    # Add optional context features if supplied
    if user_ctrs is not None:
        u_ctr = user_ctrs.reshape(n_events, candidates_per_event)[:, 0:1]
        features.append(u_ctr)
    if time_buckets is not None:
        tb = time_buckets.reshape(n_events, candidates_per_event)[:, 0:1]
        features.append(tb)
    if day_types is not None:
        dt = day_types.reshape(n_events, candidates_per_event)[:, 0:1]
        features.append(dt)
        
    return np.hstack(features).astype(np.float32)


class HierarchicalEventCalibrator:
    """Calibrates and recomposes conditional choice probabilities with event click scale.
    
    Recomposition uses ((1-w) * raw_event_sum + w * calibrated_scale) * q[e, i].
    Raw sums may exceed one; the resulting candidate probabilities must remain
    in [0, 1]. Outputs use float64 to avoid unnecessary ranking ties.
    """

    def __init__(self, blend_weight: float = 0.5):
        """
        Args:
            blend_weight: Interpolation between raw V6 event sum and calibrated event scale.
                          0.0 = pure raw event sum, 1.0 = pure calibrated event scale.
        """
        self.blend_weight = self._checked_weight(blend_weight)
        self.weights_: Optional[np.ndarray] = None
        self.bias_: float = 0.0

    @staticmethod
    def _checked_weight(value):
        try:
            if np.ndim(value) != 0:
                raise ValueError
            weight = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("blend_weight must be a finite scalar in [0, 1].") from error
        if not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("blend_weight must be a finite scalar in [0, 1].")
        return weight

    def fit_logistic_scale(
        self,
        event_features: np.ndarray,
        event_labels: np.ndarray,
        lr: float = 0.01,
        epochs: int = 100,
        l2: float = 1.0
    ):
        """Fit a simple logistic regressor predicting event-level click (1 if event had click, 0 otherwise).
        
        Args:
            event_features: Shape [N_events, D].
            event_labels: Shape [N_events] binary (1 if any candidate clicked, 0 otherwise).
        """
        n, d = event_features.shape
        w = np.zeros(d, dtype=np.float64)
        b = 0.0
        
        # Standardize features
        mean = event_features.mean(axis=0)
        std = np.maximum(event_features.std(axis=0), 1e-6)
        x_norm = (event_features - mean) / std
        y = event_labels.astype(np.float64)
        
        # Batch gradient descent
        for _ in range(epochs):
            logits = np.dot(x_norm, w) + b
            preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
            err = preds - y
            
            grad_w = (np.dot(x_norm.T, err) / n) + (l2 * w / n)
            grad_b = float(np.mean(err))
            
            w -= lr * grad_w
            b -= lr * grad_b
            
        self.weights_ = w / std
        self.bias_ = b - float(np.dot(mean / std, w))

    def predict_event_propensity(self, event_features: np.ndarray) -> np.ndarray:
        """Predict probability that an event contains at least one click."""
        if self.weights_ is None:
            # Fallback to the first feature (event sum of probabilities)
            return np.clip(event_features[:, 0], 0.0, 1.0)
        
        logits = np.dot(event_features, self.weights_) + self.bias_
        propensity = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        return propensity

    def recompose(
        self,
        probs: np.ndarray,
        event_propensity: Optional[np.ndarray] = None,
        candidates_per_event: int = 10
    ) -> np.ndarray:
        """Blend event scales and recompose candidate probabilities in float64.
        
        Args:
            probs: Base model probabilities [N_events, 10].
            event_propensity: Estimated event click probability [N_events] or [N_events, 1].
            candidates_per_event: 10.
            
        Returns:
            Candidate probabilities in the input shape. Zero weight or omitted
            propensity preserves original values exactly. Zero-sum events stay
            zero; positive-sum events use the blended scale. Invalid inputs or
            results outside [0, 1] raise ValueError rather than being clipped.
        """
        weight = self._checked_weight(self.blend_weight)
        if (isinstance(candidates_per_event, (bool, np.bool_))
                or not isinstance(candidates_per_event, (int, np.integer))
                or candidates_per_event <= 0):
            raise ValueError("candidates_per_event must be a positive integer.")
        original = np.asarray(probs, dtype=np.float64)
        if (original.ndim not in (1, 2) or not original.size
                or original.size % candidates_per_event
                or (original.ndim == 2 and original.shape[1] != candidates_per_event)):
            raise ValueError("Expected a nonempty vector or matrix of whole candidate events.")
        if not np.isfinite(original).all() or np.any((original < 0) | (original > 1)):
            raise ValueError("Candidate probabilities must be finite and in [0, 1].")
        p = original.reshape(-1, candidates_per_event)
        if event_propensity is None:
            return original.copy()
        calibrated = np.asarray(event_propensity, dtype=np.float64)
        if calibrated.shape not in ((len(p),), (len(p), 1)):
            raise ValueError("Expected one event scale per event, as a vector or column.")
        if not np.isfinite(calibrated).all() or np.any(calibrated < 0):
            raise ValueError("Event scales must be finite and nonnegative.")
        if weight == 0:
            return original.copy()
        event_sum = p.sum(axis=1, keepdims=True)
        choice = np.divide(p, event_sum, out=np.zeros_like(p), where=event_sum > 0)
        scale = (1 - weight) * event_sum + weight * calibrated.reshape(-1, 1)
        recomposed = scale * choice
        if not np.isfinite(recomposed).all() or np.any((recomposed < 0) | (recomposed > 1)):
            raise ValueError("Recomposition produced probabilities outside [0, 1].")
        return recomposed.reshape(original.shape)
