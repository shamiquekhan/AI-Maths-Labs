"""Binary cross-entropy losses with explicit numerical safeguards."""

from __future__ import annotations

import numpy as np


def binary_cross_entropy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """Mean binary cross-entropy from probabilities.

    Probabilities are clipped before the logarithms.  The training code uses the
    logits-based form below, which is more stable because it never forms
    ``log(0)``.
    """

    truth = _validate_labels(y_true)
    probs = _validate_probabilities(probabilities)
    if truth.shape != probs.shape:
        raise ValueError(f"shape mismatch: y_true {truth.shape}, probabilities {probs.shape}")
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must lie in (0, 0.5)")
    clipped = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(truth * np.log(clipped) + (1.0 - truth) * np.log(1.0 - clipped)))


def binary_cross_entropy_with_logits(y_true: np.ndarray, logits: np.ndarray) -> float:
    """Stable mean BCE computed directly from logits.

    Uses ``max(z, 0) - z*y + log1p(exp(-|z|))``, which avoids overflowing
    ``exp`` for large-magnitude logits.
    """

    truth = _validate_labels(y_true)
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"logits must be 1-D, got shape {values.shape}")
    if truth.shape != values.shape:
        raise ValueError(f"shape mismatch: y_true {truth.shape}, logits {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits contain NaN or infinity")
    losses = np.maximum(values, 0.0) - values * truth + np.log1p(np.exp(-np.abs(values)))
    return float(np.mean(losses))


def _validate_labels(y_true: np.ndarray) -> np.ndarray:
    truth = np.asarray(y_true)
    if truth.ndim != 1:
        raise ValueError("y_true must be 1-D")
    if not np.all(np.isin(truth, [0, 1])):
        raise ValueError("y_true must contain only 0 and 1")
    return truth.astype(np.float64)


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError("probabilities must be 1-D")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities contain NaN or infinity")
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    return probs
