"""Analytical gradients and finite-difference checks for logistic regression."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .losses import binary_cross_entropy_with_logits


def compute_gradients(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    *,
    regularization: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Return gradients of mean BCE plus L2 penalty with respect to weights and bias.

    The sigmoid/BCE simplification gives ``Xᵀ(p - y) / N`` and
    ``mean(p - y)``.  The L2 penalty differentiates to ``regularization * w``
    (and does not penalise the bias).
    """

    features = _validate_features(X)
    labels = _validate_labels(y, len(features))
    weight_vector = np.asarray(weights, dtype=np.float64)
    if weight_vector.ndim != 1 or len(weight_vector) != features.shape[1]:
        raise ValueError(
            f"weights must have shape ({features.shape[1]},), got {weight_vector.shape}"
        )
    if not np.isfinite(bias):
        raise ValueError("bias must be finite")
    if regularization < 0:
        raise ValueError("regularization must be non-negative")

    logits = features @ weight_vector + float(bias)
    probabilities = _stable_sigmoid(logits)
    residual = probabilities - labels
    gradient_weights = features.T @ residual / len(features)
    if regularization:
        gradient_weights = gradient_weights + regularization * weight_vector
    gradient_bias = float(np.mean(residual))
    return gradient_weights, gradient_bias


def loss_with_regularization(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    *,
    regularization: float = 0.0,
) -> float:
    features = _validate_features(X)
    labels = _validate_labels(y, len(features))
    weight_vector = np.asarray(weights, dtype=np.float64)
    logits = features @ weight_vector + float(bias)
    data_loss = binary_cross_entropy_with_logits(labels, logits)
    penalty = 0.5 * regularization * float(np.dot(weight_vector, weight_vector))
    return data_loss + penalty


def finite_difference_gradient(
    function: Callable[[np.ndarray], float],
    point: np.ndarray,
    *,
    h: float = 1e-5,
) -> np.ndarray:
    """Central finite-difference gradient of a scalar function."""

    values = np.asarray(point, dtype=np.float64)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be positive and finite")
    gradient = np.zeros_like(values)
    flat = values.reshape(-1)
    gradient_flat = gradient.reshape(-1)
    for index in range(flat.size):
        original = flat[index]
        flat[index] = original + h
        plus = float(function(values))
        flat[index] = original - h
        minus = float(function(values))
        flat[index] = original
        gradient_flat[index] = (plus - minus) / (2.0 * h)
    return gradient


def check_gradients(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    *,
    regularization: float = 0.0,
    h: float = 1e-5,
    atol: float = 1e-6,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    """Compare analytical gradients with central differences.

    Returns arrays and error statistics; tests assert ``passed``.  A relative
    error is reported only where the numerical magnitude is above a small
    floor, avoiding division by values that are effectively zero.
    """

    features = _validate_features(X)
    labels = _validate_labels(y, len(features))
    weight_vector = np.asarray(weights, dtype=np.float64)
    analytic_w, analytic_b = compute_gradients(
        features, labels, weight_vector, bias, regularization=regularization
    )
    numeric_w = finite_difference_gradient(
        lambda w: loss_with_regularization(
            features, labels, w, bias, regularization=regularization
        ),
        weight_vector,
        h=h,
    )
    numeric_b = finite_difference_gradient(
        lambda b: loss_with_regularization(
            features, labels, weight_vector, float(b[0]), regularization=regularization
        ),
        np.array([bias], dtype=np.float64),
        h=h,
    )[0]

    abs_w = np.abs(analytic_w - numeric_w)
    abs_b = abs(analytic_b - numeric_b)
    scale_w = np.maximum(np.abs(analytic_w) + np.abs(numeric_w), 1e-12)
    scale_b = max(abs(analytic_b) + abs(numeric_b), 1e-12)
    rel_w = abs_w / scale_w
    rel_b = abs_b / scale_b
    passed = bool(
        np.all(abs_w <= atol + rtol * np.abs(numeric_w)) and abs_b <= atol + rtol * abs(numeric_b)
    )
    return {
        "analytic_weights": analytic_w,
        "numeric_weights": numeric_w,
        "analytic_bias": analytic_b,
        "numeric_bias": float(numeric_b),
        "max_abs_weight_error": float(np.max(abs_w)) if abs_w.size else 0.0,
        "max_rel_weight_error": float(np.max(rel_w)) if rel_w.size else 0.0,
        "abs_bias_error": float(abs_b),
        "rel_bias_error": float(rel_b),
        "h": float(h),
        "atol": float(atol),
        "rtol": float(rtol),
        "passed": passed,
    }


def _stable_sigmoid(z: np.ndarray) -> np.ndarray:
    result = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z / (1.0 + exp_z)
    return result


def _validate_features(X: np.ndarray) -> np.ndarray:
    features = np.asarray(X, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {features.shape}")
    if len(features) == 0:
        raise ValueError("X must contain at least one example")
    if not np.all(np.isfinite(features)):
        raise ValueError("X contains NaN or infinity")
    return features


def _validate_labels(y: np.ndarray, n: int) -> np.ndarray:
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError("y must be 1-D")
    if len(labels) != n:
        raise ValueError(f"X and y length mismatch: {n} vs {len(labels)}")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("y must contain only 0 and 1")
    return labels.astype(np.float64)
