"""Metrics shared by the classification and language-model laboratories."""

from __future__ import annotations

from typing import Any

import numpy as np


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return ``[[TN, FP], [FN, TP]]`` for binary labels."""

    truth = _binary_vector(y_true, "y_true")
    prediction = _binary_vector(y_pred, "y_pred")
    if len(truth) != len(prediction):
        raise ValueError("y_true and y_pred must have the same length")
    tn = int(np.sum((truth == 0) & (prediction == 0)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = _binary_vector(y_true, "y_true")
    prediction = _binary_vector(y_pred, "y_pred")
    if len(truth) != len(prediction):
        raise ValueError("y_true and y_pred must have the same length")
    if len(truth) == 0:
        raise ValueError("Cannot compute accuracy on an empty set")
    return float(np.mean(truth == prediction))


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, zero_division: float = 0.0) -> float:
    matrix = confusion_matrix_binary(y_true, y_pred)
    tp = matrix[1, 1]
    fp = matrix[0, 1]
    denominator = tp + fp
    return float(tp / denominator) if denominator else float(zero_division)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, zero_division: float = 0.0) -> float:
    matrix = confusion_matrix_binary(y_true, y_pred)
    tp = matrix[1, 1]
    fn = matrix[1, 0]
    denominator = tp + fn
    return float(tp / denominator) if denominator else float(zero_division)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, zero_division: float = 0.0) -> float:
    precision = precision_score(y_true, y_pred, zero_division=zero_division)
    recall = recall_score(y_true, y_pred, zero_division=zero_division)
    denominator = precision + recall
    return float(2 * precision * recall / denominator) if denominator else float(zero_division)


def binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    truth = _binary_vector(y_true, "y_true")
    probabilities = _probability_vector(y_prob, "y_prob")
    if len(truth) != len(probabilities):
        raise ValueError("y_true and y_prob must have the same length")
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    return float(-np.mean(truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped)))


def roc_auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Rank-based ROC AUC with average ranks for ties."""

    truth = _binary_vector(y_true, "y_true")
    scores = _probability_vector(y_prob, "y_prob")
    if len(np.unique(truth)) < 2:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = average_rank
        start = end
    n_pos = int(np.sum(truth == 1))
    n_neg = int(np.sum(truth == 0))
    rank_sum = float(np.sum(ranks[truth == 1]))
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def binary_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Return a compact, JSON-serialisable classification report."""

    truth = _binary_vector(y_true, "y_true")
    probabilities = _probability_vector(y_prob, "y_prob")
    if len(truth) != len(probabilities):
        raise ValueError("y_true and y_prob must have the same length")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    prediction = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix_binary(truth, prediction)
    return {
        "n": len(truth),
        "threshold": float(threshold),
        "loss": binary_log_loss(truth, probabilities),
        "accuracy": accuracy_score(truth, prediction),
        "precision": precision_score(truth, prediction),
        "recall": recall_score(truth, prediction),
        "f1": f1_score(truth, prediction),
        "roc_auc": roc_auc_score(truth, probabilities),
        "positive_rate": float(np.mean(truth)),
        "predicted_positive_rate": float(np.mean(prediction)),
        "confusion_matrix": matrix.tolist(),
        "confusion_labels": [["TN", "FP"], ["FN", "TP"]],
    }


def calibration_table(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> list[dict[str, float]]:
    """Bin predictions and compare mean confidence with observed frequency."""

    truth = _binary_vector(y_true, "y_true")
    probabilities = _probability_vector(y_prob, "y_prob")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    for index in range(n_bins):
        low, high = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (probabilities >= low) & (probabilities <= high)
        else:
            mask = (probabilities >= low) & (probabilities < high)
        count = int(np.sum(mask))
        if count == 0:
            continue
        rows.append(
            {
                "bin_low": float(low),
                "bin_high": float(high),
                "count": float(count),
                "mean_confidence": float(np.mean(probabilities[mask])),
                "observed_frequency": float(np.mean(truth[mask])),
            }
        )
    return rows


def perplexity(mean_cross_entropy: float) -> float:
    """Convert mean natural-log cross-entropy to perplexity."""

    if not np.isfinite(mean_cross_entropy):
        raise ValueError("mean cross-entropy must be finite")
    if mean_cross_entropy < 0:
        raise ValueError("mean cross-entropy must be non-negative")
    return float(np.exp(mean_cross_entropy))


def _binary_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if not np.all(np.isin(array, [0, 1])):
        raise ValueError(f"{name} must contain only 0 and 1")
    return array.astype(int)


def _probability_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must lie in [0, 1]")
    return array
