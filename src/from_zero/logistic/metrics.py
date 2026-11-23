"""Classification metric helpers specialised for the logistic laboratory."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.metrics import binary_classification_metrics, calibration_table
from .model import LogisticRegression


def evaluate_classifier(
    model: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    *,
    threshold: float = 0.5,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Return probabilities, predictions, metrics, and calibration rows."""

    probabilities = model.predict_proba(X)
    labels = np.asarray(y).astype(int)
    metrics = binary_classification_metrics(labels, probabilities, threshold=threshold)
    metrics["calibration"] = calibration_table(labels, probabilities, n_bins=calibration_bins)
    return {
        "probabilities": probabilities,
        "predictions": (probabilities >= threshold).astype(int),
        "metrics": metrics,
    }


def format_confusion_matrix(matrix: list[list[int]] | np.ndarray) -> str:
    array = np.asarray(matrix)
    return (
        "                 predicted 0   predicted 1\n"
        f"actual 0 (TN/FP)  {array[0, 0]:>10d}   {array[0, 1]:>11d}\n"
        f"actual 1 (FN/TP)  {array[1, 0]:>10d}   {array[1, 1]:>11d}"
    )
