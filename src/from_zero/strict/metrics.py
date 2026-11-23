"""Classification metrics implemented with plain Python."""

from __future__ import annotations

from collections.abc import Sequence


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> list[list[int]]:
    """Return ``[[TN, FP], [FN, TP]]`` for binary labels."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    matrix = [[0, 0], [0, 0]]
    for truth, prediction in zip(y_true, y_pred, strict=False):
        if int(truth) not in (0, 1) or int(prediction) not in (0, 1):
            raise ValueError("confusion_matrix expects binary labels 0/1")
        matrix[int(truth)][int(prediction)] += 1
    return matrix


def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not y_true:
        return 0.0
    correct = sum(
        1
        for truth, prediction in zip(y_true, y_pred, strict=False)
        if int(truth) == int(prediction)
    )
    return correct / len(y_true)


def precision_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    matrix = confusion_matrix(y_true, y_pred)
    true_positive = matrix[1][1]
    false_positive = matrix[0][1]
    denominator = true_positive + false_positive
    return true_positive / denominator if denominator else 0.0


def recall_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    matrix = confusion_matrix(y_true, y_pred)
    true_positive = matrix[1][1]
    false_negative = matrix[1][0]
    denominator = true_positive + false_negative
    return true_positive / denominator if denominator else 0.0


def f1_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def classification_report(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": matrix,
        "support": len(y_true),
    }


def threshold_predictions(probabilities: Sequence[float], threshold: float = 0.5) -> list[int]:
    return [1 if value >= threshold else 0 for value in probabilities]


__all__ = [
    "accuracy_score",
    "classification_report",
    "confusion_matrix",
    "f1_score",
    "precision_score",
    "recall_score",
    "threshold_predictions",
]
