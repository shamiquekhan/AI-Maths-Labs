"""Binary logistic regression implemented from first principles in NumPy."""

from __future__ import annotations

from .gradients import (
    check_gradients,
    compute_gradients,
    finite_difference_gradient,
    loss_with_regularization,
)
from .losses import binary_cross_entropy, binary_cross_entropy_with_logits
from .model import LogisticRegression, sigmoid
from .training import LogisticConfig, LogisticRun, run_logistic_experiment

__all__ = [
    "LogisticConfig",
    "LogisticRegression",
    "LogisticRun",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "check_gradients",
    "compute_gradients",
    "finite_difference_gradient",
    "loss_with_regularization",
    "run_logistic_experiment",
    "sigmoid",
]
