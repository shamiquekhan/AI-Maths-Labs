"""Vanilla (stochastic) gradient descent."""

from __future__ import annotations

import numpy as np

from .base import Optimizer


class SGD(Optimizer):
    """``θ ← θ − η g``.

    Mini-batch stochasticity comes from the caller supplying gradients computed
    on a subset of examples; this class deliberately does not own data loading.
    """

    name = "sgd"

    def _update(self, parameters: list[np.ndarray], gradients: list[np.ndarray]) -> None:
        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter -= self.learning_rate * gradient
