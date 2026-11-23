"""RMSProp: scale each coordinate by a running root-mean-square gradient."""

from __future__ import annotations

import numpy as np

from .base import Optimizer


class RMSProp(Optimizer):
    name = "rmsprop"

    def __init__(self, learning_rate: float, rho: float = 0.9, eps: float = 1e-8) -> None:
        super().__init__(learning_rate)
        if not 0.0 <= rho < 1.0:
            raise ValueError("rho must be in [0, 1)")
        if eps <= 0 or not np.isfinite(eps):
            raise ValueError("eps must be positive and finite")
        self.rho = float(rho)
        self.eps = float(eps)

    def _update(self, parameters: list[np.ndarray], gradients: list[np.ndarray]) -> None:
        squares = self._state_arrays("square_avg", parameters)
        for parameter, gradient, square in zip(parameters, gradients, squares, strict=True):
            square *= self.rho
            square += (1.0 - self.rho) * gradient * gradient
            parameter -= self.learning_rate * gradient / (np.sqrt(square) + self.eps)
