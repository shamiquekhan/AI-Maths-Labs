"""Classical momentum.

Convention used here (matching PyTorch's ``momentum``):

``v_t = β v_{t−1} + g_t`` and ``θ_{t+1} = θ_t − η v_t``.

Some texts use ``v_t = β v_{t−1} + (1 − β) g_t``.  Both are valid, but mixing
them changes the effective learning rate, so this project states its convention.
"""

from __future__ import annotations

import numpy as np

from .base import Optimizer


class Momentum(Optimizer):
    name = "momentum"

    def __init__(self, learning_rate: float, beta: float = 0.9) -> None:
        super().__init__(learning_rate)
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        self.beta = float(beta)

    def _update(self, parameters: list[np.ndarray], gradients: list[np.ndarray]) -> None:
        velocities = self._state_arrays("velocity", parameters)
        for parameter, gradient, velocity in zip(parameters, gradients, velocities, strict=True):
            velocity *= self.beta
            velocity += gradient
            parameter -= self.learning_rate * velocity
