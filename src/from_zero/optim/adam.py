"""Adam and AdamW with bias correction.

Adam follows Kingma and Ba (2014).  AdamW keeps weight decay decoupled from the
adaptive gradient update, which is not the same as adding an L2 penalty to the
loss.
"""

from __future__ import annotations

import numpy as np

from .base import Optimizer


class Adam(Optimizer):
    name = "adam"

    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(learning_rate)
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        if eps <= 0 or not np.isfinite(eps):
            raise ValueError("eps must be positive and finite")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

    def _update(self, parameters: list[np.ndarray], gradients: list[np.ndarray]) -> None:
        first = self._state_arrays("m", parameters)
        second = self._state_arrays("v", parameters)
        self.state["t"] = int(self.state.get("t", 0)) + 1
        t = self.state["t"]
        bias_correction1 = 1.0 - self.beta1**t
        bias_correction2 = 1.0 - self.beta2**t
        for parameter, gradient, mean, variance in zip(
            parameters, gradients, first, second, strict=True
        ):
            mean *= self.beta1
            mean += (1.0 - self.beta1) * gradient
            variance *= self.beta2
            variance += (1.0 - self.beta2) * gradient * gradient
            corrected_mean = mean / bias_correction1
            corrected_variance = variance / bias_correction2
            parameter -= (
                self.learning_rate * corrected_mean / (np.sqrt(corrected_variance) + self.eps)
            )

    def reset_state(self) -> None:
        super().reset_state()


class AdamW(Adam):
    """Adam with decoupled weight decay: ``θ ← θ − η λ θ`` before the Adam step."""

    name = "adamw"

    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(learning_rate, beta1=beta1, beta2=beta2, eps=eps)
        if weight_decay < 0 or not np.isfinite(weight_decay):
            raise ValueError("weight_decay must be non-negative and finite")
        self.weight_decay = float(weight_decay)

    def _update(self, parameters: list[np.ndarray], gradients: list[np.ndarray]) -> None:
        if self.weight_decay:
            for parameter in parameters:
                parameter -= self.learning_rate * self.weight_decay * parameter
        super()._update(parameters, gradients)
