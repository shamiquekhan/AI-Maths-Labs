"""RMSProp: exponentially weighted average of squared gradients."""

from __future__ import annotations

import math
from collections.abc import Iterable

from ..tensor import Tensor
from .base import Optimizer


class RMSprop(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if momentum < 0:
            raise ValueError(f"Invalid momentum: {momentum}")
        super().__init__(
            parameters,
            {
                "lr": float(lr),
                "alpha": float(alpha),
                "eps": float(eps),
                "weight_decay": float(weight_decay),
                "momentum": float(momentum),
                "centered": bool(centered),
            },
        )

    def step(self) -> None:
        self.step_count += 1
        lr = self.lr
        alpha = self.defaults["alpha"]
        eps = self.defaults["eps"]
        weight_decay = self.defaults["weight_decay"]
        momentum = self.defaults["momentum"]
        centered = self.defaults["centered"]

        for index, parameter in enumerate(self.parameters):
            gradient = self._gradient(parameter)
            if gradient is None:
                continue
            if weight_decay:
                gradient = [
                    g + weight_decay * v
                    for g, v in zip(gradient, parameter._values(), strict=False)
                ]
            state = self._state(index)
            if "square_avg" not in state:
                state["square_avg"] = [0.0] * parameter.size
            state["square_avg"] = [
                alpha * s + (1.0 - alpha) * g * g
                for s, g in zip(state["square_avg"], gradient, strict=False)
            ]
            if centered:
                if "grad_avg" not in state:
                    state["grad_avg"] = [0.0] * parameter.size
                state["grad_avg"] = [
                    alpha * a + (1.0 - alpha) * g
                    for a, g in zip(state["grad_avg"], gradient, strict=False)
                ]
                denominator = [
                    math.sqrt(max(s - a * a, 0.0)) + eps
                    for s, a in zip(state["square_avg"], state["grad_avg"], strict=False)
                ]
            else:
                denominator = [math.sqrt(s) + eps for s in state["square_avg"]]

            if momentum > 0:
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = [0.0] * parameter.size
                state["momentum_buffer"] = [
                    momentum * b + g / d
                    for b, g, d in zip(
                        state["momentum_buffer"], gradient, denominator, strict=False
                    )
                ]
                deltas = [-lr * b for b in state["momentum_buffer"]]
            else:
                deltas = [-lr * g / d for g, d in zip(gradient, denominator, strict=False)]
            self._apply(parameter, deltas)


__all__ = ["RMSprop"]
