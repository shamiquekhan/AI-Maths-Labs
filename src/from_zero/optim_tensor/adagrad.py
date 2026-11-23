"""AdaGrad: per-parameter learning rates from accumulated squared gradients."""

from __future__ import annotations

import math
from collections.abc import Iterable

from ..tensor import Tensor
from .base import Optimizer


class Adagrad(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float = 0.01,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        eps: float = 1e-10,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if lr_decay < 0:
            raise ValueError(f"Invalid lr_decay: {lr_decay}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        super().__init__(
            parameters,
            {
                "lr": float(lr),
                "lr_decay": float(lr_decay),
                "weight_decay": float(weight_decay),
                "eps": float(eps),
            },
        )

    def step(self) -> None:
        self.step_count += 1
        lr = self.lr
        lr_decay = self.defaults["lr_decay"]
        weight_decay = self.defaults["weight_decay"]
        eps = self.defaults["eps"]
        effective_lr = lr / (1.0 + (self.step_count - 1) * lr_decay)

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
            if "sum" not in state:
                state["sum"] = [0.0] * parameter.size
            state["sum"] = [s + g * g for s, g in zip(state["sum"], gradient, strict=False)]
            deltas = [
                -effective_lr * g / (math.sqrt(s) + eps)
                for g, s in zip(gradient, state["sum"], strict=False)
            ]
            self._apply(parameter, deltas)


__all__ = ["Adagrad"]
