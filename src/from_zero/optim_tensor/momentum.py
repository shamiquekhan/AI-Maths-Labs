"""SGD with classical momentum: ``v = beta*v + g``."""

from __future__ import annotations

from collections.abc import Iterable

from ..tensor import Tensor
from .base import Optimizer


class Momentum(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        super().__init__(
            parameters,
            {
                "lr": float(lr),
                "momentum": float(momentum),
                "weight_decay": float(weight_decay),
            },
        )

    def step(self) -> None:
        self.step_count += 1
        lr = self.lr
        momentum = self.defaults["momentum"]
        weight_decay = self.defaults["weight_decay"]
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
            if "velocity" not in state:
                state["velocity"] = [0.0] * parameter.size
            state["velocity"] = [
                momentum * v + g for v, g in zip(state["velocity"], gradient, strict=False)
            ]
            self._apply(parameter, [-lr * v for v in state["velocity"]])


__all__ = ["Momentum"]
