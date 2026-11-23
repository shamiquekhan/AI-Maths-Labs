"""Stochastic gradient descent with optional momentum and weight decay."""

from __future__ import annotations

from collections.abc import Iterable

from ..tensor import Tensor
from .base import Optimizer


class SGD(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum > 0 and dampening == 0")
        super().__init__(
            parameters,
            {
                "lr": float(lr),
                "momentum": float(momentum),
                "dampening": float(dampening),
                "weight_decay": float(weight_decay),
                "nesterov": bool(nesterov),
            },
        )

    def step(self) -> None:
        self.step_count += 1
        lr = self.lr
        momentum = self.defaults["momentum"]
        dampening = self.defaults["dampening"]
        weight_decay = self.defaults["weight_decay"]
        nesterov = self.defaults["nesterov"]

        for index, parameter in enumerate(self.parameters):
            gradient = self._gradient(parameter)
            if gradient is None:
                continue
            if weight_decay:
                gradient = [
                    g + weight_decay * v
                    for g, v in zip(gradient, parameter._values(), strict=False)
                ]
            if momentum:
                state = self._state(index)
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = list(gradient)
                else:
                    state["momentum_buffer"] = [
                        momentum * b + (1.0 - dampening) * g
                        for b, g in zip(state["momentum_buffer"], gradient, strict=False)
                    ]
                if nesterov:
                    gradient = [
                        g + momentum * b
                        for g, b in zip(gradient, state["momentum_buffer"], strict=False)
                    ]
                else:
                    gradient = state["momentum_buffer"]
            self._apply(parameter, [-lr * g for g in gradient])


__all__ = ["SGD"]
