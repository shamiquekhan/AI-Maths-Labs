"""AdamW: Adam with decoupled weight decay."""

from __future__ import annotations

import math
from collections.abc import Iterable

from ..tensor import Tensor
from .base import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        super().__init__(
            parameters,
            {
                "lr": float(lr),
                "betas": (float(betas[0]), float(betas[1])),
                "eps": float(eps),
                "weight_decay": float(weight_decay),
                "amsgrad": bool(amsgrad),
            },
        )

    def step(self) -> None:
        self.step_count += 1
        lr = self.lr
        beta1, beta2 = self.defaults["betas"]
        eps = self.defaults["eps"]
        weight_decay = self.defaults["weight_decay"]
        amsgrad = self.defaults["amsgrad"]

        for index, parameter in enumerate(self.parameters):
            gradient = self._gradient(parameter)
            if gradient is None:
                continue
            state = self._state(index)
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = [0.0] * parameter.size
                state["exp_avg_sq"] = [0.0] * parameter.size
                if amsgrad:
                    state["max_exp_avg_sq"] = [0.0] * parameter.size
            state["step"] += 1
            step = state["step"]

            state["exp_avg"] = [
                beta1 * m + (1.0 - beta1) * g
                for m, g in zip(state["exp_avg"], gradient, strict=False)
            ]
            state["exp_avg_sq"] = [
                beta2 * v + (1.0 - beta2) * g * g
                for v, g in zip(state["exp_avg_sq"], gradient, strict=False)
            ]
            bias1 = 1.0 - beta1**step
            bias2 = 1.0 - beta2**step

            if amsgrad:
                state["max_exp_avg_sq"] = [
                    max(previous, current)
                    for previous, current in zip(
                        state["max_exp_avg_sq"], state["exp_avg_sq"], strict=False
                    )
                ]
                second = state["max_exp_avg_sq"]
            else:
                second = state["exp_avg_sq"]

            values = parameter._values()
            deltas = []
            for position, (m, v) in enumerate(zip(state["exp_avg"], second, strict=False)):
                decayed = values[position] * (1.0 - lr * weight_decay)
                update = lr * (m / bias1) / (math.sqrt(v / bias2) + eps)
                deltas.append(decayed - values[position] - update)
            self._apply(parameter, deltas)


__all__ = ["AdamW"]
