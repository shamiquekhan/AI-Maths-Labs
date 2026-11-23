"""Base optimizer operating on custom ``Parameter`` tensors."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from ..tensor import Tensor


class Optimizer:
    """Base class for optimizers over custom tensors.

    Parameters are updated in place.  Optimizer state is stored as plain
    Python lists keyed by parameter index, which keeps checkpoints
    JSON-serialisable and independent of object identity.
    """

    def __init__(self, parameters: Iterable[Tensor], defaults: dict[str, Any]) -> None:
        self.parameters: list[Tensor] = list(parameters)
        if not self.parameters:
            raise ValueError("Optimizer received no parameters")
        self.defaults: dict[str, Any] = dict(defaults)
        self.state: dict[int, dict[str, Any]] = {}
        self.step_count = 0

    # -- public API --------------------------------------------------------
    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None

    def step(self) -> None:
        raise NotImplementedError

    @property
    def lr(self) -> float:
        return float(self.defaults["lr"])

    @lr.setter
    def lr(self, value: float) -> None:
        if value < 0:
            raise ValueError("Learning rate must be non-negative")
        self.defaults["lr"] = float(value)

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _apply(parameter: Tensor, deltas: list[float]) -> None:
        if len(deltas) != parameter.size:
            raise ValueError(f"Update has {len(deltas)} values but parameter has {parameter.size}")
        data = parameter._data
        base = parameter._offset
        for index, delta in enumerate(deltas):
            data[base + index] += delta

    @staticmethod
    def _gradient(parameter: Tensor) -> list[float] | None:
        if parameter.grad is None:
            return None
        gradient = parameter.grad
        if gradient.shape != parameter.shape:
            gradient = gradient.reshape(parameter.shape)
        return gradient._values()

    def _state(self, index: int) -> dict[str, Any]:
        return self.state.setdefault(index, {})

    # -- serialisation -----------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        serialised: dict[str, Any] = {}
        for index, values in self.state.items():
            entry: dict[str, Any] = {}
            for key, value in values.items():
                if isinstance(value, Tensor):
                    entry[key] = value._values()
                elif isinstance(value, list):
                    entry[key] = list(value)
                else:
                    entry[key] = value
            serialised[str(index)] = entry
        return {
            "state": serialised,
            "defaults": dict(self.defaults),
            "step_count": self.step_count,
            "param_count": len(self.parameters),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("param_count") not in (None, len(self.parameters)):
            raise ValueError("Optimizer state was created for a different number of parameters")
        restored: dict[int, dict[str, Any]] = {}
        for key, values in state.get("state", {}).items():
            restored[int(key)] = {
                name: list(value) if isinstance(value, list) else value
                for name, value in values.items()
            }
        self.state = restored
        self.defaults.update(state.get("defaults", {}))
        self.step_count = int(state.get("step_count", 0))


def clip_grad_norm_(parameters: Iterable[Tensor], max_norm: float, norm_type: float = 2.0) -> float:
    """Clip gradients by global norm, modifying them in place."""
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    parameters = list(parameters)
    if norm_type == float("inf"):
        total = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                total = max(total, max(abs(v) for v in parameter.grad._values()))
        if total > max_norm:
            scale = max_norm / (total + 1e-12)
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad = parameter.grad * scale
        return total

    total_sq = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total_sq += sum(v * v for v in parameter.grad._values())
    total_norm = math.sqrt(total_sq)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad = parameter.grad * scale
    return total_norm


def clip_grad_value_(parameters: Iterable[Tensor], clip_value: float) -> None:
    if clip_value <= 0:
        raise ValueError("clip_value must be positive")
    for parameter in parameters:
        if parameter.grad is None:
            continue
        values = parameter.grad._values()
        parameter.grad = Tensor._from_flat(
            [max(-clip_value, min(clip_value, v)) for v in values], parameter.shape
        )


__all__ = ["Optimizer", "clip_grad_norm_", "clip_grad_value_"]
