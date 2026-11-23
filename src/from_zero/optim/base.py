"""Common optimizer interface and validation.

Optimizers consume gradients; they do not differentiate anything.  Parameters
are updated in place, which makes the association between optimizer state and
parameter arrays explicit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np


class Optimizer(ABC):
    """Base class for NumPy optimizers.

    ``step(parameters, gradients)`` updates each parameter array in place.
    State is keyed by name and is never reset implicitly during a step.
    """

    name = "optimizer"

    def __init__(self, learning_rate: float) -> None:
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, int | float):
            raise TypeError("learning_rate must be a number")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        self.learning_rate = float(learning_rate)
        self.state: dict[str, Any] = {}

    def step(
        self,
        parameters: Sequence[np.ndarray],
        gradients: Sequence[np.ndarray],
    ) -> None:
        """Validate inputs and apply one update in place."""

        params, grads = self._validate(parameters, gradients)
        self._update(params, grads)

    @abstractmethod
    def _update(self, parameters: list[np.ndarray], gradients: list[np.ndarray]) -> None:
        raise NotImplementedError

    def reset_state(self) -> None:
        """Forget all moment estimates and step counters."""

        self.state = {}

    def set_learning_rate(self, learning_rate: float) -> None:
        if not np.isfinite(learning_rate) or learning_rate < 0:
            raise ValueError("learning_rate must be non-negative and finite")
        self.learning_rate = float(learning_rate)

    def _validate(
        self,
        parameters: Sequence[np.ndarray],
        gradients: Sequence[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        params = list(parameters)
        grads = list(gradients)
        if not params:
            raise ValueError("parameters must contain at least one array")
        if len(params) != len(grads):
            raise ValueError(
                f"parameters and gradients must have the same length, got {len(params)} and {len(grads)}"
            )
        for index, (parameter, gradient) in enumerate(zip(params, grads, strict=True)):
            if not isinstance(parameter, np.ndarray) or not isinstance(gradient, np.ndarray):
                raise TypeError(f"parameter {index} and its gradient must be numpy arrays")
            if parameter.shape != gradient.shape:
                raise ValueError(
                    f"parameter {index} shape {parameter.shape} != gradient shape {gradient.shape}"
                )
            if not np.issubdtype(parameter.dtype, np.floating):
                raise TypeError(f"parameter {index} must have a floating dtype")
            if not np.all(np.isfinite(parameter)):
                raise FloatingPointError(f"parameter {index} contains NaN or infinity")
            if not np.all(np.isfinite(gradient)):
                raise FloatingPointError(f"gradient {index} contains NaN or infinity")
            if not parameter.flags.writeable:
                raise ValueError(f"parameter {index} is read-only")
        return params, grads

    def _state_arrays(self, key: str, parameters: list[np.ndarray]) -> list[np.ndarray]:
        existing = self.state.get(key)
        valid = (
            isinstance(existing, list)
            and len(existing) == len(parameters)
            and all(
                isinstance(item, np.ndarray) and item.shape == parameter.shape
                for item, parameter in zip(existing, parameters, strict=True)
            )
        )
        if not valid:
            existing = [np.zeros_like(parameter) for parameter in parameters]
            self.state[key] = existing
        return existing
