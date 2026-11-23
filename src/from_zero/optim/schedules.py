"""Learning-rate schedules.

A schedule returns the learning rate to use for a zero-based update index.  The
optimizer remains responsible for applying it, so schedules can be tested
independently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Schedule(ABC):
    """Base class for a learning-rate schedule."""

    name = "schedule"

    @abstractmethod
    def __call__(self, step: int) -> float:
        raise NotImplementedError

    @staticmethod
    def _validate_step(step: int) -> int:
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step < 0:
            raise ValueError("step must be non-negative")
        return step


class ConstantSchedule(Schedule):
    name = "constant"

    def __init__(self, learning_rate: float) -> None:
        if learning_rate <= 0 or not np.isfinite(learning_rate):
            raise ValueError("learning_rate must be positive and finite")
        self.learning_rate = float(learning_rate)

    def __call__(self, step: int) -> float:
        self._validate_step(step)
        return self.learning_rate


class StepDecay(Schedule):
    """Multiply the rate by ``gamma`` every ``step_size`` updates."""

    name = "step_decay"

    def __init__(self, learning_rate: float, step_size: int, gamma: float = 0.5) -> None:
        if learning_rate <= 0 or not np.isfinite(learning_rate):
            raise ValueError("learning_rate must be positive and finite")
        if step_size < 1:
            raise ValueError("step_size must be at least 1")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.learning_rate = float(learning_rate)
        self.step_size = int(step_size)
        self.gamma = float(gamma)

    def __call__(self, step: int) -> float:
        index = self._validate_step(step)
        return self.learning_rate * self.gamma ** (index // self.step_size)


class ExponentialDecay(Schedule):
    """Multiply the rate by ``gamma`` after every update."""

    name = "exponential_decay"

    def __init__(self, learning_rate: float, gamma: float = 0.99) -> None:
        if learning_rate <= 0 or not np.isfinite(learning_rate):
            raise ValueError("learning_rate must be positive and finite")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)

    def __call__(self, step: int) -> float:
        index = self._validate_step(step)
        return self.learning_rate * self.gamma**index


class CosineDecay(Schedule):
    """Cosine annealing from ``learning_rate`` to ``min_learning_rate``."""

    name = "cosine_decay"

    def __init__(
        self,
        learning_rate: float,
        total_steps: int,
        min_learning_rate: float = 0.0,
    ) -> None:
        if learning_rate <= 0 or not np.isfinite(learning_rate):
            raise ValueError("learning_rate must be positive and finite")
        if total_steps < 1:
            raise ValueError("total_steps must be at least 1")
        if min_learning_rate < 0 or min_learning_rate > learning_rate:
            raise ValueError("min_learning_rate must be in [0, learning_rate]")
        self.learning_rate = float(learning_rate)
        self.total_steps = int(total_steps)
        self.min_learning_rate = float(min_learning_rate)

    def __call__(self, step: int) -> float:
        index = self._validate_step(step)
        progress = min(index, self.total_steps) / self.total_steps
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return self.min_learning_rate + (self.learning_rate - self.min_learning_rate) * cosine


def build_schedule(config: dict[str, Any]) -> Schedule:
    """Build a schedule from a small configuration mapping."""

    kind = str(config.get("kind", "constant")).lower()
    learning_rate = float(config.get("learning_rate", 0.01))
    if kind == "constant":
        return ConstantSchedule(learning_rate)
    if kind == "step":
        return StepDecay(
            learning_rate,
            step_size=int(config.get("step_size", 30)),
            gamma=float(config.get("gamma", 0.5)),
        )
    if kind in {"exponential", "exp"}:
        return ExponentialDecay(learning_rate, gamma=float(config.get("gamma", 0.99)))
    if kind == "cosine":
        return CosineDecay(
            learning_rate,
            total_steps=int(config.get("total_steps", 100)),
            min_learning_rate=float(config.get("min_learning_rate", 0.0)),
        )
    raise ValueError(f"Unknown schedule kind: {kind}")
