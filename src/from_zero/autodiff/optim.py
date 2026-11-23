"""A minimal SGD optimizer for scalar ``Value`` parameters."""

from __future__ import annotations

from collections.abc import Iterable

from .value import Value


class SGD:
    """Update ``parameter.data -= learning_rate * parameter.grad``."""

    def __init__(self, parameters: Iterable[Value], learning_rate: float = 0.1) -> None:
        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("SGD requires at least one parameter")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.learning_rate = float(learning_rate)

    def step(self) -> None:
        for parameter in self.parameters:
            parameter.data -= self.learning_rate * parameter.grad

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = 0.0
