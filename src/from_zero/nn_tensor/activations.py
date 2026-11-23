"""Activation modules built on tensor primitives."""

from __future__ import annotations

from ..tensor import Tensor
from .module import Module


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    def extra_repr(self) -> str:
        return ""


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class GELU(Module):
    """Tanh approximation of GELU."""

    def forward(self, x: Tensor) -> Tensor:
        inner = (x + 0.044715 * (x * x * x)) * 0.7978845608028654
        return 0.5 * x * (1.0 + inner.tanh())


class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * x.sigmoid()


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        return x.relu() - self.negative_slope * (-x).relu()

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


class ELU(Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        negative = self.alpha * ((-((-x).relu())).exp() - 1.0)
        return x.relu() + negative

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"


class Softmax(Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(dim=self.dim)


class LogSoftmax(Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.log_softmax(dim=self.dim)


__all__ = [
    "ELU",
    "GELU",
    "LeakyReLU",
    "LogSoftmax",
    "ReLU",
    "SiLU",
    "Sigmoid",
    "Softmax",
    "Tanh",
]
