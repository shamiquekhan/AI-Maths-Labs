"""The scalar ``Value`` node and its differentiable operations.

Each node records a forward value, a gradient, its parent nodes, and a local
backward rule.  ``backward()`` performs a reverse topological traversal and
accumulates gradients with ``+=`` so shared subexpressions receive every
contribution.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


class Value:
    """A scalar node in a reverse-mode computation graph."""

    __slots__ = ("_backward", "_op", "_prev", "data", "grad")

    def __init__(
        self,
        data: float | int,
        _children: Iterable[Value] = (),
        _op: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._prev: tuple[Value, ...] = tuple(dict.fromkeys(_children))
        self._op = _op
        self._backward = lambda: None

    @property
    def parents(self) -> tuple[Value, ...]:
        return self._prev

    @property
    def op(self) -> str:
        return self._op

    @property
    def is_leaf(self) -> bool:
        return not self._prev

    def backward(self) -> None:
        """Run reverse-mode differentiation from this scalar objective.

        Leaf gradients are intentionally *accumulated*: repeated calls add
        another gradient contribution.  Call :meth:`zero_grad` to clear them.
        Intermediate gradients are recomputed for each call.
        """

        from .graph import topological_sort

        order = topological_sort(self)
        for node in order:
            if node._prev:
                node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):
            node._backward()

    def zero_grad(self) -> None:
        from .graph import zero_grad

        zero_grad(self)

    def __add__(self, other: Any) -> Value:
        other = _as_value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Any) -> Value:
        return _as_value(other) + self

    def __mul__(self, other: Any) -> Value:
        other = _as_value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Any) -> Value:
        return _as_value(other) * self

    def __neg__(self) -> Value:
        out = Value(-self.data, (self,), "neg")

        def _backward() -> None:
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other: Any) -> Value:
        return self + (-_as_value(other))

    def __rsub__(self, other: Any) -> Value:
        return _as_value(other) - self

    def __truediv__(self, other: Any) -> Value:
        other = _as_value(other)
        if other.data == 0.0:
            raise ZeroDivisionError("division by zero in Value")
        out = Value(self.data / other.data, (self, other), "/")

        def _backward() -> None:
            self.grad += out.grad / other.data
            other.grad += -self.data / (other.data * other.data) * out.grad

        out._backward = _backward
        return out

    def __rtruediv__(self, other: Any) -> Value:
        return _as_value(other) / self

    def __pow__(self, exponent: Any) -> Value:
        if isinstance(exponent, Value) or not isinstance(exponent, int | float):
            raise TypeError("Value supports only a scalar integer or float exponent")
        exponent = float(exponent)
        if self.data == 0.0 and exponent < 0.0:
            raise ValueError("negative powers of zero are undefined")
        if self.data == 0.0 and 0.0 < exponent < 1.0:
            raise ValueError("derivative of x**p is undefined at x=0 for 0 < p < 1")
        out = Value(self.data**exponent, (self,), f"**{exponent:g}")

        def _backward() -> None:
            derivative = 0.0 if exponent == 0.0 else exponent * self.data ** (exponent - 1.0)
            self.grad += derivative * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        if self.data <= 0.0:
            raise ValueError("log is defined only for positive values in this engine")
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        value = math.tanh(self.data)
        out = Value(value, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - value * value) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(max(self.data, 0.0), (self,), "relu")

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        value = _sigmoid_scalar(self.data)
        out = Value(value, (self,), "sigmoid")

        def _backward() -> None:
            self.grad += value * (1.0 - value) * out.grad

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"


def _as_value(value: Any) -> Value:
    if isinstance(value, Value):
        return value
    if isinstance(value, int | float):
        return Value(value)
    raise TypeError(f"Expected a Value or number, got {type(value).__name__}")


def _sigmoid_scalar(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)
