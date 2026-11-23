"""A tiny scalar neural-network library built on the autodiff engine."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from .value import Value

ACTIVATIONS = ("tanh", "relu", "sigmoid", "linear")


class Module:
    """Minimal parameter container."""

    def parameters(self) -> Iterator[Value]:
        yield from _iter_parameters(self)

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.grad = 0.0

    def parameter_count(self) -> int:
        return sum(1 for _ in self.parameters())


class Neuron(Module):
    """One linear unit followed by a scalar activation."""

    def __init__(
        self,
        n_inputs: int,
        *,
        activation: str = "tanh",
        rng: random.Random | None = None,
    ) -> None:
        if n_inputs < 1:
            raise ValueError("n_inputs must be at least 1")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        generator = rng or random
        limit = 1.0 / math.sqrt(n_inputs)
        self.weights = [Value(generator.uniform(-limit, limit)) for _ in range(n_inputs)]
        self.bias = Value(generator.uniform(-limit, limit))
        self.activation = activation

    def __call__(self, inputs: Sequence[Value | float]) -> Value:
        if len(inputs) != len(self.weights):
            raise ValueError(f"expected {len(self.weights)} inputs, got {len(inputs)}")
        total = self.bias
        for weight, item in zip(self.weights, inputs, strict=True):
            total = total + weight * item
        return _activate(total, self.activation)


class Layer(Module):
    """A list of independent neurons."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        *,
        activation: str = "tanh",
        rng: random.Random | None = None,
    ) -> None:
        if n_outputs < 1:
            raise ValueError("n_outputs must be at least 1")
        self.neurons = [Neuron(n_inputs, activation=activation, rng=rng) for _ in range(n_outputs)]

    def __call__(self, inputs: Sequence[Value | float]) -> list[Value]:
        return [neuron(inputs) for neuron in self.neurons]


class MLP(Module):
    """A multilayer perceptron with scalar-valued parameters.

    A single output is returned as a scalar ``Value`` for convenience; multiple
    outputs are returned as a list.
    """

    def __init__(
        self,
        n_inputs: int,
        hidden_sizes: Sequence[int],
        n_outputs: int,
        *,
        activation: str = "tanh",
        output_activation: str = "linear",
        seed: int | None = None,
    ) -> None:
        if n_inputs < 1 or n_outputs < 1:
            raise ValueError("n_inputs and n_outputs must be positive")
        if any(size < 1 for size in hidden_sizes):
            raise ValueError("hidden sizes must be positive")
        if activation not in ACTIVATIONS or output_activation not in ACTIVATIONS:
            raise ValueError(f"activations must be one of {ACTIVATIONS}")
        rng = random.Random(seed)
        sizes = [n_inputs, *hidden_sizes, n_outputs]
        self.layers: list[Layer] = []
        for index in range(len(sizes) - 1):
            is_output = index == len(sizes) - 2
            self.layers.append(
                Layer(
                    sizes[index],
                    sizes[index + 1],
                    activation=output_activation if is_output else activation,
                    rng=rng,
                )
            )

    def __call__(self, inputs: Sequence[Value | float]) -> Value | list[Value]:
        current: Sequence[Value | float] = list(inputs)
        for layer in self.layers:
            current = layer(current)
        outputs = list(current)
        return outputs[0] if len(outputs) == 1 else outputs


def _activate(value: Value, name: str) -> Value:
    if name == "tanh":
        return value.tanh()
    if name == "relu":
        return value.relu()
    if name == "sigmoid":
        return value.sigmoid()
    if name == "linear":
        return value
    raise ValueError(f"Unknown activation: {name}")


def _iter_parameters(value: Any) -> Iterator[Value]:
    if isinstance(value, Value):
        yield value
    elif isinstance(value, Module):
        for item in value.__dict__.values():
            yield from _iter_parameters(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_parameters(item)
    elif isinstance(value, Iterable) and not isinstance(value, str | bytes):
        for item in value:
            yield from _iter_parameters(item)
