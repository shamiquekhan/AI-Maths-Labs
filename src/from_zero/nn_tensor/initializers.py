"""Weight initialisation routines using explicit random generators."""

from __future__ import annotations

import math

from ..tensor import Tensor
from ..tensor.storage import RNG, default_rng, is_contiguous


def _fill(tensor: Tensor, values) -> Tensor:
    if not is_contiguous(tensor.shape, tensor.strides):
        raise ValueError("Initialisers require contiguous tensors")
    tensor._data[tensor._offset : tensor._offset + tensor.size] = [float(v) for v in values]
    return tensor


def fan_in_and_fan_out(tensor: Tensor) -> tuple:
    if tensor.ndim < 2:
        raise ValueError("fan_in/fan_out require at least 2 dimensions")
    receptive = 1
    for dim in tensor.shape[2:]:
        receptive *= dim
    return tensor.shape[1] * receptive, tensor.shape[0] * receptive


def calculate_gain(nonlinearity: str, param: float | None = None) -> float:
    if nonlinearity in {"linear", "conv", "sigmoid"}:
        return 1.0
    if nonlinearity == "tanh":
        return 5.0 / 3.0
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    if nonlinearity == "leaky_relu":
        slope = 0.01 if param is None else float(param)
        return math.sqrt(2.0 / (1.0 + slope * slope))
    if nonlinearity == "selu":
        return 0.75
    raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")


def zeros(tensor: Tensor) -> Tensor:
    return _fill(tensor, [0.0] * tensor.size)


def ones(tensor: Tensor) -> Tensor:
    return _fill(tensor, [1.0] * tensor.size)


def constant(tensor: Tensor, value: float) -> Tensor:
    return _fill(tensor, [value] * tensor.size)


def uniform(
    tensor: Tensor, low: float = 0.0, high: float = 1.0, generator: RNG | None = None
) -> Tensor:
    rng = generator or default_rng()
    return _fill(tensor, [rng.uniform(low, high) for _ in range(tensor.size)])


def normal(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    generator: RNG | None = None,
) -> Tensor:
    rng = generator or default_rng()
    return _fill(tensor, [rng.normal(mean, std) for _ in range(tensor.size)])


def xavier_uniform(tensor: Tensor, gain: float = 1.0, generator: RNG | None = None) -> Tensor:
    fan_in, fan_out = fan_in_and_fan_out(tensor)
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return uniform(tensor, -bound, bound, generator)


def xavier_normal(tensor: Tensor, gain: float = 1.0, generator: RNG | None = None) -> Tensor:
    fan_in, fan_out = fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return normal(tensor, 0.0, std, generator)


def he_uniform(
    tensor: Tensor,
    mode: str = "fan_in",
    nonlinearity: str = "relu",
    generator: RNG | None = None,
) -> Tensor:
    fan_in, fan_out = fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    bound = math.sqrt(6.0 / fan) * calculate_gain(nonlinearity)
    return uniform(tensor, -bound, bound, generator)


def he_normal(
    tensor: Tensor,
    mode: str = "fan_in",
    nonlinearity: str = "relu",
    generator: RNG | None = None,
) -> Tensor:
    fan_in, fan_out = fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    std = calculate_gain(nonlinearity) / math.sqrt(fan)
    return normal(tensor, 0.0, std, generator)


def lecun_uniform(tensor: Tensor, generator: RNG | None = None) -> Tensor:
    fan_in, _ = fan_in_and_fan_out(tensor)
    bound = math.sqrt(3.0 / fan_in)
    return uniform(tensor, -bound, bound, generator)


def lecun_normal(tensor: Tensor, generator: RNG | None = None) -> Tensor:
    fan_in, _ = fan_in_and_fan_out(tensor)
    return normal(tensor, 0.0, math.sqrt(1.0 / fan_in), generator)


INITIALIZERS = {
    "zeros": zeros,
    "ones": ones,
    "uniform": uniform,
    "normal": normal,
    "xavier_uniform": xavier_uniform,
    "xavier_normal": xavier_normal,
    "he_uniform": he_uniform,
    "he_normal": he_normal,
    "lecun_uniform": lecun_uniform,
    "lecun_normal": lecun_normal,
}


def initialize(tensor: Tensor, name: str, generator: RNG | None = None) -> Tensor:
    if name not in INITIALIZERS:
        raise ValueError(f"Unknown initializer {name!r}; choose from {sorted(INITIALIZERS)}")
    return INITIALIZERS[name](tensor, generator=generator)


__all__ = [
    "INITIALIZERS",
    "calculate_gain",
    "constant",
    "fan_in_and_fan_out",
    "he_normal",
    "he_uniform",
    "initialize",
    "lecun_normal",
    "lecun_uniform",
    "normal",
    "ones",
    "uniform",
    "xavier_normal",
    "xavier_uniform",
    "zeros",
]
