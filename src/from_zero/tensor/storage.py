"""Flat storage, shape, stride, and broadcasting utilities.

This module is deliberately dependency-free: it uses only the Python
standard library so that the tensor engine can run anywhere.

Storage layout is row-major (C order).  A tensor is described by

* ``data``    -- a flat Python list of floats,
* ``shape``   -- a tuple of non-negative integers,
* ``strides`` -- the number of flat elements between adjacent indices,
* ``offset``  -- the flat position of the first logical element.

For a shape ``(d0, d1, d2)`` the row-major strides are
``(d1 * d2, d2, 1)`` and the flat offset of ``(i0, i1, i2)`` is
``i0 * s0 + i1 * s1 + i2 * s2``.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

Shape = tuple[int, ...]


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------
def normalize_shape(shape: Sequence[int]) -> Shape:
    """Validate and normalise a shape tuple."""
    out: list[int] = []
    for dim in shape:
        if not isinstance(dim, int):
            raise TypeError(f"Shape dimensions must be integers, got {type(dim).__name__}")
        if dim < 0:
            raise ValueError(f"Shape dimensions must be non-negative, got {dim}")
        out.append(dim)
    return tuple(out)


def numel(shape: Sequence[int]) -> int:
    """Number of elements described by ``shape``."""
    total = 1
    for dim in shape:
        total *= dim
    return total


def compute_strides(shape: Sequence[int]) -> Shape:
    """Row-major strides for ``shape``."""
    shape = tuple(shape)
    strides: list[int] = [0] * len(shape)
    running = 1
    for dim in range(len(shape) - 1, -1, -1):
        strides[dim] = running
        running *= shape[dim]
    return tuple(strides)


def flatten_index(index: Sequence[int], shape: Sequence[int], strides: Sequence[int]) -> int:
    """Flat offset of a multi-dimensional ``index``."""
    if len(index) != len(shape):
        raise ValueError(f"Index rank {len(index)} does not match shape {tuple(shape)}")
    offset = 0
    for idx, dim, stride in zip(index, shape, strides, strict=False):
        if not 0 <= idx < dim:
            raise IndexError(f"Index {idx} out of range for dimension of size {dim}")
        offset += idx * stride
    return offset


def unflatten_index(offset: int, shape: Sequence[int]) -> Shape:
    """Convert a flat offset to a multi-dimensional index (row-major)."""
    if offset < 0 or offset >= numel(shape):
        raise IndexError(f"Flat index {offset} out of range for shape {tuple(shape)}")
    index: list[int] = [0] * len(shape)
    remaining = offset
    for dim in range(len(shape) - 1, -1, -1):
        index[dim] = remaining % shape[dim]
        remaining //= shape[dim]
    return tuple(index)


def is_contiguous(shape: Sequence[int], strides: Sequence[int]) -> bool:
    """True when ``strides`` describe a dense row-major layout."""
    return tuple(strides) == compute_strides(shape)


# ---------------------------------------------------------------------------
# Broadcasting
# ---------------------------------------------------------------------------
def broadcast_shape(shape_a: Sequence[int], shape_b: Sequence[int]) -> Shape:
    """Shape produced by broadcasting ``shape_a`` with ``shape_b``.

    Dimensions are aligned from the right; two dimensions are compatible
    when they are equal or one of them is 1.
    """
    a = tuple(shape_a)
    b = tuple(shape_b)
    result: list[int] = []
    for i in range(1, max(len(a), len(b)) + 1):
        da = a[-i] if i <= len(a) else 1
        db = b[-i] if i <= len(b) else 1
        if da == db:
            result.append(da)
        elif da == 1:
            result.append(db)
        elif db == 1:
            result.append(da)
        else:
            raise ValueError(f"Shapes {a} and {b} are not broadcastable")
    return tuple(reversed(result))


def is_broadcastable(shape_a: Sequence[int], shape_b: Sequence[int]) -> bool:
    try:
        broadcast_shape(shape_a, shape_b)
        return True
    except ValueError:
        return False


def broadcast_index(index: Sequence[int], shape: Sequence[int], out_shape: Sequence[int]) -> Shape:
    """Map an index in ``out_shape`` to the corresponding index in ``shape``.

    Dimensions of size 1 in ``shape`` are indexed with 0; missing leading
    dimensions are dropped.
    """
    if len(index) != len(out_shape):
        raise ValueError("Index rank does not match the broadcast shape")
    shift = len(out_shape) - len(shape)
    mapped: list[int] = []
    for dim, size in enumerate(shape):
        out_dim = index[shift + dim]
        mapped.append(0 if size == 1 else out_dim)
    return tuple(mapped)


def broadcast_flat_index(out_flat: int, out_shape: Sequence[int], in_shape: Sequence[int]) -> int:
    """Flat index in ``in_shape`` corresponding to ``out_flat`` in ``out_shape``."""
    if tuple(out_shape) == tuple(in_shape):
        return out_flat
    idx = unflatten_index(out_flat, out_shape)
    mapped = broadcast_index(idx, in_shape, out_shape)
    return flatten_index(mapped, in_shape, compute_strides(in_shape))


def broadcast_values(
    values: Sequence[float], in_shape: Sequence[int], out_shape: Sequence[int]
) -> list[float]:
    """Repeat ``values`` according to broadcasting rules."""
    if tuple(in_shape) == tuple(out_shape):
        return list(values)
    total = numel(out_shape)
    out = [0.0] * total
    if not in_shape:
        return [values[0]] * total
    for k in range(total):
        out[k] = values[broadcast_flat_index(k, out_shape, in_shape)]
    return out


def sum_to_shape(
    values: Sequence[float], src_shape: Sequence[int], target_shape: Sequence[int]
) -> list[float]:
    """Sum ``values`` over broadcast dimensions until ``target_shape``.

    ``target_shape`` must be right-aligned with ``src_shape``; every target
    dimension must equal the source dimension or be 1 (or absent).
    """
    src_shape = tuple(src_shape)
    target_shape = tuple(target_shape)
    if src_shape == target_shape:
        return list(values)
    if len(target_shape) > len(src_shape):
        raise ValueError(f"Cannot reduce shape {src_shape} to {target_shape}")

    total = numel(target_shape)
    out = [0.0] * total
    if total == 0:
        return out
    target_strides = compute_strides(target_shape)
    shift = len(src_shape) - len(target_shape)

    for k, value in enumerate(values):
        src_idx = unflatten_index(k, src_shape)
        target_idx: list[int] = []
        for dim, size in enumerate(target_shape):
            src_dim = src_idx[shift + dim]
            if size == 1:
                target_idx.append(0)
            else:
                target_idx.append(src_dim)
        out[flatten_index(target_idx, target_shape, target_strides)] += value
    return out


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
class Storage:
    """A flat float buffer shared by one or more tensor views."""

    __slots__ = ("data", "offset", "shape", "strides")

    def __init__(
        self,
        data: Iterable[float],
        shape: Sequence[int] | None = None,
        strides: Sequence[int] | None = None,
        offset: int = 0,
    ) -> None:
        self.data: list[float] = [float(v) for v in data]
        if shape is None:
            shape = (len(self.data),)
        self.shape = normalize_shape(shape)
        expected = numel(self.shape)
        if expected != len(self.data):
            raise ValueError(
                f"Storage has {len(self.data)} values but shape {self.shape} needs {expected}"
            )
        self.strides = tuple(strides) if strides is not None else compute_strides(self.shape)
        if len(self.strides) != len(self.shape):
            raise ValueError("Strides rank must match shape rank")
        self.offset = int(offset)
        if not 0 <= self.offset <= len(self.data):
            raise ValueError("Storage offset out of range")

    def clone(self) -> Storage:
        return Storage(list(self.data), self.shape, self.strides, self.offset)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"Storage(size={len(self.data)}, shape={self.shape}, offset={self.offset})"


# ---------------------------------------------------------------------------
# Random number helpers
# ---------------------------------------------------------------------------
class RNG:
    """Deterministic random source for tensors, data, dropout and sampling."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed_value = seed
        self._random = random.Random(seed)

    def reseed(self, seed: int) -> None:
        self.seed_value = seed
        self._random = random.Random(seed)

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return self._random.uniform(low, high)

    def normal(self, mean: float = 0.0, std: float = 1.0) -> float:
        return self._random.gauss(mean, std)

    def randint(self, low: int, high: int) -> int:
        return self._random.randint(low, high)

    def choice(self, population: Sequence) -> object:
        return self._random.choice(population)

    def shuffle(self, values: list) -> None:
        self._random.shuffle(values)

    def sample(self, population: Sequence, k: int) -> list:
        return self._random.sample(population, k)

    def state(self) -> object:
        return self._random.getstate()

    def set_state(self, state: object) -> None:
        self._random.setstate(state)


_default_rng = RNG()


def default_rng() -> RNG:
    return _default_rng


def seed_default_rng(seed: int) -> None:
    _default_rng.reseed(seed)


def randn(shape: Sequence[int], generator: RNG | None = None) -> list[float]:
    rng = generator or _default_rng
    return [rng.normal() for _ in range(numel(shape))]


def uniform_values(
    shape: Sequence[int], low: float = 0.0, high: float = 1.0, generator: RNG | None = None
) -> list[float]:
    rng = generator or _default_rng
    return [rng.uniform(low, high) for _ in range(numel(shape))]


__all__ = [
    "RNG",
    "Shape",
    "Storage",
    "broadcast_flat_index",
    "broadcast_index",
    "broadcast_shape",
    "broadcast_values",
    "compute_strides",
    "default_rng",
    "flatten_index",
    "is_broadcastable",
    "is_contiguous",
    "normalize_shape",
    "numel",
    "randn",
    "seed_default_rng",
    "sum_to_shape",
    "unflatten_index",
    "uniform_values",
]
