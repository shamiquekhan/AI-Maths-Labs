"""A dependency-free tensor engine with reverse-mode automatic differentiation.

The implementation is intentionally explicit: storage is a flat Python list,
shapes and strides are tuples, and every differentiable operation records a
local backward rule.  No NumPy, PyTorch, or other numerical library is used.

Design
------
* ``Tensor`` owns a flat ``list[float]`` buffer (possibly shared by views),
  a shape tuple, row-major strides, and an offset.
* Operations produce new tensors and attach a ``_backward`` closure that
  receives the upstream gradient through ``out.grad``.
* ``backward`` performs a topological sort and traverses nodes in reverse,
  accumulating gradients for tensors that appear on multiple paths.
* All arithmetic during the backward pass runs with gradient tracking
  disabled, so gradients never build a second graph.
"""

from __future__ import annotations

import builtins
import itertools
import math
from collections.abc import Callable, Sequence
from typing import Any

from .storage import (
    broadcast_flat_index,
    broadcast_shape,
    compute_strides,
    default_rng,
    flatten_index,
    is_contiguous,
    numel,
    sum_to_shape,
    unflatten_index,
)

Shape = tuple[int, ...]

# ---------------------------------------------------------------------------
# Gradient tracking switch
# ---------------------------------------------------------------------------
_GRAD_ENABLED = True


class no_grad:
    """Context manager that disables graph construction."""

    def __init__(self) -> None:
        self._previous = True

    def __enter__(self) -> no_grad:
        global _GRAD_ENABLED
        self._previous = _GRAD_ENABLED
        _GRAD_ENABLED = False
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._previous


def is_grad_enabled() -> bool:
    return _GRAD_ENABLED


# ---------------------------------------------------------------------------
# Nested-list helpers
# ---------------------------------------------------------------------------
def _nested_shape(value: Any) -> Shape:
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        first = _nested_shape(value[0])
        for item in value[1:]:
            if _nested_shape(item) != first:
                raise ValueError("Nested data must be rectangular")
        return (len(value), *first)
    return ()


def _flatten_nested(value: Any, out: list[float]) -> None:
    if isinstance(value, (list, tuple)):
        for item in value:
            _flatten_nested(item, out)
    else:
        out.append(float(value))


def _shape_and_values(data: Any) -> tuple[Shape, list[float]]:
    if isinstance(data, Tensor):
        return data.shape, data._values()
    shape = _nested_shape(data)
    values: list[float] = []
    _flatten_nested(data, values)
    return shape, values


def _resolve_shape(shape: Sequence[int], total: int) -> Shape:
    shape = tuple(shape)
    if shape.count(-1) > 1:
        raise ValueError("Only one dimension can be inferred with -1")
    if -1 in shape:
        known = 1
        for dim in shape:
            if dim != -1:
                if dim < 0:
                    raise ValueError(f"Invalid shape dimension {dim}")
                known *= dim
        if known == 0 or total % known != 0:
            raise ValueError(f"Cannot reshape {total} values into {shape}")
        shape = tuple(total // known if dim == -1 else dim for dim in shape)
    if numel(shape) != total:
        raise ValueError(f"Shape {shape} needs {numel(shape)} values but got {total}")
    return shape


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------
class Tensor:
    """Multi-dimensional float tensor with reverse-mode autograd."""

    __slots__ = (
        "_backward",
        "_data",
        "_offset",
        "_op",
        "_prev",
        "_shape",
        "_strides",
        "grad",
        "name",
        "requires_grad",
    )

    # -- construction ------------------------------------------------------
    def __init__(
        self,
        data: Tensor | Sequence | float | int = 0.0,
        shape: Sequence[int] | None = None,
        requires_grad: bool = False,
    ) -> None:
        inferred_shape, values = _shape_and_values(data)
        shape = inferred_shape if shape is None else _resolve_shape(shape, len(values))
        self._data = values
        self._shape = tuple(shape)
        self._strides = compute_strides(self._shape)
        self._offset = 0
        self.requires_grad = bool(requires_grad)
        self.grad: Tensor | None = None
        self._backward: Callable[[], None] | None = None
        self._prev: tuple[Tensor, ...] = ()
        self._op = "leaf"
        self.name: str | None = None

    @classmethod
    def _from_flat(
        cls,
        values: Sequence[float],
        shape: Sequence[int],
        requires_grad: bool = False,
        op: str = "",
        prev: tuple[Tensor, ...] = (),
    ) -> Tensor:
        obj = cls.__new__(cls)
        obj._data = [float(v) for v in values]
        obj._shape = tuple(shape)
        obj._strides = compute_strides(obj._shape)
        obj._offset = 0
        obj.requires_grad = bool(requires_grad)
        obj.grad = None
        obj._backward = None
        obj._prev = tuple(prev)
        obj._op = op
        obj.name = None
        return obj

    @classmethod
    def _view(
        cls,
        data: list[float],
        shape: Sequence[int],
        strides: Sequence[int],
        offset: int,
        requires_grad: bool = False,
        op: str = "",
        prev: tuple[Tensor, ...] = (),
    ) -> Tensor:
        obj = cls.__new__(cls)
        obj._data = data
        obj._shape = tuple(shape)
        obj._strides = tuple(strides)
        obj._offset = int(offset)
        obj.requires_grad = bool(requires_grad)
        obj.grad = None
        obj._backward = None
        obj._prev = tuple(prev)
        obj._op = op
        obj.name = None
        return obj

    # -- metadata ----------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def strides(self) -> Shape:
        return self._strides

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return numel(self._shape)

    @property
    def data(self) -> list[float]:
        """Logical values in row-major order (a copy)."""
        return self._values()

    def __len__(self) -> int:
        if not self._shape:
            raise TypeError("len() of a 0-d tensor")
        return self._shape[0]

    def __repr__(self) -> str:
        if self.ndim == 0:
            return f"Tensor({self.item():.6g}, requires_grad={self.requires_grad})"
        preview = self._values()
        if len(preview) > 8:
            preview = preview[:8]
            suffix = ", ..."
        else:
            suffix = ""
        return (
            f"Tensor(shape={self.shape}, values=[{', '.join(f'{v:.6g}' for v in preview)}{suffix}], "
            f"requires_grad={self.requires_grad})"
        )

    def __hash__(self) -> int:
        return id(self)

    def __float__(self) -> float:
        return self.item()

    # -- value access ------------------------------------------------------
    def _values(self) -> list[float]:
        """Logical values in row-major order."""
        n = self.size
        if n == 0:
            return []
        ndim = self.ndim
        if ndim == 0:
            return [self._data[self._offset]]
        data = self._data
        out = [0.0] * n
        index = [0] * ndim
        offset = self._offset
        for k in range(n):
            out[k] = data[offset]
            for dim in range(ndim - 1, -1, -1):
                index[dim] += 1
                offset += self._strides[dim]
                if index[dim] < self._shape[dim]:
                    break
                offset -= self._strides[dim] * self._shape[dim]
                index[dim] = 0
        return out

    def _at(self, index: Sequence[int]) -> float:
        offset = self._offset
        for idx, stride in zip(index, self._strides, strict=False):
            offset += idx * stride
        return self._data[offset]

    def item(self) -> float:
        if self.size != 1:
            raise ValueError(f"item() requires a single element, got size {self.size}")
        return self._values()[0]

    def tolist(self) -> Any:
        if self.ndim == 0:
            return self.item()
        if self.ndim == 1:
            return self._values()
        values = self._values()
        return _nest(values, self._shape)

    def clone(self) -> Tensor:
        return Tensor._from_flat(self._values(), self._shape, self.requires_grad, op="clone")

    def detach(self) -> Tensor:
        return Tensor._view(
            self._data, self._shape, self._strides, self._offset, False, op="detach"
        )

    # -- gradient machinery ------------------------------------------------
    def _accumulate(self, gradient: Tensor) -> None:
        if not self.requires_grad or gradient is None:
            return
        if gradient.shape != self._shape:
            gradient = gradient.reshape(self._shape)
        if self.grad is None:
            self.grad = gradient
        else:
            self.grad = self.grad + gradient

    def zero_grad(self) -> None:
        self.grad = None

    def backward(self, gradient: Tensor | None = None) -> None:
        """Backpropagate from this tensor.

        Scalar outputs seed the gradient with 1.  Non-scalar outputs require
        an explicit upstream ``gradient`` with the same shape.
        """
        if gradient is not None:
            if not isinstance(gradient, Tensor):
                gradient = Tensor(gradient)
            if gradient.shape != self.shape:
                raise ValueError(
                    f"Upstream gradient shape {gradient.shape} does not match output shape {self.shape}"
                )
        elif self.size != 1:
            raise RuntimeError("backward() requires an explicit gradient for non-scalar outputs")

        order: list[Tensor] = []
        visited = set()

        def visit(node: Tensor) -> None:
            key = id(node)
            if key in visited:
                return
            visited.add(key)
            for parent in node._prev:
                visit(parent)
            order.append(node)

        visit(self)

        global _GRAD_ENABLED
        previous = _GRAD_ENABLED
        _GRAD_ENABLED = False
        try:
            seed = (
                gradient
                if gradient is not None
                else Tensor._from_flat([1.0] * self.size, self._shape)
            )
            if self.grad is None:
                self.grad = seed
            else:
                self.grad = self.grad + seed
            for node in reversed(order):
                if node._backward is not None:
                    node._backward()
        finally:
            _GRAD_ENABLED = previous

    # -- indexing ----------------------------------------------------------
    def __getitem__(self, key: Any) -> Tensor:
        if not isinstance(key, tuple):
            key = (key,)
        if key.count(Ellipsis) > 1:
            raise IndexError("Only one ellipsis is allowed")
        if Ellipsis in key:
            pos = key.index(Ellipsis)
            fill = self.ndim - (len(key) - 1)
            key = key[:pos] + (slice(None),) * fill + key[pos + 1 :]
        if len(key) > self.ndim:
            raise IndexError(f"Too many indices for tensor of dimension {self.ndim}")
        key = key + (slice(None),) * (self.ndim - len(key))

        ranges: list[list[int]] = []
        out_shape: list[int] = []
        for dim, selector in enumerate(key):
            size = self._shape[dim]
            if isinstance(selector, int):
                index = selector + size if selector < 0 else selector
                if not 0 <= index < size:
                    raise IndexError(f"Index {selector} out of range for dimension {dim}")
                ranges.append([index])
            elif isinstance(selector, slice):
                indices = list(range(*selector.indices(size)))
                ranges.append(indices)
                out_shape.append(len(indices))
            else:
                raise TypeError(f"Unsupported index type {type(selector).__name__}")

        values: list[float] = []
        logical_offsets: list[int] = []
        self_strides = compute_strides(self._shape)
        for combination in itertools.product(*ranges):
            offset = self._offset
            logical = 0
            for idx, stride, logical_stride in zip(
                combination, self._strides, self_strides, strict=False
            ):
                offset += idx * stride
                logical += idx * logical_stride
            values.append(self._data[offset])
            logical_offsets.append(logical)

        requires = _GRAD_ENABLED and self.requires_grad
        out = Tensor._from_flat(values, tuple(out_shape), requires, op="getitem", prev=(self,))
        if requires:

            def _backward() -> None:
                upstream = out.grad._values()
                buffer = [0.0] * self.size
                for logical, value in zip(logical_offsets, upstream, strict=False):
                    buffer[logical] += value
                self._accumulate(Tensor._from_flat(buffer, self._shape))

            out._backward = _backward
        return out

    def __setitem__(self, key: Any, value: Any) -> None:
        """In-place scalar assignment (not differentiable)."""
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (self.ndim - len(key))
        ranges = []
        for dim, selector in enumerate(key):
            if isinstance(selector, int):
                ranges.append([selector + self._shape[dim] if selector < 0 else selector])
            elif isinstance(selector, slice):
                ranges.append(list(range(*selector.indices(self._shape[dim]))))
            else:
                raise TypeError(f"Unsupported index type {type(selector).__name__}")
        replacement = value._values() if isinstance(value, Tensor) else None
        for counter, combination in enumerate(itertools.product(*ranges)):
            offset = self._offset + builtins.sum(
                i * s for i, s in zip(combination, self._strides, strict=False)
            )
            if replacement is not None:
                self._data[offset] = replacement[counter % len(replacement)]
            else:
                self._data[offset] = float(value)

    # -- shape operations --------------------------------------------------
    def reshape(self, *shape: Any) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            target = tuple(shape[0])
        else:
            target = tuple(shape)
        target = _resolve_shape(target, self.size)

        requires = _GRAD_ENABLED and self.requires_grad
        if is_contiguous(self._shape, self._strides):
            out = Tensor._view(
                self._data,
                target,
                compute_strides(target),
                self._offset,
                requires,
                op="reshape",
                prev=(self,),
            )
        else:
            out = Tensor._from_flat(self._values(), target, requires, op="reshape", prev=(self,))
        if requires:

            def _backward() -> None:
                self._accumulate(out.grad.reshape(self._shape))

            out._backward = _backward
        return out

    def view(self, *shape: Any) -> Tensor:
        return self.reshape(*shape)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Tensor:
        ndim = self.ndim
        if ndim == 0:
            return self.reshape(1)
        start = start_dim + ndim if start_dim < 0 else start_dim
        end = end_dim + ndim if end_dim < 0 else end_dim
        if not 0 <= start <= end < ndim:
            raise ValueError("Invalid flatten dimensions")
        new_shape = (
            *self._shape[:start],
            numel(self._shape[start : end + 1]),
            *self._shape[end + 1 :],
        )
        return self.reshape(new_shape)

    def transpose(self, dim0: int = -2, dim1: int = -1) -> Tensor:
        if self.ndim < 2:
            if dim0 == dim1:
                return self
            raise ValueError("transpose requires at least 2 dimensions")
        d0 = dim0 + self.ndim if dim0 < 0 else dim0
        d1 = dim1 + self.ndim if dim1 < 0 else dim1
        if not 0 <= d0 < self.ndim or not 0 <= d1 < self.ndim:
            raise ValueError("Transpose dimensions out of range")
        new_shape = list(self._shape)
        new_strides = list(self._strides)
        new_shape[d0], new_shape[d1] = new_shape[d1], new_shape[d0]
        new_strides[d0], new_strides[d1] = new_strides[d1], new_strides[d0]
        requires = _GRAD_ENABLED and self.requires_grad
        out = Tensor._view(
            self._data, new_shape, new_strides, self._offset, requires, op="transpose", prev=(self,)
        )
        if requires:

            def _backward() -> None:
                self._accumulate(out.grad.transpose(d0, d1))

            out._backward = _backward
        return out

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def permute(self, *dims: int) -> Tensor:
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        if len(dims) != self.ndim:
            raise ValueError(f"Expected {self.ndim} dimensions, got {len(dims)}")
        normalized = tuple(d + self.ndim if d < 0 else d for d in dims)
        if sorted(normalized) != list(range(self.ndim)):
            raise ValueError("Invalid permutation")
        new_shape = tuple(self._shape[d] for d in normalized)
        new_strides = tuple(self._strides[d] for d in normalized)
        requires = _GRAD_ENABLED and self.requires_grad
        out = Tensor._view(
            self._data, new_shape, new_strides, self._offset, requires, op="permute", prev=(self,)
        )
        if requires:
            inverse = [0] * self.ndim
            for position, dim in enumerate(normalized):
                inverse[dim] = position

            def _backward() -> None:
                self._accumulate(out.grad.permute(*inverse))

            out._backward = _backward
        return out

    def squeeze(self, dim: int | None = None) -> Tensor:
        if dim is None:
            new_shape = tuple(d for d in self._shape if d != 1)
        else:
            d = dim + self.ndim if dim < 0 else dim
            if not 0 <= d < self.ndim:
                raise ValueError("Squeeze dimension out of range")
            if self._shape[d] != 1:
                return self
            new_shape = self._shape[:d] + self._shape[d + 1 :]
        return self.reshape(new_shape)

    def unsqueeze(self, dim: int) -> Tensor:
        d = dim + self.ndim + 1 if dim < 0 else dim
        if not 0 <= d <= self.ndim:
            raise ValueError("Unsqueeze dimension out of range")
        return self.reshape((*self._shape[:d], 1, *self._shape[d:]))

    # -- reductions --------------------------------------------------------
    def sum(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        if dim is None:
            total = builtins.sum(self._values())
            requires = _GRAD_ENABLED and self.requires_grad
            out = Tensor._from_flat([total], (), requires, op="sum", prev=(self,))
            if requires:

                def _backward() -> None:
                    value = out.grad.item()
                    self._accumulate(Tensor._from_flat([value] * self.size, self._shape))

                out._backward = _backward
            return out

        d = dim + self.ndim if dim < 0 else dim
        if not 0 <= d < self.ndim:
            raise ValueError("Sum dimension out of range")
        out_shape = (
            (*self._shape[:d], 1, *self._shape[d + 1 :])
            if keepdim
            else self._shape[:d] + self._shape[d + 1 :]
        )
        out_strides = compute_strides(out_shape)
        mapping: list[int] = []
        values = [0.0] * numel(out_shape)
        for k, value in enumerate(self._values()):
            index = unflatten_index(k, self._shape)
            out_index = (*index[:d], 0, *index[d + 1 :]) if keepdim else index[:d] + index[d + 1 :]
            flat = flatten_index(out_index, out_shape, out_strides)
            mapping.append(flat)
            values[flat] += value

        requires = _GRAD_ENABLED and self.requires_grad
        out = Tensor._from_flat(values, out_shape, requires, op="sum", prev=(self,))
        if requires:

            def _backward() -> None:
                upstream = out.grad._values()
                buffer = [upstream[flat] for flat in mapping]
                self._accumulate(Tensor._from_flat(buffer, self._shape))

            out._backward = _backward
        return out

    def mean(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        if dim is None:
            count = self.size
        else:
            d = dim + self.ndim if dim < 0 else dim
            count = self._shape[d]
        if count == 0:
            raise ValueError("mean of an empty tensor is undefined")
        return self.sum(dim=dim, keepdim=keepdim) / float(count)

    def max(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        values = self._values()
        if dim is None:
            if not values:
                raise ValueError("max of an empty tensor is undefined")
            best = builtins.max(values)
            arg = values.index(best)
            requires = _GRAD_ENABLED and self.requires_grad
            out = Tensor._from_flat([best], (), requires, op="max", prev=(self,))
            if requires:

                def _backward() -> None:
                    buffer = [0.0] * self.size
                    buffer[arg] = out.grad.item()
                    self._accumulate(Tensor._from_flat(buffer, self._shape))

                out._backward = _backward
            return out

        d = dim + self.ndim if dim < 0 else dim
        if not 0 <= d < self.ndim:
            raise ValueError("Max dimension out of range")
        out_shape = (
            (*self._shape[:d], 1, *self._shape[d + 1 :])
            if keepdim
            else self._shape[:d] + self._shape[d + 1 :]
        )
        out_strides = compute_strides(out_shape)
        total = numel(out_shape)
        best = [-math.inf] * total
        arg = [0] * total
        for k, value in enumerate(values):
            index = unflatten_index(k, self._shape)
            out_index = (*index[:d], 0, *index[d + 1 :]) if keepdim else index[:d] + index[d + 1 :]
            flat = flatten_index(out_index, out_shape, out_strides)
            if value > best[flat]:
                best[flat] = value
                arg[flat] = k

        requires = _GRAD_ENABLED and self.requires_grad
        out = Tensor._from_flat(best, out_shape, requires, op="max", prev=(self,))
        if requires:

            def _backward() -> None:
                upstream = out.grad._values()
                buffer = [0.0] * self.size
                for flat, source in enumerate(arg):
                    buffer[source] += upstream[flat]
                self._accumulate(Tensor._from_flat(buffer, self._shape))

            out._backward = _backward
        return out

    def argmax(self, dim: int | None = None) -> int | list[int]:
        values = self._values()
        if dim is None:
            return values.index(builtins.max(values))
        d = dim + self.ndim if dim < 0 else dim
        out_shape = self._shape[:d] + self._shape[d + 1 :]
        out_strides = compute_strides(out_shape)
        best = [-math.inf] * numel(out_shape)
        arg = [0] * numel(out_shape)
        for k, value in enumerate(values):
            index = unflatten_index(k, self._shape)
            flat = flatten_index(index[:d] + index[d + 1 :], out_shape, out_strides)
            if value > best[flat]:
                best[flat] = value
                arg[flat] = index[d]
        return arg

    # -- elementwise unary ops --------------------------------------------
    def _unary(
        self,
        forward: Callable[[float], float],
        backward: Callable[[float, float, float], float],
        op: str,
    ) -> Tensor:
        values = self._values()
        out_values = [forward(v) for v in values]
        requires = _GRAD_ENABLED and self.requires_grad
        out = Tensor._from_flat(out_values, self._shape, requires, op=op, prev=(self,))
        if requires:

            def _backward() -> None:
                upstream = out.grad._values()
                local = [
                    backward(upstream[k], values[k], out_values[k]) for k in range(len(values))
                ]
                self._accumulate(Tensor._from_flat(local, self._shape))

            out._backward = _backward
        return out

    def exp(self) -> Tensor:
        return self._unary(math.exp, lambda g, x, y: g * y, "exp")

    def log(self) -> Tensor:
        return self._unary(math.log, lambda g, x, y: g / x, "log")

    def log1p(self) -> Tensor:
        return self._unary(lambda x: math.log1p(x), lambda g, x, y: g / (1.0 + x), "log1p")

    def sqrt(self) -> Tensor:
        return self._unary(math.sqrt, lambda g, x, y: g / (2.0 * y), "sqrt")

    def abs(self) -> Tensor:
        return self._unary(abs, lambda g, x, y: g if x > 0 else (-g if x < 0 else 0.0), "abs")

    def tanh(self) -> Tensor:
        return self._unary(math.tanh, lambda g, x, y: g * (1.0 - y * y), "tanh")

    def sigmoid(self) -> Tensor:
        def stable(x: float) -> float:
            if x >= 0:
                return 1.0 / (1.0 + math.exp(-x))
            z = math.exp(x)
            return z / (1.0 + z)

        return self._unary(stable, lambda g, x, y: g * y * (1.0 - y), "sigmoid")

    def relu(self) -> Tensor:
        return self._unary(
            lambda x: x if x > 0 else 0.0, lambda g, x, y: g if x > 0 else 0.0, "relu"
        )

    def softplus(self) -> Tensor:
        """Stable ``log(1 + exp(x))`` with derivative ``sigmoid(x)``.

        The explicit backward rule avoids the zero derivative that the
        ``max(x, 0) + log1p(exp(-abs(x)))`` decomposition would produce at
        ``x == 0``, where the true derivative is ``0.5``.
        """

        def forward(x: float) -> float:
            if x > 0:
                return x + math.log1p(math.exp(-x))
            return math.log1p(math.exp(x))

        def backward(g: float, x: float, y: float) -> float:
            if x >= 0:
                return g / (1.0 + math.exp(-x))
            z = math.exp(x)
            return g * z / (1.0 + z)

        return self._unary(forward, backward, "softplus")

    def clamp(self, min_value: float | None = None, max_value: float | None = None) -> Tensor:
        def forward(x: float) -> float:
            if min_value is not None and x < min_value:
                return min_value
            if max_value is not None and x > max_value:
                return max_value
            return x

        def backward(g: float, x: float, y: float) -> float:
            if min_value is not None and x < min_value:
                return 0.0
            if max_value is not None and x > max_value:
                return 0.0
            return g

        return self._unary(forward, backward, "clamp")

    def pow(self, exponent: float) -> Tensor:
        if not isinstance(exponent, (int, float)):
            raise TypeError("pow exponent must be a scalar")

        def forward(x: float) -> float:
            return x**exponent

        def backward(g: float, x: float, y: float) -> float:
            if exponent == 0:
                return 0.0
            return g * exponent * (x ** (exponent - 1))

        return self._unary(forward, backward, f"pow:{exponent}")

    # -- softmax -----------------------------------------------------------
    def softmax(self, dim: int = -1) -> Tensor:
        maximum = self.max(dim=dim, keepdim=True).detach()
        shifted = (self - maximum).exp()
        return shifted / shifted.sum(dim=dim, keepdim=True)

    def log_softmax(self, dim: int = -1) -> Tensor:
        maximum = self.max(dim=dim, keepdim=True).detach()
        shifted = self - maximum
        return shifted - shifted.exp().sum(dim=dim, keepdim=True).log()

    # -- binary operations -------------------------------------------------
    def _binary(
        self,
        other: Tensor | float | int,
        forward: Callable[[float, float], float],
        grad_self: Callable[[float, float, float], float],
        grad_other: Callable[[float, float, float], float],
        op: str,
    ) -> Tensor:
        other = as_tensor(other)
        out_shape = broadcast_shape(self._shape, other._shape)
        left = self._values()
        right = other._values()
        total = numel(out_shape)
        out_values = [0.0] * total
        left_index = [0] * total
        right_index = [0] * total
        for k in range(total):
            li = broadcast_flat_index(k, out_shape, self._shape)
            ri = broadcast_flat_index(k, out_shape, other._shape)
            left_index[k] = li
            right_index[k] = ri
            out_values[k] = forward(left[li], right[ri])

        requires = _GRAD_ENABLED and (self.requires_grad or other.requires_grad)
        out = Tensor._from_flat(out_values, out_shape, requires, op=op, prev=(self, other))
        if requires:

            def _backward() -> None:
                upstream = out.grad._values()
                if self.requires_grad:
                    local = [
                        grad_self(upstream[k], left[left_index[k]], right[right_index[k]])
                        for k in range(total)
                    ]
                    reduced = sum_to_shape(local, out_shape, self._shape)
                    self._accumulate(Tensor._from_flat(reduced, self._shape))
                if other.requires_grad:
                    local = [
                        grad_other(upstream[k], left[left_index[k]], right[right_index[k]])
                        for k in range(total)
                    ]
                    reduced = sum_to_shape(local, out_shape, other._shape)
                    other._accumulate(Tensor._from_flat(reduced, other._shape))

            out._backward = _backward
        return out

    def __add__(self, other: Tensor | float | int) -> Tensor:
        return self._binary(other, lambda a, b: a + b, lambda g, a, b: g, lambda g, a, b: g, "add")

    def __radd__(self, other: Tensor | float | int) -> Tensor:
        return as_tensor(other).__add__(self)

    def __sub__(self, other: Tensor | float | int) -> Tensor:
        return self._binary(other, lambda a, b: a - b, lambda g, a, b: g, lambda g, a, b: -g, "sub")

    def __rsub__(self, other: Tensor | float | int) -> Tensor:
        return as_tensor(other).__sub__(self)

    def __mul__(self, other: Tensor | float | int) -> Tensor:
        return self._binary(
            other, lambda a, b: a * b, lambda g, a, b: g * b, lambda g, a, b: g * a, "mul"
        )

    def __rmul__(self, other: Tensor | float | int) -> Tensor:
        return as_tensor(other).__mul__(self)

    def __truediv__(self, other: Tensor | float | int) -> Tensor:
        return self._binary(
            other,
            lambda a, b: a / b,
            lambda g, a, b: g / b,
            lambda g, a, b: -g * a / (b * b),
            "div",
        )

    def __rtruediv__(self, other: Tensor | float | int) -> Tensor:
        return as_tensor(other).__truediv__(self)

    def __neg__(self) -> Tensor:
        return self * (-1.0)

    def __pow__(self, exponent: float) -> Tensor:
        return self.pow(exponent)

    # -- matrix multiplication --------------------------------------------
    def matmul(self, other: Tensor | Sequence) -> Tensor:
        other = as_tensor(other)
        if self.ndim == 0 or other.ndim == 0:
            raise ValueError("matmul requires at least 1-dimensional tensors")

        a = self
        b = other
        left_vector = a.ndim == 1
        right_vector = b.ndim == 1
        if left_vector:
            a = a.unsqueeze(0)
        if right_vector:
            b = b.unsqueeze(-1)

        out = _matmul_nd(a, b)
        if left_vector and right_vector:
            return out.reshape(())
        if left_vector:
            return out.squeeze(0)
        if right_vector:
            return out.squeeze(-1)
        return out

    def __matmul__(self, other: Tensor | Sequence) -> Tensor:
        return self.matmul(other)

    def __rmatmul__(self, other: Sequence) -> Tensor:
        return as_tensor(other).matmul(self)

    def dot(self, other: Tensor | Sequence) -> Tensor:
        return (self * other).sum()

    # -- convenience -------------------------------------------------------
    def numel(self) -> int:
        return self.size

    def contiguous(self) -> Tensor:
        if is_contiguous(self._shape, self._strides) and self._offset == 0:
            return self
        return Tensor._from_flat(
            self._values(), self._shape, self.requires_grad, op="contiguous", prev=(self,)
        )

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError("Cannot iterate over a 0-d tensor")
        return (self[index] for index in range(self._shape[0]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _nest(values: Sequence[float], shape: Shape) -> Any:
    if not shape:
        return values[0]
    if len(shape) == 1:
        return list(values)
    step = numel(shape[1:])
    return [_nest(values[i * step : (i + 1) * step], shape[1:]) for i in range(shape[0])]


def as_tensor(value: Tensor | float | int | Sequence) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return Tensor(value)


def tensor(data: Any, requires_grad: bool = False, shape: Sequence[int] | None = None) -> Tensor:
    return Tensor(data, shape=shape, requires_grad=requires_grad)


def zeros(shape: int | Sequence[int], requires_grad: bool = False) -> Tensor:
    if isinstance(shape, int):
        shape = (shape,)
    return Tensor._from_flat([0.0] * numel(shape), shape, requires_grad, op="zeros")


def ones(shape: int | Sequence[int], requires_grad: bool = False) -> Tensor:
    if isinstance(shape, int):
        shape = (shape,)
    return Tensor._from_flat([1.0] * numel(shape), shape, requires_grad, op="ones")


def full(shape: int | Sequence[int], value: float, requires_grad: bool = False) -> Tensor:
    if isinstance(shape, int):
        shape = (shape,)
    return Tensor._from_flat([float(value)] * numel(shape), shape, requires_grad, op="full")


def randn(
    shape: int | Sequence[int],
    requires_grad: bool = False,
    generator: Any | None = None,
    mean: float = 0.0,
    std: float = 1.0,
) -> Tensor:
    if isinstance(shape, int):
        shape = (shape,)
    rng = generator or default_rng()
    values = [rng.normal(mean, std) for _ in range(numel(shape))]
    return Tensor._from_flat(values, shape, requires_grad, op="randn")


def rand(
    shape: int | Sequence[int],
    requires_grad: bool = False,
    generator: Any | None = None,
) -> Tensor:
    if isinstance(shape, int):
        shape = (shape,)
    rng = generator or default_rng()
    values = [rng.uniform(0.0, 1.0) for _ in range(numel(shape))]
    return Tensor._from_flat(values, shape, requires_grad, op="rand")


def uniform(
    shape: int | Sequence[int],
    low: float = 0.0,
    high: float = 1.0,
    requires_grad: bool = False,
    generator: Any | None = None,
) -> Tensor:
    if isinstance(shape, int):
        shape = (shape,)
    rng = generator or default_rng()
    values = [rng.uniform(low, high) for _ in range(numel(shape))]
    return Tensor._from_flat(values, shape, requires_grad, op="uniform")


def eye(size: int) -> Tensor:
    values = [0.0] * (size * size)
    for i in range(size):
        values[i * size + i] = 1.0
    return Tensor._from_flat(values, (size, size), op="eye")


def arange(start: float, stop: float | None = None, step: float = 1.0) -> Tensor:
    if stop is None:
        start, stop = 0.0, start
    if step == 0:
        raise ValueError("arange step must be non-zero")
    values: list[float] = []
    value = float(start)
    if step > 0:
        while value < stop:
            values.append(value)
            value += step
    else:
        while value > stop:
            values.append(value)
            value += step
    return Tensor._from_flat(values, (len(values),), op="arange")


# ---------------------------------------------------------------------------
# Batched matrix multiplication
# ---------------------------------------------------------------------------
def _matmul_nd(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication for tensors with at least 2 dimensions."""
    if a.ndim < 2 or b.ndim < 2:
        raise ValueError("_matmul_nd requires tensors with at least 2 dimensions")
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(
            f"Incompatible matmul shapes {a.shape} and {b.shape}: "
            f"inner dimensions {a.shape[-1]} and {b.shape[-2]} differ"
        )

    batch_shape = broadcast_shape(a.shape[:-2], b.shape[:-2])
    m, k = a.shape[-2:]
    _, n = b.shape[-2:]
    out_shape = (*batch_shape, m, n)
    batch_total = numel(batch_shape)

    a_values = a._values()
    b_values = b._values()
    out_values = [0.0] * (batch_total * m * n)

    for batch in range(batch_total):
        a_batch = broadcast_flat_index(batch, batch_shape, a.shape[:-2])
        b_batch = broadcast_flat_index(batch, batch_shape, b.shape[:-2])
        a_base = a_batch * m * k
        b_base = b_batch * k * n
        out_base = batch * m * n
        for i in range(m):
            a_row = a_base + i * k
            out_row = out_base + i * n
            for j in range(n):
                total = 0.0
                for t in range(k):
                    total += a_values[a_row + t] * b_values[b_base + t * n + j]
                out_values[out_row + j] = total

    requires = _GRAD_ENABLED and (a.requires_grad or b.requires_grad)
    out = Tensor._from_flat(out_values, out_shape, requires, op="matmul", prev=(a, b))
    if requires:

        def _backward() -> None:
            upstream = out.grad
            if a.requires_grad:
                grad_a = _matmul_nd(upstream, b.transpose(-1, -2))
                reduced = sum_to_shape(grad_a._values(), grad_a.shape, a.shape)
                a._accumulate(Tensor._from_flat(reduced, a.shape))
            if b.requires_grad:
                grad_b = _matmul_nd(a.transpose(-1, -2), upstream)
                reduced = sum_to_shape(grad_b._values(), grad_b.shape, b.shape)
                b._accumulate(Tensor._from_flat(reduced, b.shape))

        out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# Embedding lookup and target gathering
# ---------------------------------------------------------------------------
def _indices_list(indices: Any) -> list[int]:
    if isinstance(indices, Tensor):
        return [round(v) for v in indices._values()]
    if isinstance(indices, (list, tuple)):
        flat: list[int] = []

        def walk(value: Any) -> None:
            if isinstance(value, (list, tuple)):
                for item in value:
                    walk(item)
            else:
                flat.append(round(float(value)))

        walk(indices)
        return flat
    return [round(float(indices))]


def embedding_lookup(weight: Tensor, indices: Any) -> Tensor:
    """Gather rows of ``weight`` (vocab, dim) for integer ``indices``.

    The output shape is ``indices.shape + (dim,)``.  Gradients scatter-add
    back into the embedding table, so repeated indices accumulate correctly.
    """
    if weight.ndim != 2:
        raise ValueError("Embedding weight must be 2-dimensional")
    vocab, dim = weight.shape
    index_values = _indices_list(indices)
    if isinstance(indices, Tensor):
        out_shape = (*indices.shape, dim)
    else:
        out_shape = (*_nested_shape(indices), dim) if isinstance(indices, (list, tuple)) else (dim,)

    weight_values = weight._values()
    out_values = [0.0] * (len(index_values) * dim)
    for position, index in enumerate(index_values):
        if not 0 <= index < vocab:
            raise IndexError(f"Embedding index {index} out of range for vocabulary size {vocab}")
        base = index * dim
        out_values[position * dim : (position + 1) * dim] = weight_values[base : base + dim]

    requires = _GRAD_ENABLED and weight.requires_grad
    out = Tensor._from_flat(out_values, out_shape, requires, op="embedding", prev=(weight,))
    if requires:

        def _backward() -> None:
            upstream = out.grad._values()
            buffer = [0.0] * (vocab * dim)
            for position, index in enumerate(index_values):
                base = index * dim
                for d in range(dim):
                    buffer[base + d] += upstream[position * dim + d]
            weight._accumulate(Tensor._from_flat(buffer, weight.shape))

        out._backward = _backward
    return out


def cat_last(a: Tensor, b: Tensor) -> Tensor:
    """Concatenate two tensors along their last dimension."""
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError("cat_last requires tensors with at least one dimension")
    if a.shape[:-1] != b.shape[:-1]:
        raise ValueError(f"cat_last requires matching leading shapes, got {a.shape} and {b.shape}")
    left = a.shape[-1]
    right = b.shape[-1]
    out_shape = (*a.shape[:-1], left + right)
    a_values = a._values()
    b_values = b._values()
    rows = numel(a.shape[:-1])
    out_values = [0.0] * (rows * (left + right))
    for row in range(rows):
        base = row * (left + right)
        out_values[base : base + left] = a_values[row * left : (row + 1) * left]
        out_values[base + left : base + left + right] = b_values[row * right : (row + 1) * right]

    requires = _GRAD_ENABLED and (a.requires_grad or b.requires_grad)
    out = Tensor._from_flat(out_values, out_shape, requires, op="cat", prev=(a, b))
    if requires:

        def _backward() -> None:
            upstream = out.grad._values()
            if a.requires_grad:
                grad_a = [0.0] * (rows * left)
                for row in range(rows):
                    source = row * (left + right)
                    grad_a[row * left : (row + 1) * left] = upstream[source : source + left]
                a._accumulate(Tensor._from_flat(grad_a, a.shape))
            if b.requires_grad:
                grad_b = [0.0] * (rows * right)
                for row in range(rows):
                    source = row * (left + right) + left
                    grad_b[row * right : (row + 1) * right] = upstream[source : source + right]
                b._accumulate(Tensor._from_flat(grad_b, b.shape))

        out._backward = _backward
    return out


def gather_last(values: Tensor, targets: Any) -> Tensor:
    """Gather ``values[..., target]`` for integer targets."""
    if values.ndim < 1:
        raise ValueError("gather_last requires at least 1 dimension")
    vocab = values.shape[-1]
    prefix_shape = values.shape[:-1]
    target_values = _indices_list(targets)
    if len(target_values) != numel(prefix_shape):
        raise ValueError(
            f"Expected {numel(prefix_shape)} targets for shape {prefix_shape}, got {len(target_values)}"
        )

    flat = values._values()
    out_values = [0.0] * len(target_values)
    for position, target in enumerate(target_values):
        if not 0 <= target < vocab:
            raise IndexError(f"Target {target} out of range for vocabulary size {vocab}")
        out_values[position] = flat[position * vocab + target]

    requires = _GRAD_ENABLED and values.requires_grad
    out = Tensor._from_flat(out_values, prefix_shape, requires, op="gather", prev=(values,))
    if requires:

        def _backward() -> None:
            upstream = out.grad._values()
            buffer = [0.0] * values.size
            for position, target in enumerate(target_values):
                buffer[position * vocab + target] += upstream[position]
            values._accumulate(Tensor._from_flat(buffer, values.shape))

        out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def mse_loss(prediction: Tensor, target: Tensor) -> Tensor:
    difference = prediction - target
    return (difference * difference).mean()


def binary_cross_entropy_with_logits(logits: Tensor, targets: Tensor) -> Tensor:
    """Stable BCE-with-logits written as ``y*softplus(-z) + (1-y)*softplus(z)``."""
    targets = as_tensor(targets)
    per_element = targets * (-logits).softplus() + (1.0 - targets) * logits.softplus()
    return per_element.mean()


def cross_entropy(logits: Tensor, targets: Any) -> Tensor:
    """Mean categorical cross-entropy using a stable log-sum-exp."""
    maximum = logits.max(dim=-1, keepdim=True).detach()
    shifted = logits - maximum
    log_normaliser = shifted.exp().sum(dim=-1, keepdim=True).log() + maximum
    log_probabilities = logits - log_normaliser
    return -gather_last(log_probabilities, targets).mean()


__all__ = [
    "Tensor",
    "arange",
    "as_tensor",
    "binary_cross_entropy_with_logits",
    "cat_last",
    "cross_entropy",
    "embedding_lookup",
    "eye",
    "full",
    "gather_last",
    "is_grad_enabled",
    "mse_loss",
    "no_grad",
    "ones",
    "rand",
    "randn",
    "tensor",
    "uniform",
    "zeros",
]
