"""Neural-network layers built on the custom tensor engine."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..tensor import Tensor, embedding_lookup, ones, zeros
from ..tensor.storage import RNG, default_rng
from .initializers import normal, xavier_uniform
from .initializers import zeros as init_zeros
from .module import Module, Parameter


class Linear(Module):
    """Affine transform ``y = x @ W^T + b``.

    ``W`` has shape ``(out_features, in_features)`` and ``b`` has shape
    ``(out_features,)``.  Inputs may have any leading dimensions.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        generator: RNG | None = None,
        initializer: str = "xavier_uniform",
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Linear dimensions must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.weight = Parameter(zeros((out_features, in_features)), name="weight")
        if initializer == "xavier_uniform":
            xavier_uniform(self.weight, generator=generator)
        elif initializer == "zeros":
            init_zeros(self.weight)
        else:
            from .initializers import INITIALIZERS

            if initializer not in INITIALIZERS:
                raise ValueError(f"Unknown initializer {initializer!r}")
            INITIALIZERS[initializer](self.weight, generator=generator)

        if bias:
            self.bias: Parameter | None = Parameter(zeros((out_features,)), name="bias")
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Linear expected last dimension {self.in_features}, got {x.shape}")
        output = x.matmul(self.weight.transpose(-1, -2))
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}"


class Embedding(Module):
    """Lookup table mapping integer token ids to dense vectors."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        generator: RNG | None = None,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("Embedding dimensions must be positive")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.weight = Parameter(zeros((num_embeddings, embedding_dim)), name="weight")
        normal(self.weight, mean=0.0, std=1.0, generator=generator)
        if padding_idx is not None:
            if not 0 <= padding_idx < num_embeddings:
                raise ValueError("padding_idx out of range")
            base = padding_idx * embedding_dim
            self.weight._data[base : base + embedding_dim] = [0.0] * embedding_dim

    def forward(self, indices: Tensor | Sequence[int]) -> Tensor:
        return embedding_lookup(self.weight, indices)

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"


class LayerNorm(Module):
    """Layer normalisation over the last dimension."""

    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        normalized_shape = tuple(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("LayerNorm currently supports a single normalised dimension")
        if normalized_shape[0] <= 0:
            raise ValueError("normalized_shape must be positive")
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight: Parameter | None = Parameter(ones(normalized_shape[0]), name="weight")
            self.bias: Parameter | None = Parameter(zeros(normalized_shape[0]), name="bias")
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.normalized_shape[0]:
            raise ValueError(
                f"LayerNorm expected last dimension {self.normalized_shape[0]}, got {x.shape}"
            )
        mean = x.mean(dim=-1, keepdim=True)
        centred = x - mean
        variance = (centred * centred).mean(dim=-1, keepdim=True)
        normalised = centred / (variance + self.eps).sqrt()
        if self.weight is not None and self.bias is not None:
            return normalised * self.weight + self.bias
        return normalised

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}"


class Dropout(Module):
    """Inverted dropout with an explicit random generator."""

    def __init__(self, p: float = 0.5, generator: RNG | None = None) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("Dropout probability must be in [0, 1)")
        self.p = p
        self.rng = generator or default_rng()

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        scale = 1.0 / (1.0 - self.p)
        values = [scale if self.rng.uniform(0.0, 1.0) >= self.p else 0.0 for _ in range(x.size)]
        mask = Tensor._from_flat(values, x.shape, False, op="dropout_mask")
        return x * mask

    def extra_repr(self) -> str:
        return f"p={self.p}"


class Flatten(Module):
    """Flatten a contiguous range of dimensions."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim, self.end_dim)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Sequential(Module):
    """Apply modules in order."""

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        if len(modules) == 1 and isinstance(modules[0], (list, tuple)):
            modules = tuple(modules[0])  # type: ignore[assignment]
        for index, module in enumerate(modules):
            if not isinstance(module, Module):
                raise TypeError("Sequential expects Module instances")
            self._modules[str(index)] = module

    def forward(self, x: Any) -> Any:
        for module in self._modules.values():
            x = module(x)
        return x

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, index: int) -> Module:
        return list(self._modules.values())[index]

    def append(self, module: Module) -> Sequential:
        self._modules[str(len(self._modules))] = module
        return self

    def extend(self, modules: Sequence[Module]) -> Sequential:
        for module in modules:
            self.append(module)
        return self


__all__ = [
    "Dropout",
    "Embedding",
    "Flatten",
    "Identity",
    "LayerNorm",
    "Linear",
    "Sequential",
]
