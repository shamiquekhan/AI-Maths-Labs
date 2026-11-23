"""Loss modules using the custom tensor engine."""

from __future__ import annotations

from typing import Any

from ..tensor import (
    Tensor,
    binary_cross_entropy_with_logits,
    cross_entropy,
)
from .module import Module


def _validate_reduction(reduction: str) -> None:
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")


class MSELoss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        _validate_reduction(reduction)
        self.reduction = reduction

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        difference = prediction - target
        squared = difference * difference
        if self.reduction == "none":
            return squared
        if self.reduction == "sum":
            return squared.sum()
        return squared.mean()


class BCEWithLogitsLoss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        _validate_reduction(reduction)
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        loss = binary_cross_entropy_with_logits(logits, targets)
        if self.reduction == "mean":
            return loss
        if self.reduction == "sum":
            return loss * float(logits.size)
        targets_tensor = targets if isinstance(targets, Tensor) else Tensor(targets)
        return logits.relu() - logits * targets_tensor + (-logits.abs()).exp().log1p()


class CrossEntropyLoss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        _validate_reduction(reduction)
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Any) -> Tensor:
        loss = cross_entropy(logits, targets)
        if self.reduction == "mean":
            return loss
        if self.reduction == "sum":
            return loss * float(logits.size // logits.shape[-1])
        raise ValueError("CrossEntropyLoss supports 'mean' and 'sum' reductions")


__all__ = ["BCEWithLogitsLoss", "CrossEntropyLoss", "MSELoss"]
