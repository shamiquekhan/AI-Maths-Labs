"""Composable operations and losses built on :class:`Value`."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .value import Value


def mean(values: Iterable[Value | float]) -> Value:
    items = [_as_value(value) for value in values]
    if not items:
        raise ValueError("mean requires at least one value")
    total = Value(0.0)
    for item in items:
        total = total + item
    return total / float(len(items))


def softplus(value: Value | float) -> Value:
    """Stable ``log(1 + exp(z))`` with the correct derivative."""

    z = _as_value(value)
    if z.data >= 0.0:
        return z + (1.0 + (-z).exp()).log()
    return (1.0 + z.exp()).log()


def mse_loss(
    predictions: Sequence[Value | float] | Value,
    targets: Sequence[float] | float,
) -> Value:
    predictions_list = _as_sequence(predictions)
    targets_list = _as_sequence(targets)
    if len(predictions_list) != len(targets_list):
        raise ValueError("predictions and targets must have the same length")
    squared = [
        (prediction - target) ** 2
        for prediction, target in zip(predictions_list, targets_list, strict=True)
    ]
    return mean(squared)


def binary_cross_entropy_with_logits(
    logits: Sequence[Value | float] | Value,
    targets: Sequence[float] | float,
) -> Value:
    """Mean binary cross-entropy from logits, computed with softplus."""

    logits_list = _as_sequence(logits)
    targets_list = _as_sequence(targets)
    if len(logits_list) != len(targets_list):
        raise ValueError("logits and targets must have the same length")
    losses = [
        softplus(logit) - logit * target
        for logit, target in zip(logits_list, targets_list, strict=True)
    ]
    return mean(losses)


def _as_value(value: Value | float) -> Value:
    if isinstance(value, Value):
        return value
    if isinstance(value, int | float):
        return Value(value)
    raise TypeError(f"Expected Value or number, got {type(value).__name__}")


def _as_sequence(value: Sequence[Value | float] | Value | float) -> list[Value]:
    if isinstance(value, Value):
        return [value]
    if isinstance(value, int | float):
        return [Value(value)]
    if isinstance(value, Sequence):
        return [_as_value(item) for item in value]
    raise TypeError(f"Expected a Value, number, or sequence, got {type(value).__name__}")
