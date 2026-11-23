"""Finite-difference gradient checks for the strict tensor engine."""

from __future__ import annotations

import math
import random

import pytest

from from_zero.autograd_tensor import gradcheck
from from_zero.tensor import (
    Tensor,
    binary_cross_entropy_with_logits,
    cat_last,
    cross_entropy,
    embedding_lookup,
    gather_last,
    mse_loss,
    tensor,
)


def _random_tensor(shape, seed: int = 0, low: float = -1.0, high: float = 1.0) -> Tensor:
    rng = random.Random(seed)
    values = [rng.uniform(low, high) for _ in range(math.prod(shape))]
    return Tensor._from_flat(values, shape, requires_grad=True)


@pytest.mark.parametrize(
    "build",
    [
        lambda x: (x * x + x).sum(),
        lambda x: (x / (x * x + 1.5)).sum(),
        lambda x: (x.exp() + (x.abs() + 1.0).log()).sum(),
        lambda x: (x.tanh() + x.sigmoid()).sum(),
        lambda x: x.softplus().sum(),
        lambda x: (x.relu() + 0.1 * x).sum(),
        lambda x: x.sum() + x.mean(),
        lambda x: x.max(),
        lambda x: (x * x).mean(dim=0).sum(),
        lambda x: x.softmax().pow(2).sum(),
        lambda x: (x.log_softmax() * tensor([1.0, 2.0, 3.0])).sum(),
    ],
)
def test_elementwise_and_reduction_gradients(build) -> None:
    x = _random_tensor((3,))
    assert gradcheck(lambda: build(x), [x])


def test_matrix_multiplication_gradients() -> None:
    a = _random_tensor((2, 3), seed=1)
    b = _random_tensor((3, 2), seed=2)
    assert gradcheck(lambda: (a @ b).sum(), [a, b])


def test_batched_matmul_gradients() -> None:
    a = _random_tensor((2, 2, 3), seed=3)
    b = _random_tensor((2, 3, 2), seed=4)
    assert gradcheck(lambda: (a @ b).sum(), [a, b])


def test_broadcast_gradients() -> None:
    x = _random_tensor((2, 3), seed=5)
    bias = _random_tensor((3,), seed=6)
    assert gradcheck(lambda: ((x + bias) ** 2).sum(), [x, bias])

    row = _random_tensor((2, 1), seed=7)
    column = _random_tensor((1, 3), seed=8)
    assert gradcheck(lambda: (row * column).sum(), [row, column])


def test_shape_operation_gradients() -> None:
    x = _random_tensor((2, 3), seed=9)
    weights = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert gradcheck(lambda: (x.reshape(3, 2).T * weights).sum(), [x])

    cube = _random_tensor((2, 2, 2), seed=10)
    assert gradcheck(lambda: cube.permute(2, 0, 1).sum(), [cube])

    z = _random_tensor((2, 3), seed=11)
    assert gradcheck(lambda: (z[:, 1:] * 2.0).sum(), [z])


def test_softmax_and_cross_entropy_gradients() -> None:
    logits = _random_tensor((4, 3), seed=12)
    assert gradcheck(lambda: cross_entropy(logits, [0, 2, 1, 0]), [logits])
    assert gradcheck(lambda: cross_entropy(logits.softmax(), [1, 1, 0, 2]), [logits])

    scores = _random_tensor((2, 3), seed=13)
    assert gradcheck(lambda: (scores.softmax(dim=1) * scores).sum(), [scores])
    assert gradcheck(lambda: (scores.log_softmax(dim=1) * scores).sum(), [scores])


def test_bce_and_mse_gradients() -> None:
    logits = _random_tensor((4,), seed=14)
    targets = tensor([1.0, 0.0, 1.0, 0.0])
    assert gradcheck(lambda: binary_cross_entropy_with_logits(logits, targets), [logits])

    zero_logits = Tensor._from_flat([0.0, 0.0, 0.0], (3,), requires_grad=True)
    assert gradcheck(
        lambda: binary_cross_entropy_with_logits(zero_logits, tensor([1.0, 0.0, 1.0])),
        [zero_logits],
    )

    predictions = _random_tensor((3,), seed=15)
    assert gradcheck(lambda: mse_loss(predictions, tensor([0.2, -0.4, 0.7])), [predictions])


def test_embedding_and_gather_gradients() -> None:
    weight = _random_tensor((4, 2), seed=16)
    assert gradcheck(lambda: embedding_lookup(weight, [0, 2, 0, 3]).sum(), [weight])

    values = _random_tensor((3, 4), seed=17)
    assert gradcheck(lambda: gather_last(values, [0, 3, 1]).sum(), [values])


def test_cat_gradients() -> None:
    a = _random_tensor((2, 2), seed=18)
    b = _random_tensor((2, 1), seed=19)
    assert gradcheck(lambda: (cat_last(a, b) ** 2).sum(), [a, b])


def test_layer_norm_composition_gradients() -> None:
    x = _random_tensor((3, 4), seed=20)
    scale = _random_tensor((4,), seed=21, low=0.5, high=1.5)
    shift = _random_tensor((4,), seed=22)

    def normalise() -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        centred = x - mean
        variance = (centred * centred).mean(dim=-1, keepdim=True)
        return (centred / (variance + 1e-5).sqrt() * scale + shift).sum()

    assert gradcheck(normalise, [x, scale, shift])


def test_gradcheck_detects_bad_gradients() -> None:
    x = _random_tensor((2,), seed=23)

    def wrong() -> Tensor:
        output = x.sum()
        output._backward = lambda: x._accumulate(Tensor._from_flat([0.0, 0.0], (2,)))
        return output

    with pytest.raises(AssertionError, match="Gradient check failed"):
        gradcheck(wrong, [x], raise_on_failure=True)
