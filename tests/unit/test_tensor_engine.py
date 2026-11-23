"""Unit tests for the strict tensor engine and its autograd graph."""

from __future__ import annotations

import math

import pytest

from from_zero.tensor import (
    Tensor,
    arange,
    binary_cross_entropy_with_logits,
    cat_last,
    cross_entropy,
    embedding_lookup,
    eye,
    full,
    gather_last,
    is_grad_enabled,
    mse_loss,
    no_grad,
    ones,
    tensor,
    zeros,
)


def test_construction_and_metadata() -> None:
    x = tensor([[1, 2, 3], [4, 5, 6]])
    assert x.shape == (2, 3)
    assert x.strides == (3, 1)
    assert x.ndim == 2
    assert x.size == 6
    assert x.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert x[1, 2].item() == 6.0
    assert len(x) == 2
    assert x.data == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_shape_inference_and_errors() -> None:
    assert tensor(3.5).shape == ()
    assert tensor([1, 2]).reshape(2, 1).shape == (2, 1)
    assert tensor([1, 2, 3, 4]).reshape(-1, 2).shape == (2, 2)
    with pytest.raises(ValueError):
        tensor([1, 2, 3]).reshape(2, 2)
    with pytest.raises(ValueError):
        tensor([[1, 2], [3]])
    with pytest.raises(ValueError):
        tensor([1, 2, 3]).reshape(-1, -1)
    with pytest.raises(ValueError):
        _ = tensor([1.0, 2.0]).item()


def test_factories() -> None:
    assert zeros((2, 2)).tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert ones(3).tolist() == [1.0, 1.0, 1.0]
    assert full((2,), 7.0).tolist() == [7.0, 7.0]
    assert eye(2).tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert arange(5).tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert arange(1.0, 2.0, 0.5).tolist() == [1.0, 1.5]


def test_reshape_transpose_permute_and_flatten() -> None:
    x = tensor([[1, 2, 3], [4, 5, 6]])
    assert x.reshape(3, 2).tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert x.T.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    cube = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert cube.permute(1, 2, 0).shape == (2, 2, 2)
    assert cube.flatten().tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert tensor([1, 2]).unsqueeze(0).shape == (1, 2)
    assert tensor([[[1]]]).squeeze().shape == ()
    assert tensor([1, 2]).squeeze(0).shape == (2,)


def test_getitem_and_setitem() -> None:
    x = tensor([[1, 2, 3], [4, 5, 6]])
    assert x[0].tolist() == [1.0, 2.0, 3.0]
    assert x[1, 1:].tolist() == [5.0, 6.0]
    assert x[:, ::2].tolist() == [[1.0, 3.0], [4.0, 6.0]]
    assert x[-1, -1].item() == 6.0
    x[0, 0] = 9.0
    assert x[0, 0].item() == 9.0
    with pytest.raises(IndexError):
        _ = x[2, 0]
    with pytest.raises(TypeError):
        _ = x[{"bad": 1}]


def test_broadcasting_arithmetic() -> None:
    x = tensor([[1, 2, 3], [4, 5, 6]])
    y = tensor([10, 20, 30])
    assert (x + y).tolist() == [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]]
    assert (x * 2.0).tolist() == [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]
    assert (2.0 - x).tolist() == [[1.0, 0.0, -1.0], [-2.0, -3.0, -4.0]]
    assert (x / 2.0)[0].tolist() == [0.5, 1.0, 1.5]
    assert (-x)[0].tolist() == [-1.0, -2.0, -3.0]


def test_reductions_dim_and_keepdim() -> None:
    x = tensor([[1, 2, 3], [4, 5, 6]])
    assert x.sum().item() == 21.0
    assert x.sum(dim=0).tolist() == [5.0, 7.0, 9.0]
    assert x.sum(dim=1).tolist() == [6.0, 15.0]
    assert x.sum(dim=1, keepdim=True).tolist() == [[6.0], [15.0]]
    assert x.mean().item() == pytest.approx(3.5)
    assert x.mean(dim=0).tolist() == [2.5, 3.5, 4.5]
    assert x.max().item() == 6.0
    assert x.max(dim=0).tolist() == [4.0, 5.0, 6.0]
    assert x.max(dim=1, keepdim=True).tolist() == [[3.0], [6.0]]
    assert x.argmax() == 5
    assert x.argmax(dim=1) == [2, 2]


def test_matmul_variants() -> None:
    a = tensor([[1, 2], [3, 4]])
    b = tensor([[5, 6], [7, 8]])
    assert (a @ b).tolist() == [[19.0, 22.0], [43.0, 50.0]]
    assert a.matmul(tensor([1, 1])).tolist() == [3.0, 7.0]
    assert tensor([1, 2]).matmul(a).tolist() == [7.0, 10.0]
    assert tensor([1, 2]).matmul(tensor([3, 4])).item() == 11.0

    batched = Tensor._from_flat(list(range(1, 13)), (2, 2, 3))
    weights = Tensor._from_flat(list(range(1, 7)), (2, 3, 1))
    result = batched @ weights
    assert result.shape == (2, 2, 1)
    assert result[0, 0, 0].item() == 14.0
    assert result[1, 1, 0].item() == 167.0

    with pytest.raises(ValueError):
        _ = a @ tensor([[1, 2, 3]])
    with pytest.raises(ValueError):
        _ = tensor(1.0) @ tensor(2.0)


def test_softmax_and_log_softmax() -> None:
    x = tensor([1.0, 2.0, 3.0])
    probabilities = x.softmax()
    assert probabilities.sum().item() == pytest.approx(1.0)
    assert probabilities[2].item() > probabilities[0].item()
    log_probabilities = x.log_softmax()
    for index in range(3):
        assert log_probabilities[index].item() == pytest.approx(
            math.log(probabilities[index].item())
        )
    matrix = tensor([[1.0, 2.0], [3.0, 4.0]])
    assert matrix.softmax(dim=1).sum(dim=1).tolist() == pytest.approx([1.0, 1.0])
    shifted = tensor([1000.0, 1001.0, 1002.0])
    assert shifted.softmax().sum().item() == pytest.approx(1.0)


def test_elementwise_math() -> None:
    x = tensor([0.0, 1.0, 4.0])
    assert x.exp().tolist()[0] == 1.0
    assert x.log1p().tolist()[0] == 0.0
    assert tensor([4.0]).sqrt().item() == 2.0
    assert tensor([-2.0, 2.0]).abs().tolist() == [2.0, 2.0]
    assert tensor([0.0]).tanh().item() == 0.0
    assert tensor([0.0]).sigmoid().item() == 0.5
    assert tensor([-1.0, 2.0]).relu().tolist() == [0.0, 2.0]
    assert tensor([-5.0, 0.5, 5.0]).clamp(0.0, 1.0).tolist() == [0.0, 0.5, 1.0]
    assert tensor([3.0]).pow(2).item() == 9.0


def test_backward_accumulates_shared_nodes() -> None:
    x = tensor([1.0, 2.0], requires_grad=True)
    y = x * x + x
    y.sum().backward()
    assert x.grad.tolist() == [3.0, 5.0]


def test_backward_requires_gradient_for_non_scalar() -> None:
    x = tensor([1.0, 2.0], requires_grad=True)
    y = x * 2.0
    with pytest.raises(RuntimeError):
        y.backward()
    y.backward(tensor([1.0, 0.0]))
    assert x.grad.tolist() == [2.0, 0.0]


def test_gradient_shapes_with_broadcasting() -> None:
    x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    b = tensor([1.0, 1.0, 1.0], requires_grad=True)
    (x + b).sum().backward()
    assert x.grad.shape == (2, 3)
    assert b.grad.tolist() == [2.0, 2.0, 2.0]

    row = tensor([[1.0], [2.0]], requires_grad=True)
    column = tensor([[1.0, 1.0, 1.0]], requires_grad=True)
    (row * column).sum().backward()
    assert row.grad.tolist() == [[3.0], [3.0]]
    assert column.grad.tolist() == [[3.0, 3.0, 3.0]]


def test_slice_gradient_scatter_adds() -> None:
    x = tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x[1:] * 2.0
    y.sum().backward()
    assert x.grad.tolist() == [0.0, 2.0, 2.0]

    z = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    (z[:, 0] * z[:, 0]).sum().backward()
    assert z.grad.tolist() == [[2.0, 0.0], [6.0, 0.0]]


def test_reshape_transpose_gradients() -> None:
    x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    y = x.reshape(3, 2).T
    weights = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    (y * weights).sum().backward()
    assert x.grad.tolist() == [[1.0, 4.0, 2.0], [5.0, 3.0, 6.0]]


def test_embedding_lookup_gradients_accumulate() -> None:
    weight = tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], requires_grad=True)
    indices = [0, 2, 0]
    output = embedding_lookup(weight, indices)
    assert output.shape == (3, 2)
    assert output[0].tolist() == [0.1, 0.2]
    output.sum().backward()
    assert weight.grad.tolist() == [[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]]


def test_embedding_validation() -> None:
    weight = tensor([[0.1, 0.2]])
    with pytest.raises(IndexError):
        embedding_lookup(weight, [1])
    with pytest.raises(ValueError):
        embedding_lookup(tensor([1.0, 2.0]), [0])


def test_gather_last_and_cross_entropy() -> None:
    values = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    gathered = gather_last(values, [2, 0])
    assert gathered.tolist() == [3.0, 4.0]
    gathered.sum().backward()
    assert values.grad.tolist() == [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]

    logits = tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)
    loss = cross_entropy(logits, [0, 1])
    assert loss.item() == pytest.approx(math.log(2.0))
    loss.backward()
    assert logits.grad.tolist()[0] == pytest.approx([-0.25, 0.25])
    assert logits.grad.tolist()[1] == pytest.approx([0.25, -0.25])

    with pytest.raises(IndexError):
        cross_entropy(tensor([[1.0, 2.0]]), [2])


def test_losses() -> None:
    prediction = tensor([1.0, 2.0])
    target = tensor([1.0, 4.0])
    assert mse_loss(prediction, target).item() == pytest.approx(2.0)

    logits = tensor([0.0, 0.0])
    assert binary_cross_entropy_with_logits(logits, tensor([1.0, 0.0])).item() == pytest.approx(
        math.log(2.0)
    )
    assert binary_cross_entropy_with_logits(tensor([10.0]), tensor([1.0])).item() == pytest.approx(
        0.0, abs=1e-4
    )


def test_cat_last_values_and_gradients() -> None:
    a = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = tensor([[5.0], [6.0]], requires_grad=True)
    joined = cat_last(a, b)
    assert joined.tolist() == [[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]
    (joined * tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).sum().backward()
    assert a.grad.tolist() == [[1.0, 2.0], [4.0, 5.0]]
    assert b.grad.tolist() == [[3.0], [6.0]]
    with pytest.raises(ValueError):
        cat_last(tensor([[1.0]]), tensor([[1.0], [2.0]]))


def test_no_grad_and_detach() -> None:
    x = tensor([1.0, 2.0], requires_grad=True)
    assert is_grad_enabled()
    with no_grad():
        assert not is_grad_enabled()
        y = x * 2.0
        assert not y.requires_grad
        assert y._backward is None
    assert is_grad_enabled()
    detached = (x * 3.0).detach()
    assert not detached.requires_grad
    assert detached.tolist() == [3.0, 6.0]
    assert (x + 1.0).requires_grad


def test_zero_grad_and_clone() -> None:
    x = tensor([1.0], requires_grad=True)
    (x * 2.0).sum().backward()
    assert x.grad is not None
    x.zero_grad()
    assert x.grad is None
    copy = x.clone()
    copy[0] = 5.0
    assert x[0].item() == 1.0
    assert copy[0].item() == 5.0


def test_repr_and_iteration() -> None:
    assert "Tensor" in repr(tensor(1.0))
    assert "shape=(2, 2)" in repr(tensor([[1, 2], [3, 4]]))
    rows = [row.tolist() for row in tensor([[1, 2], [3, 4]])]
    assert rows == [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(TypeError):
        iter(tensor(1.0))


def test_mutation_is_visible_through_views() -> None:
    x = tensor([1.0, 2.0, 3.0])
    view = x.reshape(3, 1)
    x[1] = 9.0
    assert view[1, 0].item() == 9.0
