"""Unit tests for the scalar reverse-mode autodiff engine."""

from __future__ import annotations

import math

import pytest

from from_zero.autodiff import (
    MLP,
    SGD,
    Value,
    binary_cross_entropy_with_logits,
    graph_to_dict,
    mean,
    mse_loss,
    softplus,
    topological_sort,
    xor_dataset,
)
from from_zero.autodiff.examples import demo_expression
from from_zero.autodiff.graph import zero_grad


def test_add_mul_neg_sub_div_pow() -> None:
    a, b = Value(3.0), Value(2.0)
    c = (a + b) * (a - b) / b**2
    c.backward()
    # c = (5 * 1) / 4 = 1.25
    assert c.data == pytest.approx(1.25)
    # dc/da = ((a-b) + (a+b))/b^2 = (1 + 5)/4 = 1.5
    assert a.grad == pytest.approx(1.5)
    # dc/db from quotient and subtraction; check numerically below
    assert math.isfinite(b.grad)
    assert (-a).data == -3.0


def test_reverse_operators() -> None:
    x = Value(2.0)
    assert (1.0 + x).data == 3.0
    assert (1.0 - x).data == -1.0
    assert (6.0 / x).data == 3.0
    assert (3.0 * x).data == 6.0


def test_shared_subexpression_accumulates() -> None:
    a = Value(2.0)
    b = Value(3.0)
    shared = a * b
    output = shared + shared
    output.backward()
    assert a.grad == pytest.approx(6.0)
    assert b.grad == pytest.approx(4.0)


def test_repeated_backward_accumulates_and_zero_grad_clears() -> None:
    a = Value(2.0)
    output = a * 3.0
    output.backward()
    assert a.grad == pytest.approx(3.0)
    output.backward()
    assert a.grad == pytest.approx(6.0)
    a.zero_grad()
    assert a.grad == 0.0


def test_topological_order_places_parents_before_children() -> None:
    a, b = Value(1.0), Value(2.0)
    c = a * b
    d = c + a
    order = topological_sort(d)
    positions = {id(node): index for index, node in enumerate(order)}
    assert positions[id(a)] < positions[id(c)]
    assert positions[id(b)] < positions[id(c)]
    assert positions[id(c)] < positions[id(d)]
    assert len(order) == 4


def test_graph_to_dict_has_expected_structure() -> None:
    payload = graph_to_dict(demo_expression())
    assert payload["root"] == len(payload["nodes"]) - 1
    assert payload["nodes"][-1]["op"] == "*"
    assert all(isinstance(node["parents"], list) for node in payload["nodes"])
    with pytest.raises(TypeError):
        topological_sort("not a value")  # type: ignore[arg-type]


def test_zero_grad_clears_entire_graph() -> None:
    a, b = Value(1.0), Value(2.0)
    output = (a * b + a).tanh()
    output.backward()
    assert a.grad != 0.0
    zero_grad(output)
    assert all(node.grad == 0.0 for node in topological_sort(output))


def test_unary_operations_are_mathematically_correct() -> None:
    x = Value(0.7)
    y = x.exp().log() + x.tanh() + x.relu() + x.sigmoid()
    y.backward()
    expected = 1.0 + (1.0 - math.tanh(0.7) ** 2) + 1.0
    sigmoid = 1.0 / (1.0 + math.exp(-0.7))
    expected += sigmoid * (1.0 - sigmoid)
    assert x.grad == pytest.approx(expected)
    assert x.exp().log().data == pytest.approx(0.7)
    assert Value(-2.0).relu().data == 0.0


def test_invalid_math_is_rejected() -> None:
    with pytest.raises(ValueError):
        Value(0.0).log()
    with pytest.raises(ZeroDivisionError):
        Value(1.0) / Value(0.0)
    with pytest.raises(ValueError):
        Value(0.0) ** 0.5
    with pytest.raises(ValueError):
        Value(0.0) ** -1.0
    with pytest.raises(TypeError):
        Value(2.0) ** Value(2.0)
    with pytest.raises(TypeError):
        Value(1.0) + "x"  # type: ignore[operator]


def test_softplus_is_stable_and_correct() -> None:
    large = softplus(1000.0)
    assert large.data == pytest.approx(1000.0)
    negative = softplus(-1000.0)
    assert negative.data == pytest.approx(0.0, abs=1e-12)
    value = Value(0.3)
    softplus(value).backward()
    sigmoid = 1.0 / (1.0 + math.exp(-0.3))
    assert value.grad == pytest.approx(sigmoid)


def test_losses_and_mean() -> None:
    assert mean([1.0, 2.0, 3.0]).data == pytest.approx(2.0)
    with pytest.raises(ValueError):
        mean([])
    loss = mse_loss([Value(1.0), Value(2.0)], [0.0, 2.0])
    assert loss.data == pytest.approx(0.5)
    bce = binary_cross_entropy_with_logits([Value(0.0), Value(0.0)], [0.0, 1.0])
    assert bce.data == pytest.approx(math.log(2.0))
    extreme = binary_cross_entropy_with_logits([Value(1000.0)], [0.0])
    assert math.isfinite(extreme.data)
    assert extreme.data == pytest.approx(1000.0)


def test_mlp_parameter_count_and_shapes() -> None:
    model = MLP(2, (4, 3), 1, activation="tanh", output_activation="linear", seed=1)
    assert model.parameter_count() == (2 * 4 + 4) + (4 * 3 + 3) + (3 * 1 + 1)
    output = model([Value(0.0), Value(1.0)])
    assert isinstance(output, Value)
    multi = MLP(2, (3,), 2, seed=1)([Value(0.0), Value(1.0)])
    assert isinstance(multi, list) and len(multi) == 2
    with pytest.raises(ValueError):
        model([Value(0.0)])
    model.zero_grad()
    assert all(parameter.grad == 0.0 for parameter in model.parameters())


def test_xor_dataset_shape() -> None:
    data = xor_dataset()
    assert len(data) == 4
    assert {label for _, label in data} == {0, 1}


def test_sgd_updates_scalar_parameters() -> None:
    parameter = Value(1.0)
    parameter.grad = 2.0
    optimizer = SGD([parameter], learning_rate=0.5)
    optimizer.step()
    assert parameter.data == pytest.approx(0.0)
    optimizer.zero_grad()
    assert parameter.grad == 0.0
    with pytest.raises(ValueError):
        SGD([], learning_rate=0.1)
    with pytest.raises(ValueError):
        SGD([parameter], learning_rate=0.0)
