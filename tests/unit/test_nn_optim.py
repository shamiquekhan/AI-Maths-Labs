"""Unit tests for the strict neural-network and optimizer stacks."""

from __future__ import annotations

import json
import math

import pytest

from from_zero.nn_tensor import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    Module,
    MSELoss,
    Parameter,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
    calculate_gain,
    he_normal,
    xavier_uniform,
)
from from_zero.optim_tensor import (
    SGD,
    Adagrad,
    Adam,
    AdamW,
    CosineDecay,
    ExponentialDecay,
    Momentum,
    RMSprop,
    StepDecay,
    clip_grad_norm_,
    clip_grad_value_,
)
from from_zero.tensor import Tensor, no_grad, tensor
from from_zero.tensor.storage import RNG


def _parameter(values, shape=None) -> Parameter:
    data = Tensor(values) if shape is None else Tensor._from_flat(values, shape)
    return Parameter(data)


def test_parameter_is_a_tensor() -> None:
    parameter = _parameter([1.0, 2.0])
    assert isinstance(parameter, Tensor)
    assert parameter.requires_grad
    assert parameter.shape == (2,)
    assert "Parameter" in repr(parameter)


def test_linear_shapes_and_values() -> None:
    layer = Linear(2, 3)
    layer.weight._data[:] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    layer.bias._data[:] = [0.5, -0.5, 0.0]
    output = layer(tensor([[1.0, 2.0]]))
    assert output.shape == (1, 3)
    assert output.tolist() == [[5.5, 10.5, 17.0]]

    without_bias = Linear(2, 1, bias=False)
    without_bias.weight._data[:] = [1.0, 1.0]
    assert without_bias(tensor([[2.0, 3.0]])).item() == 5.0
    with pytest.raises(ValueError):
        layer(tensor([[1.0, 2.0, 3.0]]))


def test_linear_gradients() -> None:
    layer = Linear(2, 1)
    layer.weight._data[:] = [1.0, 2.0]
    layer.bias._data[:] = [0.0]
    features = tensor([[1.0, 2.0], [3.0, 4.0]])
    layer(features).sum().backward()
    assert layer.weight.grad.tolist() == [[4.0, 6.0]]
    assert layer.bias.grad.tolist() == [2.0]


def test_embedding_lookup() -> None:
    embedding = Embedding(3, 2, generator=RNG(0))
    output = embedding([0, 2])
    assert output.shape == (2, 2)
    assert output[0].tolist() == embedding.weight[0].tolist()
    padding = Embedding(3, 2, padding_idx=1, generator=RNG(0))
    assert padding.weight[1].tolist() == [0.0, 0.0]


def test_layer_norm_statistics() -> None:
    layer = LayerNorm(3)
    output = layer(tensor([[1.0, 2.0, 3.0]]))
    values = output[0].tolist()
    assert sum(values) / 3 == pytest.approx(0.0, abs=1e-6)
    variance = sum((value - sum(values) / 3) ** 2 for value in values) / 3
    assert variance == pytest.approx(1.0, rel=1e-3)
    with pytest.raises(ValueError):
        layer(tensor([[1.0, 2.0]]))


def test_dropout_modes() -> None:
    dropout = Dropout(0.5, generator=RNG(0))
    x = tensor([[1.0, 2.0, 3.0, 4.0]])
    with no_grad():
        training = dropout(x).tolist()[0]
        for value, original in zip(training, x.tolist()[0], strict=False):
            assert value in (0.0, 2.0 * original)
        dropout.eval()
        assert dropout(x).tolist() == x.tolist()


def test_sequential_registration_and_modes() -> None:
    model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
    names = [name for name, _ in model.named_parameters()]
    assert names == ["0.weight", "0.bias", "2.weight", "2.bias"]
    assert len(model.parameters()) == 4
    model.eval()
    assert all(not module.training for module in model.modules())
    model.train()
    assert all(module.training for module in model.modules())


def test_module_state_dict_roundtrip() -> None:
    model = Sequential(Linear(2, 2), Tanh(), Linear(2, 1))
    state = model.state_dict()
    values = {name: parameter.tolist() for name, parameter in state.items()}
    with no_grad():
        for parameter in model.parameters():
            parameter._data[:] = [0.0] * parameter.size
    model.load_state_dict(state)
    for name, parameter in model.named_parameters():
        assert parameter.tolist() == values[name]
    with pytest.raises(KeyError):
        model.load_state_dict({"0.weight": tensor([[1.0, 2.0]])})


def test_initializers() -> None:
    weight = Parameter(tensor([[0.0] * 10 for _ in range(20)]))
    xavier_uniform(weight, generator=RNG(0))
    bound = math.sqrt(6.0 / 30.0)
    assert all(abs(value) <= bound for value in weight._values())
    assert len(set(weight._values())) > 1

    he = Parameter(tensor([[0.0] * 400 for _ in range(400)]))
    he_normal(he, generator=RNG(1))
    values = he._values()
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    assert abs(mean) < 0.01
    assert std == pytest.approx(calculate_gain("relu") / 20.0, rel=0.1)


def test_loss_modules() -> None:
    mse = MSELoss()
    assert mse(tensor([1.0, 2.0]), tensor([1.0, 4.0])).item() == pytest.approx(2.0)
    assert MSELoss(reduction="sum")(tensor([1.0, 2.0]), tensor([1.0, 4.0])).item() == 4.0

    bce = BCEWithLogitsLoss()
    assert bce(tensor([0.0]), tensor([1.0])).item() == pytest.approx(math.log(2.0))
    assert bce(tensor([0.0, 0.0]), tensor([1.0, 0.0])).item() == pytest.approx(math.log(2.0))
    assert bce(tensor([0.0, 0.0]), tensor([1.0, 0.0])).item() != pytest.approx(
        BCEWithLogitsLoss(reduction="sum")(tensor([0.0, 0.0]), tensor([1.0, 0.0])).item()
    )

    ce = CrossEntropyLoss()
    assert ce(tensor([[0.0, 0.0]]), [1]).item() == pytest.approx(math.log(2.0))
    assert CrossEntropyLoss(reduction="sum")(tensor([[0.0, 0.0]]), [1]).item() == pytest.approx(
        math.log(2.0)
    )


def test_sgd_first_step_and_in_place_update() -> None:
    parameter = _parameter([1.0, 2.0])
    pointer = id(parameter)
    parameter.grad = tensor([0.5, -1.0])
    optimizer = SGD([parameter], lr=0.1)
    optimizer.step()
    assert id(parameter) == pointer
    assert parameter.tolist() == pytest.approx([0.95, 2.1])
    assert optimizer.step_count == 1


def test_momentum_two_steps() -> None:
    parameter = _parameter([1.0, 2.0])
    optimizer = Momentum([parameter], lr=0.1, momentum=0.9)
    parameter.grad = tensor([0.5, -1.0])
    optimizer.step()
    assert parameter.tolist() == pytest.approx([0.95, 2.1])
    parameter.grad = tensor([0.5, -1.0])
    optimizer.step()
    assert parameter.tolist() == pytest.approx([0.855, 2.29])


def test_adagrad_first_step() -> None:
    parameter = _parameter([1.0, 2.0])
    parameter.grad = tensor([0.5, -1.0])
    Adagrad([parameter], lr=0.1).step()
    assert parameter.tolist() == pytest.approx([0.9, 2.1])


def test_rmsprop_first_step() -> None:
    parameter = _parameter([1.0, 2.0])
    parameter.grad = tensor([0.5, -1.0])
    RMSprop([parameter], lr=0.01, alpha=0.9).step()
    expected = [1.0 - 0.01 * 0.5 / math.sqrt(0.025), 2.0 + 0.01 / math.sqrt(0.1)]
    assert parameter.tolist() == pytest.approx(expected)


def test_adam_first_step() -> None:
    parameter = _parameter([1.0, 2.0])
    parameter.grad = tensor([0.5, -1.0])
    optimizer = Adam([parameter], lr=0.1)
    optimizer.step()
    assert parameter.tolist() == pytest.approx([0.9, 2.1], abs=1e-6)
    assert optimizer.state_dict()["step_count"] == 1


def test_adamw_decoupled_weight_decay() -> None:
    parameter = _parameter([1.0, 2.0])
    parameter.grad = tensor([0.5, -1.0])
    AdamW([parameter], lr=0.1, weight_decay=0.01).step()
    assert parameter.tolist() == pytest.approx([0.899, 2.098], abs=1e-6)


def test_optimizer_state_is_json_safe() -> None:
    parameter = _parameter([1.0, 2.0])
    parameter.grad = tensor([0.5, -1.0])
    optimizer = Adam([parameter], lr=0.1)
    optimizer.step()
    payload = json.dumps(optimizer.state_dict())
    restored = Adam([_parameter([0.0, 0.0])], lr=0.1)
    restored.load_state_dict(json.loads(payload))
    assert restored.step_count == 1
    assert restored.state[0]["step"] == 1
    assert restored.state[0]["exp_avg"] == pytest.approx([0.05, -0.1])


def test_zero_grad_and_clipping() -> None:
    first = _parameter([0.0, 0.0])
    second = _parameter([0.0])
    first.grad = tensor([3.0, 4.0])
    second.grad = tensor([12.0])
    total = clip_grad_norm_([first, second], max_norm=1.0)
    assert total == pytest.approx(13.0)
    assert first.grad.tolist() == pytest.approx([3.0 / 13.0, 4.0 / 13.0])
    assert second.grad.tolist() == pytest.approx([12.0 / 13.0])

    clip_grad_value_([first], 0.1)
    assert first.grad.tolist() == pytest.approx([0.1, 0.1])

    optimizer = SGD([first, second], lr=0.1)
    optimizer.zero_grad()
    assert first.grad is None and second.grad is None


def test_schedule_values() -> None:
    optimizer = SGD([_parameter([0.0])], lr=1.0)
    schedule = StepDecay(optimizer, step_size=2, gamma=0.5)
    assert schedule.last_epoch == 0
    assert optimizer.lr == pytest.approx(1.0)
    schedule.step()
    assert optimizer.lr == pytest.approx(1.0)
    schedule.step()
    assert optimizer.lr == pytest.approx(0.5)

    optimizer = SGD([_parameter([0.0])], lr=1.0)
    exponential = ExponentialDecay(optimizer, gamma=0.1)
    exponential.step()
    exponential.step()
    assert optimizer.lr == pytest.approx(0.01)

    optimizer = SGD([_parameter([0.0])], lr=1.0)
    cosine = CosineDecay(optimizer, T_max=10, eta_min=0.1)
    assert optimizer.lr == pytest.approx(1.0)
    for _ in range(10):
        cosine.step()
    assert optimizer.lr == pytest.approx(0.1, abs=1e-6)


def test_gradients_flow_through_sequential_classifier() -> None:
    model = Sequential(Linear(2, 4, generator=RNG(0)), ReLU(), Linear(4, 1, generator=RNG(0)))
    x = tensor([[0.0, 0.0], [1.0, 1.0]])
    target = tensor([[0.0], [1.0]])
    loss = BCEWithLogitsLoss()(model(x), target)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(
        len(gradient._values()) == parameter.size
        for gradient, parameter in zip(gradients, model.parameters(), strict=False)
    )


def test_custom_module_composition() -> None:
    class Scale(Module):
        def __init__(self, factor: float) -> None:
            super().__init__()
            self.factor = factor

        def forward(self, x: Tensor) -> Tensor:
            return x * self.factor

    model = Sequential(Scale(2.0), Sigmoid())
    output = model(tensor([0.0]))
    assert output.item() == pytest.approx(0.5)
