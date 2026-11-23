"""Unit tests for the NumPy optimizer library."""

from __future__ import annotations

import numpy as np
import pytest

from from_zero.optim import (
    SGD,
    Adam,
    AdamW,
    ConstantSchedule,
    CosineDecay,
    ExponentialDecay,
    Momentum,
    RMSProp,
    StepDecay,
    make_objective,
    make_optimizer,
    run_optimizer,
)
from from_zero.optim.schedules import build_schedule


def test_sgd_step() -> None:
    parameter = np.array([1.0, -2.0])
    gradient = np.array([0.5, -0.25])
    optimizer = SGD(0.1)
    optimizer.step([parameter], [gradient])
    assert np.allclose(parameter, [0.95, -1.975])
    assert optimizer.state == {}


def test_momentum_first_and_second_step() -> None:
    parameter = np.array([0.0])
    optimizer = Momentum(0.1, beta=0.9)
    optimizer.step([parameter], [np.array([1.0])])
    assert parameter[0] == pytest.approx(-0.1)
    optimizer.step([parameter], [np.array([1.0])])
    # v = 0.9 * 1 + 1 = 1.9
    assert parameter[0] == pytest.approx(-0.29)
    assert np.allclose(optimizer.state["velocity"][0], [1.9])


def test_rmsprop_first_step() -> None:
    parameter = np.array([1.0])
    optimizer = RMSProp(0.1, rho=0.9, eps=1e-8)
    optimizer.step([parameter], [np.array([2.0])])
    expected_square = 0.1 * 4.0
    expected = 1.0 - 0.1 * 2.0 / (np.sqrt(expected_square) + 1e-8)
    assert parameter[0] == pytest.approx(expected)
    assert optimizer.state["square_avg"][0][0] == pytest.approx(expected_square)


def test_adam_first_step_is_sign_like() -> None:
    parameter = np.array([1.0, -1.0])
    optimizer = Adam(0.1)
    optimizer.step([parameter], [np.array([2.0, -4.0])])
    assert np.allclose(parameter, [0.9, -0.9], atol=1e-7)
    assert optimizer.state["t"] == 1
    optimizer.reset_state()
    assert optimizer.state == {}


def test_adamw_decouples_weight_decay() -> None:
    parameter = np.array([2.0])
    optimizer = AdamW(0.1, weight_decay=0.5)
    optimizer.step([parameter], [np.array([0.0])])
    # Decay first: 2 * (1 - 0.1 * 0.5) = 1.9; a zero gradient adds no Adam step.
    assert parameter[0] == pytest.approx(1.9)
    assert np.allclose(optimizer.state["m"][0], [0.0])


def test_optimizer_input_validation() -> None:
    optimizer = SGD(0.1)
    with pytest.raises(ValueError):
        optimizer.step([], [])
    with pytest.raises(ValueError):
        optimizer.step([np.zeros(2)], [np.zeros(3)])
    with pytest.raises(TypeError):
        optimizer.step([np.zeros(2, dtype=int)], [np.zeros(2, dtype=int)])
    with pytest.raises(FloatingPointError):
        optimizer.step([np.array([np.nan])], [np.array([0.0])])
    with pytest.raises(FloatingPointError):
        optimizer.step([np.array([0.0])], [np.array([np.inf])])
    read_only = np.zeros(1)
    read_only.flags.writeable = False
    with pytest.raises(ValueError):
        optimizer.step([read_only], [np.zeros(1)])


def test_make_optimizer_names_and_unknown() -> None:
    assert isinstance(make_optimizer("sgd", 0.1), SGD)
    assert isinstance(make_optimizer("momentum", 0.1), Momentum)
    assert isinstance(make_optimizer("rmsprop", 0.1), RMSProp)
    assert isinstance(make_optimizer("adam", 0.1), Adam)
    assert isinstance(make_optimizer("adamw", 0.1), AdamW)
    with pytest.raises(ValueError):
        make_optimizer("nope", 0.1)


def test_schedules_return_expected_values() -> None:
    assert ConstantSchedule(0.1)(0) == 0.1
    step = StepDecay(0.1, step_size=2, gamma=0.5)
    assert [step(index) for index in range(5)] == pytest.approx([0.1, 0.1, 0.05, 0.05, 0.025])
    exponential = ExponentialDecay(0.1, gamma=0.5)
    assert exponential(3) == pytest.approx(0.0125)
    cosine = CosineDecay(0.1, total_steps=10, min_learning_rate=0.0)
    assert cosine(0) == pytest.approx(0.1)
    assert cosine(10) == pytest.approx(0.0)
    assert cosine(20) == pytest.approx(0.0)
    with pytest.raises(TypeError):
        cosine(1.5)
    with pytest.raises(ValueError):
        cosine(-1)


def test_build_schedule_from_mapping() -> None:
    assert isinstance(build_schedule({"kind": "constant", "learning_rate": 0.1}), ConstantSchedule)
    assert isinstance(build_schedule({"kind": "step", "learning_rate": 0.1}), StepDecay)
    assert isinstance(
        build_schedule({"kind": "exponential", "learning_rate": 0.1}), ExponentialDecay
    )
    assert isinstance(
        build_schedule({"kind": "cosine", "learning_rate": 0.1, "total_steps": 5}), CosineDecay
    )
    with pytest.raises(ValueError):
        build_schedule({"kind": "unknown"})


def test_run_optimizer_decreases_quadratic_loss() -> None:
    objective = make_objective("quadratic", curvature=(1.0, 10.0))
    optimizer = SGD(0.1)
    result = run_optimizer(objective, optimizer, steps=100)
    assert result.steps >= 1
    assert result.final_loss < result.losses[0]
    assert result.trajectory.shape[0] == len(result.losses)
    assert np.all(np.isfinite(result.losses))
    assert result.metadata["learning_rate"] == 0.1


def test_run_optimizer_stops_on_gradient_tolerance() -> None:
    objective = make_objective("quadratic", curvature=(1.0, 1.0))
    result = run_optimizer(objective, SGD(0.5), steps=500, grad_tol=1e-6)
    assert result.steps < 500
    assert result.final_loss < 1e-8


def test_objective_evaluate_validation() -> None:
    objective = make_objective("rosenbrock")
    with pytest.raises(ValueError):
        objective.evaluate(np.zeros(3))
    with pytest.raises(ValueError):
        objective.evaluate(np.array([np.nan, 1.0]))
    with pytest.raises(ValueError):
        make_objective("unknown")
