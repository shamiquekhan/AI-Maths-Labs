"""Numerical gradient checks for the from-scratch components."""

from __future__ import annotations

import numpy as np
import pytest

from from_zero.autodiff import Value
from from_zero.logistic import check_gradients, finite_difference_gradient, loss_with_regularization
from from_zero.optim import make_objective


def _central_difference(function, point: np.ndarray, h: float = 1e-6) -> np.ndarray:
    values = np.asarray(point, dtype=np.float64).copy()
    gradient = np.zeros_like(values)
    for index in range(values.size):
        original = values.flat[index]
        values.flat[index] = original + h
        plus = float(function(values))
        values.flat[index] = original - h
        minus = float(function(values))
        values.flat[index] = original
        gradient.flat[index] = (plus - minus) / (2.0 * h)
    return gradient


def _autodiff_function(values: np.ndarray) -> Value:
    a, b, c = (Value(float(item)) for item in values)
    first = (a * b + c).tanh()
    second = (a**2 + 0.5 * b).sigmoid()
    third = (c.exp() + 1.0).log()
    return first + second + third + (a / (b + 2.0)) - 0.25 * c**2


def test_autodiff_matches_central_differences() -> None:
    point = np.array([0.7, 1.3, -0.4])
    # Read leaf gradients from the graph by rebuilding the leaves.
    a, b, c = (Value(float(item)) for item in point)
    first = (a * b + c).tanh()
    second = (a**2 + 0.5 * b).sigmoid()
    third = (c.exp() + 1.0).log()
    output = first + second + third + (a / (b + 2.0)) - 0.25 * c**2
    output.backward()
    analytic = np.array([a.grad, b.grad, c.grad])
    numeric = _central_difference(lambda values: _autodiff_function(values).data, point)
    assert np.allclose(analytic, numeric, atol=1e-6, rtol=1e-5)


def test_logistic_gradients_match_central_differences() -> None:
    rng = np.random.default_rng(21)
    for _ in range(5):
        X = rng.normal(size=(12, 3))
        y = (X[:, 0] - 0.5 * X[:, 1] + 0.3 * rng.normal(size=12) > 0).astype(int)
        weights = rng.normal(size=3)
        bias = 0.2
        result = check_gradients(X, y, weights, bias, regularization=0.03)
        assert result["passed"]
        numeric_w = finite_difference_gradient(
            lambda w: loss_with_regularization(X, y, w, bias, regularization=0.03), weights
        )
        assert np.allclose(result["analytic_weights"], numeric_w, atol=1e-6, rtol=1e-5)


def test_optimizer_objective_gradients_match_central_differences() -> None:
    rng = np.random.default_rng(9)
    for name in ("quadratic", "rosenbrock", "logistic"):
        objective = make_objective(name, seed=3)
        point = objective.initial + rng.normal(scale=0.1, size=objective.initial.shape)
        _, analytic = objective.evaluate(point)
        numeric = _central_difference(objective.function, point)
        assert np.allclose(analytic, numeric, atol=1e-5, rtol=1e-4), name


def test_quadratic_and_rosenbrock_have_known_minima() -> None:
    quadratic = make_objective("quadratic", curvature=(2.0, 5.0))
    assert quadratic.function(np.zeros(2)) == pytest.approx(0.0)
    rosenbrock = make_objective("rosenbrock")
    assert rosenbrock.function(np.array([1.0, 1.0])) == pytest.approx(0.0)
    assert np.allclose(rosenbrock.gradient(np.array([1.0, 1.0])), 0.0, atol=1e-12)
