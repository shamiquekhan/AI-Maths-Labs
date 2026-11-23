"""Unit tests for the NumPy logistic-regression implementation."""

from __future__ import annotations

import numpy as np
import pytest

from from_zero.logistic import (
    LogisticRegression,
    binary_cross_entropy,
    binary_cross_entropy_with_logits,
    check_gradients,
    compute_gradients,
    sigmoid,
)
from from_zero.logistic.losses import _validate_probabilities


def test_sigmoid_is_stable_at_extreme_logits() -> None:
    values = sigmoid(np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0]))
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert values[2] == 0.5
    assert np.all(np.isfinite(values))
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert sigmoid(0.0) == 0.5


def test_bce_from_logits_is_finite_for_extreme_logits() -> None:
    logits = np.array([-1000.0, 1000.0])
    loss = binary_cross_entropy_with_logits(np.array([0, 1]), logits)
    assert np.isfinite(loss)
    assert loss == pytest.approx(0.0, abs=1e-6)
    wrong = binary_cross_entropy_with_logits(np.array([1, 0]), logits)
    assert wrong == pytest.approx(1000.0, rel=1e-6)


def test_bce_from_probabilities_clips_and_validates() -> None:
    loss = binary_cross_entropy(np.array([0, 1]), np.array([0.0, 1.0]))
    assert np.isfinite(loss)
    with pytest.raises(ValueError):
        binary_cross_entropy(np.array([0, 1]), np.array([0.2, 1.2]))
    with pytest.raises(ValueError):
        _validate_probabilities(np.array([0.5, np.nan]))


def test_gradient_check_passes_with_regularization() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 4))
    y = (X[:, 0] - X[:, 1] + 0.2 * rng.normal(size=30) > 0).astype(int)
    weights = rng.normal(size=4) * 0.3
    result = check_gradients(X, y, weights, 0.1, regularization=0.01)
    assert result["passed"]
    assert result["max_abs_weight_error"] < 1e-6
    assert result["abs_bias_error"] < 1e-6


def test_compute_gradients_matches_closed_form() -> None:
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, -1.0]])
    y = np.array([0, 1, 1, 0])
    weights = np.array([0.2, -0.3])
    bias = 0.05
    gradient_w, gradient_b = compute_gradients(X, y, weights, bias, regularization=0.1)
    probabilities = 1.0 / (1.0 + np.exp(-(X @ weights + bias)))
    expected_w = X.T @ (probabilities - y) / len(y) + 0.1 * weights
    expected_b = np.mean(probabilities - y)
    assert np.allclose(gradient_w, expected_w)
    assert gradient_b == pytest.approx(expected_b)


def test_logistic_regression_learns_separable_data() -> None:
    rng = np.random.default_rng(11)
    X = np.vstack([rng.normal(loc=-2.0, size=(60, 2)), rng.normal(loc=2.0, size=(60, 2))])
    y = np.array([0] * 60 + [1] * 60)
    model = LogisticRegression(
        learning_rate=0.1,
        epochs=300,
        batch_size=None,
        early_stopping_patience=None,
        seed=11,
    ).fit(X, y)
    assert model.weights_ is not None
    assert model.score(X, y) > 0.98
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (120,)
    assert np.all((probabilities > 0.0) & (probabilities < 1.0))
    assert model.n_iter_ == 300


def test_logistic_regression_early_stopping_uses_validation_loss() -> None:
    rng = np.random.default_rng(5)
    X = rng.normal(size=(80, 2))
    y = (X[:, 0] + 0.1 * rng.normal(size=80) > 0).astype(int)
    model = LogisticRegression(
        learning_rate=0.2,
        epochs=1000,
        batch_size=None,
        early_stopping_patience=10,
        min_delta=1e-3,
        seed=5,
    ).fit(X, y, X_val=X, y_val=y)
    assert model.converged_
    assert model.n_iter_ < 1000
    assert 1 <= model.best_epoch_ <= model.n_iter_


def test_prediction_requires_fit_and_matching_width() -> None:
    model = LogisticRegression()
    with pytest.raises(RuntimeError):
        model.predict_proba(np.zeros((2, 2)))
    X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    y = np.array([0, 0, 1, 1])
    model.fit(X, y, X_val=X, y_val=y)
    with pytest.raises(ValueError):
        model.predict_proba(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        model.predict(X, threshold=1.5)
    assert set(model.predict(X)).issubset({0, 1})
    assert 0.0 <= model.loss(X, y) <= 5.0


def test_invalid_training_data_is_rejected() -> None:
    model = LogisticRegression()
    with pytest.raises(ValueError):
        model.fit(np.zeros((3, 2)), np.array([0, 0, 0]))
    with pytest.raises(ValueError):
        model.fit(np.zeros((3, 2)), np.array([0, 1]))
    with pytest.raises(ValueError):
        model.fit(np.array([[0.0, np.nan]]), np.array([0]))
