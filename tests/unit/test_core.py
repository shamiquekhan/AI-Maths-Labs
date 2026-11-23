"""Tests for shared core utilities."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from from_zero.core.datasets import StandardScaler, prepare_classification, split_classification
from from_zero.core.metrics import (
    binary_classification_metrics,
    confusion_matrix_binary,
    perplexity,
)
from from_zero.core.seeds import set_seed
from from_zero.core.validation import (
    ConfigError,
    check_bool,
    check_choice,
    check_float,
    check_int,
    deep_merge,
    load_config,
    reject_unknown_keys,
    save_config,
)


def test_set_seed_repeats_python_numpy_and_torch() -> None:
    set_seed(123)
    first = (random.random(), float(np.random.rand()), None)
    try:
        import torch

        first = (first[0], first[1], float(torch.rand(1).item()))
    except ImportError:
        pass
    set_seed(123)
    second = (random.random(), float(np.random.rand()), None)
    try:
        import torch

        second = (second[0], second[1], float(torch.rand(1).item()))
    except ImportError:
        pass
    assert first == second


def test_validation_helpers_accept_and_reject() -> None:
    assert check_int("n", 3, minimum=1, maximum=5) == 3
    assert check_float("x", 0.5, minimum=0.0, maximum=1.0) == 0.5
    assert check_choice("mode", "fast", ("fast", "slow")) == "fast"
    assert check_bool("flag", True) is True
    with pytest.raises(ConfigError):
        check_int("n", True)
    with pytest.raises(ConfigError):
        check_int("n", 0, minimum=1)
    with pytest.raises(ConfigError):
        check_float("x", float("nan"), minimum=0.0)
    with pytest.raises(ConfigError):
        check_choice("mode", "medium", ("fast", "slow"))
    with pytest.raises(ConfigError):
        check_bool("flag", 1)
    with pytest.raises(ConfigError):
        reject_unknown_keys({"a": 1, "b": 2}, ("a",), "section")


def test_deep_merge_is_recursive_and_non_mutating() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    merged = deep_merge(base, {"a": {"c": 9}, "e": 4})
    assert merged == {"a": {"b": 1, "c": 9}, "d": 3, "e": 4}
    assert base == {"a": {"b": 1, "c": 2}, "d": 3}


def test_yaml_config_round_trip(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    save_config({"name": "demo", "values": [1, 2], "nested": {"x": 0.5}}, path)
    assert load_config(path) == {"name": "demo", "values": [1, 2], "nested": {"x": 0.5}}
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_standard_scaler_uses_fitted_statistics() -> None:
    scaler = StandardScaler().fit(np.array([[1.0, 2.0], [3.0, 6.0]]))
    transformed = scaler.transform(np.array([[1.0, 2.0], [3.0, 6.0]]))
    assert np.allclose(transformed.mean(axis=0), 0.0)
    assert np.allclose(transformed.std(axis=0), 1.0)
    with pytest.raises(ValueError):
        scaler.transform(np.array([[1.0, 2.0, 3.0]]))


def test_prepare_classification_shapes_and_metadata() -> None:
    prepared = prepare_classification(
        name="moons",
        n_samples=120,
        noise=0.1,
        test_fraction=0.2,
        val_fraction=0.2,
        seed=7,
    )
    split = prepared.split
    assert split.X_train.shape[0] + split.X_val.shape[0] + split.X_test.shape[0] == 120
    assert split.X_train.shape[1] == 2
    assert set(np.unique(split.y_train)).issubset({0, 1})
    metadata = prepared.as_metadata()
    assert metadata["splits"]["n_train"] == len(split.y_train)
    assert metadata["generator"] == "moons"
    assert np.allclose(split.X_train.mean(axis=0), 0.0, atol=1e-8)


def test_split_classification_rejects_bad_fractions() -> None:
    X = np.zeros((10, 2))
    y = np.zeros(10, dtype=int)
    with pytest.raises(ValueError):
        split_classification(X, y, test_fraction=0.6, val_fraction=0.5)


def test_confusion_matrix_and_metrics() -> None:
    matrix = confusion_matrix_binary([0, 0, 1, 1], [0, 1, 1, 1])
    assert matrix.tolist() == [[1, 1], [0, 2]]
    metrics = binary_classification_metrics([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9])
    assert metrics["accuracy"] == 0.75
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["loss"] <= 5.0
    assert perplexity(0.0) == pytest.approx(1.0)
    assert math.isclose(perplexity(math.log(4.0)), 4.0, rel_tol=1e-12)
