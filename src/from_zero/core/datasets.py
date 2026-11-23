"""Dataset generation, splitting, and leakage-safe preprocessing.

The classification helpers are deliberately small and dependency-free.  The
preprocessing contract is the important part: fit the scaler on the training
split only, then apply the frozen parameters to validation and test data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    """A stratified train/validation/test split of tabular classification data."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("X_train", "X_val", "X_test"):
            array = getattr(self, name)
            if array.ndim != 2:
                raise ValueError(f"{name} must be 2-D, got shape {array.shape}")
        for name in ("y_train", "y_val", "y_test"):
            array = getattr(self, name)
            if array.ndim != 1:
                raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
        n_features = self.X_train.shape[1]
        if self.X_val.shape[1] != n_features or self.X_test.shape[1] != n_features:
            raise ValueError("All feature matrices must have the same number of columns")
        if len(self.X_train) != len(self.y_train):
            raise ValueError("X_train and y_train length mismatch")
        if len(self.X_val) != len(self.y_val):
            raise ValueError("X_val and y_val length mismatch")
        if len(self.X_test) != len(self.y_test):
            raise ValueError("X_test and y_test length mismatch")


@dataclass
class StandardScaler:
    """Feature-wise standardization fitted on training data only."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> StandardScaler:
        array = _as_2d_float(X, "X")
        self.mean_ = array.mean(axis=0)
        scale = array.std(axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before transform")
        array = _as_2d_float(X, "X")
        if array.shape[1] != len(self.mean_):
            raise ValueError(
                f"X has {array.shape[1]} features, scaler was fitted with {len(self.mean_)}"
            )
        return (array - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before inverse_transform")
        array = _as_2d_float(X, "X")
        return array * self.scale_ + self.mean_

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "scale": None if self.scale_ is None else self.scale_.tolist(),
        }


@dataclass(frozen=True)
class PreparedClassification:
    """A scaled split plus the frozen preprocessing and generation metadata."""

    split: DatasetSplit
    scaler: StandardScaler
    dataset_name: str
    metadata: dict[str, Any]

    def as_metadata(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "dataset": self.dataset_name,
            "preprocessing": {
                "kind": "standardize",
                "fit_on": "train",
                **self.scaler.to_dict(),
            },
        }


def make_classification_dataset(
    name: str = "linearly_separable",
    n_samples: int = 600,
    noise: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one of the small two-dimensional teaching datasets.

    Available datasets: ``linearly_separable``, ``moons``, ``circles``, ``xor``.
    """

    if n_samples < 8:
        raise ValueError("n_samples must be at least 8")
    if noise < 0:
        raise ValueError("noise must be non-negative")
    rng = np.random.default_rng(seed)
    name = name.lower()

    if name == "linearly_separable":
        n0 = n_samples // 2
        n1 = n_samples - n0
        X0 = rng.normal(loc=(-1.2, -0.6), scale=noise, size=(n0, 2))
        X1 = rng.normal(loc=(1.2, 0.6), scale=noise, size=(n1, 2))
        X = np.vstack([X0, X1])
        y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
        order = rng.permutation(len(y))
        return X[order].astype(np.float64), y[order]

    if name == "moons":
        n0 = n_samples // 2
        n1 = n_samples - n0
        t0 = np.linspace(0.0, np.pi, n0)
        t1 = np.linspace(0.0, np.pi, n1)
        X0 = np.column_stack([np.cos(t0), np.sin(t0)])
        X1 = np.column_stack([1.0 - np.cos(t1), 0.5 - np.sin(t1)])
        X0 += rng.normal(scale=noise, size=X0.shape)
        X1 += rng.normal(scale=noise, size=X1.shape)
        X = np.vstack([X0, X1])
        y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
        order = rng.permutation(len(y))
        return X[order].astype(np.float64), y[order]

    if name == "circles":
        n0 = n_samples // 2
        n1 = n_samples - n0
        t0 = rng.uniform(0, 2 * np.pi, n0)
        t1 = rng.uniform(0, 2 * np.pi, n1)
        X0 = np.column_stack([np.cos(t0), np.sin(t0)])
        X1 = 0.45 * np.column_stack([np.cos(t1), np.sin(t1)])
        X0 += rng.normal(scale=noise, size=X0.shape)
        X1 += rng.normal(scale=noise, size=X1.shape)
        X = np.vstack([X0, X1])
        y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
        order = rng.permutation(len(y))
        return X[order].astype(np.float64), y[order]

    if name == "xor":
        centers = np.array([[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]])
        labels = np.array([0, 0, 1, 1])
        counts = _balanced_counts(n_samples, 4)
        features: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for center, label, count in zip(centers, labels, counts, strict=True):
            features.append(rng.normal(loc=center, scale=max(noise, 0.05), size=(count, 2)))
            targets.append(np.full(count, label, dtype=int))
        X = np.vstack(features)
        y = np.concatenate(targets)
        order = rng.permutation(len(y))
        return X[order].astype(np.float64), y[order]

    raise ValueError(f"Unknown dataset {name!r}; choose linearly_separable, moons, circles, or xor")


def split_classification(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_fraction: float = 0.2,
    val_fraction: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
) -> DatasetSplit:
    """Split features and labels into train/validation/test sets.

    Splitting happens *before* any preprocessing.  Stratification keeps the
    class balance stable in small educational datasets.
    """

    X = _as_2d_float(X, "X")
    y = _as_1d_labels(y, len(X))
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1)")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if test_fraction + val_fraction >= 1.0:
        raise ValueError("test_fraction + val_fraction must be < 1")
    if len(X) < 10:
        raise ValueError("At least 10 samples are required for a three-way split")

    rng = np.random.default_rng(seed)
    if stratify:
        train_idx: list[int] = []
        val_idx: list[int] = []
        test_idx: list[int] = []
        for label in np.unique(y):
            indices = np.flatnonzero(y == label)
            rng.shuffle(indices)
            n = len(indices)
            n_test = max(1, round(n * test_fraction)) if test_fraction > 0 else 0
            n_val = max(1, round(n * val_fraction)) if val_fraction > 0 else 0
            if n - n_test - n_val < 1:
                raise ValueError(
                    "Not enough samples per class for the requested split; "
                    "reduce fractions or increase n_samples"
                )
            test_idx.extend(indices[:n_test].tolist())
            val_idx.extend(indices[n_test : n_test + n_val].tolist())
            train_idx.extend(indices[n_test + n_val :].tolist())
    else:
        permutation = rng.permutation(len(X))
        n_test = max(1, round(len(X) * test_fraction)) if test_fraction > 0 else 0
        n_val = max(1, round(len(X) * val_fraction)) if val_fraction > 0 else 0
        test_idx = permutation[:n_test].tolist()
        val_idx = permutation[n_test : n_test + n_val].tolist()
        train_idx = permutation[n_test + n_val :].tolist()

    if not train_idx:
        raise ValueError("The training split is empty")

    def take(indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        ordered = np.asarray(indices, dtype=int)
        return X[ordered], y[ordered]

    X_train, y_train = take(train_idx)
    X_val, y_val = take(val_idx)
    X_test, y_test = take(test_idx)
    return DatasetSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=("x1", "x2"),
        metadata={
            "n_total": len(X),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "stratified": bool(stratify),
            "split_seed": seed,
        },
    )


def prepare_classification(
    name: str = "linearly_separable",
    n_samples: int = 600,
    noise: float = 0.15,
    test_fraction: float = 0.2,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> PreparedClassification:
    """Generate, split, and scale a classification dataset without leakage."""

    X, y = make_classification_dataset(name, n_samples=n_samples, noise=noise, seed=seed)
    split = split_classification(
        X,
        y,
        test_fraction=test_fraction,
        val_fraction=val_fraction,
        seed=seed + 1,
    )
    scaler = StandardScaler().fit(split.X_train)
    scaled = DatasetSplit(
        X_train=scaler.transform(split.X_train),
        y_train=split.y_train,
        X_val=scaler.transform(split.X_val),
        y_val=split.y_val,
        X_test=scaler.transform(split.X_test),
        y_test=split.y_test,
        feature_names=split.feature_names,
        metadata=split.metadata,
    )
    metadata = {
        "generator": name,
        "n_samples": int(n_samples),
        "noise": float(noise),
        "class_balance": {
            "negative": int(np.sum(y == 0)),
            "positive": int(np.sum(y == 1)),
        },
        "splits": split.metadata,
    }
    return PreparedClassification(
        split=scaled,
        scaler=scaler,
        dataset_name=name,
        metadata=metadata,
    )


def _balanced_counts(total: int, groups: int) -> list[int]:
    base = total // groups
    counts = [base] * groups
    for index in range(total % groups):
        counts[index] += 1
    return counts


def _as_2d_float(X: Any, name: str) -> np.ndarray:
    array = np.asarray(X, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _as_1d_labels(y: Any, n: int) -> np.ndarray:
    array = np.asarray(y)
    if array.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {array.shape}")
    if len(array) != n:
        raise ValueError(f"X and y length mismatch: {n} vs {len(array)}")
    if not np.all(np.isin(array, [0, 1])):
        raise ValueError("y must contain only binary labels 0 and 1")
    return array.astype(int)
