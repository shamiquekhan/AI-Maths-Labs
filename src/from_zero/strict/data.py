"""Datasets, batching, splits, and standardisation without third-party libraries."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from typing import Any

from ..tensor import Tensor


class Dataset:
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError


class TensorDataset(Dataset):
    """In-memory dataset of feature vectors and scalar targets."""

    def __init__(self, features: Sequence[Sequence[float]], targets: Sequence[float]) -> None:
        if len(features) != len(targets):
            raise ValueError("features and targets must have the same length")
        self.features = [list(map(float, row)) for row in features]
        self.targets = [float(value) for value in targets]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[list[float], float]:
        return self.features[index], self.targets[index]


class SequenceDataset(Dataset):
    """Character/sequence dataset producing shifted next-token pairs."""

    def __init__(self, token_ids: Sequence[int], block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.token_ids = [int(token) for token in token_ids]
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.token_ids) - self.block_size)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        chunk = self.token_ids[index : index + self.block_size + 1]
        return chunk[:-1], chunk[1:]


class DataLoader:
    """Deterministic mini-batch iteration over a dataset."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

    def __len__(self) -> int:
        total = len(self.dataset)
        if self.drop_last:
            return total // self.batch_size
        return math.ceil(total / self.batch_size) if total else 0

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed).shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start : start + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                continue
            features = []
            targets = []
            for index in batch:
                feature, target = self.dataset[index]
                features.append(feature)
                targets.append([target])
            yield Tensor(features), Tensor(targets)


def split_indices(
    n_samples: int,
    train_fraction: float = 0.6,
    valid_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[int], list[int], list[int]]:
    """Reproducible train/validation/test index split."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 <= valid_fraction < 1.0:
        raise ValueError("valid_fraction must be in [0, 1)")
    if train_fraction + valid_fraction >= 1.0:
        raise ValueError("train_fraction + valid_fraction must be < 1")

    indices = list(range(n_samples))
    random.Random(seed).shuffle(indices)
    train_end = int(n_samples * train_fraction)
    valid_end = train_end + int(n_samples * valid_fraction)
    return indices[:train_end], indices[train_end:valid_end], indices[valid_end:]


def subset(features: Sequence, targets: Sequence, indices: Sequence[int]) -> TensorDataset:
    return TensorDataset([features[i] for i in indices], [targets[i] for i in indices])


def standardize(
    train_features: Sequence[Sequence[float]],
    *other_sets: Sequence[Sequence[float]],
) -> tuple[list[float], list[float], list[list[list[float]]]]:
    """Fit mean/std on training data only, then transform every set."""
    if not train_features:
        raise ValueError("Training features are empty")
    n_features = len(train_features[0])
    if any(len(row) != n_features for row in train_features):
        raise ValueError("Inconsistent feature dimensions")

    means = [0.0] * n_features
    for row in train_features:
        for j, value in enumerate(row):
            means[j] += value
    means = [value / len(train_features) for value in means]

    variances = [0.0] * n_features
    for row in train_features:
        for j, value in enumerate(row):
            difference = value - means[j]
            variances[j] += difference * difference
    stds = [math.sqrt(value / len(train_features)) or 1.0 for value in variances]

    def transform(rows: Sequence[Sequence[float]]) -> list[list[float]]:
        return [[(value - means[j]) / stds[j] for j, value in enumerate(row)] for row in rows]

    transformed = [transform(train_features)]
    transformed.extend(transform(rows) for rows in other_sets)
    return means, stds, transformed


def shuffle_together(
    features: Sequence[Sequence[float]], targets: Sequence[float], seed: int = 0
) -> tuple[list[list[float]], list[float]]:
    indices = list(range(len(features)))
    random.Random(seed).shuffle(indices)
    return [list(features[i]) for i in indices], [float(targets[i]) for i in indices]


__all__ = [
    "DataLoader",
    "Dataset",
    "SequenceDataset",
    "TensorDataset",
    "shuffle_together",
    "split_indices",
    "standardize",
    "subset",
]
