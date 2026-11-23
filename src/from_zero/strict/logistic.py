"""Logistic regression built entirely on the strict tensor engine."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from ..nn_tensor import BCEWithLogitsLoss, Module, Parameter
from ..optim_tensor import SGD, Adam, Optimizer
from ..tensor import Tensor, zeros
from .data import TensorDataset, split_indices, standardize, subset
from .metrics import classification_report, threshold_predictions
from .training import History, Trainer


class LogisticRegression(Module):
    """Binary logistic regression: ``p(y=1|x) = sigmoid(x @ w + b)``."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        self.n_features = n_features
        self.weight = Parameter(zeros((n_features, 1)), name="weight")
        self.bias = Parameter(zeros((1,)), name="bias")

    def logits(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {features.shape[-1]}")
        return features.matmul(self.weight) + self.bias

    def forward(self, features: Tensor) -> Tensor:
        """Return logits; use :meth:`predict_proba` for probabilities."""
        return self.logits(features)

    def predict_proba(self, features: Tensor) -> Tensor:
        return self.logits(features).sigmoid()

    def extra_repr(self) -> str:
        return f"n_features={self.n_features}"


def analytical_gradients(
    features: Sequence[Sequence[float]],
    targets: Sequence[float],
    weight: Sequence[float],
    bias: float,
    l2: float = 0.0,
) -> tuple[list[float], float]:
    """Pedagogical analytical gradients for logistic regression.

    ``dL/dz = sigmoid(z) - y``; ``dL/dw = X^T (p - y) / N + l2 * w / N``;
    ``dL/db = mean(p - y)``.  The bias is never regularised.
    """
    n_samples = len(features)
    if n_samples == 0:
        raise ValueError("No training samples")
    n_features = len(features[0])
    weight_gradient = [0.0] * n_features
    bias_gradient = 0.0
    for row, target in zip(features, targets, strict=False):
        if len(row) != n_features:
            raise ValueError("Inconsistent feature dimensions")
        z = bias + sum(w * x for w, x in zip(weight, row, strict=False))
        probability = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
        delta = probability - float(target)
        for j in range(n_features):
            weight_gradient[j] += delta * row[j]
        bias_gradient += delta
    weight_gradient = [
        value / n_samples + l2 * weight[j] / n_samples for j, value in enumerate(weight_gradient)
    ]
    return weight_gradient, bias_gradient / n_samples


def make_classification(
    n_samples: int = 200,
    n_features: int = 2,
    separation: float = 3.0,
    noise: float = 0.5,
    seed: int = 0,
) -> tuple[list[list[float]], list[float]]:
    """Synthetic, roughly linearly separable two-class dataset."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if n_features < 1:
        raise ValueError("n_features must be positive")
    rng = random.Random(seed)
    features: list[list[float]] = []
    targets: list[float] = []
    for index in range(n_samples):
        label = index % 2
        centre = separation / 2.0 if label == 1 else -separation / 2.0
        row = [centre + rng.gauss(0.0, noise)]
        for _ in range(n_features - 1):
            row.append(rng.gauss(0.0, noise))
        features.append(row)
        targets.append(float(label))
    order = list(range(n_samples))
    rng.shuffle(order)
    return [features[i] for i in order], [targets[i] for i in order]


def fit_logistic_regression(
    features: Sequence[Sequence[float]],
    targets: Sequence[float],
    epochs: int = 200,
    learning_rate: float = 0.1,
    l2: float = 0.0,
    optimizer: str = "sgd",
    seed: int = 0,
) -> tuple[LogisticRegression, History]:
    """Train logistic regression with the custom engine."""
    dataset = TensorDataset(features, targets)
    model = LogisticRegression(len(features[0]))
    if optimizer == "adam":
        optimiser: Optimizer = Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        optimiser = SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("optimizer must be 'sgd' or 'adam'")

    loss_fn = BCEWithLogitsLoss()
    if l2:
        weight = model.weight

        class RegularisedBCE(Module):
            def forward(self, logits: Tensor, target: Tensor) -> Tensor:
                base = loss_fn(logits, target)
                penalty = (weight * weight).sum() * (l2 / (2.0 * len(dataset)))
                return base + penalty

        criterion = RegularisedBCE()
    else:
        criterion = loss_fn

    trainer = Trainer(
        model=model,
        optimizer=optimiser,
        loss_fn=criterion,
        batch_size=len(dataset),
        shuffle=False,
        seed=seed,
        probability_fn=lambda output: output.sigmoid(),
    )
    history = trainer.fit(dataset, epochs=epochs)
    return model, history


def evaluate_logistic(
    model: LogisticRegression,
    features: Sequence[Sequence[float]],
    targets: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, object]:
    dataset = TensorDataset(features, targets)
    trainer = Trainer(
        model=model,
        optimizer=SGD(model.parameters(), lr=0.0),
        loss_fn=BCEWithLogitsLoss(),
        probability_fn=lambda output: output.sigmoid(),
    )
    probabilities = trainer.predict_proba(dataset)
    predictions = threshold_predictions(probabilities, threshold)
    truth = [round(value) for value in targets]
    report = classification_report(truth, predictions)
    report["probabilities"] = probabilities
    return report


def train_valid_test_experiment(
    n_samples: int = 300,
    n_features: int = 2,
    separation: float = 3.0,
    noise: float = 0.6,
    epochs: int = 200,
    learning_rate: float = 0.1,
    l2: float = 0.0,
    seed: int = 0,
) -> dict[str, object]:
    """Reproducible split, standardisation, training, and evaluation."""
    features, targets = make_classification(
        n_samples=n_samples, n_features=n_features, separation=separation, noise=noise, seed=seed
    )
    train_idx, valid_idx, test_idx = split_indices(len(features), seed=seed)
    means, stds, (train_x, valid_x, test_x) = standardize(
        [features[i] for i in train_idx],
        [features[i] for i in valid_idx],
        [features[i] for i in test_idx],
    )
    train_set = subset(train_x, [targets[i] for i in train_idx], list(range(len(train_idx))))
    valid_set = subset(valid_x, [targets[i] for i in valid_idx], list(range(len(valid_idx))))
    test_set = subset(test_x, [targets[i] for i in test_idx], list(range(len(test_idx))))

    model = LogisticRegression(n_features)
    optimiser = SGD(model.parameters(), lr=learning_rate)
    trainer = Trainer(
        model=model,
        optimizer=optimiser,
        loss_fn=BCEWithLogitsLoss(),
        batch_size=min(64, len(train_set)),
        shuffle=True,
        seed=seed,
        probability_fn=lambda output: output.sigmoid(),
    )
    history = trainer.fit(train_set, epochs=epochs, valid_dataset=valid_set)

    test_probabilities = trainer.predict_proba(test_set)
    test_predictions = threshold_predictions(test_probabilities)
    test_truth = [round(value) for value in test_set.targets]
    report = classification_report(test_truth, test_predictions)
    return {
        "model": model,
        "history": history,
        "means": means,
        "stds": stds,
        "splits": {"train": train_idx, "valid": valid_idx, "test": test_idx},
        "test": report,
        "test_probabilities": test_probabilities,
        "test_targets": test_truth,
    }


__all__ = [
    "LogisticRegression",
    "analytical_gradients",
    "evaluate_logistic",
    "fit_logistic_regression",
    "make_classification",
    "train_valid_test_experiment",
]
