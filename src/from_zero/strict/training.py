"""Reusable training loop for the FROM ZERO strict core."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..optim_tensor import clip_grad_norm_
from ..tensor import Tensor, no_grad
from .data import DataLoader, Dataset
from .metrics import accuracy_score, threshold_predictions


class History:
    """Recorded losses and metrics per epoch."""

    def __init__(self) -> None:
        self.train_loss: list[float] = []
        self.valid_loss: list[float] = []
        self.train_accuracy: list[float] = []
        self.valid_accuracy: list[float] = []

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "train_loss": list(self.train_loss),
            "valid_loss": list(self.valid_loss),
            "train_accuracy": list(self.train_accuracy),
            "valid_accuracy": list(self.valid_accuracy),
        }

    @property
    def final_train_loss(self) -> float:
        return self.train_loss[-1] if self.train_loss else float("nan")

    @property
    def final_valid_loss(self) -> float:
        return self.valid_loss[-1] if self.valid_loss else float("nan")


class Trainer:
    """Forward, loss, backward, optional clipping, and optimizer step."""

    def __init__(
        self,
        model,
        optimizer,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 0,
        grad_clip: float | None = None,
        probability_fn: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.grad_clip = grad_clip
        self.probability_fn = probability_fn

    def _probabilities(self, output: Tensor) -> list[float]:
        """Convert model outputs to probabilities when a classifier is used."""
        if self.probability_fn is None:
            return []
        return self.probability_fn(output)._values()

    def _iterate(self, dataset: Dataset, epoch: int, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            seed=self.seed + epoch,
        )

    def fit(
        self,
        train_dataset: Dataset,
        epochs: int,
        valid_dataset: Dataset | None = None,
        on_epoch: Callable[[int, History], None] | None = None,
    ) -> History:
        if epochs < 0:
            raise ValueError("epochs must be non-negative")
        history = History()
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            total_examples = 0
            predictions: list[int] = []
            targets: list[int] = []

            for features, target in self._iterate(train_dataset, epoch, self.shuffle):
                self.optimizer.zero_grad()
                output = self.model(features)
                loss = self.loss_fn(output, target)
                loss.backward()
                if self.grad_clip is not None:
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                total_loss += float(loss.item()) * len(target)
                total_examples += len(target)
                probabilities = self._probabilities(output)
                if probabilities:
                    predictions.extend(threshold_predictions(probabilities))
                    targets.extend(round(value) for value in target._values())

            history.train_loss.append(total_loss / total_examples if total_examples else 0.0)
            if predictions:
                history.train_accuracy.append(accuracy_score(targets, predictions))

            if valid_dataset is not None:
                evaluation = self.evaluate(valid_dataset)
                history.valid_loss.append(evaluation["loss"])
                if evaluation["accuracy"] is not None:
                    history.valid_accuracy.append(evaluation["accuracy"])

            if on_epoch is not None:
                on_epoch(epoch, history)

        return history

    def predict_proba(self, dataset: Dataset) -> list[float]:
        if self.probability_fn is None:
            raise ValueError("predict_proba requires probability_fn (for example a sigmoid)")
        self.model.eval()
        outputs: list[float] = []
        with no_grad():
            for features, _ in DataLoader(dataset, batch_size=self.batch_size, shuffle=False):
                output = self.model(features)
                if output.shape[-1] != 1:
                    raise ValueError("predict_proba expects a single output per example")
                outputs.extend(self._probabilities(output))
        return outputs

    def predict(self, dataset: Dataset, threshold: float = 0.5) -> list[int]:
        return threshold_predictions(self.predict_proba(dataset), threshold)

    def evaluate(self, dataset: Dataset) -> dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        total_examples = 0
        probabilities: list[float] = []
        targets: list[int] = []
        with no_grad():
            for features, target in DataLoader(dataset, batch_size=self.batch_size, shuffle=False):
                output = self.model(features)
                loss = self.loss_fn(output, target)
                total_loss += float(loss.item()) * len(target)
                total_examples += len(target)
                batch_probabilities = self._probabilities(output)
                if batch_probabilities:
                    probabilities.extend(batch_probabilities)
                    targets.extend(round(value) for value in target._values())

        result: dict[str, Any] = {
            "loss": total_loss / total_examples if total_examples else 0.0,
            "accuracy": None,
        }
        if probabilities:
            result["accuracy"] = accuracy_score(targets, threshold_predictions(probabilities))
        return result


__all__ = ["History", "Trainer"]
