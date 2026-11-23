"""A transparent NumPy logistic regression classifier."""

from __future__ import annotations

from typing import Any

import numpy as np

from .gradients import compute_gradients
from .losses import binary_cross_entropy_with_logits


def sigmoid(z: Any) -> float | np.ndarray:
    """Numerically stable logistic sigmoid for scalars and arrays.

    For negative logits we evaluate ``eᶻ / (1 + eᶻ)`` instead of
    ``1 / (1 + e⁻ᶻ)``, so large-magnitude inputs do not overflow.
    """

    array = np.asarray(z, dtype=np.float64)
    result = np.empty_like(array, dtype=np.float64)
    positive = array >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exp_z = np.exp(array[~positive])
    result[~positive] = exp_z / (1.0 + exp_z)
    if result.ndim == 0:
        return float(result)
    return result


class LogisticRegression:
    """Binary logistic regression trained with mini-batch gradient descent.

    The model stores raw weights and a bias.  Standardisation belongs to the
    data preparation layer so the same fitted scaler can be reused at inference
    time.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        epochs: int = 1000,
        regularization: float = 0.0,
        seed: int = 42,
        batch_size: int | None = 64,
        threshold: float = 0.5,
        early_stopping_patience: int | None = 100,
        min_delta: float = 1e-6,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if regularization < 0:
            raise ValueError("regularization must be non-negative")
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive or None")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if early_stopping_patience is not None and early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive or None")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")

        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.regularization = float(regularization)
        self.seed = int(seed)
        self.batch_size = batch_size
        self.threshold = float(threshold)
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = float(min_delta)

        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.history_: list[dict[str, float]] = []
        self.n_iter_: int = 0
        self.best_epoch_: int = 0
        self.converged_: bool = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = False,
    ) -> LogisticRegression:
        features, labels = _validate_training_data(X, y)
        if (X_val is None) != (y_val is None):
            raise ValueError("Provide both X_val and y_val, or neither")
        if X_val is not None and y_val is not None:
            val_features, val_labels = _validate_training_data(X_val, y_val)
            if val_features.shape[1] != features.shape[1]:
                raise ValueError(
                    "Validation features must have the same width as training features"
                )
        else:
            val_features, val_labels = None, None

        n_samples, n_features = features.shape
        self.weights_ = np.zeros(n_features, dtype=np.float64)
        self.bias_ = 0.0
        self.history_ = []
        self.n_iter_ = 0
        self.best_epoch_ = 0
        self.converged_ = False

        rng = np.random.default_rng(self.seed)
        batch_size = n_samples if self.batch_size is None else min(self.batch_size, n_samples)
        best_score = float("inf")
        best_weights = self.weights_.copy()
        best_bias = self.bias_
        epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):
            permutation = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch = permutation[start : start + batch_size]
                gradient_w, gradient_b = compute_gradients(
                    features[batch],
                    labels[batch],
                    self.weights_,
                    self.bias_,
                    regularization=self.regularization,
                )
                self.weights_ -= self.learning_rate * gradient_w
                self.bias_ -= self.learning_rate * gradient_b

            train_probabilities = self.predict_proba(features)
            train_loss = binary_cross_entropy_with_logits(labels, self.decision_function(features))
            train_accuracy = float(np.mean((train_probabilities >= self.threshold) == labels))
            row: dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_accuracy": train_accuracy,
            }
            monitored = train_loss
            if val_features is not None and val_labels is not None:
                val_probabilities = self.predict_proba(val_features)
                val_loss = binary_cross_entropy_with_logits(
                    val_labels, self.decision_function(val_features)
                )
                row["val_loss"] = float(val_loss)
                row["val_accuracy"] = float(
                    np.mean((val_probabilities >= self.threshold) == val_labels)
                )
                monitored = val_loss
            self.history_.append(row)

            if monitored < best_score - self.min_delta:
                best_score = monitored
                best_weights = self.weights_.copy()
                best_bias = self.bias_
                self.best_epoch_ = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if verbose and (epoch == 1 or epoch % 100 == 0 or epoch == self.epochs):
                message = f"epoch {epoch:4d} train_loss={train_loss:.5f}"
                if "val_loss" in row:
                    message += f" val_loss={row['val_loss']:.5f}"
                print(message)

            if (
                self.early_stopping_patience is not None
                and val_features is not None
                and epochs_without_improvement >= self.early_stopping_patience
            ):
                self.converged_ = True
                break

        self.n_iter_ = len(self.history_)
        if val_features is not None:
            self.weights_ = best_weights
            self.bias_ = best_bias
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return raw logits ``Xw + b``."""

        features = self._validate_features(X)
        if self.weights_ is None:
            raise RuntimeError("LogisticRegression must be fitted before decision_function")
        return features @ self.weights_ + self.bias_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``P(y=1 | x)`` for each row of ``X``."""

        return np.asarray(sigmoid(self.decision_function(X)), dtype=np.float64)

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Return hard binary predictions using a probability threshold."""

        cutoff = self.threshold if threshold is None else float(threshold)
        if not 0.0 <= cutoff <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return (self.predict_proba(X) >= cutoff).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        labels = np.asarray(y)
        predictions = self.predict(X)
        if len(labels) != len(predictions):
            raise ValueError("X and y length mismatch")
        return float(np.mean(predictions == labels))

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        labels = np.asarray(y)
        logits = self.decision_function(X)
        if len(labels) != len(logits):
            raise ValueError("X and y length mismatch")
        data_loss = binary_cross_entropy_with_logits(labels, logits)
        if self.weights_ is None:
            return data_loss
        return data_loss + 0.5 * self.regularization * float(self.weights_ @ self.weights_)

    def get_params(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "regularization": self.regularization,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "threshold": self.threshold,
            "early_stopping_patience": self.early_stopping_patience,
            "min_delta": self.min_delta,
        }

    def _validate_features(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("LogisticRegression must be fitted before prediction")
        features = np.asarray(X, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {features.shape}")
        if features.shape[1] != len(self.weights_):
            raise ValueError(
                f"X has {features.shape[1]} features, model expects {len(self.weights_)}"
            )
        if not np.all(np.isfinite(features)):
            raise ValueError("X contains NaN or infinity")
        return features


def _validate_training_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y)
    if features.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {features.shape}")
    if labels.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {labels.shape}")
    if len(features) != len(labels):
        raise ValueError(f"X and y length mismatch: {len(features)} vs {len(labels)}")
    if len(features) == 0:
        raise ValueError("Training data is empty")
    if not np.all(np.isfinite(features)):
        raise ValueError("X contains NaN or infinity")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("y must contain only binary labels 0 and 1")
    if len(np.unique(labels)) < 2:
        raise ValueError("Training data must contain both classes")
    return features, labels.astype(np.float64)
