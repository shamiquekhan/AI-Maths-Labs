#!/usr/bin/env python3
"""
Logistic Regression Example

Demonstrates the NumPy logistic regression implementation:
- Synthetic dataset generation
- Stable sigmoid and BCE-with-logits
- Analytical gradient derivation
- Training with early stopping
- Evaluation metrics
"""

import sys

sys.path.insert(0, "src")

import matplotlib.pyplot as plt
import numpy as np

from from_zero.core.datasets import prepare_classification
from from_zero.core.metrics import (
    binary_classification_metrics,
    confusion_matrix_binary,
)
from from_zero.core.plotting import (
    plot_confusion_matrix,
    plot_decision_boundary,
    plot_sigmoid,
)
from from_zero.logistic import (
    LogisticRegression,
    binary_cross_entropy_with_logits,
    check_gradients,
    compute_gradients,
    sigmoid,
)


def test_sigmoid_stability():
    """Test sigmoid numerical stability."""
    print("=" * 60)
    print("Logistic Regression: Sigmoid Stability")
    print("=" * 60)

    # Extreme values
    z = np.array([-1000.0, -100.0, -10.0, 0.0, 10.0, 100.0, 1000.0])
    sig = sigmoid(z)
    print(f"z = {z}")
    print(f"sigmoid(z) = {sig}")
    print(f"All finite: {np.all(np.isfinite(sig))}")
    print(f"All in [0,1]: {np.all((sig >= 0) & (sig <= 1))}")


def test_bce_from_logits():
    """Test stable BCE from logits."""
    print("\n" + "=" * 60)
    print("Logistic Regression: BCE from Logits")
    print("=" * 60)

    # Perfect predictions
    y = np.array([0, 1])
    logits = np.array([-100.0, 100.0])
    loss = binary_cross_entropy_with_logits(y, logits)
    print(f"Perfect predictions loss: {loss:.6f} (should be ~0)")

    # Wrong predictions
    logits_wrong = np.array([100.0, -100.0])
    loss_wrong = binary_cross_entropy_with_logits(y, logits_wrong)
    print(f"Wrong predictions loss: {loss_wrong:.6f} (should be ~100)")

    # Uncertain predictions
    logits_uncertain = np.array([0.0, 0.0])
    loss_uncertain = binary_cross_entropy_with_logits(y, logits_uncertain)
    print(f"Uncertain predictions loss: {loss_uncertain:.6f} (should be ~0.693)")


def test_gradient_check():
    """Test analytical gradients against finite differences."""
    print("\n" + "=" * 60)
    print("Logistic Regression: Gradient Check")
    print("=" * 60)

    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 4))
    y = (X[:, 0] - X[:, 1] + 0.2 * rng.normal(size=30) > 0).astype(int)
    weights = rng.normal(size=4) * 0.3
    bias = 0.1

    result = check_gradients(X, y, weights, bias, regularization=0.01)
    print(f"Gradient check passed: {result['passed']}")
    print(f"Max weight error: {result['max_abs_weight_error']:.2e}")
    print(f"Bias error: {result['abs_bias_error']:.2e}")


def test_analytical_gradients():
    """Test compute_gradients matches manual derivation."""
    print("\n" + "=" * 60)
    print("Logistic Regression: Analytical Gradients")
    print("=" * 60)

    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, -1.0]])
    y = np.array([0, 1, 1, 0])
    weights = np.array([0.2, -0.3])
    bias = 0.05

    grad_w, grad_b = compute_gradients(X, y, weights, bias, regularization=0.1)

    # Manual derivation
    logits = X @ weights + bias
    probs = sigmoid(logits)
    expected_w = X.T @ (probs - y) / len(y) + 0.1 * weights
    expected_b = np.mean(probs - y)

    print(f"Analytical weights: {grad_w}")
    print(f"Expected weights:   {expected_w}")
    print(f"Match: {np.allclose(grad_w, expected_w)}")
    print(f"Analytical bias: {grad_b:.6f}")
    print(f"Expected bias:   {expected_b:.6f}")
    print(f"Match: {abs(grad_b - expected_b) < 1e-10}")


def train_logistic_model():
    """Train logistic regression on synthetic data."""
    print("\n" + "=" * 60)
    print("Logistic Regression: Training")
    print("=" * 60)

    # Generate linearly separable data
    prepared = prepare_classification(
        name="linearly_separable",
        n_samples=800,
        noise=0.22,
        test_fraction=0.2,
        val_fraction=0.2,
        seed=42,
    )
    split = prepared.split
    print(f"Train: {split.X_train.shape}, Val: {split.X_val.shape}, Test: {split.X_test.shape}")

    # Train
    model = LogisticRegression(
        learning_rate=0.08,
        epochs=400,
        batch_size=64,
        regularization=0.001,
        early_stopping_patience=80,
        seed=42,
    )

    model.fit(split.X_train, split.y_train, X_val=split.X_val, y_val=split.y_val)

    print(f"\nEpochs run: {model.n_iter_}")
    print(f"Best epoch: {model.best_epoch_}")
    print(f"Converged: {model.converged_}")
    print(f"Train accuracy: {model.score(split.X_train, split.y_train):.4f}")
    print(f"Val accuracy:   {model.score(split.X_val, split.y_val):.4f}")
    print(f"Test accuracy:  {model.score(split.X_test, split.y_test):.4f}")

    # Evaluation metrics
    probs = model.predict_proba(split.X_test)
    preds = model.predict(split.X_test)
    metrics = binary_classification_metrics(split.y_test, probs)
    cm = confusion_matrix_binary(split.y_test, preds)

    print("\nTest Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"Confusion Matrix:\n{cm}")

    return model, split


def test_moons_dataset():
    """Test on non-linearly separable moons dataset."""
    print("\n" + "=" * 60)
    print("Logistic Regression: Moons Dataset (Non-linear)")
    print("=" * 60)

    prepared = prepare_classification(
        name="moons",
        n_samples=600,
        noise=0.15,
        test_fraction=0.2,
        val_fraction=0.2,
        seed=42,
    )
    split = prepared.split

    model = LogisticRegression(
        learning_rate=0.1,
        epochs=500,
        batch_size=64,
        regularization=0.001,
        seed=42,
    )

    model.fit(split.X_train, split.y_train, X_val=split.X_val, y_val=split.y_val)

    test_acc = model.score(split.X_test, split.y_test)
    print(f"Test accuracy on moons: {test_acc:.4f}")
    print("(Linear model cannot perfectly separate moons - this is expected)")

    return model, split


def visualize_results(model, split):
    """Create visualizations."""
    print("\n" + "=" * 60)
    print("Logistic Regression: Visualizations")
    print("=" * 60)

    # Sigmoid plot
    fig = plot_sigmoid()
    fig.savefig("/tmp/sigmoid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Sigmoid plot saved to /tmp/sigmoid.png")

    # Decision boundary
    fig = plot_decision_boundary(
        model, split.X_test, split.y_test, title="Test Set Decision Boundary"
    )
    fig.savefig("/tmp/decision_boundary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Decision boundary plot saved to /tmp/decision_boundary.png")

    # Confusion matrix
    preds = model.predict(split.X_test)
    fig = plot_confusion_matrix(split.y_test, preds, title="Test Confusion Matrix")
    fig.savefig("/tmp/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Confusion matrix plot saved to /tmp/confusion_matrix.png")


def main():
    test_sigmoid_stability()
    test_bce_from_logits()
    test_gradient_check()
    test_analytical_gradients()

    model, split = train_logistic_model()
    _moons_model, _moons_split = test_moons_dataset()

    # Visualize (optional - requires matplotlib)
    try:
        visualize_results(model, split)
    except Exception as e:
        print(f"\nVisualization skipped: {e}")

    print("\n" + "=" * 60)
    print("All logistic regression examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
