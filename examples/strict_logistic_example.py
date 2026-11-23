#!/usr/bin/env python3
"""Strict logistic regression: analytical gradients, training, evaluation."""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from from_zero.nn_tensor import BCEWithLogitsLoss
from from_zero.strict import (
    LogisticRegression,
    analytical_gradients,
    evaluate_logistic,
    fit_logistic_regression,
    make_classification,
)
from from_zero.tensor import Tensor


def main() -> None:
    print("=" * 66)
    print("FROM ZERO - strict logistic regression")
    print("=" * 66)

    features, targets = make_classification(
        n_samples=200, n_features=2, separation=3.0, noise=0.5, seed=0
    )

    print("\n1. Analytical gradient versus autograd")
    print("-" * 40)
    sample_x = features[:5]
    sample_y = targets[:5]
    weight = [0.3, -0.2]
    bias = 0.1
    analytical_weight, analytical_bias = analytical_gradients(
        sample_x, sample_y, weight, bias, l2=0.1
    )

    model = LogisticRegression(2)
    model.weight._data[:] = weight
    model.bias._data[:] = [bias]
    loss = BCEWithLogitsLoss()(model(Tensor(sample_x)), Tensor([[y] for y in sample_y]))
    loss = loss + (model.weight * model.weight).sum() * (0.1 / (2.0 * len(sample_x)))
    loss.backward()

    print(f"analytical  w={[round(v, 6) for v in analytical_weight]} b={analytical_bias:.6f}")
    print(
        "autograd    w="
        f"{[round(row[0], 6) for row in model.weight.grad.tolist()]} "
        f"b={model.bias.grad.tolist()[0]:.6f}"
    )

    print("\n2. Training with the custom engine")
    print("-" * 40)
    trained, history = fit_logistic_regression(
        features, targets, epochs=300, learning_rate=0.5, seed=0
    )
    print(f"first loss  {history.train_loss[0]:.4f}")
    print(f"final loss  {history.train_loss[-1]:.4f}")
    print(f"accuracy    {history.train_accuracy[-1]:.1%}")

    print("\n3. Evaluation")
    print("-" * 40)
    report = evaluate_logistic(trained, features, targets)
    print(f"accuracy    {report['accuracy']:.3f}")
    print(f"precision   {report['precision']:.3f}")
    print(f"recall      {report['recall']:.3f}")
    print(f"f1          {report['f1']:.3f}")
    print(f"confusion   {report['confusion_matrix']}")
    print(f"weights     {[round(row[0], 4) for row in trained.weight.tolist()]}")
    print(f"bias        {trained.bias.tolist()[0]:.4f}")

    print("\nStrict logistic example complete.")


if __name__ == "__main__":
    main()
