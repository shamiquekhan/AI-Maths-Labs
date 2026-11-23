#!/usr/bin/env python3
"""Tensor engine and autograd example using only the Python standard library."""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from from_zero.autograd_tensor import format_graph, gradcheck
from from_zero.tensor import Tensor, no_grad, tensor


def main() -> None:
    print("=" * 66)
    print("FROM ZERO - strict tensor engine (no NumPy, no PyTorch)")
    print("=" * 66)

    print("\n1. Storage, shape, strides")
    print("-" * 40)
    x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"values      {x.tolist()}")
    print(f"shape       {x.shape}")
    print(f"strides     {x.strides}")
    print(f"transpose   {x.T.tolist()}  (strides {x.T.strides}, no data copied)")

    print("\n2. Broadcasting")
    print("-" * 40)
    bias = tensor([10.0, 20.0, 30.0])
    print(f"x + bias    {(x + bias).tolist()}")

    print("\n3. Matrix multiplication")
    print("-" * 40)
    weights = tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    print(f"x @ W       {(x @ weights).tolist()}")

    print("\n4. Reverse-mode autograd")
    print("-" * 40)
    parameter = tensor([0.5, -1.5, 2.0], requires_grad=True)
    loss = ((parameter * parameter) + parameter).sum()
    loss.backward()
    print(f"loss        {loss.item():.6f}")
    print(f"dL/dx       {parameter.grad.tolist()}  (expected 2x + 1)")
    print("\nGraph nodes (parents before children):")
    print(format_graph(loss))

    print("\n5. Finite-difference check")
    print("-" * 40)
    check = tensor([0.2, -0.4, 0.9], requires_grad=True)
    passed = gradcheck(lambda: (check.tanh() + check.sigmoid()).sum(), [check])
    print(f"gradcheck   {'PASS' if passed else 'FAIL'}")

    print("\n6. no_grad")
    print("-" * 40)
    with no_grad():
        quiet = check * 2.0
    print(f"requires_grad inside no_grad: {quiet.requires_grad}")

    print("\n7. Losses and softmax")
    print("-" * 40)
    logits = tensor([[1.0, 2.0, 3.0]])
    print(f"softmax     {[round(value, 5) for value in logits.softmax()[0].tolist()]}")
    print(f"softmax sum {logits.softmax().sum().item():.6f}")

    print("\nStrict tensor example complete.")


if __name__ == "__main__":
    main()
