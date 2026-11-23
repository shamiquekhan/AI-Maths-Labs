#!/usr/bin/env python3
"""
Tensor Operations Example (using NumPy)

This example demonstrates tensor operations using NumPy, which is the
numerical backend used by the FROM ZERO labs for logistic regression,
optimizers, and other NumPy-based modules.

For the custom autodiff tensor engine, see autograd_example.py which
uses the Value class from from_zero.autodiff.
"""

import sys
import math

sys.path.insert(0, "src")

import numpy as np

from from_zero.logistic.model import sigmoid


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    max_val = np.max(x)
    exps = np.exp(x - max_val)
    return exps / np.sum(exps)


def log_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax."""
    max_val = np.max(x)
    log_sum = max_val + math.log(np.sum(np.exp(x - max_val)))
    return x - log_sum


def main():
    print("=" * 60)
    print("FROM ZERO - Tensor Operations (NumPy)")
    print("=" * 60)

    # 1. Construction
    print("\n1. Tensor Construction")
    print("-" * 40)

    # Scalar
    s = np.array(3.14)
    print(f"Scalar: {s}, shape: {s.shape}, ndim: {s.ndim}")

    # Vector
    v = np.array([1.0, 2.0, 3.0, 4.0])
    print(f"Vector: {v}, shape: {v.shape}")

    # Matrix
    m = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    print(f"Matrix: {m}, shape: {m.shape}")

    # 3D tensor
    t3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(f"3D tensor: shape {t3d.shape}, size: {t3d.size}")

    # 2. Elementwise operations
    print("\n2. Elementwise Operations")
    print("-" * 40)

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a - b = {a - b}")
    print(f"a * b = {a * b}")
    print(f"a / b = {a / b}")
    print(f"-a = {-a}")

    # 3. Broadcasting
    print("\n3. Broadcasting")
    print("-" * 40)

    m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    v = np.array([10.0, 20.0, 30.0])  # (3,)
    print(f"Matrix (2,3): {m}")
    print(f"Vector (3,): {v}")
    print(f"m + v = {m + v}  # broadcasts v to each row")

    # 4. Matrix multiplication
    print("\n4. Matrix Multiplication")
    print("-" * 40)

    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    C = A @ B
    print(f"A = {A}")
    print(f"B = {B}")
    print(f"A @ B = {C}")

    # Batched matmul
    A_batch = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    B_batch = np.array([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]])
    C_batch = A_batch @ B_batch
    print(f"Batched A @ B shape: {C_batch.shape}")

    # 5. Reductions
    print("\n5. Reductions")
    print("-" * 40)

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"x = {x}, shape: {x.shape}")
    print(f"sum() = {x.sum()}")
    print(f"sum(axis=0) = {x.sum(axis=0)}")
    print(f"sum(axis=1) = {x.sum(axis=1)}")
    print(f"mean() = {x.mean()}")
    print(f"mean(axis=0) = {x.mean(axis=0)}")

    # 6. Reshape and transpose
    print("\n6. Reshape and Transpose")
    print("-" * 40)

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = x.reshape(2, 3)
    print(f"Original: {x.shape} -> Reshaped: {y.shape}")
    print(f"Data: {y}")

    z = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    zt = z.T
    print(f"Transpose {z.shape} -> {zt.shape}")
    print(f"Data: {zt}")

    # 7. Stable mathematical functions
    print("\n7. Stable Mathematical Functions")
    print("-" * 40)

    # Sigmoid (from logistic model)
    print(f"sigmoid(100) = {sigmoid(np.array(100.0)):.6f} (should be ~1.0)")
    print(f"sigmoid(-100) = {sigmoid(np.array(-100.0)):.6f} (should be ~0.0)")
    print(f"sigmoid(0) = {sigmoid(np.array(0.0)):.6f} (should be 0.5)")

    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    print(f"tanh({x}) = {np.tanh(x)}")
    print(f"relu({x}) = {np.maximum(x, 0)}")

    # Softmax stability (numpy implementation)
    def softmax(x: np.ndarray) -> np.ndarray:
        max_val = np.max(x)
        exps = np.exp(x - max_val)
        return exps / np.sum(exps)

    def log_softmax(x: np.ndarray) -> np.ndarray:
        max_val = np.max(x)
        log_sum = max_val + math.log(np.sum(np.exp(x - max_val)))
        return x - log_sum

    # Softmax stability
    logits = np.array([1000.0, 1001.0, 999.0])  # Large values
    sm = softmax(logits)
    print(f"softmax([1000, 1001, 999]) = {sm}")
    print(f"  sum = {sum(sm):.6f} (should be 1.0)")

    # Log-softmax
    lsm = log_softmax(logits)
    print(f"log_softmax = {lsm}")
    print(f"  exp sum = {sum([math.exp(v) for v in lsm]):.6f}")

    print("\n" + "=" * 60)
    print("All tensor examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
