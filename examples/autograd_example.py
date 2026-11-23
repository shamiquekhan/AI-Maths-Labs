#!/usr/bin/env python3
"""
Autograd Engine Example

Demonstrates the custom automatic differentiation engine:
- Building computational graphs
- Reverse-mode differentiation
- Gradient accumulation
- Finite-difference gradient checking
"""

import sys

sys.path.insert(0, "src")

import math

from from_zero.autodiff import (
    Value,
    graph_to_dict,
    mse_loss,
    topological_sort,
    zero_grad,
)
from from_zero.autodiff.nn import MLP
from from_zero.autodiff.optim import SGD


def demo_expression():
    """Create a sample expression for visualization."""
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = a * b + c
    e = d.tanh()
    return e


def test_scalar_operations():
    """Test basic scalar operations and their gradients."""
    print("=" * 60)
    print("Autograd: Basic Operations")
    print("=" * 60)

    # Addition
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    c.backward()
    print(f"Addition: {a.data} + {b.data} = {c.data}")
    print(f"  da/dc = {a.grad}, db/dc = {b.grad} (expected: 1.0, 1.0)")

    # Multiplication
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()
    print(f"\nMultiplication: {a.data} * {b.data} = {c.data}")
    print(f"  da/dc = {a.grad}, db/dc = {b.grad} (expected: 3.0, 2.0)")

    # Chain rule: (a * b + c).tanh()
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = a * b + c
    e = d.tanh()
    e.backward()
    print(f"\nChain rule: tanh({a.data} * {b.data} + {c.data}) = {e.data:.4f}")
    print(f"  da/de = {a.grad:.4f}")
    print(f"  db/de = {b.grad:.4f}")
    print(f"  dc/de = {c.grad:.4f}")


def test_shared_subexpression():
    """Test gradient accumulation for shared subexpressions."""
    print("\n" + "=" * 60)
    print("Autograd: Shared Subexpressions")
    print("=" * 60)

    # Shared subexpression: a * b appears twice
    a = Value(2.0)
    b = Value(3.0)
    shared = a * b
    c = shared + shared  # 2 * (a * b)
    c.backward()

    print("c = (a * b) + (a * b) = 2 * a * b")
    print(f"  c = {c.data}")
    print(f"  dc/da = {a.grad} (expected: 2 * b = 6.0)")
    print(f"  dc/db = {b.grad} (expected: 2 * a = 4.0)")


def test_graph_visualization():
    """Test graph building and topological sorting."""
    print("\n" + "=" * 60)
    print("Autograd: Graph Visualization")
    print("=" * 60)

    expr = demo_expression()
    expr.backward()

    # Topological sort
    order = topological_sort(expr)
    print(f"Graph has {len(order)} nodes:")
    for i, node in enumerate(order):
        parents = [id(p) for p in node.parents]
        print(
            f"  {i}: op={node.op or 'leaf'}, data={node.data:.4f}, grad={node.grad:.4f}, parents={parents}"
        )

    # Graph to dict for serialization
    graph_dict = graph_to_dict(expr)
    print(f"\nGraph dict has {len(graph_dict['nodes'])} nodes")
    print(f"Root index: {graph_dict['root']}")


def test_gradient_checking():
    """Test finite-difference gradient checking."""
    print("\n" + "=" * 60)
    print("Autograd: Finite-Difference Gradient Check")
    print("=" * 60)

    def autodiff_fn(vals):
        """Returns (output, leaf_nodes) for gradient checking."""
        a, b, c = [Value(v) for v in vals]
        first = (a * b + c).tanh()
        second = (a**2 + 0.5 * b).sigmoid()
        third = (c.exp() + 1.0).log()
        out = first + second + third + (a / (b + 2.0)) - 0.25 * c**2
        return out, (a, b, c)

    point = [0.7, 1.3, -0.4]

    # Autodiff gradients
    out, (a, b, c) = autodiff_fn(point)
    out.backward()
    analytic = [a.grad, b.grad, c.grad]

    # Finite differences
    def central_diff(fn, x, h=1e-6):
        grad = []
        for i in range(len(x)):
            x_plus = x[:]
            x_minus = x[:]
            x_plus[i] += h
            x_minus[i] -= h
            out_plus, _ = fn(x_plus)
            out_minus, _ = fn(x_minus)
            grad.append((out_plus.data - out_minus.data) / (2 * h))
        return grad

    numeric = central_diff(autodiff_fn, point)

    print(f"Point: {point}")
    print(f"Autodiff:  {[f'{g:.8f}' for g in analytic]}")
    print(f"Numeric:   {[f'{g:.8f}' for g in numeric]}")
    max_err = max(abs(a - n) for a, n in zip(analytic, numeric, strict=False))
    print(f"Max error: {max_err:.2e} {'✓ PASS' if max_err < 1e-6 else '✗ FAIL'}")


def test_activation_gradients():
    """Test gradients of activation functions."""
    print("\n" + "=" * 60)
    print("Autograd: Activation Function Gradients")
    print("=" * 60)

    x = Value(0.7)

    # Tanh
    y = x.tanh()
    y.backward()
    expected = 1 - math.tanh(0.7) ** 2
    print(f"tanh({x.data}) = {y.data:.6f}")
    print(f"  dy/dx = {x.grad:.6f} (expected: {expected:.6f})")

    x = Value(0.7)
    # Sigmoid
    y = x.sigmoid()
    y.backward()
    sig = 1 / (1 + math.exp(-0.7))
    expected = sig * (1 - sig)
    print(f"sigmoid({x.data}) = {y.data:.6f}")
    print(f"  dy/dx = {x.grad:.6f} (expected: {expected:.6f})")

    x = Value(-0.5)
    # ReLU
    y = x.relu()
    y.backward()
    print(f"relu({x.data}) = {y.data:.6f}")
    print(f"  dy/dx = {x.grad:.6f} (expected: 0.0)")

    x = Value(0.5)
    y = x.relu()
    y.backward()
    print(f"relu({x.data}) = {y.data:.6f}")
    print(f"  dy/dx = {x.grad:.6f} (expected: 1.0)")


def test_zero_grad():
    """Test zero_grad functionality."""
    print("\n" + "=" * 60)
    print("Autograd: zero_grad")
    print("=" * 60)

    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()
    print(f"After first backward: a.grad={a.grad}, b.grad={b.grad}")

    # Call backward again - gradients accumulate
    c.backward()
    print(f"After second backward: a.grad={a.grad}, b.grad={b.grad}")

    # Zero grad
    zero_grad(c)
    print(f"After zero_grad: a.grad={a.grad}, b.grad={b.grad}")

    # Backward again
    c.backward()
    print(f"After backward again: a.grad={a.grad}, b.grad={b.grad}")


def test_mlp_training():
    """Test MLP training on XOR."""
    print("\n" + "=" * 60)
    print("Autograd: MLP Training on XOR")
    print("=" * 60)

    # XOR dataset
    data = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]

    model = MLP(2, [4, 4], 1, activation="tanh", output_activation="linear", seed=42)
    optimizer = SGD(model.parameters(), learning_rate=0.2)

    print(f"Model parameters: {model.parameter_count()}")
    print("Training...")

    for epoch in range(5000):
        epoch_loss = 0.0
        for xi, yi in data:
            x_vals = [Value(v) for v in xi]
            y_pred = model(x_vals)
            loss = mse_loss([y_pred], [Value(float(yi))])
            model.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data

        if epoch % 1000 == 0:
            print(f"  Epoch {epoch}: loss = {epoch_loss / len(data):.6f}")

    # Final evaluation
    correct = 0
    print("\nFinal predictions:")
    for xi, yi in data:
        x_vals = [Value(v) for v in xi]
        y_pred = model(x_vals)
        pred = 1 if y_pred.data > 0.5 else 0
        correct += pred == yi
        print(f"  {xi} -> {y_pred.data:.4f} (target: {yi}, pred: {pred})")

    print(f"\nAccuracy: {correct}/4 = {correct / 4:.2f}")


def main():
    test_scalar_operations()
    test_shared_subexpression()
    test_graph_visualization()
    test_gradient_checking()
    test_activation_gradients()
    test_zero_grad()
    test_mlp_training()

    print("\n" + "=" * 60)
    print("All autograd examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
