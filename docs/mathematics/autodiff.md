# Reverse-Mode Automatic Differentiation: A Scalar Engine

## Core Idea

Reverse-mode autodiff computes gradients of a scalar output with respect to many inputs by:
1. **Forward pass**: Build a directed acyclic graph (DAG) of operations.
2. **Backward pass**: Traverse the graph in reverse topological order, applying the chain rule.

For a scalar loss $\mathcal{L}$, this gives $\nabla_\theta \mathcal{L}$ in time proportional to the forward pass (plus a constant factor), independent of the number of parameters.

---

## The Computation Graph

Each node $v$ in the graph stores:
- `data`: The forward value (float).
- `grad`: Accumulated gradient $\frac{\partial \mathcal{L}}{\partial v}$ (float, starts at 0).
- `parents`: Tuple of parent nodes (the inputs to the operation that produced $v$).
- `op`: String label (e.g., `"+"`, `"*"`, `"tanh"`).
- `_backward`: A closure that propagates gradients to parents.

**Shared subexpressions:** If a node feeds multiple children, its gradient accumulates contributions from all of them (hence `+=` in `_backward`).

---

## Topological Order

A valid reverse traversal requires parents to be processed **after** all their children. This is a topological sort of the DAG.

Algorithm (iterative, no recursion):
```
1. DFS from root to collect all nodes (parents before children in visitation order).
2. Reverse the list → children before parents.
3. Set root.grad = 1.0 (dL/dL = 1).
4. For each node in reversed order: call node._backward().
```

This ensures when we process a node, its `grad` already holds the total $\frac{\partial \mathcal{L}}{\partial \text{node}}$.

---

## Local Derivative Rules

For an operation $c = f(a, b)$, the backward rule adds to parents' gradients:

| Operation | Forward | Backward contribution |
|-----------|---------|----------------------|
| Addition | $c = a + b$ | $\bar{a} \mathrel{+}= \bar{c}$, $\bar{b} \mathrel{+}= \bar{c}$ |
| Multiplication | $c = a b$ | $\bar{a} \mathrel{+}= b \bar{c}$, $\bar{b} \mathrel{+}= a \bar{c}$ |
| Power | $c = a^p$ | $\bar{a} \mathrel{+}= p a^{p-1} \bar{c}$ |
| Exp | $c = e^a$ | $\bar{a} \mathrel{+}= e^a \bar{c} = c \bar{c}$ |
| Log | $c = \log a$ | $\bar{a} \mathrel{+}= \bar{c} / a$ |
| Tanh | $c = \tanh a$ | $\bar{a} \mathrel{+}= (1 - \tanh^2 a) \bar{c} = (1 - c^2) \bar{c}$ |
| Sigmoid | $c = \sigma(a)$ | $\bar{a} \mathrel{+}= \sigma(a)(1-\sigma(a)) \bar{c} = c(1-c) \bar{c}$ |
| ReLU | $c = \max(0, a)$ | $\bar{a} \mathrel{+}= \mathbb{1}_{a > 0} \bar{c}$ |
| Division | $c = a / b$ | $\bar{a} \mathrel{+}= \bar{c}/b$, $\bar{b} \mathrel{+}= -a \bar{c}/b^2$ |
| Negation | $c = -a$ | $\bar{a} \mathrel{+}= -\bar{c}$ |

(Notation: $\bar{x}$ denotes the accumulated gradient of $x$.)

---

## Gradient Accumulation and `zero_grad()`

- `backward()` **adds** to existing `.grad` values.
- To compute a fresh gradient, either:
  - Create a new graph (new `Value` objects), or
  - Call `zero_grad(root)` which traverses the graph and sets all `.grad = 0.0`.

The engine does **not** auto-reset—explicit is better than implicit.

---

## Numerical Gradient Check

For a function $f: \mathbb{R}^n \to \mathbb{R}$, compare autodiff with central differences:

$$\frac{\partial f}{\partial x_i} \approx \frac{f(x + h e_i) - f(x - h e_i)}{2h}$$

The test (`tests/numerical/test_gradients.py`):
- Uses random points in the domain.
- Checks absolute tolerance $10^{-6}$ and relative tolerance $10^{-5}$.
- Documents floating-point error (small $h$ → truncation error; large $h$ → rounding error).

---

## Building a Tiny MLP

```python
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        return act.tanh()  # activation


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]


class MLP:
    def __init__(self, nin, nouts):
        self.layers = [Layer(nin if i == 0 else nouts[i - 1], nouts[i]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x
```

Training loop (XOR dataset):
```python
model = MLP(2, [4, 4, 1])
for epoch in range(epochs):
    for xi, yi in dataset:
        y_pred = model([Value(v) for v in xi])
        loss = (y_pred - Value(yi)) ** 2
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad
```

---

## Scope and Limitations

| Aspect | This Engine | PyTorch/JAX |
|--------|-------------|-------------|
| Data type | Scalar `float` | N-dimensional tensors |
| Graph | Explicit Python objects | Implicit (trace) or explicit (FX) |
| Speed | Python loops, per-op overhead | Vectorized C++/CUDA kernels |
| Use case | **Teaching** | Production |
| Operations | 12 elementary ops | 1000+ ops |
| Autograd | Reverse-mode, manual | Reverse-mode, automatic |

**This engine is not a tensor framework.** It exists to make reverse-mode differentiation inspectable.

---

## Stability Choices

1. **Log** rejects non-positive inputs (raises `ValueError`).
2. **Power** at zero: $0^p$ is defined for $p>0$, derivative undefined for $0<p<1$ (raises).
3. **BCE with logits** uses stable `softplus`, not `log(sigmoid(z))`.
4. **All operations** return `Value` with `float` data—no silent NaN propagation.

---

## Visualisation

The graph drawer (`autodiff/graph.py`) produces a Matplotlib figure:
- Nodes show `op`, `data`, `grad`.
- Edges show parent → child dependencies.
- Layout uses topological levels (parents left, children right).

Example output: `artifacts/experiments/autodiff_xor/plots/autodiff_graph.png`

---

## References

- Rumelhart, Hinton & Williams, *Learning representations by back-propagating errors* (1986)
- Baydin et al., *Automatic differentiation in machine learning: a survey* (2018)
- Karpathy, *micrograd* (https://github.com/karpathy/micrograd)
- PyTorch autograd mechanics: https://pytorch.org/docs/stable/notes/autograd.html