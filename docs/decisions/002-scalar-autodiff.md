# ADR 002: Scalar Reverse-Mode Autodiff Engine

**Date:** 2024-01-18
**Status:** Accepted
**Deciders:** Project maintainer

## Context

After implementing manual gradients for logistic regression, the next module should explain how frameworks like PyTorch compute gradients automatically. A full tensor autodiff system is out of scope.

## Decision

Build a **scalar** reverse-mode autodiff engine (`Value` class) with:
- Explicit computation graph (DAG) built during forward pass.
- Local backward functions (closures) per node.
- Iterative topological sort for reverse traversal.
- Gradient accumulation (`+=`) for shared subexpressions.
- Supported ops: `+`, `-`, `*`, `/`, `**`, `exp`, `log`, `tanh`, `relu`, `sigmoid`.

## Rationale

| Scope | Scalar Engine | Tensor Engine |
|-------|---------------|---------------|
| Implementation effort | ~500 lines | ~10,000+ lines |
| Conceptual clarity | Every node is a Python object | Batched operations obscure graph |
| Teaching value | Can print entire graph, step backward | Hard to inspect |
| XOR training | Works (slow but visible) | Overkill |
| Extensibility | Clear where to add ops | Complex dispatch |

The scalar engine makes the chain rule **visible**: `node._backward()` shows exactly what $\frac{\partial \text{out}}{\partial \text{in}}$ is for each operation.

## Consequences

- **Positive:** Learner can trace every gradient; numerical checks are straightforward; graph visualisation is trivial.
- **Negative:** Not a tensor framework; cannot train GPT/diffusion; Python overhead makes it 1000× slower than PyTorch.
- **Explicit non-goal:** Do not extend to tensors. The engine is a teaching artifact.

## Alternatives Considered

1. **PyTorch `autograd` with hooks:** Shows gradients but not the graph structure or local derivatives.
2. **Micrograd (Karpathy) as dependency:** Copied philosophy, but implemented from scratch to match project conventions (type hints, validation, tests).
3. **JAX-style trace-based:** Too abstract for "first principles" goal.

## Implementation Notes

- `src/from_zero/autodiff/value.py` — `Value` class with `__add__`, `__mul__`, etc.
- `src/from_zero/autodiff/graph.py` — `topological_sort`, `draw_graph`, `graph_to_dict`.
- `src/from_zero/autodiff/nn.py` — `Neuron`, `Layer`, `MLP`, `Module`, `SGD`.
- `tests/numerical/test_gradients.py` — Central-difference checks on random graphs.
- `tests/unit/test_autodiff.py` — Unit tests for each op, shared subexpressions, `zero_grad`.

## References

- Karpathy, *micrograd* (https://github.com/karpathy/micrograd)
- PyTorch autograd mechanics: https://pytorch.org/docs/stable/notes/autograd.html
- Project blueprint §10 "Module 03 — Automatic differentiation engine"