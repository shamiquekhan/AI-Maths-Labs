# ADR 001: NumPy for Foundational Modules

**Date:** 2024-01-15
**Status:** Accepted
**Deciders:** Project maintainer

## Context

The project includes three foundational modules: logistic regression, optimizers, and a scalar autodiff engine. These are intended to teach the mathematics by exposing every computational step.

## Decision

Implement logistic regression, optimizers, and the scalar autodiff engine using **NumPy only** (no PyTorch, no JAX, no TensorFlow).

## Rationale

| Factor | NumPy | PyTorch/JAX |
|--------|-------|-------------|
| Gradient visibility | Manual implementation required | Hidden in `backward()` |
| Learning value | Forces derivation | Abstracts away the math |
| Dependencies | Light, universal | Heavy, CUDA complexity |
| Inspection | Every array is a `np.ndarray` | Tensor metadata, autograd graph |
| Debugging | Print intermediate values | Requires hooks, breakpoints |

For teaching "what is a gradient?", the student must write the gradient. NumPy makes this unavoidable.

## Consequences

- **Positive:** Transparent mathematics; no "magic" differentiation; easy to add finite-difference checks; runs on any Python environment.
- **Negative:** No GPU acceleration (not needed for small datasets); manual batching; no built-in modules (must implement `Linear`, `LayerNorm`, etc. if needed).
- **Boundary:** GPT and diffusion modules use PyTorch (see ADR 003).

## Alternatives Considered

1. **Pure Python (lists):** Too slow, no broadcasting, no linear algebra.
2. **JAX:** Beautiful autodiff, but hides the manual gradient derivation we want to teach.
3. **Micrograd-style scalar class:** Used for the autodiff module itself (ADR 002), but too slow for logistic/optimizer loops.

## Implementation Notes

- `src/from_zero/logistic/` — all NumPy.
- `src/from_zero/optim/` — all NumPy (list of arrays API).
- `src/from_zero/autodiff/` — custom `Value` class (scalar), no NumPy in forward pass.
- Shared `core/` utilities use NumPy for metrics, datasets, plotting.

## References

- Project blueprint §5.3 "Architectural boundary: NumPy versus PyTorch"