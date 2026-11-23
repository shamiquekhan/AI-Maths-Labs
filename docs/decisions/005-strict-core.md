# ADR 005: Strict From-Scratch Tensor Core

**Date:** 2026-11-23
**Status:** Accepted
**Deciders:** Project maintainer

## Context

The original laboratories used NumPy for logistic regression, optimizers, and
scalar autodiff, and PyTorch for MiniGPT and DDPM. That kept the experiments
readable, but it left a gap: the project claimed to teach "from zero" while the
tensor-level machinery was provided by third-party libraries.

The project needed a core that:

- Implements storage, shapes, strides, broadcasting, matrix multiplication,
  reverse-mode autograd, layers, losses, initializers, and optimizers itself.
- Runs the same engine through logistic regression, a character transformer, and
  a diffusion model, so the engine is exercised beyond toy scalar examples.
- Uses only the Python standard library in the core; no NumPy, SciPy, PyTorch,
  TensorFlow, Keras, JAX, or scikit-learn.
- Keeps the existing hybrid laboratories working and honestly labelled.
- Can be verified: finite-difference gradient checks, hand-checked optimizer
  updates, causality tests, and an import audit.

## Decision

Add a strict core as separate packages, replacing the earlier placeholder
tensor/autograd implementation:

```
tensor/           flat-list storage, views, broadcasting, ops, autograd
autograd_tensor/  topological inspection and finite-difference gradcheck
nn_tensor/        Parameter/Module, layers, activations, losses, initializers
optim_tensor/     SGD, momentum, AdaGrad, RMSprop, Adam, AdamW, schedules
strict/           logistic, char GPT, diffusion, JSON checkpoints
```

The strict packages are the only place where the "no third-party numerical
library" policy applies. The hybrid packages remain available and are labelled
as NumPy/PyTorch implementations.

### Storage and views

A tensor owns a flat `list[float]` plus shape, row-major strides, and an offset.
Transpose and permute swap strides without copying. Reshape is allowed only for
contiguous views. Broadcasting aligns shapes from the right; the backward pass
uses `sum_to_shape` to reduce gradients to the original shape.

### Autograd

Every differentiable operation returns a new tensor and records `_prev` (parent
tensors) and `_backward` (a closure mapping the upstream gradient to local
gradients). `backward()` performs a reverse topological traversal and
accumulates gradients, so shared subexpressions are handled correctly. A
`no_grad()` context disables graph construction for inference.

### Models

- **Logistic regression:** custom BCE-with-logits and an analytical-gradient
  oracle used to cross-check autograd.
- **Character GPT:** token and position embeddings, pre-layer-norm blocks,
  causal multi-head self-attention, GELU MLP, and cross-entropy. A causality
  test changes a future token and asserts that earlier logits do not move.
- **Diffusion:** linear beta schedule, closed-form forward process, sinusoidal
  timestep embedding, MLP noise predictor, and a reverse sampler.

### Checkpoints

Strict checkpoints are JSON documents with format marker
`from-zero-strict-1`. No pickle is used, so loading a checkpoint cannot execute
arbitrary code.

## Rationale

| Option | Verdict |
|---|---|
| Keep NumPy/PyTorch everywhere | Rejected: does not satisfy "from scratch" |
| Reimplement only scalars | Rejected: does not exercise tensors, layers, or optimizers |
| Monkey-patch NumPy | Rejected: hides the machinery and breaks the import policy |
| Separate strict packages | **Chosen:** isolated policy, testable boundary, hybrid labs preserved |
| Rewrite the hybrid labs on the strict engine | Rejected for now: slower and would discard working PyTorch evidence |

The strict engine is intentionally simple. Flat lists and Python closures make
the cost of every operation visible, which is the pedagogical point. It is not
a performance library.

## Consequences

**Positive**

- The project can demonstrate every layer from storage to sampling without a
  numerical dependency.
- The import audit makes the policy executable rather than aspirational.
- Finite-difference checks cover every differentiable operation.
- Strict models share one engine, so a bug in broadcasting or matmul surfaces
  across logistic, GPT, and diffusion tests.

**Negative**

- Pure Python is slow; strict models must stay tiny.
- Some operations (general strided reshape, advanced indexing) are deliberately
  limited.
- Maintaining two backends increases the test surface.
- The public deployment omits PyTorch, so the hybrid GPT/diffusion pages show a
  fallback; the strict variants are the cloud-safe substitutes.

## Verification

- `tests/unit/test_tensor_engine.py` — storage, views, broadcasting, matmul,
  softmax, embeddings, losses.
- `tests/numerical/test_strict_gradients.py` — central finite differences for
  every differentiable op, including the `logit == 0` BCE edge case.
- `tests/unit/test_nn_optim.py` — layers, state dicts, first optimizer updates.
- `tests/unit/test_strict_support.py` — schedules, metrics, checkpoints.
- `tests/unit/test_strict_import_audit.py` — static and subprocess import audit.
- `tests/integration/test_strict_pipelines.py` — analytical-vs-autograd
  logistic gradients, GPT causality/training, diffusion training/sampling,
  checkpoint round-trip.

## Alternatives Considered

1. **Use a tiny third-party autograd library.** Rejected: violates the policy
   and hides the graph.
2. **Write a C extension for speed.** Rejected: portability and review cost;
   the project is educational.
3. **Keep the old tensor implementation and document its gaps.** Rejected: the
   old code exposed attributes that did not match the documented API and did
   not support the required operations.

## References

- Karpathy, *micrograd* — scalar reverse-mode autograd.
- Karpathy, *nanoGPT* — decoder-only transformer structure.
- Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models* (2020).
- PyTorch documentation — broadcasting and autograd semantics (reference only;
  not imported by the strict core).
