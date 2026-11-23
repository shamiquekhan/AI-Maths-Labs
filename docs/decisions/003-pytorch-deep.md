# ADR 003: PyTorch for Transformer and Diffusion

**Date:** 2024-01-20
**Status:** Accepted
**Deciders:** Project maintainer

## Context

The GPT and DDPM modules require:
- Multi-head attention with batched matrix multiplication.
- LayerNorm, GeLU, residual connections.
- Convolutional U-Net with skip connections.
- GPU acceleration for reasonable training time.
- Checkpointing with optimizer state.

Implementing these from scratch in NumPy would be a tensor framework project, not a mathematics teaching project.

## Decision

Use **PyTorch** for the MiniGPT and Diffusion modules.

## Rationale

| Requirement | NumPy + Custom | PyTorch |
|-------------|----------------|---------|
| Batched matmul (`B, T, d`) | Manual loops / einsum | `torch.matmul` / `@` |
| Multi-head attention | 200+ lines | `nn.MultiheadAttention` or custom ~50 lines |
| LayerNorm, GeLU | Implement + test | `nn.LayerNorm`, `F.gelu` |
| Conv U-Net | Very hard | `nn.Conv2d`, `nn.ConvTranspose2d` |
| GPU | Manual CUDA kernels | `.to('cuda')` |
| Optimizer (AdamW) | Re-implement | `torch.optim.AdamW` |
| Checkpointing | Custom serialization | `torch.save` / `torch.load` |
| Reference gradients | Manual check only | `loss.backward()` for comparison |

PyTorch is the standard for research implementations. Using it honestly (with clear labeling) teaches how to build with the framework while understanding the math.

## Consequences

- **Positive:** Realistic training times; standard patterns; easy comparison with reference implementations; checkpoint/load works reliably.
- **Negative:** Adds heavy dependency; hides some math (e.g., `LayerNorm` backward); GPU non-determinism affects reproducibility.
- **Mitigation:** Document exactly which operations are PyTorch vs. custom; expose attention weights, logits, and intermediate states; test causal masking and shapes.

## Boundary Enforcement

| Module | NumPy | PyTorch |
|--------|-------|---------|
| Logistic regression | ✅ All | ❌ |
| Optimizers | ✅ All | ❌ |
| Autodiff engine | ✅ All | ❌ |
| MiniGPT | ❌ | ✅ All |
| Diffusion | ❌ | ✅ All |

No module mixes NumPy arrays and PyTorch tensors in its core logic. Conversions happen only at boundaries (e.g., data loading).

## Alternatives Considered

1. **JAX:** Functional, beautiful, but less familiar; same "hides math" concern.
2. **TinyGrad / Micrograd tensor:** Not production-ready; would become the project.
3. **Pure NumPy GPT:** Possible but 50× slower; no GPU; defeats the purpose of training a generative model.

## Implementation Notes

- `src/from_zero/gpt/` — Pure PyTorch `nn.Module` hierarchy.
- `src/from_zero/diffusion/` — Pure PyTorch `nn.Module` U-Net.
- Both expose `model.num_parameters()` and return attention weights when requested.
- Tests verify: causal mask (`tests/unit/test_gpt.py`), schedule identities (`tests/numerical/test_diffusion_math.py`).

## References

- Project blueprint §11 "Module 04 — Mini GPT" and §12 "Module 05 — Diffusion"
- PyTorch documentation: https://pytorch.org/docs/