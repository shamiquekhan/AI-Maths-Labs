# Experiment Report: Strict From-Scratch Core

**Run name:** `strict_core` (examples + integration tests)
**Date:** 2026-11-23
**Backend:** Python standard library only
**Seed:** fixed per run (recorded below)
**Hardware:** CPU

This report covers the strict engine and its three end-to-end pipelines. It is
separate from the hybrid NumPy/PyTorch experiment reports. The numbers come
from the runnable examples and the integration suite; they demonstrate
correctness and learning, not benchmark quality.

## Policy under test

The strict packages (`tensor`, `autograd_tensor`, `nn_tensor`,
`optim_tensor`, `strict`) may import only the Python standard library and each
other. `tests/unit/test_strict_import_audit.py` parses every module and rejects
NumPy, SciPy, PyTorch, TensorFlow, Keras, JAX, and scikit-learn. A subprocess
test imports the packages with those libraries blocked and confirms that none
is loaded.

## 1. Strict logistic regression

**Command**

```bash
python examples/strict_logistic_example.py
```

**Setup**

- 200 synthetic samples, 2 features, separation 3.0, noise 0.5, seed 0.
- Full-batch training, 300 epochs, learning rate 0.5.
- Custom BCE-with-logits; L2 on weights.

**Gradient cross-check**

The analytical oracle and autograd agree to the printed precision:

```
analytical  w=[-0.558502, -0.197] b=0.273794
autograd    w=[-0.558502, -0.197] b=0.273794
```

**Training**

| Metric | Value |
|---|---|
| First loss | 0.6931 |
| Final loss | 0.0106 |
| Training accuracy | 100.0% |
| Precision / recall / F1 | 1.000 / 1.000 / 1.000 |
| Confusion matrix | `[[100, 0], [0, 100]]` |

A train/validation/test split run (300 samples, seed 7) reached validation loss
0.0096 from 0.4394 and test accuracy 1.000. The data is linearly separable, so
perfect accuracy is expected; the point is the verified gradient path, not the
score.

## 2. Strict character GPT

**Command**

```bash
python examples/strict_gpt_example.py
```

**Setup**

- Corpus: `"the quick brown fox jumps over the lazy dog. "` repeated three
  times; 28-character vocabulary.
- Context 8, embedding 8, 2 heads (head dimension 4), 1 pre-LN block.
- 1,376 parameters; dropout 0.
- 40 optimizer steps, batch size 4, learning rate 5e-3, seed 0.

**Causality check**

Changing the last token of `"the quic"` to `"the quia"`:

| Region | Maximum logit change |
|---|---|
| Positions before the edit | `0.00e+00` |
| Edited position | `2.77e+00` |

The exact zero confirms the causal mask: no future token influences an earlier
position.

**Training**

| Step | Loss |
|---|---|
| 0 | 3.6104 |
| 20 | 3.3334 |
| 36 | 3.0244 |
| 39 (final) | 2.9559 |

Sampling is character-level and reflects the tiny corpus:

```
temperature 0.0: 'the lhkrhkhkhkhkhkhkhkhkhkhkhkhkhk'
temperature 0.8: 'the dxrgkhnr. vkr kogxu  mviek emm'
```

This is a correctness demonstration, not a language model.

## 3. Strict diffusion

**Command**

```bash
python examples/strict_diffusion_example.py
```

**Setup**

- 8 synthetic 4×4 pattern images, seed 0.
- Linear schedule with 20 timesteps, `β` from `1e-4` to `0.02`.
- Sinusoidal timestep embedding (8 dims), MLP denoiser (hidden 32).
- 100 optimizer steps, batch size 8, learning rate 0.01, seed 0.

**Training**

| Step | Loss |
|---|---|
| 0 | 1.2227 |
| 30 | 1.0024 |
| 60 | 0.8582 |
| 90 | 0.9202 |

The loss decreases overall but is noisy because each step samples a random
timestep and noise. The sampler records 21 frames for a 20-step schedule
(initial noise plus one frame per reverse step), and the final 4×4 sample is
rendered as ASCII in the example output.

## 4. Checkpoint round-trip

Strict checkpoints are JSON with format marker `from-zero-strict-1`:

```python
from from_zero.strict import load_checkpoint, save_checkpoint

save_checkpoint("model.json", model, optimizer=optimizer, metadata={"epoch": 4})
payload = load_checkpoint("model.json")
```

The integration test asserts that every parameter, optimizer state entry, and
metadata field survives the round-trip. No pickle is used.

## 5. Automated coverage

| Test | Result |
|---|---|
| `tests/unit/test_tensor_engine.py` | pass |
| `tests/numerical/test_strict_gradients.py` | pass (finite differences) |
| `tests/unit/test_nn_optim.py` | pass |
| `tests/unit/test_strict_support.py` | pass |
| `tests/unit/test_strict_import_audit.py` | pass |
| `tests/integration/test_strict_pipelines.py` | pass |
| `tests/integration/test_app_pages.py` (strict page + fallback) | pass |

## Limitations

1. **Scale:** pure Python is slow; models are tiny and trained briefly.
2. **GPT quality:** a 28-character vocabulary and 40 steps cannot produce
   fluent text. The test target is causality and a decreasing loss.
3. **Diffusion quality:** an MLP on 4×4 patterns is not a U-Net on MNIST.
4. **Single seed:** the examples fix seeds for reproducibility, not statistical
   significance.
5. **No GPU:** the strict core is CPU-only by design.

## Reproduction

```bash
python examples/strict_tensor_example.py
python examples/strict_logistic_example.py
python examples/strict_gpt_example.py
python examples/strict_diffusion_example.py
python -m pytest tests/unit/test_tensor_engine.py tests/numerical/test_strict_gradients.py \
  tests/unit/test_nn_optim.py tests/unit/test_strict_support.py \
  tests/unit/test_strict_import_audit.py tests/integration/test_strict_pipelines.py -q
```
