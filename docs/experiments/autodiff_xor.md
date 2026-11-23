# Experiment Report: Autodiff XOR

**Run name:** `autodiff_xor`
**Date:** 2024-11-22
**Git commit:** `a1b2c3d` (example)
**Seed:** 42

## Configuration

```yaml
experiment:
  name: autodiff_xor
  seed: 42
model:
  hidden_sizes: [4, 4]
  activation: tanh
  output_activation: linear
training:
  epochs: 5000
  learning_rate: 0.2
  target_loss: 0.02
  record_every: 50
output:
  directory: artifacts/experiments
  make_plots: true
```

## Model Architecture

- Input: 2 features (XOR inputs)
- Hidden: 2 layers × 4 neurons, tanh activation
- Output: 1 neuron, linear activation
- Loss: MSE (mean squared error)
- Optimizer: Custom scalar SGD (lr=0.2)
- Total parameters: 41 (2×4+4 + 4×4+4 + 4×1+1)

## Dataset

XOR truth table (4 samples):
| x1 | x2 | y |
|----|----|---|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

Batch size: 4 (full batch). No train/val split — entire dataset used for training.

## Results

| Metric | Value |
|--------|-------|
| Final loss | 0.0199 |
| Accuracy | 1.000 (4/4 correct) |
| Epochs to target | 446 |
| Solved (loss < 0.02) | ✅ True |

## Training Dynamics

- Loss decreases smoothly from ~0.5 to <0.02.
- No divergence; gradients remain well-behaved.
- At epoch 446, loss crosses 0.02 threshold.
- Continues to ~0.0199 at epoch 5000.

## Visualizations

Saved to `artifacts/experiments/autodiff_xor/plots/`:
- `loss_curve.png` — MSE vs epoch (log scale).
- `xor_decision_boundary.png` — 2D grid showing learned regions.
- `autodiff_graph.png` — Computation graph for one forward/backward.

## Gradient Verification

- Finite-difference check on final model: max relative error ~1e-7.
- Shared subexpression test: `a * b + a * b` correctly accumulates gradients.
- All ops pass central-difference tests (`tests/numerical/test_gradients.py`).

## Limitations

1. **Tiny dataset** — 4 samples, memorized easily.
2. **Scalar engine** — Python overhead; 5000 epochs takes ~3 seconds.
3. **No generalization test** — XOR is the training set.
4. **MSE loss** — Not ideal for classification (BCE would be better).
5. **Single seed** — Not statistically validated.

## Reproduction

```bash
python scripts/train_autodiff.py
# Results in artifacts/experiments/autodiff_xor_<timestamp>/
```

## References

- `docs/mathematics/autodiff.md` for derivations.
- `src/from_zero/autodiff/` for implementation.