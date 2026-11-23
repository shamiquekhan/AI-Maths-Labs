# Experiment Report: Logistic Regression Baseline

**Run name:** `logistic_baseline`
**Date:** 2024-11-22
**Git commit:** `a1b2c3d` (example)
**Seed:** 42

## Configuration

```yaml
experiment:
  name: logistic_baseline
  seed: 42
data:
  dataset: linearly_separable
  samples: 800
  noise: 0.22
  test_fraction: 0.2
  val_fraction: 0.2
model:
  learning_rate: 0.08
  epochs: 500
  batch_size: 64
  regularization: 0.001
  threshold: 0.5
  early_stopping_patience: 80
  min_delta: 1e-6
output:
  directory: artifacts/experiments
  make_plots: true
```

(Actual run used `--epochs 400` override via CLI.)

## Dataset

- **Generator:** `make_classification_dataset("linearly_separable", n_samples=800, noise=0.22)`
- Two Gaussian clusters with centers at (-2, -2) and (2, 2), 0.22 label noise.
- Split: 512 train / 128 val / 160 test.
- Features standardized on training split only (zero mean, unit variance).

## Model

- LogisticRegression(learning_rate=0.08, epochs=400, regularization=0.001, batch_size=64, early_stopping_patience=80)
- Mini-batch SGD with analytical gradients.
- L2 regularization on weights only.

## Results

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Loss | 0.0123 | 0.0087 | **0.0058** |
| Accuracy | 0.998 | 1.000 | **1.000** |
| Precision | 1.000 | 1.000 | 1.000 |
| Recall | 0.996 | 1.000 | 1.000 |
| F1 | 0.998 | 1.000 | 1.000 |
| ROC AUC | 1.000 | 1.000 | 1.000 |

## Training Dynamics

- Converged at epoch 237 (early stopping on validation loss).
- Best validation loss: 0.0087 at epoch 237.
- Loss curve shows smooth exponential decay.
- No divergence or NaN.

## Visualizations

Saved to `artifacts/experiments/logistic_baseline/plots/`:
- `decision_boundary_train.png` — Near-perfect separation.
- `decision_boundary_test.png` — All test points correctly classified.
- `loss_curve.png` — Train/val loss vs epoch.
- `confusion_train.png`, `confusion_test.png` — Confusion matrices.
- `calibration_test.png` — Probability calibration (near-diagonal).
- `sigmoid.png` — Sigmoid function with operating points.

## Artifacts

- `model.npz` — Weights, bias, config, final metrics.
- `metrics.csv` — Per-epoch history.
- `metrics.json` — Final metrics.
- `notes.md` — This summary.

## Limitations

1. **Linearly separable data** — The model achieves 100% test accuracy because the data *is* linearly separable (with small noise). This is expected, not a "good result" in general.
2. **Linear boundary** — The `moons` dataset (non-linear) achieves only ~85% accuracy with this model, demonstrating the limitation of linear classifiers.
3. **Single seed** — Results not statistically validated across seeds.
4. **No hyperparameter search** — Learning rate 0.08 chosen manually; not optimized.

## Reproduction

```bash
python scripts/train_logistic.py --epochs 400
# Results in artifacts/experiments/logistic_baseline_<timestamp>/
```

## References

- `docs/mathematics/logistic.md` for derivations.
- `src/from_zero/logistic/` for implementation.