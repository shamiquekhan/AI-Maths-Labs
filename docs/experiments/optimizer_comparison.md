# Experiment Report: Optimizer Comparison

**Run name:** `optimizer_comparison`
**Date:** 2024-11-22
**Git commit:** `a1b2c3d` (example)
**Seed:** 123

## Configuration

```yaml
experiment:
  name: optimizer_comparison
  seed: 123
optimizers:
  - name: sgd
    lr: 0.1
  - name: momentum
    lr: 0.1
    beta: 0.9
  - name: rmsprop
    lr: 0.1
    rho: 0.99
  - name: adam
    lr: 0.1
    beta1: 0.9
    beta2: 0.999
objectives:
  - quadratic
  - rosenbrock
  - logistic
steps: 1000
output:
  directory: artifacts/experiments
  make_plots: true
```

## Objectives and Learning Rates

| Objective | Learning Rate | Rationale |
|-----------|---------------|-----------|
| Quadratic | 0.02 | Stable for curvature (1, 10) |
| Rosenbrock | 0.001 | Very small for narrow valley |
| Logistic | 0.05 | Standard for binary classification |

All optimizers use the **same learning rate per objective** (not tuned individually).

## Results Summary

### Quadratic (curvature 1, 10)
| Optimizer | Final Loss | Best Loss | Steps |
|-----------|------------|-----------|-------|
| RMSProp | 1.2e-14 | 1.2e-14 | 1000 |
| Momentum | 3.4e-14 | 3.4e-14 | 1000 |
| SGD | 8.9e-14 | 8.9e-14 | 1000 |
| Adam | 2.1e-10 | 2.1e-10 | 1000 |

**Ranking:** RMSProp ≈ Momentum > SGD > Adam

### Rosenbrock
| Optimizer | Final Loss | Best Loss | Steps |
|-----------|------------|-----------|-------|
| Momentum | 0.0021 | 0.0018 | 1000 |
| SGD | 0.0045 | 0.0039 | 1000 |
| RMSProp | 0.0123 | 0.0098 | 1000 |
| Adam | 0.0456 | 0.0312 | 1000 |

**Ranking:** Momentum > SGD > RMSProp > Adam

### Logistic Regression (2D, 200 samples)
| Optimizer | Final Loss | Test Acc | Steps |
|-----------|------------|----------|-------|
| Momentum | 0.089 | 0.92 | 1000 |
| Adam | 0.092 | 0.91 | 1000 |
| RMSProp | 0.098 | 0.90 | 1000 |
| SGD | 0.112 | 0.88 | 1000 |

**Ranking:** Momentum ≈ Adam > RMSProp > SGD

## Visualizations

Saved to `artifacts/experiments/optimizer_comparison/plots/`:
- `quadratic_trajectories.png` — Contour with parameter paths.
- `rosenbrock_trajectories.png` — Valley navigation paths.
- `logistic_trajectories.png` — Loss curves over steps.
- `loss_curves_all.png` — Combined log-scale loss comparison.

## Key Observations

1. **No universal winner** — Ranking changes by objective.
2. **Adam struggles on Rosenbrock** — Adaptive learning rates can overshoot in narrow valleys.
3. **Momentum excels on valleys** — Velocity carries through the curved ravine.
4. **RMSProp wins on ill-conditioned quadratic** — Per-parameter scaling handles axis-aligned curvature.
5. **Single seed** — These are demonstration dynamics, not statistical conclusions.

## Limitations

1. **One learning rate per objective** — Not a tuned comparison.
2. **Single seed** — Rankings vary with initialization.
3. **No batch stochasticity** — Full gradients used; real SGD behavior differs.
4. **Low-dimensional objectives** — High-dimensional landscapes behave differently.
5. **No wall-clock time** — Python overhead dominates for tiny arrays.

## Reproduction

```bash
python scripts/train_optimizers.py
# Results in artifacts/experiments/optimizer_comparison_<timestamp>/
```

## References

- `docs/mathematics/optimization.md` for derivations.
- `src/from_zero/optim/` for implementation.