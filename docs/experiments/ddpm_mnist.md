# Experiment Report: DDPM on MNIST

**Run name:** `ddpm_mnist`
**Date:** 2024-11-22
**Git commit:** `a1b2c3d` (example)
**Seed:** 42

## Configuration

```yaml
experiment:
  name: ddpm_mnist
  seed: 42
data:
  dataset: mnist
  image_size: 28
  max_samples: 6000
  val_fraction: 0.1
diffusion:
  timesteps: 1000
  schedule: linear
  beta_start: 0.0001
  beta_end: 0.02
model:
  in_channels: 1
  base_channels: 32
  channel_mults: [1, 2, 4]
  num_res_blocks: 2
  dropout: 0.1
training:
  batch_size: 128
  learning_rate: 0.0002
  max_steps: 3000
  eval_interval: 250
  eval_batches: 10
  log_interval: 50
  weight_decay: 0.01
  grad_clip: 1.0
sampling:
  sample_count: 16
  sample_interval: 500
output:
  directory: artifacts/experiments
  make_plots: true
```

## Data

- **Dataset:** MNIST (via `torchvision.datasets.MNIST`).
- **Images:** 28×28 grayscale, normalized to [-1, 1].
- **Train:** 54,000 (from 60,000 with 10% val).
- **Val:** 6,000.
- **Download:** Automatic (cached in `data/MNIST/`, gitignored).

## Model

- **Architecture:** Compact U-Net.
- **Base channels:** 32
- **Multipliers:** 1, 2, 4 → 32, 64, 128 channels.
- **Res blocks:** 2 per resolution.
- **Attention:** Single-head spatial at bottleneck (128-ch).
- **Timestep embedding:** Sinusoidal (128) → MLP (512) → AdaGN.
- **Parameters:** 540,961
- **Weight decay:** 0.01 (AdamW)

## Diffusion

- **Schedule:** Linear (β₁=1e-4, βₜ=0.02)
- **Timesteps:** 1000
- **Objective:** Noise prediction (MSE)
- **Sampling:** DDPM reverse process (Algorithm 2 from Ho et al.)

## Training

- **Optimizer:** AdamW (lr=2e-4, β=(0.9, 0.999), eps=1e-8)
- **Grad clip:** 1.0
- **Steps:** 3000
- **Device:** CUDA (GTX 1650, 4 GB VRAM)
- **Time:** ~141 s total (~0.047 s/step)
- **Mixed precision:** No (FP32)

## Results

| Metric | Value |
|--------|-------|
| Best val loss | 0.02458 |
| Final train loss | 0.02412 |
| Step at best | 2750 |
| Train time | 141 s |

**Loss curve:** Steady decay, slight overfitting after step 2750.

## Sampling

- **Samples:** 16 (4×4 grid) every 500 steps.
- **Final samples:** Recognizable digits (0-9), some blurring.
- **Diversity:** Different noise seeds → different digit styles.

## Visualizations

Saved to `artifacts/experiments/ddpm_mnist/plots/`:
- `noise_schedule.png` — βₜ, αₜ, ᾱₜ curves.
- `forward_process.png` — x₀ noised at t=0, 250, 500, 750, 999.
- `loss_curve.png` — Train/val loss vs step.
- `samples_step_*.png` — Generated grids at intervals.

## Schedule & Forward Process Verification

Tests in `tests/numerical/test_diffusion_math.py`:
- ᾱₜ = ∏ αₛ ✓
- Posterior variance matches closed form ✓
- Empirical moments of q(xₜ|x₀) match theory ✓
- x₀/ε recovery consistent ✓
- t=0 boundary: posterior mean = x₀, var = 0 ✓

## Artifacts

- `checkpoint/best.pt` — Model, optimizer, schedule, config, step, metrics.
- `checkpoint/last.pt` — Final state.
- `samples/final_samples.npz` — 16 generated images (numpy).
- `metrics.csv` — Per-step history.

## Limitations

1. **Small scale** — 3k steps, 6k images, 540k params. Not comparable to published DDPM (100M+ params, 1M+ images, millions of steps).
2. **No FID** — Requires Inception features, reference batch, multiple seeds. Not implemented.
3. **MNIST only** — Simple digits; no complex texture/structure.
4. **Linear schedule only** — Cosine not tested in documented run.
5. **No classifier-free guidance** — Unconditional only.
6. **Single seed** — No error bars.
7. **FP32 only** — No mixed precision.

## Reproduction

```bash
python scripts/train_diffusion.py
# Results in artifacts/experiments/ddpm_mnist_<timestamp>/
```

## References

- `docs/mathematics/diffusion.md` for derivations.
- `src/from_zero/diffusion/` for implementation.
- Ho et al., *Denoising Diffusion Probabilistic Models* (2020)