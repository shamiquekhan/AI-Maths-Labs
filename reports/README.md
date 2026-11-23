# Experiment Reports

This directory contains generated experiment reports and summaries.

## Contents

Reports are generated automatically by the training scripts and saved to `artifacts/experiments/<name>/notes.md`. This directory contains curated summaries for documentation.

## Report Template

Each experiment report includes:

1. **Configuration** — Resolved YAML config
2. **Dataset** — Source, splits, preprocessing
3. **Model** — Architecture, parameters
4. **Training** — Optimizer, schedule, hardware, time
5. **Results** — Metrics, curves, samples
6. **Visualizations** — Key plots
7. **Limitations** — Honest scope statement
8. **Reproduction** — Exact command to re-run

## Current Reports

See `docs/experiments/` for the full reports:

- `logistic_baseline.md` — Logistic regression on linearly separable data
- `optimizer_comparison.md` — SGD/Momentum/RMSProp/Adam on 3 objectives
- `autodiff_xor.md` — Scalar autodiff MLP on XOR
- `mini_gpt_shakespeare.md` — MiniGPT on Tiny Shakespeare
- `ddpm_mnist.md` — DDPM on MNIST

## Generating Reports

Reports are created automatically when you run:

```bash
python scripts/train_logistic.py
python scripts/train_optimizers.py
python scripts/train_autodiff.py
python scripts/train_gpt.py
python scripts/train_diffusion.py
```

The `ExperimentRecorder` in `src/from_zero/core/experiment.py` writes:
- `config.yaml`
- `metadata.json`
- `metrics.json`
- `metrics.csv`
- `notes.md`
- `plots/*.png`
- `samples/*`
- `checkpoint/*.pt`