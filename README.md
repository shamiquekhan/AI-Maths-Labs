# FROM ZERO: A First-Principles AI Mathematics and Generative Modeling Lab

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

An educational platform that builds the mathematical path from logistic regression to optimization, automatic differentiation, transformers, and diffusion models—implemented from first principles where it matters, with honest scope and reproducible experiments.

## What this project is

- A **strict from-scratch tensor engine** in pure Python: flat-list storage, shapes and strides, broadcasting, batched matrix multiplication, reverse-mode autograd, layers, losses, initializers, and six optimizers. No NumPy, SciPy, PyTorch, TensorFlow, Keras, JAX, or scikit-learn in the core; an import audit enforces the policy.
- Strict **logistic regression, character GPT, and diffusion** pipelines built on that engine, with JSON checkpoints and finite-difference gradient checks.
- A **NumPy implementation of logistic regression** with stable sigmoid, analytical gradients, and finite-difference checks.
- A **small NumPy optimizer library** (SGD, Momentum, RMSProp, Adam, AdamW) with learning-rate schedules and controlled comparisons.
- A **scalar reverse-mode autodiff engine** with explicit graph traversal, gradient accumulation, and a tiny MLP that learns XOR.
- A **PyTorch MiniGPT** (character-level, 4-layer, 128-dim) trained on Tiny Shakespeare with causal attention tests.
- A **PyTorch DDPM** with linear/cosine schedules, a compact U-Net, and MNIST generation.
- A **Streamlit dashboard** with ten pages of theory, visualisations, bounded experiment controls, and a torch-free public mode.
- **Unit, numerical, and integration tests** (190 tests passing) with gradient checks, causal-mask verification, and AppTest coverage of every page.
- **Experiment tracking** with YAML configs, JSON/CSV metrics, plots, checkpoints, and notes.

## What this project is not

- Not a production ML library.
- Not a benchmark leaderboard.
- Not a general-purpose chatbot or image generator.
- Not a substitute for a course in probability, linear algebra, and optimization.
- Not a claim that toy-model validation loss transfers to real applications.

> **Honest scope statement:** The strict tensor engine, autograd, layers, optimizers, and the strict logistic/GPT/diffusion pipelines use only the Python standard library. The logistic regression, optimizer comparisons, and scalar autodiff labs are NumPy implementations. The hybrid MiniGPT and DDPM labs are PyTorch implementations, clearly labelled in the dashboard. The public Streamlit deployment omits PyTorch and shows an installation note on those two pages.

## Quick start

```bash
# 1. Create environment
python -m venv .venv && source .venv/bin/activate

# 2. Install with all extras
pip install -e ".[dev,deep-learning,app]"

# 3. Run tests
pytest

# 4. Run the strict pure-Python examples
python examples/strict_tensor_example.py
python examples/strict_logistic_example.py
python examples/strict_gpt_example.py
python examples/strict_diffusion_example.py

# 5. Train each module (small, fast runs)
python scripts/train_logistic.py
python scripts/train_optimizers.py
python scripts/train_autodiff.py
python scripts/train_gpt.py
python scripts/train_diffusion.py

# 6. Launch the dashboard
streamlit run src/from_zero/app/Home.py
```

### Torch-free installation

The public dashboard runs without PyTorch. To reproduce that environment:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
pytest tests/unit/test_strict_support.py tests/integration/test_app_pages.py
streamlit run src/from_zero/app/Home.py
```

The MiniGPT and Diffusion pages show a clear fallback; the strict engine and
NumPy laboratories remain available.

### Reproduce the documented experiments

```bash
# Logistic regression (400 epochs, linearly separable data)
python scripts/train_logistic.py --epochs 400

# Optimizer comparison (1000 steps per objective)
python scripts/train_optimizers.py

# Autodiff XOR (trains to ~0.02 loss)
python scripts/train_autodiff.py

# MiniGPT on Tiny Shakespeare (3000 steps)
python scripts/train_gpt.py

# DDPM on MNIST (3000 steps, ~2 min on GPU)
python scripts/train_diffusion.py
```

Results appear in `artifacts/experiments/<name>/` with config, metrics, plots, and checkpoints.

## Learning path

| Module | Core question | Implementation | Key test |
|--------|---------------|----------------|----------|
| 00 Strict engine | How do storage, strides, broadcasting, and autograd actually work? | Pure Python | Finite-difference checks for every op; import audit |
| 01 Logistic | What is a prediction, loss, and gradient? | NumPy (and strict) | Analytical vs. autograd vs. finite difference |
| 02 Optimizers | How should parameters move using gradients? | NumPy (and strict) | Hand-checked first update; Rosenbrock comparison |
| 03 Autodiff | How can gradients be computed through a graph? | NumPy scalars | Numerical vs. autodiff on shared subexpressions |
| 04 MiniGPT | How do differentiable components learn sequences? | PyTorch (strict variant available) | Causal mask: future tokens don't leak |
| 05 Diffusion | How does a net learn probabilistic denoising? | PyTorch (strict variant available) | Forward/reverse schedule identities |

## Repository structure

```
from-zero/
├── configs/              # YAML experiment configurations
├── src/from_zero/
│   ├── core/             # Seeds, validation, metrics, datasets, plotting, experiments
│   ├── tensor/           # Strict flat-storage tensor + reverse-mode autograd
│   ├── autograd_tensor/  # Graph inspection and finite-difference gradcheck
│   ├── nn_tensor/        # Strict layers, activations, losses, initializers
│   ├── optim_tensor/     # Strict SGD, momentum, AdaGrad, RMSprop, Adam, AdamW
│   ├── strict/           # Strict logistic, char GPT, diffusion, JSON checkpoints
│   ├── logistic/         # NumPy logistic regression
│   ├── optim/            # NumPy optimizers + comparison
│   ├── autodiff/         # Scalar reverse-mode engine
│   ├── gpt/              # PyTorch MiniGPT
│   ├── diffusion/        # PyTorch DDPM
│   └── app/              # Streamlit dashboard
├── tests/
│   ├── unit/             # Component tests + strict import audit
│   ├── numerical/        # Gradient checks
│   └── integration/      # End-to-end pipelines + AppTest
├── notebooks/            # Guided explorations
├── docs/
│   ├── mathematics/      # Derivations per module
│   ├── architecture/     # System design
│   ├── experiments/      # Run reports
│   ├── decisions/        # ADRs
│   ├── release-evidence/ # Gate outputs, environment, deployment record
│   ├── deployment.md     # Streamlit Cloud guide
│   └── operations.md     # Runbook and release gates
├── scripts/              # CLI entry points + secret scanner
├── data/                 # Corpus + download notes (gitignored large files)
├── artifacts/            # Local run outputs (gitignored binaries)
└── reports/              # Experiment summaries
```

## Dashboard tour

| Page | Focus |
|------|-------|
| Overview | Module status, prerequisites, quick links |
| 01 Logistic Regression | Sigmoid, decision boundary, loss curve, training demo |
| 02 Optimizers | Contour trajectories, loss curves, comparison table |
| 03 Autodiff | Computation graph, backward pass, XOR training |
| 04 Mini GPT | Attention heatmap, generation, training demo (PyTorch) |
| 05 Diffusion | Noise schedule, forward noising, sampling demo (PyTorch) |
| 06 Experiments | Browse saved runs with config, metrics, plots, notes |
| 07 Mathematics | Compact equation reference for all modules |
| 08 About | Scope, reproducibility, data licenses, limitations |
| 09 Strict Engine | Tensor/autograd demos, strict logistic/GPT/diffusion, gradient checks |

## Documentation

- **Mathematics**: `docs/mathematics/{tensor_engine,logistic,optimization,autodiff,transformer,diffusion}.md`
- **Architecture**: `docs/architecture/system.md`
- **Decisions**: `docs/decisions/{001-numpy-foundations,002-scalar-autodiff,003-pytorch-deep,004-streamlit-ui,005-strict-core}.md`
- **Experiments**: `docs/experiments/*.md` (one per documented run, plus `strict_core.md`)
- **Deployment**: `docs/deployment.md`
- **Operations and release gates**: `docs/operations.md`
- **Release evidence**: `docs/release-evidence/README.md`

## Reproducibility

Every experiment records:
- Resolved configuration (`config.yaml`)
- Git commit, branch, dirty state
- Python, NumPy, SciPy, PyTorch, CUDA versions
- Random seed and device
- Metrics (JSON + CSV history)
- Plots (PNG) and checkpoints (`.pt`/`.npz`)
- Notes and **known limitations**

> **Determinism note:** GPU kernels and some PyTorch operations do not guarantee bitwise reproducibility. The recorded seed makes runs comparable, not necessarily bit-identical.

## Data and licenses

| Dataset | Source | License | Location |
|---------|--------|---------|----------|
| Tiny Shakespeare | Andrej Karpathy | Public domain | `data/corpora/tiny_shakespeare.txt` |
| MNIST | Yann LeCun et al. | Custom (attribution) | Downloaded by `torchvision` (gitignored) |
| Synthetic | Generated by code | MIT (code) | Generated on demand |

Code is MIT licensed. Dataset licenses remain with their providers.

## Compute expectations

| Module | Device | Typical time (small config) |
|--------|--------|-----------------------------|
| Strict engine (logistic/GPT/diffusion demos) | CPU | < 10 s |
| Logistic | CPU | < 1 s |
| Optimizers | CPU | < 2 s |
| Autodiff | CPU | < 5 s |
| MiniGPT | CPU/GPU | 30–60 s (3k steps) |
| Diffusion | GPU recommended | 2–3 min (3k steps on GTX 1650) |

All modules have CPU-compatible smoke-test configs. The strict engine is pure
Python and intentionally tiny; it is slower per parameter than NumPy or PyTorch.

## Limitations (read before citing)

- **Scale:** Every model is deliberately small. Results do not extrapolate to large-scale systems.
- **Evaluation:** Most metrics use synthetic or teaching datasets. No statistical significance testing.
- **Optimizer comparison:** Single seed, one learning rate per objective. Demonstrates dynamics, not stable ranking.
- **Strict engine:** Pure Python, CPU-only, and much slower than NumPy or PyTorch. The strict GPT is trained for tens of steps on a tiny corpus; the strict diffusion model is an MLP on 4×4 patterns, not a U-Net on MNIST.
- **MiniGPT:** Character-level next-token predictor. Not instruction-tuned, aligned, or factually grounded.
- **Diffusion:** 3,000 steps on 6,000 MNIST images. No FID or human evaluation. Samples show the reverse process works, not SOTA quality.
- **Autodiff:** The `autodiff` lab uses scalar values; the `tensor` package is the tensor-level engine. Both are educational, not production frameworks.

Each experiment directory contains its own `notes.md` with run-specific limitations.

## Contributing

See `CONTRIBUTING.md`. Before opening a PR:

```bash
ruff check src tests examples scripts
ruff format --check src tests examples scripts
pytest --cov=from_zero --cov-report=term-missing
python scripts/scan_secrets.py
```

New experiments should include a YAML config, a script, tests where appropriate, and a note about limitations. Release changes must also update `docs/release-evidence/`; see `docs/operations.md` for the G0–G9 gate list.

## References

- Goodfellow, Bengio & Courville, *Deep Learning* (2016)
- Prince, *Understanding Deep Learning* (2023)
- Karpathy, *micrograd* and *nanoGPT*
- Vaswani et al., *Attention Is All You Need* (2017)
- Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models* (2020)
- Nichol & Dhariwal, *Improved Denoising Diffusion Probabilistic Models* (2021)

## License

MIT License — see `LICENSE` for details.