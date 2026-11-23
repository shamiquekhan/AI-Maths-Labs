# System Architecture

## Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Application                     │
│  Home | 01 Logistic | 02 Optimizers | 03 Autodiff           │
│  04 MiniGPT | 05 Diffusion | 06 Experiments | 07 Mathematics│
│  08 About | 09 Strict Engine                                 │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Application Services                      │
│  bounded demos | experiment catalog | checkpoint discovery   │
└───────────────┬───────────────────────┬─────────────────────┘
                │                       │
┌───────────────▼─────────────┐ ┌───────▼──────────────────────┐
│         Shared Core         │ │         Model Modules        │
│  seeds   validation         │ │  strict  logistic  optim     │
│  metrics datasets           │ │  autodiff  gpt  diffusion    │
│  plotting experiment        │ │                              │
│  checkpointing paths        │ │                              │
└───────────────┬─────────────┘ └──────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────────────────┐
│  Standard library: strict tensor/autograd/layers/optimizers    │
│  NumPy: logistic, optimizer comparisons, scalar autodiff       │
│  PyTorch (optional): hybrid MiniGPT and DDPM                   │
└───────────────────────────────────────────────────────────────┘
```

**Key principle:** The UI calls services, services call model modules. No math in the UI. The strict core has no third-party numerical imports; the hybrid modules declare their backend explicitly.

---

## Package Structure

```
src/from_zero/
├── core/                    # Shared utilities (no ML deps)
│   ├── seeds.py             # set_seed() → Python, NumPy, Torch
│   ├── validation.py        # ConfigError, check_*, load/save YAML
│   ├── metrics.py           # Classification, perplexity, calibration
│   ├── datasets.py          # Synthetic data, splits, StandardScaler
│   ├── plotting.py          # Matplotlib figures (Agg backend)
│   ├── experiment.py        # ExperimentRecorder, metadata, CSV/JSON
│   ├── checkpointing.py     # save/load with metadata, torch.load(weights_only=True)
│   └── paths.py             # project_root(), artifact_dir()
│
├── tensor/                  # Strict engine: standard library only
│   ├── storage.py           # Flat storage, RNG, strides, broadcasting
│   ├── tensor.py            # Tensor, autograd, ops, losses, factories
│   └── __init__.py          # Public tensor API
│
├── autograd_tensor/         # Strict graph utilities
│   ├── engine.py            # topological_sort, graph inspection, gradcheck
│   └── gradcheck.py         # Numerical-gradient helper
│
├── nn_tensor/               # Strict neural-network building blocks
│   ├── module.py            # Parameter/Module registration, state_dict
│   ├── layers.py            # Linear, Embedding, LayerNorm, Dropout, Sequential
│   ├── activations.py       # ReLU, GELU, tanh, sigmoid
│   ├── losses.py            # MSE, BCE-with-logits, cross-entropy
│   └── initializers.py      # Xavier, He, uniform, normal
│
├── optim_tensor/            # Strict optimizers
│   ├── base.py              # In-place updates, state serialization, clipping
│   ├── sgd.py momentum.py adagrad.py rmsprop.py adam.py adamw.py
│   └── schedules.py         # Constant, step, exponential, cosine
│
├── strict/                  # Strict end-to-end pipelines
│   ├── data.py              # Synthetic classification and 4×4 patterns
│   ├── metrics.py           # Accuracy, precision, recall, F1
│   ├── training.py          # Trainer with optional probability_fn
│   ├── logistic.py          # LogisticRegression + analytical oracle
│   ├── gpt.py               # Char tokenizer, MiniGPT, causality, generation
│   ├── diffusion.py         # Schedule, MLP denoiser, reverse sampler
│   └── checkpoint.py        # Safe JSON checkpoints (from-zero-strict-1)
│
├── logistic/                # NumPy logistic regression
│   ├── model.py             # LogisticRegression class
│   ├── losses.py            # sigmoid, BCE, BCE-with-logits
│   ├── gradients.py         # compute_gradients, check_gradients
│   ├── training.py          # LogisticConfig, run_logistic_experiment
│   └── metrics.py           # (re-exports core.metrics)
│
├── optim/                   # NumPy optimizers
│   ├── base.py              # Optimizer ABC
│   ├── sgd.py               # SGD
│   ├── momentum.py          # Momentum
│   ├── rmsprop.py           # RMSProp
│   ├── adam.py              # Adam, AdamW
│   ├── schedules.py         # Schedule ABC + Constant/Step/Exp/Cosine
│   ├── benchmarks.py        # Objective, run_optimizer, comparison
│   └── compare.py           # OptimizerComparisonConfig, compare_optimizers
│
├── autodiff/                # Scalar reverse-mode engine
│   ├── value.py             # Value class, ops, backward
│   ├── operations.py        # Softplus, BCE, MSE, mean
│   ├── graph.py             # topological_sort, draw_graph, graph_to_dict
│   ├── nn.py                # Neuron, Layer, MLP, Module, SGD
│   └── training.py          # AutodiffConfig, run_autodiff_experiment
│
├── gpt/                     # PyTorch MiniGPT
│   ├── tokenizer.py         # CharTokenizer (encode/decode/save/load)
│   ├── dataset.py           # TextCorpus, get_batch (shifted targets)
│   ├── embeddings.py        # TokenEmbedding, PositionalEmbedding
│   ├── attention.py         # CausalSelfAttention (multi-head)
│   ├── transformer.py       # Block (pre-LN), FeedForward
│   ├── model.py             # MiniGPT, generate(), num_parameters()
│   ├── generation.py        # generate_text (temp, top-k, greedy)
│   └── training.py          # GPTTrainingConfig, train_gpt, load_gpt_checkpoint
│
├── diffusion/               # PyTorch DDPM
│   ├── schedules.py         # DiffusionSchedule, make_schedule
│   ├── process.py           # q_sample, posterior_mean, predict_x0
│   ├── model.py             # UNet, UNetConfig, ResBlock, Attention
│   ├── losses.py            # diffusion_loss (MSE noise prediction)
│   ├── sampling.py          # p_sample, sample_ddpm
│   └── training.py          # DiffusionTrainingConfig, train_diffusion, load_diffusion_checkpoint
│
└── app/                     # Streamlit dashboard
    ├── Home.py              # Entry point
    ├── common.py            # setup_page, hero, figure, metric_cards, sidebar
    ├── services.py          # run_*_demo, experiment_catalog, find_checkpoints
    └── pages/
        ├── 1_Logistic_Regression.py
        ├── 2_Optimizers.py
        ├── 3_Autodiff.py
        ├── 4_Mini_GPT.py
        ├── 5_Diffusion.py
        ├── 6_Experiments.py
        ├── 7_Mathematics.py
        ├── 8_About.py
        └── 9_Strict_Engine.py
```

---

## Strict-core boundary

The strict packages are deliberately isolated from the hybrid ones:

| Package | Imports allowed | Purpose |
|---|---|---|
| `tensor` | `math`, `random`, `itertools`, `builtins`, typing | Storage, views, ops, autograd |
| `autograd_tensor` | `tensor` + standard library | Graph inspection, finite differences |
| `nn_tensor` | `tensor` + standard library | Layers, losses, initializers |
| `optim_tensor` | `tensor` + standard library | Optimizer update rules |
| `strict` | all strict packages + standard library | End-to-end models |

`tests/unit/test_strict_import_audit.py` parses every strict module and rejects
NumPy, SciPy, PyTorch, TensorFlow, Keras, JAX, and scikit-learn imports. A
subprocess test imports the strict packages with those libraries blocked and
confirms that none is loaded.

The hybrid modules (`logistic`, `optim`, `autodiff`, `gpt`, `diffusion`) keep
their existing NumPy/PyTorch backends. They are not presented as from-scratch
implementations in the UI.

---

## Data Flow

### Experiment Run (CLI)

```
scripts/train_*.py
    │
    ▼
Config (YAML) ──▶ ConfigClass.from_dict()
    │
    ▼
run_*_experiment(config)
    │
    ├──▶ ExperimentRecorder(directory)
    │       ├──▶ save_config(config)
    │       ├──▶ log(metrics, step)
    │       ├──▶ save_figure(fig, name)
    │       └──▶ save_checkpoint(state, name)
    │
    ├──▶ Model training (core loop)
    │
    └──▶ Returns ExperimentResult(directory, metrics, model, ...)
```

### Dashboard Experiment

```
User clicks "Run" on page
    │
    ▼
services.run_*_demo(params)
    │
    ├──▶ Creates bounded config (max steps, small batch)
    ├──▶ Runs training with progress callback
    ├──▶ Stores result in st.session_state
    └──▶ Returns metrics, plots, samples
```

---

## Configuration System

- **YAML files** in `configs/` for documented runs.
- **Dataclasses** with `from_dict()` for validation.
- **Deep merge** for CLI overrides (not yet implemented; YAML is source of truth).
- **Rejection** of unknown keys (strict schema).

Example:
```yaml
# configs/logistic_regression.yaml
experiment:
  name: logistic_baseline
  seed: 42
data:
  dataset: linearly_separable
  samples: 800
model:
  learning_rate: 0.08
  epochs: 500
output:
  directory: artifacts/experiments
  make_plots: true
```

---

## Experiment Artifacts

Every run produces:
```
artifacts/experiments/<name>/
├── config.yaml          # Resolved configuration
├── metadata.json        # Git, env, seed, timestamps
├── metrics.json         # Final metrics (JSON-serializable)
├── metrics.csv          # Step-by-step history
├── notes.md             # Human-readable summary + limitations
├── plots/               # PNG figures
├── samples/             # Generated text, images, npz
└── checkpoint/          # .pt or .npz (gitignored)
```

---

## Checkpointing

- **Logistic/Autodiff:** `np.savez` (weights, bias, config).
- **GPT/Diffusion:** `torch.save` with `weights_only=True` for safety.
- **Metadata stored in checkpoint:** `model_family`, `config`, `step`, `best_metric`, `seed`.
- **Discovery:** `services.find_checkpoints(model_family)` filters by metadata.

---

## Plotting

- **Backend:** Matplotlib `Agg` (non-interactive, works headless).
- **Style:** Consistent colors, labeled axes, explanatory titles.
- **Output:** `ExperimentRecorder.save_figure(fig, name)` → `plots/<name>.png`.
- **No Plotly** in saved artifacts (static PNG for reproducibility).

---

## Testing Layers

| Layer | Location | Focus |
|-------|----------|-------|
| Unit | `tests/unit/` | Individual functions, shapes, validation |
| Numerical | `tests/numerical/` | Finite-difference gradients, math identities |
| Integration | `tests/integration/` | End-to-end pipelines (tiny runs) |
| UI Smoke | `tests/test_app_smoke.py` | Streamlit page loads, no exceptions |

Run: `pytest` (all layers).

---

## Reproducibility

- `core.seeds.set_seed(seed, deterministic=True)` sets Python, NumPy, Torch, CUDA.
- Metadata records: seed, versions, Git commit, device, CUDA.
- **Warning:** GPU non-determinism may cause bitwise differences across runs/hardware.

---

## Dependency Boundaries

| Package | Used by | Not used by |
|---------|---------|-------------|
| NumPy | core, logistic, optim, autodiff | gpt, diffusion |
| PyTorch | gpt, diffusion | core, logistic, optim, autodiff |
| SciPy | core (special functions) | — |
| Streamlit | app | core, logistic, optim, autodiff, gpt, diffusion |
| Matplotlib | core.plotting, app | — |
| Plotly | app (interactive only) | — |

This ensures the NumPy modules can run without PyTorch installed.

---

## Extension Points

To add a new module:
1. Create `src/from_zero/newmodule/` with `model.py`, `training.py`, etc.
2. Add config class with `from_dict()`.
3. Add `run_newmodule_experiment()` returning `ExperimentResult`.
4. Add page in `app/pages/`.
5. Add tests in `tests/unit/`, `tests/numerical/`, `tests/integration/`.
6. Document math in `docs/mathematics/newmodule.md`.