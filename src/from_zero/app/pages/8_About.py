"""About page: scope, reproducibility, licensing, and limitations."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st  # noqa: E402

from from_zero.app import common  # noqa: E402

common.setup_page("About", "◈")
common.sidebar()

common.hero(
    "A laboratory, not a product.",
    "FROM ZERO is a reproducible teaching repository about the mathematics that makes "
    "modern machine learning work. It is designed to be read, run, tested, and criticized.",
    ("CC BY 4.0", "Seeded experiments", "CPU-first", "Documented limits"),
)

scope, reproducibility, data, limitations, contributing = st.tabs(
    ["Scope", "Reproducibility", "Data & licenses", "Limitations", "Contributing"]
)

with scope:
    left, right = st.columns(2)
    with left:
        st.markdown("### What this repository is")
        st.markdown(
            "- A strict from-scratch tensor engine: flat storage, strides, broadcasting, "
            "reverse-mode autograd, layers, losses, and six optimizers in pure Python.\n"
            "- Strict logistic, character GPT, and diffusion pipelines on that engine.\n"
            "- A NumPy implementation of logistic regression and five optimizers.\n"
            "- A scalar reverse-mode autodiff engine with explicit graph traversal.\n"
            "- A PyTorch decoder-only transformer trained on public-domain Shakespeare.\n"
            "- A PyTorch DDPM with a compact U-Net, trained on MNIST.\n"
            "- A Streamlit dashboard, tests, notebooks, and saved experiment artifacts.\n"
            "- Derivations and decision records that explain the choices."
        )
        st.markdown("### Design principles")
        st.markdown(
            "1. **Mathematics before UI.** Every model is implemented and tested before it "
            "appears in the dashboard.\n"
            "2. **Honest scope.** From-scratch components and PyTorch components are labeled "
            "clearly.\n"
            "3. **Small enough to inspect.** Toy models, small datasets, and short training "
            "budgets keep the whole pipeline readable.\n"
            "4. **Evidence over claims.** Each experiment saves config, metrics, history, "
            "plots, and notes."
        )
    with right:
        st.markdown("### What this repository is not")
        st.markdown(
            "- Not a production ML library.\n"
            "- Not a benchmark leaderboard.\n"
            "- Not a general-purpose chatbot or image generator.\n"
            "- Not a substitute for a course in probability, linear algebra, and optimization.\n"
            "- Not a claim that a toy model's validation loss transfers to real applications."
        )
        common.callout(
            "The most important number is not accuracy.",
            "On a separable synthetic dataset, 100% accuracy is easy. The interesting "
            "questions are whether the gradient is correct, whether the implementation "
            "reproduces, and what the model cannot do.",
        )
        st.markdown("### Repository map")
        st.code(
            "configs/                   YAML experiment configurations\n"
            "src/from_zero/core/        shared data, metrics, plotting, records\n"
            "src/from_zero/tensor/      strict flat-storage tensor + autograd\n"
            "src/from_zero/autograd_tensor/ graph inspection + gradcheck\n"
            "src/from_zero/nn_tensor/   strict layers, losses, initializers\n"
            "src/from_zero/optim_tensor/ strict optimizers and schedules\n"
            "src/from_zero/strict/      strict logistic, GPT, diffusion, checkpoints\n"
            "src/from_zero/logistic/    NumPy logistic regression\n"
            "src/from_zero/optim/       NumPy optimizers and benchmarks\n"
            "src/from_zero/autodiff/    scalar reverse-mode engine\n"
            "src/from_zero/gpt/         PyTorch MiniGPT (hybrid)\n"
            "src/from_zero/diffusion/   PyTorch DDPM (hybrid)\n"
            "src/from_zero/app/         Streamlit dashboard\n"
            "tests/                     unit, numerical, integration, AppTest\n"
            "docs/                      mathematics, architecture, decisions, release evidence\n"
            "notebooks/                 guided explorations\n"
            "scripts/                   experiment entry points + secret scanner\n"
            "artifacts/                 local run outputs (binaries ignored by git)",
            "text",
        )

with reproducibility:
    st.markdown("### Reproduce from a clean environment")
    common.code_block(
        """python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,deep-learning,app]"
ruff check src tests examples scripts
ruff format --check src tests examples scripts
pytest --cov=from_zero
python examples/strict_logistic_example.py
python examples/strict_gpt_example.py
python examples/strict_diffusion_example.py
python scripts/train_logistic.py
python scripts/train_gpt.py
streamlit run src/from_zero/app/Home.py""",
        "bash",
    )
    st.markdown("### What is recorded")
    st.markdown(
        "- The resolved configuration and a slugified run name.\n"
        "- Python, platform, NumPy, SciPy, PyTorch, CUDA, and GPU information.\n"
        "- Git commit, branch, and dirty state when available.\n"
        "- Metrics as JSON, history as CSV, plots as PNG, checkpoints as `.pt` or `.npz`.\n"
        "- A notes file and a list of known limitations."
    )
    st.markdown("### Determinism")
    st.markdown(
        "Every entry point calls a shared seeding helper for Python, NumPy, and PyTorch. "
        "GPU kernels can still be nondeterministic, and some PyTorch operations do not "
        "guarantee bitwise reproducibility. The recorded seed makes runs comparable, not "
        "necessarily bit-identical."
    )
    st.markdown("### Backends")
    from from_zero.app.services import torch_status

    torch_available, torch_reason = torch_status()
    if torch_available:
        import torch

        st.write(
            f"PyTorch {torch.__version__}. CUDA available: **{torch.cuda.is_available()}**"
            + (f", {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "")
        )
    else:
        st.write(
            "PyTorch is not installed in this deployment. The strict engine and NumPy "
            "laboratories are fully available; the hybrid MiniGPT and Diffusion pages "
            "show installation instructions."
        )
        st.caption(torch_reason)
    st.write("Strict tensor engine: **pure Python standard library, CPU only**.")

with data:
    st.markdown("### Datasets")
    st.markdown(
        "- **Tiny Shakespeare:** a public-domain concatenation of Shakespeare's works, "
        "distributed by Andrej Karpathy as a language-modeling corpus. The file is stored "
        "at `data/corpora/tiny_shakespeare.txt`.\n"
        "- **MNIST:** downloaded through `torchvision` for the diffusion lab. MNIST is "
        "distributed by its original authors; check the current terms before redistributing "
        "the raw data. The repository does not commit the downloaded archive.\n"
        "- **Synthetic data:** generated deterministically by the code for logistic "
        "regression, optimizer objectives, and the diffusion fallback."
    )
    st.markdown("### Code license")
    st.markdown(
        "The code in this repository is released under the Creative Commons "
        "Attribution 4.0 International License (CC BY 4.0). See `LICENSE`. "
        "Dataset licenses are separate and remain with their original providers."
    )
    st.markdown("### Artifact policy")
    st.markdown(
        "Datasets, checkpoints, and large generated artifacts are ignored by git. Small "
        "metrics and plots may be kept locally for inspection. A fresh clone can regenerate "
        "them with the scripts."
    )

with limitations:
    st.markdown("### Limitations that apply to the whole project")
    st.markdown(
        "- **Scale:** every model is deliberately small. Results do not extrapolate to "
        "large-scale systems.\n"
        "- **Evaluation:** most metrics are computed on small synthetic or teaching datasets. "
        "No statistical significance testing is performed.\n"
        "- **Baselines:** the optimizer comparison uses one learning rate per objective and "
        "one seed. It is a controlled demonstration, not a tuning study.\n"
        "- **Language model:** MiniGPT is a character-level next-token predictor. It is not "
        "instruction-tuned, aligned, or factually grounded.\n"
        "- **Diffusion:** the DDPM is trained for 3,000 steps on 6,000 MNIST images. No FID "
        "or human evaluation is reported.\n"
        "- **Strict engine:** pure Python is CPU-only and much slower than NumPy or "
        "PyTorch. The strict GPT and diffusion demos are deliberately tiny.\n"
        "- **Autodiff:** the `autodiff` lab is scalar; the `tensor` package is the "
        "tensor-level engine. Neither is a production framework.\n"
        "- **Hardware:** CUDA availability changes runtime, and GPU nondeterminism can change "
        "exact numbers."
    )
    common.callout(
        "Read the notes file in each experiment directory.",
        "Each run records its own known limitations alongside its metrics. That is the "
        "most reliable place to understand what a particular artifact does and does not show.",
        tone="warn",
    )

with contributing:
    st.markdown("### Contributing")
    st.markdown(
        "Contributions are welcome if they preserve the teaching goals: clear mathematics, "
        "reproducible experiments, and honest claims. Before opening a pull request, run "
        "`ruff check src tests examples scripts`, `ruff format --check src tests examples "
        "scripts`, `pytest --cov=from_zero`, and `python scripts/scan_secrets.py`."
    )
    st.markdown(
        "See `CONTRIBUTING.md` for the workflow and `CODE_OF_CONDUCT.md` for community "
        "expectations. New experiments should include a YAML config, a script, tests where "
        "appropriate, and a note about limitations."
    )
    st.markdown("### References and further reading")
    st.markdown(
        "- Goodfellow, Bengio & Courville, *Deep Learning* (2016).\n"
        "- Prince, *Understanding Deep Learning* (2023).\n"
        "- Karpathy, *micrograd* and *nanoGPT*.\n"
        "- Vaswani et al., *Attention Is All You Need* (2017).\n"
        "- Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models* (2020).\n"
        "- Nichol & Dhariwal, *Improved Denoising Diffusion Probabilistic Models* (2021)."
    )
    st.markdown("### Version")
    st.code("from-zero-ai-lab 0.1.0, CC BY 4.0, Python >= 3.10", "text")

common.footer()
