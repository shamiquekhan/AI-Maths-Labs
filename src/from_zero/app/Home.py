"""FROM ZERO dashboard entry point."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st  # noqa: E402

from from_zero.app import common  # noqa: E402
from from_zero.app.services import experiment_catalog  # noqa: E402

common.setup_page("Home", "∑")
common.sidebar()

common.hero(
    "Six laboratories. One honest path from equations to generative models.",
    "Start with a strict pure-Python tensor engine: storage, strides, broadcasting, "
    "autograd, layers, and optimizers with no numerical dependency. Continue through "
    "NumPy foundations and PyTorch transformers and diffusion. Every lab has runnable "
    "code, a mathematical explanation, tests, and saved evidence.",
    (
        "Strict pure-Python core",
        "NumPy foundations",
        "PyTorch where it matters",
        "CPU-first",
    ),
)

catalog = experiment_catalog()
gpt_metrics = common.read_json("artifacts/experiments/mini_gpt_shakespeare/metrics.json")
ddpm_metrics = common.read_json("artifacts/experiments/ddpm_mnist/metrics.json")
logistic_metrics = common.read_json("artifacts/experiments/logistic_baseline/metrics.json")

gpt_parameters = gpt_metrics.get("model", {}).get("parameters", "—")
ddpm_parameters = ddpm_metrics.get("model", {}).get("parameters", "—")
logistic_accuracy = logistic_metrics.get("test", {}).get("accuracy", "—")

common.metric_cards(
    [
        (
            "Learning labs",
            "6",
            "Strict engine, logistic regression, optimizers, autodiff, GPT, diffusion.",
        ),
        (
            "MiniGPT parameters",
            f"{gpt_parameters:,}" if isinstance(gpt_parameters, int) else str(gpt_parameters),
            "Character-level, 4 layers, 128 embedding dimensions.",
        ),
        (
            "DDPM parameters",
            f"{ddpm_parameters:,}" if isinstance(ddpm_parameters, int) else str(ddpm_parameters),
            "Compact U-Net trained on 6,000 MNIST images.",
        ),
        (
            "Saved experiments",
            str(len(catalog)),
            "Each directory contains config, metrics, plots, notes, and checkpoints.",
        ),
    ]
)

st.markdown("## Choose a laboratory")

columns = st.columns(2)
labs = [
    (
        "00 · Strict tensor engine",
        "Flat-list storage, row-major strides, broadcasting, reverse-mode autograd, "
        "layers, losses, and six optimizers in plain Python. Strict logistic, character "
        "GPT, and diffusion demos run without NumPy or PyTorch.",
        "pages/9_Strict_Engine.py",
        "Open strict engine",
    ),
    (
        "01 · Logistic regression",
        "Sigmoid, binary cross-entropy, analytical gradients, and a finite-difference check. "
        "Implemented in NumPy, including the optimizer loop.",
        "pages/1_Logistic_Regression.py",
        "Open logistic lab",
    ),
    (
        "02 · Optimizers",
        "SGD, momentum, RMSProp, Adam, and AdamW on quadratic, Rosenbrock, and logistic "
        "objectives. Update equations are explicit and tested.",
        "pages/2_Optimizers.py",
        "Open optimizer lab",
    ),
    (
        "03 · Autodiff",
        "A scalar reverse-mode engine: values, local backward rules, topological traversal, "
        "gradient accumulation, and a tiny MLP that learns XOR.",
        "pages/3_Autodiff.py",
        "Open autodiff lab",
    ),
    (
        "04 · MiniGPT",
        "A decoder-only transformer in PyTorch with explicit causal masking, shifted targets, "
        "temperature/top-k generation, and a reloadable checkpoint.",
        "pages/4_Mini_GPT.py",
        "Open transformer lab",
    ),
    (
        "05 · Diffusion",
        "A DDPM with linear and cosine schedules, a compact U-Net noise predictor, and the "
        "full reverse chain. Trained on MNIST with a synthetic fallback.",
        "pages/5_Diffusion.py",
        "Open diffusion lab",
    ),
    (
        "06 · Experiments",
        "Inspect every saved run: configuration, metrics, history, plots, notes, and "
        "downloadable evidence.",
        "pages/6_Experiments.py",
        "Browse evidence",
    ),
]
for index, (title, body, page, label) in enumerate(labs):
    with columns[index % 2], st.container(border=True):
        st.markdown(f"### {title}")
        st.write(body)
        st.page_link(page, label=label)

st.markdown("## What is built from scratch?")

st.markdown(
    """
| Component | Implementation | Why |
|---|---|---|
| Strict tensor engine | Pure Python | Storage, strides, broadcasting, autograd, layers, and optimizers should be inspectable end to end. |
| Strict logistic / GPT / diffusion | Pure Python | The same engine is exercised across three model families. |
| Logistic regression | NumPy | The gradient is short enough to derive and check by hand. |
| Optimizers | NumPy | Update rules should be visible, not hidden in a library. |
| Scalar autodiff | Pure Python scalars | Reverse-mode differentiation is easier to inspect one scalar at a time. |
| MiniGPT | PyTorch (hybrid) | Attention and autograd at tensor scale are the point of the exercise. |
| DDPM | PyTorch (hybrid) | Convolutional U-Nets and long reverse chains need tensor autograd. |
| Dashboard | Streamlit | The UI calls the same application services as the scripts. |
"""
)

left, right = st.columns([1.15, 1])
with left:
    st.markdown("## Reproduce the evidence")
    st.write(
        "Install the package, run the test suite, then execute any experiment script. "
        "Every run writes a self-describing directory under `artifacts/experiments/`."
    )
    common.code_block(
        """python -m pip install -e ".[dev,deep-learning,app]"
ruff check src tests examples scripts
pytest --cov=from_zero
python examples/strict_logistic_example.py
python examples/strict_gpt_example.py
python examples/strict_diffusion_example.py
python scripts/train_logistic.py
python scripts/train_gpt.py
streamlit run src/from_zero/app/Home.py""",
        "bash",
    )

with right:
    st.markdown("## Honest scope")
    common.callout(
        "This is a teaching laboratory, not a benchmark leaderboard.",
        "MiniGPT is a small character-level model trained on Tiny Shakespeare. "
        "The DDPM is a compact U-Net trained for a few thousand steps on MNIST. "
        "Neither is a general-purpose assistant or a state-of-the-art generator. "
        "The value is the visible path from equations to working code.",
        tone="warn",
    )
    st.markdown(
        "**Read next:** the [strict engine](pages/9_Strict_Engine.py) for the "
        "from-scratch core, the [mathematics page](pages/7_Mathematics.py) for "
        "derivations, and the [about page](pages/8_About.py) for scope and licensing."
    )

common.footer()
