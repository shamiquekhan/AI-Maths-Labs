"""Strict from-scratch engine laboratory page."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st  # noqa: E402

from from_zero.app import common  # noqa: E402
from from_zero.app.services import (  # noqa: E402
    run_strict_diffusion_demo,
    run_strict_gpt_demo,
    run_strict_logistic_demo,
    strict_gradient_report,
)

common.setup_page("Strict Engine", "∑")
common.sidebar()

common.hero(
    "No NumPy. No PyTorch. Just lists and calculus.",
    "The strict core implements storage, strides, broadcasting, reverse-mode autograd, "
    "layers, losses, and optimizers in plain Python. Every gradient on this page is "
    "checked against a central finite difference.",
    ("Python standard library", "Reverse-mode autograd", "Finite-difference checked", "CPU only"),
)

learn, build, visualise, evidence = st.tabs(["Learn", "Build", "Visualise", "Evidence"])

with learn:
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("### A tensor is a list with a map")
        st.markdown(
            "Storage is one flat `list[float]`. A shape, row-major strides, and an offset "
            "turn that list into an n-dimensional view. Transposing swaps strides without "
            "moving data; reshaping a contiguous tensor only changes the shape metadata."
        )
        st.latex(r"\text{offset}(i_1,\dots,i_n) = o + \sum_{k=1}^{n} i_k s_k")
        st.markdown(
            "Broadcasting aligns shapes from the right. Dimensions of size one stretch "
            "without copying, and the backward pass sums the gradient back down to the "
            "original shape."
        )
        st.markdown("### The graph")
        st.markdown(
            "Every differentiable operation returns a new tensor and records a closure "
            "that maps the upstream gradient to local gradients. `backward()` visits nodes "
            "in reverse topological order and accumulates gradients for tensors that "
            "appear on more than one path."
        )
        st.latex(
            r"\frac{\partial L}{\partial x} = \sum_{p \in \mathrm{paths}(x \to L)} \prod_{(u,v)\in p} \frac{\partial v}{\partial u}"
        )
        st.markdown("### Optimizers")
        st.latex(r"w \leftarrow w - \eta\, \frac{m_t}{\sqrt{v_t}+\epsilon}")
        st.markdown(
            "SGD, momentum, AdaGrad, RMSprop, Adam, and AdamW update parameter storage "
            "in place. State is plain lists, so checkpoints are JSON, never pickle."
        )
    with right:
        common.code_block(
            "from from_zero.tensor import tensor\n"
            "from from_zero.nn_tensor import Linear, ReLU, Sequential\n"
            "from from_zero.optim_tensor import Adam\n"
            "\n"
            "model = Sequential(Linear(2, 8), ReLU(), Linear(8, 1))\n"
            "optimizer = Adam(model.parameters(), lr=0.05)\n"
            "\n"
            "loss = ((model(x) - y) ** 2).mean()\n"
            "loss.backward()\n"
            "optimizer.step()\n"
            "optimizer.zero_grad()",
            language="python",
        )
        common.callout(
            "The policy is tested, not just documented.",
            "An import audit parses every strict-core module and fails the suite if "
            "NumPy, SciPy, PyTorch, TensorFlow, Keras, JAX, or scikit-learn appears. A "
            "subprocess test also confirms that importing the core loads none of them.",
        )

with build:
    st.markdown("### Gradient check")
    st.markdown(
        "Analytic gradients from the custom engine versus central finite differences "
        "with $h=10^{-6}$. The table is computed on page load; it is cheap."
    )
    report = strict_gradient_report()
    st.dataframe(
        [
            {
                "expression": row["case"],
                "analytic": [round(value, 6) for value in row["analytic"]],
                "finite difference": [round(value, 6) for value in row["finite_difference"]],
                "max error": f"{row['max_error']:.2e}",
            }
            for row in report
        ],
        hide_index=True,
        width="stretch",
    )
    worst = max(row["max_error"] for row in report)
    st.caption(f"Worst absolute error: {worst:.2e}")

    st.divider()
    st.markdown("### Strict logistic regression")
    st.markdown(
        "A 160-sample synthetic dataset, full-batch training, and the custom "
        "BCE-with-logits loss. Training runs only when you press the button."
    )
    logistic_columns = st.columns([1, 1, 1, 1])
    with logistic_columns[0]:
        epochs = st.slider("Epochs", 20, 300, 150, step=10, key="strict_logistic_epochs")
    with logistic_columns[1]:
        learning_rate = st.slider(
            "Learning rate", 0.01, 1.0, 0.5, step=0.01, key="strict_logistic_lr"
        )
    with logistic_columns[2]:
        seed = st.slider("Seed", 0, 99, 7, key="strict_logistic_seed")
    with logistic_columns[3]:
        st.write("")
        if st.button("Train strict logistic", key="strict_logistic_button"):
            with st.spinner("Running the pure-Python forward and backward passes…"):
                st.session_state["strict_logistic"] = run_strict_logistic_demo(
                    epochs=epochs, learning_rate=learning_rate, seed=seed
                )
    result = st.session_state.get("strict_logistic")
    if result:
        metrics = st.columns(4)
        metrics[0].metric("Training accuracy", f"{result['accuracy']:.1%}")
        metrics[1].metric("Precision", f"{result['precision']:.3f}")
        metrics[2].metric("Recall", f"{result['recall']:.3f}")
        metrics[3].metric("Final loss", f"{result['losses'][-1]:.4f}")
        st.line_chart({"training loss": result["losses"]})
        st.caption(
            f"Learned weights: {[round(value, 3) for value in result['weights']]}, "
            f"bias {result['bias']:.3f}. Engine: {result['engine']}."
        )
    else:
        st.info("Press **Train strict logistic** to run the bounded demo.")

    st.divider()
    st.markdown("### Strict character GPT")
    st.markdown(
        "A one-layer, two-head transformer with learned token and position embeddings. "
        "The corpus is tiny, so this demonstrates the pipeline and the causal mask, not "
        "language quality."
    )
    gpt_columns = st.columns([1, 1, 2])
    with gpt_columns[0]:
        gpt_steps = st.slider("Optimizer steps", 4, 40, 12, step=2, key="strict_gpt_steps")
    with gpt_columns[1]:
        gpt_seed = st.slider("Seed", 0, 99, 3, key="strict_gpt_seed")
    with gpt_columns[2]:
        st.write("")
        if st.button("Train strict GPT", key="strict_gpt_button"):
            with st.spinner("Running causal attention and backprop in pure Python…"):
                st.session_state["strict_gpt"] = run_strict_gpt_demo(steps=gpt_steps, seed=gpt_seed)
    gpt_result = st.session_state.get("strict_gpt")
    if gpt_result:
        columns = st.columns(3)
        columns[0].metric("Parameters", f"{gpt_result['parameters']:,}")
        columns[1].metric("Vocabulary", gpt_result["vocab_size"])
        columns[2].metric("Final loss", f"{gpt_result['losses'][-1]:.3f}")
        st.line_chart({"training loss": gpt_result["losses"]})
        st.code(gpt_result["sample"], language=None)
        st.caption(
            "The causal mask is verified by a test that changes a future token and "
            "asserts that earlier logits do not move."
        )
    else:
        st.info("Press **Train strict GPT** to run a few bounded steps.")

    st.divider()
    st.markdown("### Strict diffusion")
    st.markdown(
        "A linear beta schedule, the closed-form forward process, and a small MLP that "
        "predicts the added noise. The sampler walks the reverse chain one timestep at a "
        "time."
    )
    diffusion_columns = st.columns([1, 1, 2])
    with diffusion_columns[0]:
        diffusion_steps = st.slider(
            "Optimizer steps", 20, 120, 60, step=10, key="strict_diffusion_steps"
        )
    with diffusion_columns[1]:
        diffusion_seed = st.slider("Seed", 0, 99, 2, key="strict_diffusion_seed")
    with diffusion_columns[2]:
        st.write("")
        if st.button("Train strict diffusion", key="strict_diffusion_button"):
            with st.spinner("Training the denoiser and walking the reverse chain…"):
                st.session_state["strict_diffusion"] = run_strict_diffusion_demo(
                    steps=diffusion_steps, seed=diffusion_seed
                )
    diffusion_result = st.session_state.get("strict_diffusion")
    if diffusion_result:
        columns = st.columns(3)
        columns[0].metric("Parameters", f"{diffusion_result['parameters']:,}")
        columns[1].metric("Reverse steps", diffusion_result["frames"])
        columns[2].metric("Final loss", f"{diffusion_result['losses'][-1]:.3f}")
        st.line_chart({"training loss": diffusion_result["losses"]})
        st.markdown("Final sample, reshaped to $4\\times4$ (brighter is larger):")
        shades = " ·░▒▓█"
        lines = []
        for row in diffusion_result["grid"]:
            lines.append("".join(shades[min(len(shades) - 1, int(value * 5))] for value in row))
        st.code("\n".join(lines), language=None)
        st.caption(
            "Values are clamped to [0, 1] for display. This is a tiny pattern model, "
            "not a competitive image generator."
        )
    else:
        st.info("Press **Train strict diffusion** to run the bounded demo.")

with visualise:
    st.markdown("### Strides in motion")
    st.markdown(
        "The table below shows how the same six storage values are interpreted by "
        "different shape/stride pairs. No data moves between views."
    )
    st.dataframe(
        [
            {"view": "shape (2, 3)", "strides": "(3, 1)", "row 0": "1 2 3", "row 1": "4 5 6"},
            {"view": "transpose", "strides": "(1, 3)", "row 0": "1 4", "row 1": "2 5"},
            {"view": "reshape (3, 2)", "strides": "(2, 1)", "row 0": "1 2", "row 1": "3 4"},
        ],
        hide_index=True,
        width="stretch",
    )
    st.markdown("### Recorded losses")
    logistic_result = st.session_state.get("strict_logistic")
    gpt_result = st.session_state.get("strict_gpt")
    diffusion_result = st.session_state.get("strict_diffusion")
    if logistic_result:
        st.markdown("**Strict logistic**")
        st.line_chart({"loss": logistic_result["losses"]})
    if gpt_result:
        st.markdown("**Strict GPT**")
        st.line_chart({"loss": gpt_result["losses"]})
    if diffusion_result:
        st.markdown("**Strict diffusion**")
        st.line_chart({"loss": diffusion_result["losses"]})
    if not any((logistic_result, gpt_result, diffusion_result)):
        st.info("Run a demo in the **Build** tab to populate this view.")

with evidence:
    st.markdown("### What is verified")
    st.markdown(
        "- The strict tensor engine has unit tests for storage, strides, views, "
        "broadcasting, reductions, matmul, softmax, embeddings, and losses.\n"
        "- Numerical tests compare analytic gradients with central finite differences "
        "for every differentiable operation.\n"
        "- Optimizer tests check the first update by hand for SGD, momentum, AdaGrad, "
        "RMSprop, Adam, and AdamW.\n"
        "- An import audit rejects third-party numerical libraries in the strict core.\n"
        "- End-to-end tests train logistic regression, a character GPT, and an MLP "
        "diffusion model on the custom engine."
    )
    st.markdown("### Reproduce")
    common.code_block(
        "source .venv/bin/activate\n"
        "python -m pytest -q\n"
        "ruff check src tests examples scripts\n"
        "ruff format --check src tests examples scripts",
        language="bash",
    )
    st.markdown("### Honest limitations")
    st.markdown(
        "- Pure Python is CPU-only and much slower than NumPy or PyTorch; the models "
        "are deliberately tiny.\n"
        "- The strict GPT is a character model trained for a few steps; it is a "
        "correctness demonstration, not an assistant.\n"
        "- The strict diffusion model uses an MLP denoiser on 4×4 patterns, not a U-Net "
        "on MNIST.\n"
        "- The hybrid transformer and diffusion laboratories remain available and are "
        "labelled as PyTorch implementations."
    )
