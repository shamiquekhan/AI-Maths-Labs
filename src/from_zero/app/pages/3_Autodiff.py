"""Scalar automatic differentiation laboratory page."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st  # noqa: E402

from from_zero.app import common  # noqa: E402
from from_zero.app.services import run_autodiff_demo  # noqa: E402
from from_zero.autodiff import Value, draw_graph, graph_to_dict  # noqa: E402

common.setup_page("Autodiff", "∂")
common.sidebar()

common.hero(
    "Reverse-mode autodiff, one scalar at a time.",
    "Each Value stores a number, a gradient, its parents, and a local backward rule. "
    "A reverse topological traversal applies the chain rule and accumulates gradients "
    "at every shared subexpression.",
    ("Pure Python", "Reverse mode", "Gradient accumulation", "XOR"),
)

learn, build, visualise, evidence = st.tabs(["Learn", "Build", "Visualise", "Evidence"])


def _save_graph(root: Value) -> str:
    """Render the inspected graph to a temporary artifact used by the dashboard."""

    destination = Path("artifacts/experiments/ui_autodiff_graph.png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = draw_graph(root, title="Inspected scalar computation graph")
    figure.savefig(destination, dpi=150, bbox_inches="tight", facecolor="white")
    return str(destination)


with learn:
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("### A graph, not a formula string")
        st.markdown(
            "For $f = (a b + c) \\cdot f_0$, the engine records intermediate values "
            "and the local derivative of each operation. The backward pass starts with "
            "$\\partial f / \\partial f = 1$ and walks the graph in reverse topological order."
        )
        st.latex(
            r"\frac{\partial f}{\partial a} = \frac{\partial f}{\partial (ab+c)}"
            r"\frac{\partial (ab+c)}{\partial (ab)}\frac{\partial (ab)}{\partial a}"
        )
        st.markdown("### Local rules")
        st.markdown(
            "| operation | forward | local derivative |\n"
            "|---|---|---|\n"
            "| `a + b` | $a+b$ | $1, 1$ |\n"
            "| `a * b` | $ab$ | $b, a$ |\n"
            "| `tanh(a)` | $\\tanh a$ | $1-\\tanh^2 a$ |\n"
            "| `exp(a)` | $e^a$ | $e^a$ |\n"
            "| `log(a)` | $\\log a$ | $1/a$ |\n"
            "| `relu(a)` | $\\max(0,a)$ | $1$ if $a>0$ else $0$ |"
        )
        st.markdown("### Why accumulation matters")
        st.markdown(
            "If a node is used twice, its gradient receives two contributions. The engine "
            "uses `+=` in every backward closure. Intermediate gradients are reset at the "
            "start of each `backward()` call; leaf gradients are deliberately accumulated "
            "until `zero_grad()` is called."
        )
    with right:
        common.callout(
            "This is a scalar teaching engine.",
            "Every operation allocates a Python object. It is excellent for seeing the "
            "chain rule and hopeless for a real tensor workload. That is why the transformer "
            "and diffusion labs use PyTorch autograd.",
        )
        st.markdown("### What the XOR experiment shows")
        st.markdown(
            "- A hidden layer changes the representable function class.\n"
            "- A linear model cannot solve XOR, but a small tanh MLP can.\n"
            "- Training loss alone is not enough; the lab also checks all four predictions.\n"
            "- Success on four points is a demonstration, not a generalization result."
        )
        st.code(
            "a = Value(2.0)\nb = Value(-3.0)\nf = (a * b + Value(10.0)) * Value(-2.0)\n"
            "f.backward()\nprint(a.grad)  # -2.0 * b = 6.0",
            "python",
        )

with build:
    st.markdown("### Inspect a graph")
    expression = st.selectbox(
        "Preset expression",
        [
            "shared subexpression: (a*b + c) * f",
            "tanh polynomial: tanh(x**3 + 2x)",
            "sigmoid neuron: sigmoid(w*x + b)",
        ],
    )
    if expression.startswith("shared"):
        a, b, c = Value(2.0), Value(-3.0), Value(10.0)
        f0 = Value(-2.0)
        root = (a * b + c) * f0
        description = "a = 2, b = -3, c = 10, f = -2"
    elif expression.startswith("tanh"):
        x = Value(0.7)
        root = (x**3 + 2.0 * x).tanh()
        description = "x = 0.7"
    else:
        w, x, b = Value(-1.5), Value(0.8), Value(0.25)
        root = (w * x + b).sigmoid()
        description = "w = -1.5, x = 0.8, b = 0.25"
    root.backward()
    payload = graph_to_dict(root)
    st.caption(description)
    col1, col2 = st.columns([1.35, 1])
    with col1:
        common.figure(
            _save_graph(root),
            f"Forward value {root.data:.4f}; gradient {root.grad:.4f}",
        )
    with col2:
        st.markdown("#### Node table")
        st.dataframe(
            [
                {
                    "id": node["id"],
                    "op": node["op"],
                    "value": round(node["data"], 5),
                    "grad": round(node["grad"], 5),
                    "parents": node["parents"],
                }
                for node in payload["nodes"]
            ],
            hide_index=True,
            width="stretch",
        )

    st.divider()
    st.markdown("### Train the XOR MLP")
    with st.form("autodiff_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            hidden = st.selectbox("Hidden layers", ["(4,)", "(4, 4)", "(8,)", "(8, 8)"], index=1)
            epochs = st.slider("Maximum epochs", 200, 20_000, 6000, step=200)
        with col2:
            learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.2, step=0.01)
            seed = st.number_input("Seed", min_value=0, max_value=10_000, value=42)
        with col3:
            st.caption(
                "The run stops early when loss ≤ 0.02 and all four XOR predictions are correct."
            )
        submitted = st.form_submit_button("Train scalar MLP", type="primary")

    if submitted:
        hidden_sizes = tuple(
            int(item.strip()) for item in hidden.strip("()").split(",") if item.strip()
        )
        with st.spinner("Running the scalar backward pass…"):
            try:
                st.session_state["autodiff_run"] = run_autodiff_demo(
                    epochs=int(epochs),
                    learning_rate=float(learning_rate),
                    hidden_sizes=hidden_sizes,
                    seed=int(seed),
                )
            except Exception as error:
                st.error(f"Training failed: {error}")

    run = st.session_state.get("autodiff_run")
    if run is not None:
        st.success(f"Finished in {run.result.directory}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final loss", f"{run.metrics['final_loss']:.4f}")
        col2.metric("XOR accuracy", f"{run.metrics['accuracy']:.0%}")
        col3.metric("Epochs run", str(run.metrics["epochs_run"]))
        col4.metric("Parameters", str(run.metrics["parameter_count"]))

with visualise:
    run = st.session_state.get("autodiff_run")
    source = "Dashboard run" if run is not None else "Saved baseline run"
    st.caption(f"Showing: **{source}**")
    if run is not None:
        artifacts = run.artifacts
        predictions = run.metrics.get("final_predictions", [])
    else:
        artifacts = {
            "xor_data": "artifacts/experiments/autodiff_xor/plots/xor_data.png",
            "loss_curve": "artifacts/experiments/autodiff_xor/plots/xor_loss.png",
            "decision_boundary": "artifacts/experiments/autodiff_xor/plots/xor_decision_boundary.png",
            "computation_graph": "artifacts/experiments/autodiff_xor/plots/autodiff_graph.png",
        }
        metrics = common.read_json("artifacts/experiments/autodiff_xor/metrics.json")
        predictions = metrics.get("final_predictions", [])

    col1, col2 = st.columns(2)
    with col1:
        common.figure(artifacts.get("decision_boundary", ""), "Learned XOR decision surface")
        common.figure(artifacts.get("loss_curve", ""), "Loss trajectory")
    with col2:
        common.figure(artifacts.get("xor_data", ""), "XOR has no separating line")
        common.figure(artifacts.get("computation_graph", ""), "A small autodiff graph")

    if predictions:
        st.markdown("### Final XOR predictions")
        st.dataframe(predictions, hide_index=True, width="stretch")

with evidence:
    st.markdown("### Saved XOR experiment")
    st.write(
        "`artifacts/experiments/autodiff_xor` contains the trained scalar MLP, its "
        "parameter dump, the loss history, a graph rendering, and the decision surface."
    )
    metrics = common.read_json("artifacts/experiments/autodiff_xor/metrics.json")
    if metrics:
        st.json(metrics, expanded=False)
    else:
        st.info("Run `python scripts/train_autodiff.py` to create the XOR artifacts.")
    notes = common.load_text("artifacts/experiments/autodiff_xor/notes.md")
    if notes:
        st.markdown("#### Notes")
        st.markdown(notes)

common.footer()
