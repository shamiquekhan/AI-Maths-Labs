"""Logistic regression laboratory page."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st  # noqa: E402

from from_zero.app import common  # noqa: E402
from from_zero.app.services import run_logistic_demo  # noqa: E402

common.setup_page("Logistic Regression", "σ")
common.sidebar()

common.hero(
    "Logistic regression, with every gradient visible.",
    "A linear score becomes a probability through the sigmoid. Binary cross-entropy "
    "gives a gradient simple enough to derive on paper, implement in NumPy, and verify "
    "against finite differences.",
    ("NumPy", "BCE from logits", "Gradient check", "Calibration"),
)

learn, build, visualise, evidence = st.tabs(["Learn", "Build", "Visualise", "Evidence"])

with learn:
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("### The model")
        st.markdown(
            "For features $x \\in \\mathbb{R}^d$, weights $w$, and bias $b$, the model "
            "computes a logit $z$ and a probability $p$:"
        )
        st.latex(r"z = w^\top x + b, \qquad p = \sigma(z) = \frac{1}{1+e^{-z}}")
        st.markdown(
            "The sigmoid is evaluated in a numerically stable form. For large positive "
            "$z$ it uses $1/(1+e^{-z})$; for negative $z$ it uses $e^z/(1+e^z)$."
        )
        st.markdown("### The loss")
        st.markdown(
            "Binary cross-entropy is computed from the logit, not from a clipped "
            "probability. This keeps the loss finite even when the model is confidently wrong."
        )
        st.latex(
            r"\mathcal{L}(w,b) = \frac{1}{N}\sum_{i=1}^{N}"
            r"\left[\log(1+e^{z_i}) - y_i z_i\right] + \frac{\lambda}{2}\|w\|_2^2"
        )
        st.markdown("### The gradient")
        st.latex(
            r"\frac{\partial \mathcal{L}}{\partial z_i} = \frac{1}{N}(\sigma(z_i)-y_i), "
            r"\qquad \nabla_w \mathcal{L} = X^\top \frac{\partial \mathcal{L}}{\partial z} + \lambda w"
        )
        st.markdown(
            "The implementation checks this analytical gradient against a central "
            "finite difference before training. The test suite repeats the check."
        )

    with right:
        st.markdown("### Why logits?")
        common.callout(
            "Probabilities are not the right place to compute the loss.",
            "Computing log(p) after sigmoid can underflow for very negative logits. "
            "The log-sum-exp form keeps the same mathematics while staying finite.",
        )
        st.markdown("### What this lab does not claim")
        st.markdown(
            "- A linear decision boundary cannot solve XOR or circles perfectly.\n"
            "- Perfect accuracy on a separable synthetic dataset is not evidence of "
            "real-world generalization.\n"
            "- Calibration is shown on the test split, but the dataset is small and synthetic."
        )
        st.markdown("### Files")
        st.code(
            "src/from_zero/logistic/model.py\n"
            "src/from_zero/logistic/losses.py\n"
            "src/from_zero/logistic/gradients.py\n"
            "src/from_zero/logistic/training.py",
            "text",
        )

with build:
    st.markdown("### Configure a bounded run")
    with st.form("logistic_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            dataset = st.selectbox(
                "Dataset",
                ["linearly_separable", "moons", "circles", "xor"],
                help="Moons, circles, and XOR are deliberately harder for a linear model.",
            )
            n_samples = st.slider("Samples", 100, 2000, 600, step=50)
            noise = st.slider("Noise", 0.0, 0.5, 0.15, step=0.01)
        with col2:
            epochs = st.slider("Epochs", 20, 1500, 400, step=20)
            learning_rate = st.slider("Learning rate", 0.001, 0.5, 0.05, step=0.001)
        with col3:
            seed = st.number_input("Seed", min_value=0, max_value=10_000, value=42)
            st.caption(
                "The run writes to `artifacts/experiments/ui_logistic_demo` and replaces "
                "the previous dashboard run with the same name."
            )
        submitted = st.form_submit_button("Train logistic regression", type="primary")

    if submitted:
        with st.spinner("Fitting the NumPy model and writing artifacts…"):
            try:
                st.session_state["logistic_run"] = run_logistic_demo(
                    dataset=dataset,
                    n_samples=int(n_samples),
                    noise=float(noise),
                    epochs=int(epochs),
                    learning_rate=float(learning_rate),
                    seed=int(seed),
                )
            except Exception as error:
                st.error(f"Training failed: {error}")

    run = st.session_state.get("logistic_run")
    if run is not None:
        st.success(f"Finished in {run.result.directory}")
        weights = (run.metrics.get("parameters") or {}).get("weights") or []
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Test accuracy", f"{run.test_eval['metrics']['accuracy']:.3f}")
        col2.metric("Test loss", f"{run.test_eval['metrics']['loss']:.4f}")
        col3.metric("ROC AUC", f"{run.test_eval['metrics'].get('roc_auc', float('nan')):.3f}")
        col4.metric("Parameters", str(len(weights) + 1))

with visualise:
    run = st.session_state.get("logistic_run")
    source = "Dashboard run" if run is not None else "Saved baseline experiment"
    st.caption(f"Showing: **{source}**")
    if run is not None:
        artifacts = run.artifacts
        test_eval = run.test_eval["metrics"]
    else:
        metrics = common.read_json("artifacts/experiments/logistic_baseline/metrics.json")
        test_eval = metrics.get("test", {})
        artifacts = {
            "decision_test": "artifacts/experiments/logistic_baseline/plots/decision_boundary_test.png",
            "loss_curve": "artifacts/experiments/logistic_baseline/plots/loss_curve.png",
            "calibration_test": "artifacts/experiments/logistic_baseline/plots/calibration_test.png",
            "confusion_test": "artifacts/experiments/logistic_baseline/plots/confusion_matrix_test.png",
            "sigmoid": "artifacts/experiments/logistic_baseline/plots/sigmoid.png",
        }

    col1, col2 = st.columns(2)
    with col1:
        common.figure(artifacts.get("decision_test", ""), "Decision boundary on held-out data")
        common.figure(artifacts.get("calibration_test", ""), "Reliability diagram")
    with col2:
        common.figure(artifacts.get("loss_curve", ""), "Training and validation loss")
        common.figure(artifacts.get("confusion_test", ""), "Confusion matrix")

    if test_eval:
        st.markdown("### Held-out metrics")
        st.dataframe(
            {
                "metric": ["accuracy", "precision", "recall", "F1", "ROC AUC", "log loss"],
                "value": [
                    test_eval.get("accuracy"),
                    test_eval.get("precision"),
                    test_eval.get("recall"),
                    test_eval.get("f1"),
                    test_eval.get("roc_auc"),
                    test_eval.get("loss"),
                ],
            },
            width="stretch",
            hide_index=True,
        )

with evidence:
    st.markdown("### Saved baseline")
    st.write(
        "`artifacts/experiments/logistic_baseline` was produced by "
        "`python scripts/train_logistic.py`. Its configuration, metrics, model, "
        "and plots are kept together so the result can be inspected without retraining."
    )
    metrics = common.read_json("artifacts/experiments/logistic_baseline/metrics.json")
    if metrics:
        training = metrics.get("training", {})
        st.json(
            {
                "dataset": metrics.get("dataset"),
                "parameters": metrics.get("parameters"),
                "training": training,
                "test": {
                    key: metrics.get("test", {}).get(key)
                    for key in ("accuracy", "loss", "f1", "roc_auc")
                },
            },
            expanded=False,
        )
    else:
        st.info("Run `python scripts/train_logistic.py` to create the baseline artifacts.")
    notes = common.load_text("artifacts/experiments/logistic_baseline/notes.md")
    if notes:
        st.markdown("#### Notes")
        st.markdown(notes)

common.footer()
