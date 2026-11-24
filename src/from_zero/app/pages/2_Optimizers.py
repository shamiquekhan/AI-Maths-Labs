"""Optimizer laboratory page."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st  # noqa: E402

from from_zero.app import common  # noqa: E402
from from_zero.app.services import run_optimizer_demo  # noqa: E402

common.setup_page("Optimizers", "↘")
common.sidebar()

common.hero(
    "Optimizers are update rules, not magic.",
    "The same gradients meet five different state machines. Momentum remembers direction, "
    "RMSProp rescales by recent magnitude, Adam combines both, and AdamW decouples weight decay.",
    ("NumPy", "SGD, Momentum, RMSProp, Adam, AdamW", "Rosenbrock", "Controlled budgets"),
)

learn, build, visualise, evidence = st.tabs(["Learn", "Build", "Visualise", "Evidence"])

with learn:
    left, right = st.columns(2)
    with left:
        st.markdown("### Stochastic gradient descent")
        st.latex(r"w_{t+1} = w_t - \eta \, g_t")
        st.markdown("### Momentum")
        st.latex(r"v_{t+1} = \beta v_t + g_t, \qquad w_{t+1} = w_t - \eta v_{t+1}")
        st.markdown("### RMSProp")
        st.latex(
            r"s_{t+1} = \rho s_t + (1-\rho) g_t^2, \qquad "
            r"w_{t+1} = w_t - \frac{\eta}{\sqrt{s_{t+1}}+\epsilon} g_t"
        )
    with right:
        st.markdown("### Adam")
        st.latex(
            r"m_{t+1} = \beta_1 m_t + (1-\beta_1) g_t, \qquad "
            r"v_{t+1} = \beta_2 v_t + (1-\beta_2) g_t^2"
        )
        st.latex(
            r"\hat m_{t+1} = \frac{m_{t+1}}{1-\beta_1^{t+1}}, \qquad "
            r"\hat v_{t+1} = \frac{v_{t+1}}{1-\beta_2^{t+1}}"
        )
        st.latex(r"w_{t+1} = w_t - \eta \frac{\hat m_{t+1}}{\sqrt{\hat v_{t+1}}+\epsilon}")
        st.markdown("### AdamW")
        st.latex(
            r"w_{t+1} = (1-\eta\lambda) w_t - \eta "
            r"\frac{\hat m_{t+1}}{\sqrt{\hat v_{t+1}}+\epsilon}"
        )

    st.markdown("### The three objectives")
    st.markdown(
        "- **Quadratic:** an axis-aligned bowl with curvature 1 and 10. The narrow direction "
        "shows why plain SGD can zig-zag.\n"
        "- **Rosenbrock:** a curved, non-convex valley. Progress requires following a narrow "
        "curved path rather than descending each coordinate independently.\n"
        "- **Logistic:** a small two-moons classification objective. It connects the optimizer "
        "lab back to the logistic-regression lab."
    )
    common.callout(
        "Read the ranking carefully.",
        "These are short runs on small teaching objectives with one learning rate per "
        "objective, not a per-algorithm hyperparameter search. A ranking here is not a claim "
        "about universal optimizer superiority.",
        tone="warn",
    )

with build:
    st.markdown("### Compare optimizers")
    with st.form("optimizer_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            steps = st.slider("Steps per optimizer", 50, 2000, 500, step=50)
            seed = st.number_input("Seed", min_value=0, max_value=10_000, value=42)
        with col2:
            optimizers = st.multiselect(
                "Optimizers",
                ["sgd", "momentum", "rmsprop", "adam", "adamw"],
                default=["sgd", "momentum", "rmsprop", "adam"],
            )
        with col3:
            objectives = st.multiselect(
                "Objectives",
                ["quadratic", "rosenbrock", "logistic"],
                default=["quadratic", "rosenbrock", "logistic"],
            )
        submitted = st.form_submit_button("Run comparison", type="primary")

    if submitted:
        if not optimizers or not objectives:
            st.error("Choose at least one optimizer and one objective.")
        else:
            with st.spinner("Running every optimizer on every objective…"):
                try:
                    st.session_state["optimizer_run"] = run_optimizer_demo(
                        steps=int(steps),
                        optimizers=tuple(optimizers),
                        objectives=tuple(objectives),
                        seed=int(seed),
                    )
                except Exception as error:
                    st.error(f"Comparison failed: {error}")

    run = st.session_state.get("optimizer_run")
    if run is not None:
        st.success(f"Finished in {run.result.directory}")
        for objective, optimizer_runs in run.runs.items():
            st.markdown(f"**{objective.title()}**")
            st.dataframe(
                {
                    "optimizer": list(optimizer_runs),
                    "final loss": [round(run_.final_loss, 6) for run_ in optimizer_runs.values()],
                    "best loss": [round(run_.best_loss, 6) for run_ in optimizer_runs.values()],
                    "steps": [run_.steps for run_ in optimizer_runs.values()],
                },
                hide_index=True,
                width="stretch",
            )

with visualise:
    run = st.session_state.get("optimizer_run")
    source = "Dashboard comparison" if run is not None else "Saved comparison"
    st.caption(f"Showing: **{source}**")
    if run is not None:
        artifacts = run.artifacts
        metrics = run.metrics
    else:
        artifacts = {
            "quadratic_losses": "artifacts/experiments/optimizer_comparison/plots/quadratic_losses.png",
            "rosenbrock_losses": "artifacts/experiments/optimizer_comparison/plots/rosenbrock_losses.png",
            "logistic_losses": "artifacts/experiments/optimizer_comparison/plots/logistic_losses.png",
            "quadratic_trajectories": "artifacts/experiments/optimizer_comparison/plots/quadratic_trajectories.png",
            "rosenbrock_trajectories": "artifacts/experiments/optimizer_comparison/plots/rosenbrock_trajectories.png",
        }
        metrics = common.read_json("artifacts/experiments/optimizer_comparison/metrics.json")

    objective_metrics = metrics.get("objectives", {})
    if objective_metrics:
        st.markdown("### Ranking by best objective value")
        rows = []
        for objective, payload in objective_metrics.items():
            rows.append(
                {
                    "objective": objective,
                    "ranking": " → ".join(payload.get("best_loss_ranking", [])),
                    "description": payload.get("description", ""),
                }
            )
        st.dataframe(rows, hide_index=True, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        common.figure(artifacts.get("quadratic_losses", ""), "Quadratic: loss by step")
        common.figure(artifacts.get("logistic_losses", ""), "Logistic: loss by step")
    with col2:
        common.figure(artifacts.get("rosenbrock_losses", ""), "Rosenbrock: loss by step")
        common.figure(
            artifacts.get("rosenbrock_trajectories", ""),
            "Rosenbrock: paths through parameter space",
        )

    if run is not None:
        st.markdown("### Raw loss history")
        rows = {}
        for objective, optimizer_runs in run.runs.items():
            for optimizer, result in optimizer_runs.items():
                rows[f"{objective}, {optimizer}"] = {
                    str(index): value
                    for index, value in enumerate(result.losses)
                    if value == value and abs(value) != float("inf")
                }
        if rows:
            st.line_chart(rows, x_label="step", y_label="objective value")

with evidence:
    st.markdown("### Saved comparison")
    st.write(
        "The saved run uses 1,000 steps for each optimizer–objective pair. Learning rates "
        "are chosen per objective (`quadratic=0.02`, `rosenbrock=0.001`, `logistic=0.05`) "
        "so the comparison is controlled without pretending one rate fits everything."
    )
    metrics = common.read_json("artifacts/experiments/optimizer_comparison/metrics.json")
    if metrics:
        for objective, payload in metrics.get("objectives", {}).items():
            with st.expander(f"{objective.title()} details", expanded=False):
                st.markdown(payload.get("description", ""))
                st.json(payload.get("optimizers", {}), expanded=False)
    else:
        st.info("Run `python scripts/train_optimizers.py` to create the comparison artifacts.")
    notes = common.load_text("artifacts/experiments/optimizer_comparison/notes.md")
    if notes:
        st.markdown("#### Notes")
        st.markdown(notes)

common.footer()
