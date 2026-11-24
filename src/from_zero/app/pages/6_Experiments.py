"""Experiment explorer page."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st  # noqa: E402

from from_zero.app import common  # noqa: E402
from from_zero.app.services import experiment_catalog, experiment_detail  # noqa: E402

common.setup_page("Experiments", "▤")
common.sidebar()

common.hero(
    "Every result leaves a trail.",
    "Each experiment directory contains the exact configuration, environment metadata, "
    "metrics, history, plots, notes, and checkpoints. Nothing depends on a hidden notebook state.",
    ("Config", "Metrics", "History", "Artifacts"),
)

catalog = experiment_catalog()
if not catalog:
    st.warning(
        "No experiments found. Run one of the training scripts, for example "
        "`python scripts/train_logistic.py`, then refresh this page."
    )
    common.footer()
    st.stop()

st.markdown("### Catalog")
st.dataframe(
    [
        {
            "name": item["name"],
            "created": item.get("created_at", ""),
            "family": item.get("metadata", {}).get("model_family", ""),
            "seed": item.get("metadata", {}).get("seed"),
            "history points": item.get("metadata", {}).get("history_points"),
        }
        for item in catalog
    ],
    hide_index=True,
    width="stretch",
)

names = [item["name"] for item in catalog]
selected = st.selectbox("Open experiment", names)
detail = experiment_detail(selected)
directory = Path(detail["directory"])
metadata = detail.get("metadata", {})
metrics = detail.get("metrics", {})

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model family", metadata.get("model_family", "—"))
col2.metric("Seed", str(metadata.get("seed", "—")))
col3.metric("History points", str(metadata.get("history_points", "—")))
col4.metric("Created", str(metadata.get("created_at", "—"))[:10])

overview, configuration, metrics_tab, history_tab, artifacts_tab, notes_tab = st.tabs(
    ["Overview", "Configuration", "Metrics", "History", "Artifacts", "Notes"]
)

with overview:
    st.markdown("### Run metadata")
    st.json(
        {
            "name": detail.get("name"),
            "directory": detail.get("directory"),
            "created_at": metadata.get("created_at"),
            "finished_at": metadata.get("finished_at"),
            "environment": metadata.get("environment"),
            "git": metadata.get("git"),
            "known_limitations": metadata.get("known_limitations", []),
        },
        expanded=False,
    )
    if metadata.get("dataset"):
        st.markdown("### Dataset")
        st.json(metadata["dataset"], expanded=False)

with configuration:
    st.markdown("### Saved configuration")
    st.code(common.load_text(directory / "config.yaml") or "No config.yaml", "yaml")
    st.download_button(
        "Download config.yaml",
        data=(directory / "config.yaml").read_bytes()
        if (directory / "config.yaml").exists()
        else b"",
        file_name=f"{selected}_config.yaml",
        mime="text/yaml",
    )

with metrics_tab:
    st.markdown("### Metrics")
    if metrics:
        st.json(metrics, expanded=False)
        st.download_button(
            "Download metrics.json",
            data=(directory / "metrics.json").read_bytes(),
            file_name=f"{selected}_metrics.json",
            mime="application/json",
        )
    else:
        st.info("No metrics.json found for this experiment.")

with history_tab:
    st.markdown("### Logged history")
    history = detail.get("history", [])
    if history:
        st.dataframe(history, hide_index=True, width="stretch")
        common.show_history(history)
    else:
        st.info("This experiment has no CSV history.")

with artifacts_tab:
    st.markdown("### Plots and samples")
    plot_files = (
        sorted((directory / "plots").rglob("*.png")) if (directory / "plots").exists() else []
    )
    if plot_files:
        for start in range(0, len(plot_files), 2):
            columns = st.columns(2)
            for column, path in zip(columns, plot_files[start : start + 2], strict=False):
                with column:
                    common.figure(path, path.name)
    else:
        st.info("No plots were saved for this experiment.")

    sample_files = (
        sorted((directory / "samples").glob("*")) if (directory / "samples").exists() else []
    )
    for path in sample_files:
        if path.suffix == ".txt":
            with st.expander(path.name, expanded=False):
                st.text(common.load_text(path, limit=6000))
        elif path.suffix in {".json", ".npz"}:
            st.caption(f"Sample artifact: `{path.name}` ({path.stat().st_size / 1024:.1f} KiB)")

    checkpoints = (
        sorted((directory / "checkpoint").glob("*.pt"))
        if (directory / "checkpoint").exists()
        else []
    )
    if checkpoints:
        st.markdown("#### Checkpoints")
        for checkpoint in checkpoints:
            st.write(f"`{checkpoint.name}` ({checkpoint.stat().st_size / 1_048_576:.1f} MiB)")
        st.caption(
            "Checkpoints are intentionally ignored by git; retrain locally to recreate them."
        )

with notes_tab:
    st.markdown("### Notes")
    st.markdown(detail.get("notes") or "No notes were recorded for this experiment.")
    if metadata.get("known_limitations"):
        st.markdown("### Known limitations")
        for limitation in metadata["known_limitations"]:
            st.markdown(f"- {limitation}")

common.footer()
