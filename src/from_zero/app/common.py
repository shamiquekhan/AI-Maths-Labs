"""Shared presentation helpers for the FROM ZERO Streamlit dashboard."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR.parents[1]
PROJECT_ROOT = APP_DIR.parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

ACCENT = "#0F766E"
ACCENT_WARM = "#C2410C"
INK = "#1F2933"
MUTED = "#52606D"


def setup_page(title: str, icon: str = "∑") -> None:
    """Configure the page and inject the shared visual language."""

    st.set_page_config(
        page_title=f"{title} · FROM ZERO",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)


def sidebar(title: str = "FROM ZERO") -> None:
    with st.sidebar:
        st.markdown(
            f"<div class='fz-sidebar-title'>{title}</div>"
            "<div class='fz-sidebar-sub'>A first-principles AI mathematics laboratory</div>",
            unsafe_allow_html=True,
        )
        st.divider()
        st.caption(
            "The strict core — tensors, autograd, layers, and optimizers — is pure "
            "Python. Logistic and optimizer labs use NumPy; the transformer and "
            "diffusion labs use PyTorch when it is installed."
        )


def hero(title: str, subtitle: str, badges: Iterable[str] = ()) -> None:
    badge_html = "".join(f"<span class='fz-badge'>{badge}</span>" for badge in badges)
    st.markdown(
        f"""
        <section class="fz-hero">
          <div class="fz-kicker">FROM ZERO · MATHEMATICS FIRST</div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
          <div class="fz-badges">{badge_html}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, body: str = "") -> None:
    st.markdown(f"### {title}")
    if body:
        st.markdown(body)


def metric_cards(items: Iterable[tuple[str, str, str]]) -> None:
    entries = list(items)
    columns = st.columns(len(entries))
    for column, (label, value, help_text) in zip(columns, entries, strict=True):
        with column:
            st.metric(label, value, help=help_text)


def callout(title: str, body: str, *, tone: str = "note") -> None:
    st.markdown(
        f"<div class='fz-callout fz-{tone}'><strong>{title}</strong><br/>{body}</div>",
        unsafe_allow_html=True,
    )


def figure(path: str | Path, caption: str | None = None) -> None:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if candidate.exists():
        st.image(str(candidate), caption=caption, width="stretch")
    else:
        st.info(f"Figure not found yet: `{path}`")


def code_block(code: str, language: str = "python") -> None:
    st.code(code, language=language, line_numbers=False)


def read_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if not candidate.exists():
        return {}
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_text(path: str | Path, limit: int = 12_000) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if not candidate.exists():
        return ""
    return candidate.read_text(encoding="utf-8")[:limit]


def show_history(
    history: list[dict[str, Any]], keys: tuple[str, ...] = ("train_loss", "val_loss")
) -> None:
    if not history:
        st.info("No logged history is available for this run.")
        return
    rows: dict[str, dict[str, float]] = {}
    for row in history:
        step = row.get("step")
        if step is None:
            continue
        try:
            x = float(step)
        except (TypeError, ValueError):
            continue
        for key in keys:
            value = row.get(key)
            if value in (None, ""):
                continue
            try:
                rows.setdefault(key, {})[str(int(x))] = float(value)
            except (TypeError, ValueError):
                continue
    if not rows:
        st.info("This run did not log the requested metric columns.")
        return
    st.line_chart(rows, x_label="step", y_label="loss")


def checkpoint_options(model_family: str) -> list[str]:
    from from_zero.app.services import find_checkpoints

    return [str(path) for path in find_checkpoints(model_family)]


def footer() -> None:
    st.divider()
    st.caption(
        "FROM ZERO · built for learning. Artifacts are local, experiments are seeded, "
        "and limitations are documented rather than hidden."
    )


_CSS = """
<style>
  :root {
    --fz-accent: #0F766E;
    --fz-warm: #C2410C;
    --fz-ink: #1F2933;
    --fz-muted: #52606D;
    --fz-paper: #FBF7F0;
    --fz-radius: 12px;
    --fz-radius-sm: 8px;
    --fz-gap: 1.5rem;
    --fz-gap-sm: 0.75rem;
    --fz-container: 1200px;
    --fz-transition: 150ms ease;
  }
  html, body, .stApp {
    overflow-x: clip;
  }
  *:focus-visible {
    outline: 2px solid var(--fz-accent) !important;
    outline-offset: 2px !important;
  }
  *:active {
    transform: scale(0.98);
  }
  .stApp {
    background:
      radial-gradient(circle at 8% 0%, rgba(15,118,110,0.08), transparent 30rem),
      radial-gradient(circle at 95% 8%, rgba(194,65,12,0.06), transparent 28rem),
      #FFFFFF;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    line-height: 1.6;
    color: var(--fz-ink);
  }
  .stApp *:focus-visible {
    outline: 2px solid var(--fz-accent) !important;
    outline-offset: 2px !important;
  }
  .stApp *:active {
    transform: scale(0.98);
  }
  h1, h2, h3 {
    font-family: Georgia, 'Times New Roman', serif;
    letter-spacing: -0.015em;
    color: var(--fz-ink);
    line-height: 1.2;
  }
  h1 { font-size: clamp(2rem, 4vw, 2.9rem) !important; line-height: 1.08 !important; }
  h2 { font-size: clamp(1.5rem, 3vw, 2rem); }
  h3 { font-size: clamp(1.25rem, 2.5vw, 1.5rem); }
  p { color: var(--fz-muted); max-width: 70ch; }
  .fz-hero {
    padding: clamp(1.5rem, 3vw, 2.4rem) clamp(1.5rem, 3vw, 2.6rem) clamp(1.25rem, 2.5vw, 2.1rem) clamp(1.5rem, 3vw, 2.6rem);
    border: 1px solid #E4D9C8;
    border-radius: var(--fz-radius);
    background: linear-gradient(135deg, #FFFDF8 0%, #F6EFE3 100%);
    margin-bottom: var(--fz-gap);
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
  }
  .fz-hero h1 { margin: 0.35rem 0 0.7rem 0; max-width: 18ch; }
  .fz-hero p { color: var(--fz-muted); font-size: 1.06rem; max-width: 70ch; margin: 0; }
  .fz-kicker {
    color: var(--fz-accent);
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
  }
  .fz-badges { margin-top: 1.15rem; display: flex; flex-wrap: wrap; gap: 0.45rem; }
  .fz-badge {
    border: 1px solid #D8CCBA;
    background: #FFFFFF;
    color: #3E4C59;
    border-radius: 999px;
    padding: 0.26rem 0.68rem;
    font-size: 0.76rem;
    letter-spacing: 0.02em;
    transition: var(--fz-transition);
  }
  .fz-badge:hover { border-color: var(--fz-accent); color: var(--fz-accent); }
  .fz-callout {
    border-left: 4px solid var(--fz-accent);
    background: #F5FAF9;
    padding: 1rem 1.25rem;
    border-radius: 0 var(--fz-radius) var(--fz-radius) 0;
    margin: var(--fz-gap-sm) 0 var(--fz-gap) 0;
    color: #334E4B;
    line-height: 1.7;
  }
  .fz-warn { border-left-color: var(--fz-warm); background: #FFF6EE; color: #6B3A1F; }
  .fz-sidebar-title {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--fz-ink);
    letter-spacing: -0.01em;
  }
  .fz-sidebar-sub { color: var(--fz-muted); font-size: 0.78rem; line-height: 1.35; }
  div[data-testid="stMetric"] {
    background: #FFFDF8;
    border: 1px solid #E7DDCE;
    padding: 1rem 1.25rem;
    border-radius: var(--fz-radius);
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    transition: var(--fz-transition);
  }
  div[data-testid="stMetric"]:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.06); }
  div[data-testid="stMetricLabel"] { color: var(--fz-muted); font-size: 0.875rem; text-transform: lowercase; letter-spacing: 0.02em; }
  div[data-testid="stMetricValue"] { color: var(--fz-ink); font-weight: 600; }
  .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; background: transparent; }
  .stTabs [data-baseweb="tab"] {
    border-radius: var(--fz-radius-sm) var(--fz-radius-sm) 0 0;
    padding: 0.625rem 1.25rem;
    font-weight: 500;
    color: var(--fz-muted);
    background: transparent;
    border: 1px solid transparent;
    border-bottom: none;
    transition: var(--fz-transition);
  }
  .stTabs [data-baseweb="tab"]:hover { color: var(--fz-ink); background: var(--fz-paper); }
  .stTabs [aria-selected="true"] { color: var(--fz-accent); background: var(--fz-paper); border-color: #E4D9C8; border-bottom-color: #FFFFFF; }
  .stTabs [data-baseweb="tab-panel"] { padding-top: var(--fz-gap); }
  code { font-size: 0.84rem; font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace; background: var(--fz-paper); padding: 0.125rem 0.375rem; border-radius: 4px; }
  .fz-formula {
    background: #FFFDF8;
    border: 1px solid #E7DDCE;
    border-radius: var(--fz-radius);
    padding: 1rem 1.25rem;
    margin-bottom: var(--fz-gap);
    overflow-x: auto;
    font-size: 0.9rem;
    line-height: 1.7;
  }
  .stButton > button {
    border-radius: var(--fz-radius-sm) !important;
    padding: 0.625rem 1.25rem !important;
    font-weight: 500 !important;
    transition: var(--fz-transition) !important;
    border: 1px solid transparent !important;
  }
  .stButton > button[kind="primary"] {
    background: var(--fz-accent) !important;
    color: white !important;
  }
  .stButton > button[kind="primary"]:hover {
    background: #0d6d5e !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(15,118,110,0.3);
  }
  .stButton > button[kind="secondary"] {
    background: white !important;
    color: var(--fz-ink) !important;
    border-color: #D8CCBA !important;
  }
  .stButton > button[kind="secondary"]:hover {
    border-color: var(--fz-accent) !important;
    color: var(--fz-accent) !important;
  }
  .stTextInput > div > div > input,
  .stNumberInput > div > div > input,
  .stSelectbox > div > div > div {
    border-radius: var(--fz-radius-sm) !important;
    border: 1px solid #D8CCBA !important;
    transition: var(--fz-transition) !important;
  }
  .stTextInput > div > div > input:focus,
  .stNumberInput > div > div > input:focus,
  .stSelectbox > div > div > div:focus-within {
    border-color: var(--fz-accent) !important;
    box-shadow: 0 0 0 3px rgba(15,118,110,0.15) !important;
  }
  .stSlider > div > div > div > div {
    background: var(--fz-accent) !important;
  }
  .stExpander { border-radius: var(--fz-radius) !important; border: 1px solid #E4D9C8 !important; }
  .stExpander summary { font-weight: 500 !important; }
  .stDataFrame { border-radius: var(--fz-radius) !important; overflow: hidden !important; }
  .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { margin-top: var(--fz-gap) !important; margin-bottom: var(--fz-gap-sm) !important; }
  .stMarkdown ul, .stMarkdown ol { padding-left: 1.5rem !important; }
  .stMarkdown li { margin-bottom: 0.375rem !important; }
  .block-container { max-width: var(--fz-container) !important; padding-top: 2rem !important; padding-bottom: 3rem !important; }
  .stSidebar { background: #FAF8F4 !important; border-right: 1px solid #E4D9C8 !important; }
  .stSidebar [data-testid="stSidebarNav"] { padding-top: 1rem !important; }
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; transition-duration: 0.01ms !important; }
  }
</style>
"""
