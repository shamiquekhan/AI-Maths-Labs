"""Shared presentation helpers for the FROM ZERO Streamlit dashboard."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit.errors import StreamlitPageNotFoundError
from streamlit.runtime.scriptrunner import get_script_run_ctx

APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR.parents[1]
PROJECT_ROOT = APP_DIR.parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def setup_page(title: str, icon: str = "∑") -> None:
    """Configure the page and inject the shared visual language."""

    st.set_page_config(
        page_title=f"{title} | FROM ZERO",
        page_icon=icon,
        layout="wide",
        # auto keeps the menu open on desktop and collapsed over the content
        # on a phone, where an expanded overlay would hide the page.
        initial_sidebar_state="auto",
    )
    # The sidebar highlights the current page by matching this title.
    st.session_state["_fz_page_title"] = title
    # st.markdown silently truncates a <style> block at roughly 2 KB, which
    # leaves the tokens applied but every component rule dead. st.html keeps
    # the whole stylesheet, so the design is injected there instead.
    st.html(_CSS)


def _entrypoint() -> Path | None:
    """Return the running entrypoint script path, if available."""

    ctx = get_script_run_ctx(suppress_warning=True)
    raw = getattr(ctx, "main_script_path", None) if ctx else None
    if not raw:
        return None
    try:
        return Path(raw).resolve()
    except OSError:
        return None


_NAV_GROUPS: tuple[tuple[str, tuple[tuple[str, str, str, str], ...]], ...] = (
    (
        "Overview",
        (("Home", "Home.py", "Home", ":material/home:"),),
    ),
    (
        "Foundations",
        (
            ("Strict Engine", "pages/9_Strict_Engine.py", "Strict Engine", ":material/memory:"),
            ("Mathematics", "pages/7_Mathematics.py", "Mathematics", ":material/calculate:"),
        ),
    ),
    (
        "Learning labs",
        (
            (
                "Logistic Regression",
                "pages/1_Logistic_Regression.py",
                "Logistic Regression",
                ":material/functions:",
            ),
            ("Optimizers", "pages/2_Optimizers.py", "Optimizers", ":material/tune:"),
            ("Autodiff", "pages/3_Autodiff.py", "Autodiff", ":material/account_tree:"),
        ),
    ),
    (
        "Generative labs",
        (
            ("MiniGPT", "pages/4_Mini_GPT.py", "MiniGPT", ":material/psychology:"),
            ("Diffusion", "pages/5_Diffusion.py", "Diffusion", ":material/blur_on:"),
        ),
    ),
    (
        "Evidence",
        (
            (
                "Experiments",
                "pages/6_Experiments.py",
                "Experiments",
                ":material/table_chart:",
            ),
            ("About", "pages/8_About.py", "About", ":material/info:"),
        ),
    ),
)


def _slug(page_path: str) -> str:
    """Return the URL slug Streamlit assigns to a page file.

    ``pages/1_Logistic_Regression.py`` becomes ``Logistic_Regression`` and the
    entry point becomes the empty string. The slug is used to highlight the
    active menu entry, so it must match the href Streamlit emits.
    """

    if page_path == "Home.py":
        return ""
    stem = Path(page_path).stem
    return stem.split("_", 1)[1] if "_" in stem else stem


def _nav_styles(active_slug: str) -> str:
    """Highlight the current page without relying on hashed emotion classes."""

    base = 'a[data-testid="stPageLink-NavLink"]'
    rules = [
        # Hide the auto-generated page list; the menu below replaces it.
        '[data-testid="stSidebarNav"] { display: none !important; }',
        f"{base} {{ border-radius: var(--radius-pill); padding: var(--space-2) var(--space-3);"
        " margin: 0; font-size: 0.9rem; color: var(--color-ink-2) !important;"
        " text-decoration: none; white-space: nowrap;"
        " transition: background-color var(--dur-short) var(--ease-out),"
        " color var(--dur-short) var(--ease-out); }",
        f"{base}:hover {{ background: var(--color-paper-2) !important;"
        " color: var(--color-ink) !important; }",
        f'{base}[href="{active_slug}"] {{ background: var(--color-paper-2) !important;'
        " color: var(--color-ink) !important; font-weight: 600;"
        " box-shadow: inset 2px 0 0 var(--color-accent); }",
        # Padding, not margin: a margin collapses out of the markdown wrapper
        # and lets the next link paint over the group label.
        ".fz-nav-group { margin: 0;"
        " padding: var(--space-4) var(--space-3) var(--space-1);"
        " font-size: 0.78rem; font-weight: 500;"
        " letter-spacing: 0.01em; color: var(--color-muted); }",
        '[data-testid="stSidebarUserContent"] { padding-top: 0 !important; }',
        '[data-testid="stSidebarUserContent"] [data-testid="stElementContainer"]'
        " { margin-top: 0 !important; margin-bottom: 0 !important; }",
        # Streamlit pulls markdown and page links up with negative margins;
        # inside the menu that makes rows collide, so neutralise it here.
        '[data-testid="stSidebarUserContent"] [data-testid="stMarkdownContainer"]'
        " { margin-top: 0 !important; margin-bottom: 0 !important; }",
        '[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"]'
        " { gap: var(--space-1) !important; }",
    ]
    return "<style>" + "".join(rules) + "</style>"


def sidebar(title: str = "FROM ZERO") -> None:
    current = st.session_state.get("_fz_page_title", "")

    with st.sidebar:
        st.markdown(
            f"<div class='fz-sidebar-title'>{title}</div>"
            "<div class='fz-sidebar-sub'>A first-principles AI mathematics laboratory</div>",
            unsafe_allow_html=True,
        )

        # st.page_link resolves paths relative to the *entrypoint*, so the menu
        # is only valid when the app is launched from Home.py. A page opened
        # directly has itself as the entrypoint and cannot resolve any page
        # path, so the menu is skipped there instead of raising.
        if _entrypoint() != (APP_DIR / "Home.py").resolve():
            st.divider()
            st.caption("Open `Home.py` to use the full navigation.")
            return

        try:
            active_slug = next(
                _slug(path)
                for _, items in _NAV_GROUPS
                for _, path, page_title, _ in items
                if page_title == current
            )
        except StopIteration:
            active_slug = ""

        st.markdown(_nav_styles(active_slug), unsafe_allow_html=True)

        try:
            for group, items in _NAV_GROUPS:
                st.markdown(f"<div class='fz-nav-group'>{group}</div>", unsafe_allow_html=True)
                for label, path, _, icon in items:
                    st.page_link(path, label=label, icon=icon)
        except StreamlitPageNotFoundError:
            # Running as a single page; the built-in page list is absent.
            st.caption("Open `Home.py` to use the full navigation.")

        st.divider()
        st.caption(
            "The strict core (tensors, autograd, layers, optimizers) is pure Python. "
            "Logistic and optimizer labs use NumPy. The transformer and diffusion labs "
            "use PyTorch when it is installed."
        )


def hero(title: str, subtitle: str, badges: Iterable[str] = ()) -> None:
    badge_html = "".join(f"<span class='fz-badge'>{badge}</span>" for badge in badges)
    st.markdown(
        f"""
        <section class="fz-hero">
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
        "FROM ZERO is built for learning. Artifacts are local, experiments are seeded, "
        "and limitations are documented rather than hidden."
    )


_CSS = """
<style>
/* Hallmark · v2.0.0 · theme: coral (light, by request) · genre: modern-minimal · scope: app
   Hallmark · pre-emit critique: P5 H4 E5 S4 R5 V4 · weakest: S, the dark ground made the saved figures read as holes; warm paper lets the plots sit in the page
   Hallmark · shape: app screen · nav: sidebar, grouped by lab family · footer: one line, no link columns
   Hallmark · redesign: full · kept: every page, link, helper and figure
   Hallmark · posture: restrained */
  @import url("https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap");
  :root {
    /* Coral, warm-paper colourway: coral marks the settled state, teal reads as an instrument. */
    --color-paper: #F4F0E8;
    --color-paper-2: #FCFAF6;
    --color-paper-3: #EBE4D8;
    --color-ink: #211C17;
    --color-ink-2: #443B32;
    --color-muted: #6B6156;
    --color-rule: #DDD4C6;
    --color-rule-2: #C4B8A6;
    --color-accent: #B23A26;
    --color-accent-ink: #FFF9F5;
    --color-accent-soft: #F6DED6;
    --color-focus: #B23A26;
    --color-cool: #2C5F5A;
    --color-cool-soft: #DDEBE8;
    --color-ok: #2F6B4F;
    --color-ok-soft: #E2EFE6;
    --color-bad: #A32B22;
    --color-bad-soft: #F7E2DF;

    --font-display: "Hanken Grotesk", ui-sans-serif, system-ui, sans-serif;
    --font-body: var(--font-display);
    --font-outlier: "Geist Mono", ui-monospace, monospace;

    --radius-card: 10px;
    --radius-pill: 6px;
    --rule-hair: 1px;
    --tracking-display: -0.02em;
    --display-weight: 600;

    --space-1: 4px;
    --space-2: 8px;
    --space-3: 12px;
    --space-4: 16px;
    --space-5: 24px;
    --space-6: 32px;
    --space-7: 48px;
    --space-8: 64px;

    --dur-micro: 90ms;
    --dur-short: 160ms;
    --ease-out: cubic-bezier(0.22, 0.61, 0.36, 1);

    --container: 1200px;
  }

  html, body {
    overflow-x: clip;
  }
  html {
    scrollbar-color: var(--color-rule-2) var(--color-paper-3);
  }
  body {
    caret-color: var(--color-accent);
  }
  ::selection {
    background: color-mix(in oklab, var(--color-accent) 25%, transparent);
  }

  *:focus-visible {
    outline: 2px solid var(--color-focus) !important;
    outline-offset: 2px !important;
  }
  /* NOTE: never scale elements on :active. Applying transform to every
     element makes the transforms compound, which moves a link out from
     under the cursor between mousedown and mouseup so the click is lost. */

  .stApp {
    background: var(--color-paper);
    font-family: var(--font-body);
    font-size: 16px;
    line-height: 1.55;
    color: var(--color-ink);
  }
  /* Streamlit styles its own headings and markdown text with class-level
     selectors, so the element selector alone loses. These stay scoped and
     specific enough to win without touching the whole document. */
  html body .stApp p,
  html body .stApp li {
    font-family: var(--font-body) !important;
    color: var(--color-ink-2);
  }

  html body .stApp h1,
  html body .stApp h2,
  html body .stApp h3,
  html body .stApp h4 {
    font-family: var(--font-display) !important;
    font-weight: var(--display-weight) !important;
    letter-spacing: var(--tracking-display) !important;
    line-height: 1.08 !important;
    color: var(--color-ink) !important;
    overflow-wrap: break-word;
    min-width: 0;
  }
  html body .stApp h1 { font-size: clamp(1.9rem, 3.4vw, 2.6rem) !important; }
  html body .stApp h2 { font-size: clamp(1.4rem, 2.2vw, 1.75rem) !important; }
  html body .stApp h3 { font-size: 1.15rem !important; }
  html body .stApp h4 { font-size: 1rem !important; }
  /* Three weights only: 400 reading, 500 interface, 600 display. */
  html body .stApp strong,
  html body .stApp b,
  html body .stApp th { font-weight: 600 !important; }

  /* ---- Hero: one statement, then the chips ---- */
  .fz-hero {
    padding: var(--space-7) var(--space-6);
    border: var(--rule-hair) solid var(--color-rule);
    border-radius: var(--radius-card);
    background: var(--color-paper-2);
    margin-bottom: var(--space-5);
  }
  html body .stApp .fz-hero h1 {
    margin: 0 0 var(--space-4) 0;
    max-width: 24ch;
    font-size: clamp(1.9rem, 3.6vw, 2.75rem) !important;
  }
  html body .stApp .fz-hero p {
    color: var(--color-ink-2) !important;
    font-size: 1rem;
    max-width: 62ch;
    margin: 0;
  }
  .fz-badges {
    margin-top: var(--space-5);
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
  }
  .fz-badge {
    border: var(--rule-hair) solid var(--color-rule-2);
    background: var(--color-paper-3);
    color: var(--color-ink-2);
    border-radius: var(--radius-pill);
    padding: var(--space-1) var(--space-3);
    font-size: 0.82rem;
  }

  /* ---- Callout: a hairline panel with a rule spine ---- */
  .fz-callout {
    border: var(--rule-hair) solid var(--color-rule);
    border-left: 2px solid var(--color-cool);
    background: var(--color-paper-2);
    padding: var(--space-4) var(--space-5);
    border-radius: var(--radius-card);
    margin: var(--space-3) 0 var(--space-5) 0;
    color: var(--color-ink-2);
    line-height: 1.6;
  }
  .fz-callout strong { color: var(--color-ink); }
  .fz-warn {
    border-left-color: var(--color-accent);
    background: var(--color-accent-soft);
    color: var(--color-ink);
  }

  /* ---- Sidebar ---- */
  .fz-sidebar-title {
    font-family: var(--font-display);
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--color-ink);
    letter-spacing: var(--tracking-display);
  }
  .fz-sidebar-sub {
    color: var(--color-muted);
    font-size: 0.82rem;
    line-height: 1.4;
    margin-top: var(--space-1);
  }

  /* ---- Metric: the instrument reading, in the measured face ---- */
  div[data-testid="stMetric"] {
    background: var(--color-paper-2);
    border: var(--rule-hair) solid var(--color-rule);
    padding: var(--space-4);
    border-radius: var(--radius-card);
  }
  div[data-testid="stMetricLabel"] {
    color: var(--color-muted);
    font-size: 0.85rem;
  }
  div[data-testid="stMetricValue"] {
    font-family: var(--font-outlier);
    font-size: 1.6rem;
    font-weight: 400;
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums slashed-zero;
    color: var(--color-ink);
  }

  /* ---- Tabs ---- */
  .stTabs [data-baseweb="tab-list"] {
    gap: var(--space-1);
    background: transparent;
    border-bottom: var(--rule-hair) solid var(--color-rule);
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 0;
    padding: var(--space-2) var(--space-4);
    font-weight: 400;
    color: var(--color-muted);
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    transition: color var(--dur-short) var(--ease-out),
      border-color var(--dur-short) var(--ease-out);
  }
  .stTabs [data-baseweb="tab"]:hover { color: var(--color-ink); }
  .stTabs [aria-selected="true"] {
    color: var(--color-ink);
    border-bottom-color: var(--color-accent);
  }
  .stTabs [data-baseweb="tab-panel"] { padding-top: var(--space-5); }

  /* ---- Code: the deepest step ---- */
  code {
    font-family: var(--font-outlier);
    font-size: 0.84rem;
    background: var(--color-paper-3);
    color: var(--color-cool);
    padding: var(--space-1) var(--space-2);
    border-radius: var(--radius-pill);
  }
  .stCode, pre, [data-testid="stCode"] {
    background: var(--color-paper-3) !important;
    border: var(--rule-hair) solid var(--color-rule) !important;
    border-radius: var(--radius-card) !important;
  }
  /* Streamlit's default token palette is tuned for its own grey surface and
     drops below 4.5:1 on warm paper. Re-map it to the page tokens. */
  .stCode .token.parameter, .stCode .token.variable,
  .stCode .token.number, .stCode .token.boolean {
    color: var(--color-cool) !important;
  }
  .stCode .token.function, .stCode .token.keyword, .stCode .token.builtin {
    color: var(--color-accent) !important;
  }
  .stCode .token.string { color: var(--color-ok) !important; }
  .stCode .token.operator, .stCode .token.punctuation {
    color: var(--color-ink-2) !important;
  }
  .stCode .token.comment { color: var(--color-muted) !important; }
  .stCode .token { font-weight: 400 !important; }

  /* ---- Buttons: the pair, coral fill beside an outline ---- */
  .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    min-height: 36px;
    border-radius: var(--radius-pill) !important;
    padding: var(--space-2) var(--space-4) !important;
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    background: var(--color-paper-2) !important;
    color: var(--color-ink) !important;
    border: var(--rule-hair) solid var(--color-rule-2) !important;
    transition: background-color var(--dur-short) var(--ease-out),
      border-color var(--dur-short) var(--ease-out) !important;
  }
  .stButton > button:hover, .stDownloadButton > button:hover,
  .stFormSubmitButton > button:hover {
    background: var(--color-paper-3) !important;
    border-color: var(--color-accent) !important;
  }
  .stButton > button:active, .stDownloadButton > button:active,
  .stFormSubmitButton > button:active {
    background: var(--color-paper-3) !important;
  }
  .stButton > button:disabled, .stDownloadButton > button:disabled {
    opacity: 0.5 !important;
    cursor: not-allowed !important;
  }
  .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
    background: var(--color-accent) !important;
    border-color: var(--color-accent) !important;
    color: var(--color-accent-ink) !important;
  }
  .stButton > button[kind="primary"]:hover,
  .stFormSubmitButton > button[kind="primary"]:hover {
    background: color-mix(in oklab, var(--color-accent) 86%, var(--color-ink)) !important;
  }

  /* ---- Inputs: paper, inside a hairline ---- */
  .stTextInput > div > div > input,
  .stNumberInput > div > div > input,
  .stSelectbox > div > div > div {
    min-height: 36px;
    border-radius: var(--radius-pill) !important;
    background: var(--color-paper-2) !important;
    color: var(--color-ink) !important;
    border: var(--rule-hair) solid var(--color-rule-2) !important;
  }
  .stTextInput > div > div > input:focus,
  .stNumberInput > div > div > input:focus,
  .stSelectbox > div > div > div:focus-within {
    border-color: var(--color-accent) !important;
    box-shadow: 0 0 0 2px var(--color-accent-soft) !important;
  }
  .stSlider > div > div > div > div {
    background: var(--color-accent) !important;
  }

  /* ---- Panels and surfaces ---- */
  .stExpander {
    border-radius: var(--radius-card) !important;
    border: var(--rule-hair) solid var(--color-rule) !important;
    background: var(--color-paper-2);
  }
  .stExpander summary { font-weight: 500 !important; color: var(--color-ink) !important; }
  .stDataFrame {
    border-radius: var(--radius-card) !important;
    overflow: hidden !important;
    border: var(--rule-hair) solid var(--color-rule) !important;
  }
  [data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: var(--radius-card) !important;
  }
  [data-testid="stVerticalBlockBorderWrapper"] > div {
    border-color: var(--color-rule) !important;
  }
  .stAlert {
    border-radius: var(--radius-card) !important;
    border: var(--rule-hair) solid var(--color-rule) !important;
    background: var(--color-paper-2) !important;
    color: var(--color-ink-2) !important;
  }
  .stAlert p { color: var(--color-ink-2) !important; }

  /* ---- Chrome ---- */
  /* The header floats over the scroll container. Keep it transparent so a
     scrolled page never hides a heading under an opaque band; the toolbar
     buttons stay legible on the paper. */
  [data-testid="stHeader"] {
    background: transparent;
    border-bottom: none;
  }
  [data-testid="stToolbar"] { color: var(--color-muted); }
  .stSidebar {
    background: var(--color-paper-3) !important;
    border-right: var(--rule-hair) solid var(--color-rule) !important;
  }
  hr, [data-testid="stDivider"] hr {
    border-color: var(--color-rule) !important;
  }
  .block-container {
    max-width: var(--container) !important;
    padding-top: var(--space-6) !important;
    padding-bottom: var(--space-8) !important;
  }
  .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    margin-top: var(--space-6) !important;
    margin-bottom: var(--space-3) !important;
  }
  .stMarkdown ul, .stMarkdown ol { padding-left: var(--space-5) !important; }
  .stMarkdown li { margin-bottom: var(--space-1) !important; }
  .stCaption, [data-testid="stCaptionContainer"] p {
    color: var(--color-muted) !important;
    font-size: 0.82rem !important;
  }
  a { color: var(--color-cool); }
  a:hover { color: var(--color-accent); }

  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.01ms !important;
    }
  }
</style>
"""
