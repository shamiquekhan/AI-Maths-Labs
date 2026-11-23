# ADR 004: Streamlit for Educational Dashboard

**Date:** 2024-01-22
**Status:** Accepted
**Deciders:** Project maintainer

## Context

The project needs an interactive web UI for:
- Reading theory with equations.
- Visualizing mathematical objects (graphs, attention, diffusion steps).
- Running bounded experiments with parameter controls.
- Browsing saved experiment artifacts.

## Decision

Use **Streamlit** for the dashboard application.

## Rationale

| Factor | Streamlit | Dash | Gradio | FastAPI + React | Jupyter + Voila |
|--------|-----------|------|--------|-----------------|-----------------|
| Python-native | ✅ | ✅ | ✅ | ❌ (JS) | ✅ |
| Declarative UI | ✅ | ✅ | ✅ | ❌ | ❌ |
| Matplotlib/Plotly support | ✅ | ✅ | ✅ | Manual | ✅ |
| Session state | ✅ | ✅ | ⚠️ | Manual | ❌ |
| Learning curve | Low | Medium | Low | High | Low |
| Multi-page apps | ✅ (pages/) | ✅ | ⚠️ | Manual | ⚠️ |
| Deployment | Easy | Easy | Easy | Complex | Medium |

Streamlit's "script-like" model matches the project's workflow: each page is a Python file that runs top-to-bottom on every interaction. The `st.session_state` pattern handles experiment state naturally.

## Consequences

- **Positive:** Fast development; all pages in `app/pages/` auto-discovered; built-in widgets for sliders, selectors, buttons; LaTeX via `st.latex()`; Matplotlib figures via `st.pyplot()`.
- **Negative:** Full rerun on every interaction (mitigated by `st.session_state` and caching); limited layout control (fixed sidebar + main); no built-in authentication; not for production multi-user apps.
- **Mitigation:** Expensive training runs are in CLI scripts; dashboard only runs **bounded demos** (max steps, small batch).

## Architecture

```
src/from_zero/app/
├── Home.py              # Entry point, overview cards
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
    └── 8_About.py
```

- **Services layer** (`services.py`) isolates UI from training logic.
- **Pages** import `from from_zero.app import common, services` after inserting `src/` into `sys.path`.
- **Bootstrap:** `Home.py` and pages insert project root into `sys.path` for imports.

## Alternatives Considered

1. **Gradio:** Better for model demos, worse for multi-page theory/experiment mix.
2. **FastAPI + React:** Full control but 10× development time; not Python-native.
3. **Jupyter notebooks as UI:** Not interactive for parameter sweeps; hard to share.

## Implementation Notes

- **Page config:** `common.setup_page(title, icon)` sets `st.set_page_config`, theme, CSS.
- **Hero section:** `common.hero(tagline, description, badges)` — consistent header.
- **Figures:** `common.figure(path, caption)` loads PNG, displays with caption.
- **Metric cards:** `common.metric_cards(dict)` — grid of `st.metric`.
- **Safeguards:** Bounded controls (max steps ≤ 5000, max batch ≤ 128); `st.session_state` for run results; errors surfaced via `st.error()`.
- **Accessibility:** Labels on all inputs; color not sole information carrier; LaTeX alt-text via Markdown.

## References

- Streamlit docs: https://docs.streamlit.io/
- Project blueprint §17 "Dashboard implementation plan"