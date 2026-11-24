# FROM ZERO — Deployment Status

**Repository:** https://github.com/shamiquekhan/AI-Maths-Labs  
**Branch:** `main`  
**Latest tag:** `v0.1.2`  
**CI:** ✅ All jobs passing (security + tests on Python 3.11 & 3.13)

---

## Streamlit Community Cloud Deployment

### One-click deploy (if authenticated):
1. Open https://share.streamlit.io/
2. Click **New app**
3. Repository: `shamiquekhan/AI-Maths-Labs`
4. Branch: `main`
5. Main file path: `src/from_zero/app/Home.py`
6. Click **Deploy!**

### Configuration (auto-detected from repo):
| Setting | Value |
|---------|-------|
| Python version | 3.13 (from `.python-version`) |
| Dependencies | `requirements.txt` (pinned, **torch-free**) |
| Entry point | `src/from_zero/app/Home.py` |
| Theme | Light coral (configured in `.streamlit/config.toml`) |
| Secrets | None required (template in `.streamlit/secrets.toml.example`) |

### What works on the public deployment:
- ✅ All 7 learning labs (Home + 6 numbered pages)
- ✅ Strict tensor engine page (pure Python)
- ✅ Mathematics foundations page
- ✅ Experiments evidence browser
- ✅ About page
- ⚠️ MiniGPT & Diffusion pages show **installation instructions** (PyTorch not installed)

### Health checks:
- Local server: `streamlit run src/from_zero/app/Home.py` → http://localhost:8501
- All 190 tests pass
- Hallmark sloplint: **0 FAIL, 0 REFLEX, 0 WARN** (light theme)
- 10/10 sidebar nav links click through with real mouse events
- No console errors, no failed network requests

---

## Design Summary (v0.1.2)

**Theme:** Coral (light, warm paper) — per user request  
**Typography:** Hanken Grotesk (display/body) + Geist Mono (code/metrics)  
**Navigation:** Single grouped sidebar (5 sections, 10 links) with active highlight via href match  
**Icons:** Material Symbols (consistent, no emoji)  
**Contrast:** All text ≥ 4.5:1 against warm paper; Streamlit code tokens remapped to theme palette  
**Focus:** Visible `:focus-visible` on every interactive element  
**Motion:** `prefers-reduced-motion: reduce` respected  
**No Floor violations:** No middle-dot meta, no tracked-caps labels, no invented metrics, no purple-blue gradients

---

## Files Changed in v0.1.2

| File | Change |
|------|--------|
| `src/from_zero/app/common.py` | Full redesign: light tokens, Hanken Grotesk/Geist Mono, grouped sidebar with padding-based spacing (fixes label overlap), token remapping for code contrast, transparent header, three-weight limit |
| `.streamlit/config.toml` | Light coral theme config (base=light, primary=#B23A26, background=#F4F0E8) |
| `src/from_zero/app/Home.py` | Removed middle-dot meta from card titles; page title separator changed `·` → `|` |
| `src/from_zero/app/pages/2_Optimizers.py` | Removed middle-dot meta from badges and table keys |
| `src/from_zero/app/pages/6_Experiments.py` | Removed middle-dot meta from checkpoint lines |
| `src/from_zero/app/pages/8_About.py` | Removed middle-dot meta from version line and PyTorch status |
| `src/from_zero/app/pages/9_Strict_Engine.py` | Removed middle-dot from ascii ramp |

---

## Post-Deploy Verification

Once deployed, verify:
1. https://your-app.streamlit.app/ loads with coral paper background
2. Sidebar shows 5 grouped sections with Material icons
3. All 10 links navigate (page title updates, URL changes)
4. Active link has coral left border + bold label
5. Home hero headline uses Hanken Grotesk 600
6. Metric cards show Geist Mono tabular figures
7. Code blocks render with theme token colours (≥4.5:1)
7. Mobile: sidebar auto-collapses (`initial_sidebar_state="auto"`)

---

*Generated at v0.1.2 (a62c246)*
