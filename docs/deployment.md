# Deployment Guide

This project deploys as a **Streamlit Community Cloud** application. The public
instance runs the strict pure-Python engine and the NumPy laboratories; the
PyTorch-backed MiniGPT and Diffusion pages detect that PyTorch is absent and
show installation instructions instead of failing.

> **Status:** deployment configuration and local smoke tests are complete. A
> public URL has **not** been recorded because the repository has no configured
> GitHub remote and no Streamlit Community Cloud account was available in the
> build environment. See [Current deployment status](#current-deployment-status).

---

## 1. Target platform

| Item | Value |
|---|---|
| Platform | Streamlit Community Cloud |
| Runtime | Python 3.13 (`.python-version`) |
| Entry point | `src/from_zero/app/Home.py` |
| Runtime dependencies | `requirements.txt` (pinned, torch-free) |
| Optional local extra | `requirements-deep-learning.txt` (PyTorch) |
| Secrets | None required; template in `.streamlit/secrets.toml.example` |
| Persistent storage | None; artifacts are read-only copies from the repository |
| Training at startup | None; all training is button-driven and bounded |

The application is stateless. It does not accept uploads, does not write user
data, and does not load untrusted pickle files. Strict checkpoints use a safe
JSON format (`from-zero-strict-1`).

---

## 2. Why the cloud requirements exclude PyTorch

PyTorch and torchvision add several gigabytes and are not needed for the core
learning path. `requirements.txt` therefore pins only:

```
numpy==2.5.5
scipy==1.18.1
matplotlib==3.11.2
plotly==7.1.0
PyYAML==6.0.4
streamlit==1.67.0
```

The MiniGPT and Diffusion pages import PyTorch inside a guarded `try` block.
When it is missing they render a clear fallback and stop; every other page,
including the strict engine, remains fully functional. The strict character GPT
and strict diffusion demos on the **Strict Engine** page use no third-party
numerical library and are the public substitutes for those two laboratories.

To run the full hybrid stack locally:

```bash
python -m pip install -r requirements.txt -r requirements-deep-learning.txt -r requirements-dev.txt
```

---

## 3. First deployment

### 3.1 Prepare the repository

```bash
# From the repository root
git remote add origin git@github.com:<owner>/<repo>.git
git push -u origin main
```

Confirm that the following are **not** tracked:

```bash
git ls-files | grep -E '(^|/)(secrets\.toml|MNIST/|checkpoint/|.*\.pt$)' || echo "clean"
```

`.gitignore` excludes `.streamlit/secrets.toml`, datasets, checkpoints, and
generated plot/sample dumps.

### 3.2 Create the app

1. Sign in at <https://share.streamlit.io>.
2. Choose **Create app** → **Deploy a public app from GitHub**.
3. Repository: `<owner>/<repo>`; branch: `main`.
4. Main file path: `src/from_zero/app/Home.py`.
5. Advanced settings → Python version: **3.13**.
6. Secrets: leave empty. The app does not require any.
7. Deploy.

Streamlit Cloud installs `requirements.txt`, then runs the entry point. The
pages insert `src/` into `sys.path` themselves, so no package installation step
is required.

### 3.3 Configure secrets (only if an integration is added later)

Use the Streamlit Cloud **Settings → Secrets** editor. The local template is
`.streamlit/secrets.toml.example`; copy it to `.streamlit/secrets.toml` for
local development and never commit the real file.

```toml
[app]
environment = "production"

[integrations]
api_base_url = "https://example.invalid"
```

Read values defensively:

```python
import streamlit as st

value = st.secrets.get("integrations", {}).get("api_base_url", "")
```

A missing secret must never crash a page.

---

## 4. Post-deployment smoke test

Run these checks against the public URL after every deploy.

| Check | Expected |
|---|---|
| `GET <url>/_stcore/health` | `ok` |
| Home page | Renders hero, six lab cards, saved-experiment count |
| Strict Engine → Train strict logistic | Metrics appear; no exception |
| Strict Engine → gradient table | Worst error below `1e-4` |
| MiniGPT page | PyTorch fallback message, no traceback |
| Diffusion page | PyTorch fallback message, no traceback |
| Experiments page | Lists committed experiment directories |
| Browser console | No mixed-content or CORS errors |

Local equivalent:

```bash
streamlit run src/from_zero/app/Home.py --server.headless true &
curl --fail http://localhost:8501/_stcore/health
python -m pytest tests/integration/test_app_pages.py -q
```

The automated AppTest suite loads every page, clicks the strict logistic
training button, and verifies the torch-free fallback in a subprocess.

---

## 5. Resource and safety limits

- **No startup training.** Importing a page must not train a model. The strict
  gradient table is the only computation on load, and it is bounded.
- **Bounded demos.** Sliders cap epochs, optimizer steps, context length, and
  batch size. Services clamp values again server-side.
- **CPU only.** No CUDA assumptions; the cloud runtime has no GPU.
- **No uploads.** `maxUploadSize` is 25 MB as a defensive limit even though no
  upload widget exists.
- **No pickle from users.** Hybrid checkpoints are loaded with
  `torch.load(..., weights_only=True)`; strict checkpoints are JSON.
- **Error details hidden.** `client.showErrorDetails = "none"` prevents stack
  traces and filesystem paths from reaching visitors.
- **No secrets in git.** `.streamlit/secrets.toml` is ignored and a scanner
  gate checks for credential patterns.

The free tier has limited memory. If a page grows beyond the budget, reduce the
demo bounds rather than adding a GPU dependency.

---

## 6. Rollback

Streamlit Community Cloud keeps the Git history. To roll back:

1. Open the app → **Settings → Deployments** (or the app menu) and redeploy the
   last known-good commit, **or**
2. Revert the offending commit and push:

   ```bash
   git revert <bad-sha>
   git push origin main
   ```

3. Re-run the smoke test table in section 4.
4. If the failure is dependency-related, pin the previous versions in
   `requirements.txt`, push, and use **Reboot app** to clear the environment.

There is no database or user data to restore. Experiment artifacts are
regenerable from `scripts/train_*.py`; committed metadata is read-only.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: from_zero` | Entry point path wrong | Use `src/from_zero/app/Home.py` |
| MiniGPT/Diffusion page shows fallback | PyTorch absent (expected on cloud) | No action; use strict engine, or install locally |
| App exceeds memory | Too many cached figures / large artifacts | Reduce demo bounds; avoid loading checkpoints on Home |
| `ImportError` for a pinned package | Python version mismatch | Set Python 3.13 in app settings |
| Secrets error | Malformed TOML or missing key | Use `st.secrets.get` defensively; validate locally |
| Health endpoint 503 | App still starting or crashed | Check **Manage app → Logs**, then reboot |

---

## 8. Current deployment status

| Field | Value |
|---|---|
| Platform | Streamlit Community Cloud (prepared) |
| Public URL | **Pending** — no GitHub remote or Streamlit account in build environment |
| Local smoke test | Passed (`/_stcore/health` → `ok`; all pages via AppTest) |
| Deploy configuration | Complete (`requirements.txt`, `.python-version`, `.streamlit/config.toml`) |
| Rollback plan | Documented above; previous commit redeploy |

Do not claim a live deployment until the URL and timestamp are recorded in
`docs/release-evidence/deploy.md`.
