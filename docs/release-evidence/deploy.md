# Deployment Record

**Status:** local deployment verified; public Streamlit Community Cloud URL **pending**.
**Verified code commit:** `4985f75ade5fe131a0ed877c4c7d6c90d7bdcfec`
**Release tag:** `v0.1.0` (applied to the evidence commit)
**Date:** 2026-11-23
**Platform:** Streamlit Community Cloud (configuration ready)
**Runtime:** Python 3.13, torch-free `requirements.txt`

---

## What was verified locally

The deployment profile was installed in a fresh virtual environment **without
PyTorch** and the application was started with Streamlit 1.67.0:

| Check | Result |
|---|---|
| `pip install -e ".[dev,app]"` | success |
| `pip check` | no broken requirements |
| `torch` importable | **False** (expected) |
| Strict + AppTest subset | 110 passed |
| `GET /_stcore/health` | `ok` |
| `GET /` | HTTP 200 |
| Server log | Uvicorn started, no traceback |

Evidence: [`clean-install.txt`](clean-install.txt) and
[`deployment-smoke.txt`](deployment-smoke.txt).

The automated AppTest suite additionally loads every page with PyTorch blocked
in a subprocess, confirms the MiniGPT and Diffusion fallbacks, and exercises the
strict logistic training button. See [`apptest.txt`](apptest.txt).

---

## Cloud deployment checklist

The following steps require a GitHub remote and a Streamlit Community Cloud
account, neither of which was available in the build environment:

- [ ] Create/confirm the GitHub repository and push `main`.
- [ ] Confirm `git ls-files` contains no secrets, datasets, or checkpoints.
- [ ] Create the app at <https://share.streamlit.io>.
- [ ] Repository: `<owner>/<repo>`; branch: `main`.
- [ ] Main file: `src/from_zero/app/Home.py`.
- [ ] Python: `3.13`.
- [ ] Secrets: none required.
- [ ] Deploy and wait for the build to finish.
- [ ] Run the post-deployment smoke table in `docs/deployment.md`.
- [ ] Record the public URL and timestamp below.

### Public URL

```
PENDING — no remote/account access in the build environment.
```

Do not describe the app as live until this field is filled with a verified URL.

---

## Rollback notes

1. Open the app in Streamlit Community Cloud → **Manage app**.
2. Redeploy the previous known-good commit, or revert and push:

   ```bash
   git revert <bad-sha>
   git push origin main
   ```

3. Reboot the app to clear the environment.
4. Re-run:

   ```bash
   curl --fail <url>/_stcore/health
   python -m pytest tests/integration/test_app_pages.py -q
   ```

5. Confirm the strict logistic demo trains and the PyTorch fallback renders.
6. Record the rollback commit and time in this file.

There is no database or user data to restore. Experiment artifacts are
regenerable; committed metadata is read-only.

---

## Known deployment constraints

- **No PyTorch:** the cloud profile omits it. MiniGPT and Diffusion show a
  clear fallback; the strict GPT and strict diffusion demos on the Strict Engine
  page are the cloud-safe substitutes.
- **CPU only:** all demos are bounded and run on CPU.
- **No startup training:** pages render without training; the strict gradient
  table is the only load-time computation and is bounded.
- **Memory:** the free tier is limited. If a page grows, reduce demo bounds
  rather than adding dependencies.
- **Error details:** hidden by `client.showErrorDetails = "none"`; use the
  Streamlit logs for diagnosis.

See `docs/deployment.md` for the full guide and `docs/operations.md` for the
runbook.
