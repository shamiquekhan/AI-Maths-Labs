# Operations Runbook

This document describes how to operate, verify, and recover the FROM ZERO
dashboard. It is intentionally lightweight: the application is a stateless
educational site with no database, no user accounts, and no persistent writes.

---

## 1. Service overview

| Component | Location | Responsibility |
|---|---|---|
| Dashboard | `src/from_zero/app/` | Streamlit UI; no model math |
| Services | `src/from_zero/app/services.py` | Bounded demos, checkpoint discovery |
| Strict core | `src/from_zero/{tensor,autograd_tensor,nn_tensor,optim_tensor,strict}/` | Pure-Python tensors, autograd, layers, optimizers, models |
| Hybrid labs | `src/from_zero/{logistic,optim,autodiff,gpt,diffusion}/` | NumPy/PyTorch implementations |
| Artifacts | `artifacts/experiments/` | Committed metadata; ignored checkpoints/plots |
| Evidence | `docs/release-evidence/` | Release-gate outputs and deployment record |

**Dependency boundary:** the strict core imports only the standard library.
`tests/unit/test_strict_import_audit.py` parses every strict module and fails if
NumPy, SciPy, PyTorch, TensorFlow, Keras, JAX, or scikit-learn appears.

---

## 2. Health checks

### Automated (pre-release and CI)

```bash
source .venv/bin/activate
python -m pytest -q
ruff check src tests examples scripts
ruff format --check src tests examples scripts
python scripts/scan_secrets.py
pip-audit -r requirements.txt
bandit -r src -q -c bandit.yaml
```

### Runtime

```bash
curl --fail http://localhost:8501/_stcore/health   # -> ok
```

On Streamlit Community Cloud, use **Manage app → Logs** and the health endpoint
at `<public-url>/_stcore/health`.

### Page-level

`tests/integration/test_app_pages.py` loads all ten pages with AppTest, clicks
the strict logistic demo, and verifies the torch-free fallback. Run it after any
UI or service change.

---

## 3. Routine operations

### Refresh committed experiment evidence

```bash
python scripts/train_logistic.py
python scripts/train_optimizers.py
python scripts/train_autodiff.py
# PyTorch required for the next two:
python scripts/train_gpt.py
python scripts/train_diffusion.py
```

Commit only metadata, metrics, notes, and small tables. Checkpoints, `.pt`
files, generated plots, and samples are ignored by design. If a figure is
needed in documentation, copy it to `docs/images/` and reference it explicitly.

### Update dependencies

1. Update the pin in `requirements.txt` (and `pyproject.toml` lower bounds).
2. Recreate a clean environment and run the full gate suite.
3. Run `pip-audit -r requirements.txt`; document any accepted advisory.
4. Update `docs/release-evidence/` with the new versions and outputs.
5. Deploy and run the post-deployment smoke test.

Do not use floating versions in the cloud requirements file.

### Rotate secrets

The dashboard currently needs none. If an integration is added:

1. Update the value in the Streamlit Cloud **Secrets** editor.
2. Reboot the app.
3. Verify the page still works when the key is absent (defensive read).
4. Never commit `.streamlit/secrets.toml`; run the secret scanner before push.

---

## 4. Incident runbook

### App fails to start

1. Check **Manage app → Logs** for the first traceback.
2. Reproduce locally: `streamlit run src/from_zero/app/Home.py`.
3. If the failure follows a dependency change, redeploy the previous commit.
4. If the failure is a missing module, confirm the entry point is
   `src/from_zero/app/Home.py` and Python is 3.13.

### Memory pressure / app restart loop

1. Identify the page: logs usually show the last rendered page.
2. Reduce the demo bound in `services.py` (steps, epochs, batch size).
3. Avoid loading large checkpoints on Home; read only `metrics.json`.
4. Reboot the app and re-run the smoke test.

### A page raises an exception

1. `client.showErrorDetails = "none"` hides details from visitors; use logs.
2. Run the page locally with AppTest:
   `python -m pytest tests/integration/test_app_pages.py -q`.
3. Fix, add a regression test, and re-run the full suite.

### PyTorch page shows the fallback unexpectedly (local)

Install the optional stack:

```bash
python -m pip install -r requirements-deep-learning.txt
```

The fallback is correct behavior on the public cloud deployment.

### Suspected secret exposure

1. Revoke the credential at its provider immediately.
2. Remove it from Git history (for example `git filter-repo`) and force-push.
3. Rotate any dependent keys.
4. Add a regression case to `scripts/scan_secrets.py` if the pattern was missed.
5. Record the incident and remediation in the release notes.

---

## 5. Backup and recovery

- **Source of truth:** the Git repository. The app is stateless.
- **No user data:** nothing to back up beyond code and committed metadata.
- **Artifacts:** regenerable with the training scripts; ignored binaries are not
  required for the dashboard to start.
- **Recovery:** redeploy the last known-good commit, or recreate the app from the
  same repository and entry point.

---

## 6. Security posture

| Control | Implementation |
|---|---|
| No secrets in git | `.gitignore` + `scripts/scan_secrets.py` |
| Pinned dependencies | `requirements*.txt`, `pyproject.toml` |
| Dependency audit | `pip-audit -r requirements.txt` |
| Static analysis | `ruff`, `bandit -r src -c bandit.yaml` (B311 excluded: seeded RNG is for reproducibility, not cryptography) |
| Safe checkpoint loading | `weights_only=True` (hybrid), JSON (strict) |
| No arbitrary uploads | No upload widget; 25 MB defensive cap |
| XSRF/CORS | `enableXsrfProtection = true`, `enableCORS = true` |
| Error detail hiding | `client.showErrorDetails = "none"` |
| Import policy | `tests/unit/test_strict_import_audit.py` |

---

## 7. Release gates

| Gate | Command | Evidence |
|---|---|---|
| G0 Clean install | fresh venv + `pip install -e ".[dev,app]"` | `clean-install.txt` |
| G1 Dependency consistency | `pip check` | `clean-install.txt` |
| G2 Tests | `pytest -q` | `pytest.txt` |
| G3 Coverage | `pytest --cov=from_zero --cov-report=term-missing` | `coverage.txt` |
| G4 Lint/format | `ruff check` + `ruff format --check` | `ruff-*.txt` |
| G5 Numerical checks | strict gradient + analytical logistic checks | `numerical-checks.txt` |
| G6 Model behavior | causality, loss decrease, sampling, checkpoint round-trip | `model-checks.txt` |
| G7 AppTest | every page + strict demo + torch fallback | `apptest.txt` |
| G8 Security | `pip-audit`, `bandit`, secret scan | `security.txt` |
| G9 Deployment | local health endpoint + cloud smoke test | `deployment-smoke.txt`, `deploy.md` |

The consolidated record is `docs/release-evidence/README.md`.

---

## 8. Rollback checklist

1. Identify the last known-good commit from `docs/release-evidence/`.
2. Redeploy that commit in Streamlit Cloud, or `git revert` and push.
3. Run `/_stcore/health` and the page smoke test.
4. Confirm the strict logistic demo still trains and the torch fallback renders.
5. Record the rollback in `docs/release-evidence/deploy.md`.
