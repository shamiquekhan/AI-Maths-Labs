# Contributing to FROM ZERO

Thanks for helping improve this teaching laboratory.

## Ground rules

- Mathematical correctness comes before features or presentation.
- Every new operation or model component needs a test.
- Keep the strict packages (`tensor`, `autograd_tensor`, `nn_tensor`,
  `optim_tensor`, `strict`) free of NumPy, SciPy, PyTorch, TensorFlow, Keras,
  JAX, and scikit-learn. The import audit test enforces this.
- Keep the NumPy implementations readable and explicit; do not replace them with
  framework calls.
- Preserve the distinction between the strict pure-Python core, NumPy-first
  teaching code, and the PyTorch models used for GPT and diffusion.
- Do not commit datasets, checkpoints, secrets, or large generated files.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,deep-learning,app]"
ruff check src tests examples scripts
ruff format --check src tests examples scripts
pytest --cov=from_zero --cov-report=term-missing
python scripts/scan_secrets.py
python scripts/verify_release_checks.py
```

For GPU PyTorch, install the wheel recommended by the official PyTorch selector
first, then install this project with `--no-deps` if necessary.

## Change checklist

- [ ] The change has a focused scope and a clear commit message.
- [ ] Unit tests cover normal and invalid inputs.
- [ ] Numerical gradients are checked when differentiation code changes.
- [ ] Strict packages contain no forbidden imports.
- [ ] `ruff check src tests examples scripts` and `pytest` pass.
- [ ] `python scripts/scan_secrets.py` passes.
- [ ] Documentation or a module note is updated when behaviour changes.
- [ ] No artifact, credential, or large data file is added.

## Commit style

Use short conventional prefixes, for example:

- `feat(logistic): implement stable sigmoid`
- `test(autodiff): add finite-difference checks`
- `docs(diffusion): derive the reverse update`

## Reporting issues

Include the command you ran, the configuration, the Python and package versions,
the full traceback, and whether the problem is reproducible with a small seed.
