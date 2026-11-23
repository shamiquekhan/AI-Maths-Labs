"""Project path helpers.

Keeping path resolution in one place means scripts, tests, and the Streamlit app
agree about where configs, corpora, and artifacts live regardless of the
current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the repository root (the directory containing ``pyproject.toml``)."""

    override = os.environ.get("FROM_ZERO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path, *, root: str | Path | None = None) -> Path:
    """Resolve ``path`` against the project root unless it is already absolute."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    base = Path(root) if root is not None else project_root()
    return (base / candidate).resolve()


def artifacts_dir() -> Path:
    """Return the default local artifact directory, creating it if necessary."""

    path = project_root() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def experiments_dir() -> Path:
    path = artifacts_dir() / "experiments"
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoints_dir() -> Path:
    path = artifacts_dir() / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    path = project_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path
