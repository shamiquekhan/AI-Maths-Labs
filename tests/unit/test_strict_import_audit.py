"""Enforce the strict-core policy: no third-party numerical libraries."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

STRICT_PACKAGES = [
    "tensor",
    "autograd_tensor",
    "nn_tensor",
    "optim_tensor",
    "strict",
]

FORBIDDEN_ROOTS = {
    "numpy",
    "scipy",
    "torch",
    "torchvision",
    "tensorflow",
    "keras",
    "jax",
    "sklearn",
    "transformers",
    "pandas",
    "plotly",
    "matplotlib",
}

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "from_zero"


def _python_files() -> list[Path]:
    files: list[Path] = []
    for package in STRICT_PACKAGES:
        files.extend(sorted((SOURCE_ROOT / package).rglob("*.py")))
    return files


def test_strict_packages_have_no_forbidden_imports() -> None:
    violations: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in FORBIDDEN_ROOTS:
                        violations.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                if root in FORBIDDEN_ROOTS:
                    violations.append(f"{path.name}: from {node.module} import ...")
    assert not violations, "Forbidden imports in the strict core:\n" + "\n".join(violations)


def test_strict_packages_exist() -> None:
    for package in STRICT_PACKAGES:
        assert (SOURCE_ROOT / package / "__init__.py").exists(), package
    assert len(_python_files()) >= 10


@pytest.mark.parametrize(
    "module",
    [
        "from_zero.tensor",
        "from_zero.autograd_tensor",
        "from_zero.nn_tensor",
        "from_zero.optim_tensor",
        "from_zero.strict",
    ],
)
def test_importing_strict_core_does_not_load_third_party_libraries(module: str) -> None:
    script = (
        "import sys; "
        "sys.path.insert(0, 'src'); "
        f"import {module}; "
        "loaded = sorted({"
        "name.split('.')[0] for name in sys.modules"
        "} & {'numpy','scipy','torch','tensorflow','keras','jax','sklearn','transformers'}); "
        "print(','.join(loaded))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(SOURCE_ROOT.parents[1]),
        check=True,
    )
    assert result.stdout.strip() == "", f"{module} imported {result.stdout.strip()}"
