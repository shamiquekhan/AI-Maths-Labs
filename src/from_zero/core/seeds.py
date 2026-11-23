"""Reproducible seeding for Python, NumPy, and (when installed) PyTorch.

Complete determinism is not guaranteed across hardware, CUDA kernels, library
versions, or platforms.  The returned dictionary is intended to be recorded in
experiment metadata so a reader knows what was controlled.
"""

from __future__ import annotations

import contextlib
import os
import random
import sys
from typing import Any

import numpy as np


def set_seed(seed: int, *, deterministic: bool = False) -> dict[str, Any]:
    """Seed the random number generators used by the project.

    Parameters
    ----------
    seed:
        Non-negative integer seed.
    deterministic:
        Ask PyTorch for deterministic kernels when available.  This can reduce
        throughput and may raise for operations without a deterministic
        implementation; the project uses ``warn_only=True`` so an experiment
        records a warning rather than failing late in training.
    """

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)

    info: dict[str, Any] = {
        "seed": seed,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": None,
        "cuda": False,
        "deterministic_requested": deterministic,
    }

    try:  # PyTorch is optional for the first three laboratories.
        import torch
    except ImportError:
        return info

    torch.manual_seed(seed)
    info["torch"] = torch.__version__
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        info["cuda"] = True
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with contextlib.suppress(AttributeError, RuntimeError):
            torch.use_deterministic_algorithms(True, warn_only=True)
    return info


def seed_worker(worker_id: int) -> None:
    """Seed a PyTorch ``DataLoader`` worker from the parent generator."""

    import torch

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
