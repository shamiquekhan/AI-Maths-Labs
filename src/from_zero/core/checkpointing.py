"""Checkpoint conventions for the PyTorch laboratories.

Checkpoints contain a model state dict, optional optimizer state, a plain
configuration dictionary, and primitive metadata.  Loading uses
``weights_only=True`` so an untrusted checkpoint cannot execute arbitrary
Python objects during unpickling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .experiment import to_jsonable


def save_checkpoint(
    path: str | Path,
    model: Any,
    *,
    optimizer: Any | None = None,
    config: dict[str, Any] | None = None,
    step: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a PyTorch model checkpoint with plain metadata."""

    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "config": to_jsonable(config or {}),
        "step": int(step) if step is not None else None,
        "extra": to_jsonable(extra or {}),
    }
    torch.save(payload, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: Any | None = None,
    optimizer: Any | None = None,
    map_location: str | Any = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint, optionally restoring model and optimizer state."""

    import torch

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
    except TypeError as exc:  # pragma: no cover - very old torch
        raise RuntimeError(
            "This project requires a PyTorch version with weights_only support (>=1.13)."
        ) from exc
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"Not a FROM ZERO checkpoint: {source}")
    if model is not None:
        model.load_state_dict(payload["model_state"], strict=strict)
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def save_json_metadata(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def load_json_metadata(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Metadata not found: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def save_numpy_parameters(path: str | Path, **arrays: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **{key: np.asarray(value) for key, value in arrays.items()})
    return destination


def load_numpy_parameters(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Parameter file not found: {source}")
    with np.load(source) as data:
        return {key: data[key] for key in data.files}
