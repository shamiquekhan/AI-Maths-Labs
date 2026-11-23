"""Safe JSON checkpoints for strict-core models (no pickle)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..tensor import Tensor

FORMAT = "from-zero-strict-1"
DEFAULT_MAX_BYTES = 64 * 1024 * 1024


def model_state_to_json(model) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        state[name] = {
            "shape": list(parameter.shape),
            "data": parameter._values(),
        }
    return state


def model_state_from_json(state: dict[str, Any]) -> dict[str, Tensor]:
    tensors: dict[str, Tensor] = {}
    for name, entry in state.items():
        shape = tuple(int(dim) for dim in entry["shape"])
        data = [float(value) for value in entry["data"]]
        if len(data) != _product(shape):
            raise ValueError(f"Checkpoint entry {name!r} has inconsistent shape/data")
        tensors[name] = Tensor._from_flat(data, shape)
    return tensors


def _product(shape) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def save_checkpoint(
    path: str,
    model,
    optimizer=None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT,
        "model": model_state_to_json(model),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "metadata": metadata or {},
    }
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return destination


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    strict: bool = True,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    size = source.stat().st_size
    if size > max_bytes:
        raise ValueError(f"Checkpoint is {size} bytes, above the {max_bytes} byte safety limit")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Checkpoint is not valid JSON: {error}") from None

    if payload.get("format") != FORMAT:
        raise ValueError(
            f"Unsupported checkpoint format {payload.get('format')!r}; expected {FORMAT!r}"
        )
    model.load_state_dict(model_state_from_json(payload["model"]), strict=strict)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload.get("metadata", {})


__all__ = [
    "FORMAT",
    "load_checkpoint",
    "model_state_from_json",
    "model_state_to_json",
    "save_checkpoint",
]
