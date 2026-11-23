"""Small configuration and validation helpers.

The goal is not a general schema language.  It is to reject invalid settings
with an actionable error before a model begins training.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and require a mapping at the top level."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration must be a mapping, got {type(data).__name__}")
    return data


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    """Write a configuration mapping to YAML."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_to_plain(config), handle, sort_keys=False)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""

    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def reject_unknown_keys(
    config: Mapping[str, Any], allowed: Iterable[str], context: str = "config"
) -> None:
    unknown = sorted(set(config) - set(allowed))
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in {context}: {', '.join(unknown)}. "
            f"Allowed keys: {', '.join(sorted(allowed))}."
        )


def check_int(
    name: str,
    value: Any,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be <= {maximum}, got {value}")
    return value


def check_float(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    allow_minimum: bool = True,
    allow_maximum: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{name} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{name} must be finite, got {result}")
    if minimum is not None:
        bad = result < minimum if allow_minimum else result <= minimum
        if bad:
            operator = ">=" if allow_minimum else ">"
            raise ConfigError(f"{name} must be {operator} {minimum}, got {result}")
    if maximum is not None:
        bad = result > maximum if allow_maximum else result >= maximum
        if bad:
            operator = "<=" if allow_maximum else "<"
            raise ConfigError(f"{name} must be {operator} {maximum}, got {result}")
    return result


def check_choice(name: str, value: Any, choices: Iterable[Any]) -> Any:
    allowed = tuple(choices)
    if value not in allowed:
        raise ConfigError(f"{name} must be one of {allowed}, got {value!r}")
    return value


def check_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean, got {value!r}")
    return value


def get_section(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Return a copy of a required mapping section."""

    if key not in config:
        raise ConfigError(f"Missing required section: {key}")
    return dict(require_mapping(config[key], key))


def _to_plain(value: Any) -> Any:
    """Convert numpy scalars and Paths into YAML-safe primitives."""

    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            return str(value)
    return value
