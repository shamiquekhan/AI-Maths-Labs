"""Shared, framework-agnostic utilities used by every laboratory."""

from __future__ import annotations

from .datasets import (
    DatasetSplit,
    PreparedClassification,
    StandardScaler,
    make_classification_dataset,
    prepare_classification,
    split_classification,
)
from .metrics import binary_classification_metrics, confusion_matrix_binary, perplexity
from .seeds import set_seed
from .validation import ConfigError, check_choice, check_float, check_int, load_config

__all__ = [
    "ConfigError",
    "DatasetSplit",
    "PreparedClassification",
    "StandardScaler",
    "binary_classification_metrics",
    "check_choice",
    "check_float",
    "check_int",
    "confusion_matrix_binary",
    "load_config",
    "make_classification_dataset",
    "perplexity",
    "prepare_classification",
    "set_seed",
    "split_classification",
]
