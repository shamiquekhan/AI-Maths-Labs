"""Strict from-scratch models built on the custom tensor engine.

Every model in this package uses only ``from_zero.tensor``,
``from_zero.nn_tensor``, and ``from_zero.optim_tensor`` for its
mathematics.  NumPy and PyTorch are not imported anywhere in this package.
"""

from __future__ import annotations

from .checkpoint import load_checkpoint, save_checkpoint
from .data import (
    DataLoader,
    Dataset,
    SequenceDataset,
    TensorDataset,
    shuffle_together,
    split_indices,
    standardize,
    subset,
)
from .diffusion import (
    DiffusionSchedule,
    MLPDenoiser,
    make_pattern_images,
    sample_diffusion,
    sinusoidal_embedding,
    train_diffusion,
)
from .gpt import (
    Block,
    CausalSelfAttention,
    CharTokenizer,
    GPTConfig,
    MiniGPT,
    generate_text,
    make_lm_batch,
    train_gpt,
)
from .logistic import (
    LogisticRegression,
    analytical_gradients,
    evaluate_logistic,
    fit_logistic_regression,
    make_classification,
    train_valid_test_experiment,
)
from .metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    threshold_predictions,
)
from .training import History, Trainer

__all__ = [
    "Block",
    "CausalSelfAttention",
    "CharTokenizer",
    "DataLoader",
    "Dataset",
    "DiffusionSchedule",
    "GPTConfig",
    "History",
    "LogisticRegression",
    "MLPDenoiser",
    "MiniGPT",
    "SequenceDataset",
    "TensorDataset",
    "Trainer",
    "accuracy_score",
    "analytical_gradients",
    "classification_report",
    "confusion_matrix",
    "evaluate_logistic",
    "f1_score",
    "fit_logistic_regression",
    "generate_text",
    "load_checkpoint",
    "make_classification",
    "make_lm_batch",
    "make_pattern_images",
    "precision_score",
    "recall_score",
    "sample_diffusion",
    "save_checkpoint",
    "shuffle_together",
    "sinusoidal_embedding",
    "split_indices",
    "standardize",
    "subset",
    "threshold_predictions",
    "train_diffusion",
    "train_gpt",
    "train_valid_test_experiment",
]
