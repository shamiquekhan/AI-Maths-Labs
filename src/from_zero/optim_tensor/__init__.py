"""Optimizers for the FROM ZERO strict tensor core."""

from __future__ import annotations

from .adagrad import Adagrad
from .adam import Adam
from .adamw import AdamW
from .base import Optimizer, clip_grad_norm_, clip_grad_value_
from .momentum import Momentum
from .rmsprop import RMSprop
from .schedules import (
    ConstantSchedule,
    CosineAnnealingWarmRestarts,
    CosineDecay,
    ExponentialDecay,
    LinearWarmup,
    LRScheduler,
    PolynomialDecay,
    StepDecay,
)
from .sgd import SGD

__all__ = [
    "SGD",
    "Adagrad",
    "Adam",
    "AdamW",
    "ConstantSchedule",
    "CosineAnnealingWarmRestarts",
    "CosineDecay",
    "ExponentialDecay",
    "LRScheduler",
    "LinearWarmup",
    "Momentum",
    "Optimizer",
    "PolynomialDecay",
    "RMSprop",
    "StepDecay",
    "clip_grad_norm_",
    "clip_grad_value_",
]
