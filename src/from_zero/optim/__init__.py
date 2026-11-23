"""A small NumPy optimizer library with explicit update equations."""

from __future__ import annotations

from .adam import Adam, AdamW
from .base import Optimizer
from .benchmarks import Objective, make_objective, make_optimizer, run_optimizer
from .compare import OptimizerComparison, OptimizerComparisonConfig, compare_optimizers
from .momentum import Momentum
from .rmsprop import RMSProp
from .schedules import (
    ConstantSchedule,
    CosineDecay,
    ExponentialDecay,
    Schedule,
    StepDecay,
)
from .sgd import SGD

__all__ = [
    "SGD",
    "Adam",
    "AdamW",
    "ConstantSchedule",
    "CosineDecay",
    "ExponentialDecay",
    "Momentum",
    "Objective",
    "Optimizer",
    "OptimizerComparison",
    "OptimizerComparisonConfig",
    "RMSProp",
    "Schedule",
    "StepDecay",
    "compare_optimizers",
    "make_objective",
    "make_optimizer",
    "run_optimizer",
]
