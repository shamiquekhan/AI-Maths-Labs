"""A deliberately small scalar reverse-mode automatic differentiation engine."""

from __future__ import annotations

from .examples import demo_expression, train_xor, xor_dataset
from .graph import draw_graph, graph_to_dict, topological_sort, zero_grad
from .nn import MLP, Layer, Module, Neuron
from .operations import (
    binary_cross_entropy_with_logits,
    mean,
    mse_loss,
    softplus,
)
from .optim import SGD
from .value import Value

__all__ = [
    "MLP",
    "SGD",
    "Layer",
    "Module",
    "Neuron",
    "Value",
    "binary_cross_entropy_with_logits",
    "demo_expression",
    "draw_graph",
    "graph_to_dict",
    "mean",
    "mse_loss",
    "softplus",
    "topological_sort",
    "train_xor",
    "xor_dataset",
    "zero_grad",
]
