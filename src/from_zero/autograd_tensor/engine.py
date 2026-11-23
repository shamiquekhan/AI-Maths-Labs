"""Autograd utilities layered on the custom tensor engine.

The computational graph lives inside :class:`from_zero.tensor.Tensor`
(``_prev`` and ``_backward``).  This package adds graph inspection and
finite-difference gradient checking without duplicating the engine.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..tensor import Tensor, is_grad_enabled, no_grad


def topological_sort(root: Tensor) -> list[Tensor]:
    """Return graph nodes in topological order (parents before children)."""
    order: list[Tensor] = []
    visited = set()

    def visit(node: Tensor) -> None:
        key = id(node)
        if key in visited:
            return
        visited.add(key)
        for parent in node._prev:
            visit(parent)
        order.append(node)

    visit(root)
    return order


def graph_nodes(root: Tensor) -> list[dict[str, object]]:
    """Describe the graph for visualisation.

    Each entry contains the node id, operation label, shape, and the ids of
    its parents.
    """
    nodes: list[dict[str, object]] = []
    for node in topological_sort(root):
        nodes.append(
            {
                "id": id(node),
                "op": node._op or "leaf",
                "shape": node.shape,
                "requires_grad": node.requires_grad,
                "parents": [id(parent) for parent in node._prev],
            }
        )
    return nodes


def format_graph(root: Tensor) -> str:
    """Human-readable textual representation of the graph."""
    lines: list[str] = []
    for node in topological_sort(root):
        parents = ", ".join(f"{id(p) % 10000:04d}" for p in node._prev)
        lines.append(
            f"{id(node) % 10000:04d} {node._op or 'leaf':<12} shape={node.shape} parents=[{parents}]"
        )
    return "\n".join(lines)


def numerical_gradient(
    function,
    tensor: Tensor,
    epsilon: float = 1e-6,
) -> list[float]:
    """Central finite-difference gradient of ``function()`` w.r.t. ``tensor``.

    ``function`` must return a scalar tensor.  Values are perturbed in place
    and restored afterwards, so no graph is built during the check.
    """
    data = tensor._data
    base = tensor._offset
    gradients: list[float] = []
    for i in range(tensor.size):
        position = base + i
        original = data[position]
        data[position] = original + epsilon
        plus = float(function().item())
        data[position] = original - epsilon
        minus = float(function().item())
        data[position] = original
        gradients.append((plus - minus) / (2.0 * epsilon))
    return gradients


def gradcheck(
    function,
    inputs: Sequence[Tensor],
    epsilon: float = 1e-6,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    raise_on_failure: bool = True,
) -> bool:
    """Compare autograd gradients with central finite differences.

    ``function`` must accept no arguments and return a scalar tensor.
    Returns ``True`` when every element is within tolerance.
    """
    for tensor in inputs:
        tensor.grad = None

    output = function()
    if output.size != 1:
        raise ValueError("gradcheck requires a scalar-valued function")
    output.backward()

    passed = True
    details: list[str] = []
    for index, tensor in enumerate(inputs):
        if not tensor.requires_grad:
            continue
        if tensor.grad is None:
            details.append(f"input {index}: missing autograd gradient")
            passed = False
            continue
        analytic = tensor.grad._values()
        numeric = numerical_gradient(function, tensor, epsilon)
        for position, (auto, finite) in enumerate(zip(analytic, numeric, strict=False)):
            difference = abs(auto - finite)
            tolerance = atol + rtol * abs(finite)
            if difference > tolerance:
                passed = False
                details.append(
                    f"input {index}[{position}]: autograd={auto:.8g} "
                    f"finite={finite:.8g} diff={difference:.3g}"
                )

    if not passed and raise_on_failure:
        raise AssertionError("Gradient check failed:\n" + "\n".join(details))
    return passed


__all__ = [
    "Tensor",
    "format_graph",
    "gradcheck",
    "graph_nodes",
    "is_grad_enabled",
    "no_grad",
    "numerical_gradient",
    "topological_sort",
]
