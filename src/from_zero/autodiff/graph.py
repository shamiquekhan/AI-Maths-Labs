"""Topological traversal, graph inspection, and graph rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .value import Value


def topological_sort(root: Value) -> list[Value]:
    """Return parents before children in a deterministic post-order.

    The traversal is iterative so a long chain of scalar operations does not hit
    Python's recursion limit.
    """

    if not isinstance(root, Value):
        raise TypeError("root must be a Value")
    visited: set[int] = set()
    order: list[Value] = []
    stack: list[tuple[Value, bool]] = [(root, False)]
    while stack:
        node, processed = stack.pop()
        node_id = id(node)
        if processed:
            order.append(node)
            continue
        if node_id in visited:
            continue
        visited.add(node_id)
        stack.append((node, True))
        for parent in reversed(node.parents):
            if id(parent) not in visited:
                stack.append((parent, False))
    return order


def zero_grad(root: Value) -> None:
    """Set every gradient in the graph rooted at ``root`` to zero."""

    for node in topological_sort(root):
        node.grad = 0.0


def graph_to_dict(root: Value) -> dict[str, Any]:
    """Serialise the graph for the dashboard or a JSON report."""

    order = topological_sort(root)
    index = {id(node): position for position, node in enumerate(order)}
    nodes = [
        {
            "id": index[id(node)],
            "op": node.op or "leaf",
            "data": node.data,
            "grad": node.grad,
            "parents": [index[id(parent)] for parent in node.parents],
        }
        for node in order
    ]
    return {"nodes": nodes, "root": index[id(root)]}


def draw_graph(
    root: Value,
    path: str | Path | None = None,
    *,
    title: str = "Scalar computation graph",
    figure_size: tuple[float, float] | None = None,
    max_nodes: int = 60,
) -> Any:
    """Draw a small computation graph with forward values and gradients."""

    import matplotlib.pyplot as plt

    order = topological_sort(root)
    if len(order) > max_nodes:
        raise ValueError(
            f"Graph has {len(order)} nodes; increase max_nodes or draw a smaller expression"
        )
    levels: dict[int, int] = {}
    for node in order:
        if not node.parents:
            levels[id(node)] = 0
        else:
            levels[id(node)] = 1 + max(levels[id(parent)] for parent in node.parents)
    by_level: dict[int, list[Value]] = {}
    for node in order:
        by_level.setdefault(levels[id(node)], []).append(node)

    positions: dict[int, tuple[float, float]] = {}
    for level, nodes in by_level.items():
        for row, node in enumerate(nodes):
            positions[id(node)] = (float(level), -float(row) + (len(nodes) - 1) / 2.0)

    if figure_size is None:
        width = max(6.0, 2.1 * (max(levels.values()) + 1))
        height = max(3.2, 1.2 * max(len(nodes) for nodes in by_level.values()) + 1.0)
        figure_size = (width, height)
    figure, ax = plt.subplots(figsize=figure_size)
    ax.set_axis_off()

    for node in order:
        x, y = positions[id(node)]
        parent_positions = [positions[id(parent)] for parent in node.parents]
        for parent_x, parent_y in parent_positions:
            ax.annotate(
                "",
                xy=(x - 0.18, y),
                xytext=(parent_x + 0.18, parent_y),
                arrowprops={"arrowstyle": "->", "color": "#777777", "lw": 1.2},
            )

    for node in order:
        x, y = positions[id(node)]
        if node is root:
            facecolor = "#F4A261"
        elif node.is_leaf:
            facecolor = "#A8DADC"
        else:
            facecolor = "#EDF2F4"
        label = f"{node.op or 'leaf'}\n{node.data:.3f}\ngrad={node.grad:.3f}"
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": facecolor,
                "edgecolor": "#333333",
                "linewidth": 0.9,
            },
            zorder=3,
        )

    ax.set_xlim(-0.7, max(levels.values()) + 0.7)
    ax.set_ylim(
        -max(len(nodes) for nodes in by_level.values()) / 2 - 0.8,
        max(len(nodes) for nodes in by_level.values()) / 2 + 0.8,
    )
    ax.set_title(title)
    figure.tight_layout()

    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150, bbox_inches="tight", facecolor="white")
    return figure
