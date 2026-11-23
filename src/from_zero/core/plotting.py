"""Matplotlib plotting helpers with explicit labels and teaching context.

Charts in this project are part of the explanation.  Every helper therefore
sets axis labels, a title, and (where useful) a legend or colorbar.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "negative": "#0072B2",
    "positive": "#D55E00",
    "train": "#0072B2",
    "validation": "#009E73",
    "test": "#CC79A7",
    "accent": "#E69F00",
    "neutral": "#555555",
    "grid": "#D9D9D9",
}
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]


def save_figure(
    figure: plt.Figure,
    path: str | Path,
    *,
    dpi: int = 150,
    close: bool = True,
) -> Path:
    """Save a figure, creating parent directories as needed."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    if close:
        plt.close(figure)
    return destination


def plot_sigmoid(ax: plt.Axes | None = None) -> plt.Figure:
    if ax is None:
        figure, ax = plt.subplots(figsize=(6, 3.6))
    else:
        figure = ax.figure
    z = np.linspace(-10, 10, 500)
    ax.plot(z, 1.0 / (1.0 + np.exp(-z)), color=COLORS["train"], linewidth=2)
    ax.axhline(0.5, color=COLORS["neutral"], linestyle=":", linewidth=1)
    ax.axvline(0.0, color=COLORS["neutral"], linestyle=":", linewidth=1)
    ax.set_xlabel("logit z = wᵀx + b")
    ax.set_ylabel("probability σ(z)")
    ax.set_title("Sigmoid: mapping any logit to (0, 1)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def plot_classification_data(
    X: np.ndarray,
    y: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    title: str = "Binary classification data",
    feature_names: Sequence[str] = ("x₁", "x₂"),
) -> plt.Figure:
    if ax is None:
        figure, ax = plt.subplots(figsize=(6, 4.6))
    else:
        figure = ax.figure
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    for label, name, marker, color in (
        (0, "class 0", "o", COLORS["negative"]),
        (1, "class 1", "x", COLORS["positive"]),
    ):
        mask = y == label
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            s=22,
            marker=marker,
            color=color,
            alpha=0.8,
            linewidths=1.2 if marker == "x" else 0.5,
            label=name,
        )
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    ax.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def plot_decision_boundary(
    predict_proba: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    threshold: float = 0.5,
    grid_steps: int = 220,
    title: str = "Logistic regression decision boundary",
    feature_names: Sequence[str] = ("x₁", "x₂"),
) -> plt.Figure:
    """Plot a filled probability field plus the labelled observations."""

    if ax is None:
        figure, ax = plt.subplots(figsize=(6.4, 4.8))
    else:
        figure = ax.figure
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    pad_x = 0.08 * max(x_max - x_min, 1e-6)
    pad_y = 0.08 * max(y_max - y_min, 1e-6)
    xx, yy = np.meshgrid(
        np.linspace(x_min - pad_x, x_max + pad_x, grid_steps),
        np.linspace(y_min - pad_y, y_max + pad_y, grid_steps),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    probabilities = np.asarray(predict_proba(grid), dtype=float).reshape(xx.shape)
    contour = ax.contourf(
        xx, yy, probabilities, levels=20, cmap="RdBu_r", alpha=0.55, vmin=0, vmax=1
    )
    ax.contour(
        xx,
        yy,
        probabilities,
        levels=[threshold],
        colors=[COLORS["neutral"]],
        linewidths=1.6,
        linestyles="--",
    )
    for label, name, marker, color in (
        (0, "class 0", "o", COLORS["negative"]),
        (1, "class 1", "x", COLORS["positive"]),
    ):
        mask = y == label
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            s=24,
            marker=marker,
            color=color,
            edgecolor="white" if marker == "o" else None,
            linewidths=1.2 if marker == "x" else 0.7,
            label=name,
        )
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    colorbar = figure.colorbar(contour, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("P(class 1 | x)")
    figure.tight_layout()
    return figure


def plot_loss_curves(
    history: Mapping[str, Sequence[float]] | Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str] = ("train_loss", "val_loss"),
    ax: plt.Axes | None = None,
    title: str = "Training loss",
    xlabel: str = "epoch",
    log_scale: bool = False,
) -> plt.Figure:
    if ax is None:
        figure, ax = plt.subplots(figsize=(6.4, 4.0))
    else:
        figure = ax.figure
    series = _normalise_history(history)
    for index, key in enumerate(keys):
        values = series.get(key)
        if not values:
            continue
        label = key.replace("_", " ").replace("loss", "loss").title()
        ax.plot(
            np.arange(1, len(values) + 1),
            values,
            linewidth=2,
            color=PALETTE[index % len(PALETTE)],
            label=label,
        )
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cross-entropy / objective")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if any(series.get(key) for key in keys):
        ax.legend(frameon=False)
    figure.tight_layout()
    return figure


def plot_optimizer_trajectories(
    objective: Callable[[np.ndarray], float],
    trajectories: Mapping[str, np.ndarray],
    *,
    ax: plt.Axes | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    title: str = "Optimizer trajectories on a 2-D objective",
    grid_steps: int = 220,
) -> plt.Figure:
    if ax is None:
        figure, ax = plt.subplots(figsize=(7.0, 5.2))
    else:
        figure = ax.figure
    if not trajectories:
        raise ValueError("At least one trajectory is required")
    points = np.vstack([np.asarray(path, dtype=float) for path in trajectories.values()])
    if bounds is None:
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        pad_x = max(0.15 * (x_max - x_min), 0.2)
        pad_y = max(0.15 * (y_max - y_min), 0.2)
        bounds = (x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y)
    x_min, x_max, y_min, y_max = bounds
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_steps),
        np.linspace(y_min, y_max, grid_steps),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.array([objective(point) for point in grid], dtype=float).reshape(xx.shape)
    values = np.where(np.isfinite(values), values, np.nan)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Objective produced no finite values on the plotting grid")
    vmax = float(np.nanpercentile(finite_values, 99.0))
    contour = ax.contourf(xx, yy, values, levels=28, cmap="viridis", alpha=0.8, vmax=vmax)
    ax.contour(xx, yy, values, levels=12, colors="white", linewidths=0.45, alpha=0.6)
    for index, (name, path) in enumerate(trajectories.items()):
        path = np.asarray(path, dtype=float)
        ax.plot(
            path[:, 0],
            path[:, 1],
            marker="o",
            markersize=2.6,
            linewidth=1.8,
            color=PALETTE[index % len(PALETTE)],
            label=name,
        )
        ax.scatter(
            path[0, 0], path[0, 1], marker="s", s=45, color="white", edgecolor="black", zorder=5
        )
        ax.scatter(
            path[-1, 0],
            path[-1, 1],
            marker="*",
            s=100,
            color=PALETTE[index % len(PALETTE)],
            edgecolor="black",
            zorder=6,
        )
    ax.set_xlabel("θ₁")
    ax.set_ylabel("θ₂")
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    colorbar = figure.colorbar(contour, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("objective value")
    figure.tight_layout()
    return figure


def plot_optimizer_losses(
    histories: Mapping[str, Sequence[float]],
    *,
    ax: plt.Axes | None = None,
    title: str = "Objective value by optimizer step",
    log_scale: bool = True,
) -> plt.Figure:
    if ax is None:
        figure, ax = plt.subplots(figsize=(6.6, 4.2))
    else:
        figure = ax.figure
    positive_seen = False
    for index, (name, values) in enumerate(histories.items()):
        if not values:
            continue
        array = np.asarray(values, dtype=float)
        array = np.where(np.isfinite(array), array, np.nan)
        if np.any(array > 0):
            positive_seen = True
        ax.plot(
            np.arange(1, len(array) + 1),
            array,
            linewidth=1.8,
            color=PALETTE[index % len(PALETTE)],
            label=name,
        )
    use_log = log_scale and positive_seen
    if use_log:
        ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("objective (log scale)" if use_log else "objective")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if histories:
        ax.legend(frameon=False)
    figure.tight_layout()
    return figure


def plot_confusion_matrix(
    matrix: np.ndarray | Sequence[Sequence[int]],
    *,
    ax: plt.Axes | None = None,
    title: str = "Confusion matrix",
) -> plt.Figure:
    if ax is None:
        figure, ax = plt.subplots(figsize=(4.4, 3.8))
    else:
        figure = ax.figure
    array = np.asarray(matrix, dtype=int)
    if array.shape != (2, 2):
        raise ValueError("confusion matrix must be 2x2")
    image = ax.imshow(array, cmap="Blues")
    labels = [["TN", "FP"], ["FN", "TP"]]
    for row in range(2):
        for col in range(2):
            value = array[row, col]
            color = "white" if value > array.max() / 2 else "black"
            ax.text(col, row, f"{labels[row][col]}\n{value}", ha="center", va="center", color=color)
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(title)
    figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return figure


def plot_calibration(
    rows: Sequence[Mapping[str, float]],
    *,
    ax: plt.Axes | None = None,
    title: str = "Calibration: confidence vs observed frequency",
) -> plt.Figure:
    if ax is None:
        figure, ax = plt.subplots(figsize=(4.8, 4.4))
    else:
        figure = ax.figure
    if rows:
        confidence = [row["mean_confidence"] for row in rows]
        observed = [row["observed_frequency"] for row in rows]
        counts = [row["count"] for row in rows]
        ax.plot(
            [0, 1], [0, 1], linestyle="--", color=COLORS["neutral"], label="perfect calibration"
        )
        _ = ax.scatter(
            confidence,
            observed,
            s=[max(18.0, 0.7 * c) for c in counts],
            color=COLORS["train"],
            edgecolor="black",
            label="bins",
        )
        ax.legend(frameon=False)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("mean predicted probability in bin")
        ax.set_ylabel("observed positive frequency")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.text(
            0.02,
            0.98,
            "bubble size ∝ bin count",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color=COLORS["neutral"],
        )
    figure.tight_layout()
    return figure


def plot_attention(
    weights: np.ndarray,
    tokens: Sequence[str],
    *,
    ax: plt.Axes | None = None,
    title: str = "Causal self-attention",
    max_tokens: int = 48,
) -> plt.Figure:
    if ax is None:
        figure, ax = plt.subplots(figsize=(7.2, 5.8))
    else:
        figure = ax.figure
    array = np.asarray(weights, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"weights must be 2-D, got shape {array.shape}")
    n = min(array.shape[0], len(tokens), max_tokens)
    array = array[:n, :n]
    display_tokens = [token.replace("\n", "\\n") for token in tokens[:n]]
    image = ax.imshow(
        array, cmap="magma", aspect="auto", vmin=0.0, vmax=max(1e-9, float(array.max()))
    )
    tick_step = max(1, n // 12)
    ticks = np.arange(0, n, tick_step)
    ax.set_xticks(ticks, [display_tokens[index] for index in ticks], rotation=60, ha="right")
    ax.set_yticks(ticks, [display_tokens[index] for index in ticks])
    ax.set_xlabel("key / attended position")
    ax.set_ylabel("query position")
    ax.set_title(title)
    figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="attention weight")
    figure.tight_layout()
    return figure


def plot_noise_schedule(schedule: Any, *, title: str = "Diffusion noise schedule") -> plt.Figure:
    """Plot beta, alpha, and cumulative alpha-bar for a schedule object or dict."""

    if hasattr(schedule, "betas"):
        betas = _to_numpy(schedule.betas)
        alphas = _to_numpy(schedule.alphas)
        alpha_bars = _to_numpy(schedule.alpha_bars)
    else:
        betas = _to_numpy(schedule["betas"])
        alphas = _to_numpy(schedule["alphas"])
        alpha_bars = _to_numpy(schedule["alpha_bars"])
    steps = np.arange(1, len(betas) + 1)
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.4))
    axes[0].plot(steps, betas, color=COLORS["train"])
    axes[0].set_title("βₜ: noise added per step")
    axes[0].set_xlabel("timestep t")
    axes[0].set_ylabel("βₜ")
    axes[1].plot(steps, alphas, color=COLORS["positive"])
    axes[1].set_title("αₜ = 1 − βₜ: signal retained")
    axes[1].set_xlabel("timestep t")
    axes[1].set_ylabel("αₜ")
    axes[1].set_ylim(0.0, 1.02)
    axes[2].plot(steps, alpha_bars, color=COLORS["validation"])
    axes[2].set_title("ᾱₜ = ∏ αₛ: cumulative signal")
    axes[2].set_xlabel("timestep t")
    axes[2].set_ylabel("ᾱₜ")
    axes[2].set_ylim(0.0, 1.02)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle(title, y=1.02)
    figure.tight_layout()
    return figure


def plot_image_grid(
    images: Any,
    *,
    nrow: int = 4,
    title: str = "Generated samples",
    value_range: tuple[float, float] = (-1.0, 1.0),
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    array = _images_to_numpy(images, value_range=value_range)
    count, _height, _width = array.shape
    ncol = min(nrow, count)
    rows = int(np.ceil(count / ncol))
    if figsize is None:
        figsize = (2.0 * ncol, 2.0 * rows + 0.4)
    figure, axes = plt.subplots(rows, ncol, figsize=figsize, squeeze=False)
    for index in range(rows * ncol):
        row, col = divmod(index, ncol)
        axis = axes[row, col]
        axis.axis("off")
        if index < count:
            axis.imshow(array[index], cmap="gray", vmin=0.0, vmax=1.0)
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_forward_diffusion(
    images: Any,
    timesteps: Sequence[int],
    *,
    title: str = "Forward diffusion: clean image → noise",
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> plt.Figure:
    array = _images_to_numpy(images, value_range=value_range)
    if array.shape[0] != len(timesteps):
        raise ValueError("images and timesteps must have the same length")
    figure, axes = plt.subplots(
        1, len(timesteps), figsize=(2.1 * len(timesteps), 2.6), squeeze=False
    )
    for index, (axis, timestep) in enumerate(zip(axes[0], timesteps, strict=True)):
        axis.imshow(array[index], cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(f"t = {timestep}")
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_training_curves(
    history: Mapping[str, Sequence[float]] | Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str] = ("train_loss", "val_loss"),
    title: str = "Training curves",
    xlabel: str = "step",
    log_scale: bool = False,
) -> plt.Figure:
    return plot_loss_curves(history, keys=keys, title=title, xlabel=xlabel, log_scale=log_scale)


def _normalise_history(
    history: Mapping[str, Sequence[float]] | Sequence[Mapping[str, Any]],
) -> dict[str, list[float]]:
    if isinstance(history, Mapping):
        return {
            str(key): [
                float(value) if np.isfinite(float(value)) else float("nan") for value in values
            ]
            for key, values in history.items()
        }
    series: dict[str, list[float]] = {}
    for row in history:
        for key, value in row.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                number = float(value)
                series.setdefault(str(key), []).append(
                    number if np.isfinite(number) else float("nan")
                )
    return series


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _images_to_numpy(images: Any, *, value_range: tuple[float, float]) -> np.ndarray:
    array = _to_numpy(images)
    if array.ndim == 4:
        if array.shape[1] == 1:
            array = array[:, 0]
        elif array.shape[-1] in (1, 3):
            array = array[..., 0]
        else:
            array = array[:, 0]
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"images must be (N, H, W) or (N, C, H, W), got {array.shape}")
    low, high = value_range
    if high == low:
        raise ValueError("value_range must have distinct bounds")
    array = (array - low) / (high - low)
    return np.clip(array, 0.0, 1.0)
