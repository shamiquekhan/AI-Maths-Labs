"""Controlled objective functions and a generic optimizer runner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.datasets import prepare_classification
from ..logistic.model import sigmoid
from .adam import Adam, AdamW
from .base import Optimizer
from .momentum import Momentum
from .rmsprop import RMSProp
from .schedules import Schedule
from .sgd import SGD


@dataclass(frozen=True)
class Objective:
    """A scalar objective with an analytic gradient and a starting point."""

    name: str
    function: Callable[[np.ndarray], float]
    gradient: Callable[[np.ndarray], np.ndarray]
    initial: np.ndarray
    bounds: tuple[float, float, float, float] | None = None
    optimum: np.ndarray | None = None
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        return int(np.asarray(self.initial).size)

    def evaluate(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        point = np.asarray(theta, dtype=np.float64)
        if point.shape != np.asarray(self.initial).shape:
            raise ValueError(
                f"{self.name}: expected parameter shape {np.asarray(self.initial).shape}, got {point.shape}"
            )
        if not np.all(np.isfinite(point)):
            raise ValueError(f"{self.name}: parameters contain NaN or infinity")
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(self.function(point))
            gradient = np.asarray(self.gradient(point), dtype=np.float64)
        if gradient.shape != point.shape:
            raise ValueError(f"{self.name}: gradient shape mismatch")
        return value, gradient


@dataclass
class OptimizerRun:
    """Trace of one optimizer on one objective."""

    optimizer: str
    objective: str
    trajectory: np.ndarray
    losses: list[float]
    grad_norms: list[float]
    diverged: bool
    steps: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_loss(self) -> float:
        return float(self.losses[-1])

    @property
    def best_loss(self) -> float:
        return float(np.nanmin(self.losses))


def quadratic(theta: np.ndarray, curvature: tuple[float, float] = (1.0, 10.0)) -> float:
    values = np.asarray(theta, dtype=np.float64)
    c1, c2 = curvature
    return float(0.5 * (c1 * values[0] ** 2 + c2 * values[1] ** 2))


def quadratic_gradient(
    theta: np.ndarray, curvature: tuple[float, float] = (1.0, 10.0)
) -> np.ndarray:
    values = np.asarray(theta, dtype=np.float64)
    return np.array([curvature[0] * values[0], curvature[1] * values[1]])


def rosenbrock(theta: np.ndarray, a: float = 1.0, b: float = 100.0) -> float:
    x, y = np.asarray(theta, dtype=np.float64)
    return float((a - x) ** 2 + b * (y - x**2) ** 2)


def rosenbrock_gradient(theta: np.ndarray, a: float = 1.0, b: float = 100.0) -> np.ndarray:
    x, y = np.asarray(theta, dtype=np.float64)
    dx = -2.0 * (a - x) - 4.0 * b * x * (y - x**2)
    dy = 2.0 * b * (y - x**2)
    return np.array([dx, dy], dtype=np.float64)


def make_objective(
    name: str,
    *,
    seed: int = 42,
    curvature: tuple[float, float] = (1.0, 10.0),
) -> Objective:
    """Build a quadratic, Rosenbrock, or small logistic-regression objective."""

    name = name.lower()
    if name in {"quadratic", "ill_conditioned_quadratic"}:
        return Objective(
            name="quadratic",
            function=lambda theta: quadratic(theta, curvature),
            gradient=lambda theta: quadratic_gradient(theta, curvature),
            initial=np.array([8.0, 8.0], dtype=np.float64),
            bounds=(-10.0, 10.0, -10.0, 10.0),
            optimum=np.zeros(2),
            description=(
                "Axis-aligned quadratic with curvature "
                f"{curvature[0]:g} and {curvature[1]:g}; the narrow valley exposes "
                "the effect of momentum and adaptive scaling."
            ),
            metadata={"curvature": list(curvature)},
        )
    if name == "rosenbrock":
        return Objective(
            name="rosenbrock",
            function=rosenbrock,
            gradient=rosenbrock_gradient,
            initial=np.array([-1.2, 1.0], dtype=np.float64),
            bounds=(-2.0, 2.0, -1.0, 3.0),
            optimum=np.array([1.0, 1.0]),
            description=(
                "Rosenbrock's banana-shaped valley; a non-convex objective with a "
                "global minimum at (1, 1)."
            ),
        )
    if name in {"logistic", "logistic_regression"}:
        prepared = prepare_classification(
            name="moons",
            n_samples=400,
            noise=0.2,
            test_fraction=0.2,
            val_fraction=0.2,
            seed=seed,
        )
        split = prepared.split
        X = np.vstack([split.X_train, split.X_val])
        y = np.concatenate([split.y_train, split.y_val]).astype(np.float64)
        augmented = np.column_stack([X, np.ones(len(X))])
        regularization = 0.01

        def logistic_loss(theta: np.ndarray) -> float:
            logits = augmented @ theta
            losses = np.maximum(logits, 0.0) - logits * y + np.log1p(np.exp(-np.abs(logits)))
            return float(np.mean(losses) + 0.5 * regularization * np.dot(theta[:-1], theta[:-1]))

        def logistic_gradient(theta: np.ndarray) -> np.ndarray:
            probabilities = np.asarray(sigmoid(augmented @ theta), dtype=np.float64)
            gradient = augmented.T @ (probabilities - y) / len(y)
            gradient[:-1] += regularization * theta[:-1]
            return gradient

        return Objective(
            name="logistic",
            function=logistic_loss,
            gradient=logistic_gradient,
            initial=np.zeros(3, dtype=np.float64),
            bounds=None,
            optimum=None,
            description=(
                "Logistic regression on a standardized two-moons split. Parameters are "
                "[w₁, w₂, b]; this is a 3-D objective, so only loss curves are plotted."
            ),
            metadata={"regularization": regularization, "n_examples": len(y)},
        )
    raise ValueError(f"Unknown objective {name!r}; choose quadratic, rosenbrock, or logistic")


def make_optimizer(name: str, learning_rate: float, **kwargs: Any) -> Optimizer:
    """Instantiate an optimizer by name with explicit default hyperparameters."""

    key = name.lower()
    if key in {"sgd", "gradient_descent"}:
        return SGD(learning_rate, **kwargs)
    if key in {"momentum", "sgd_momentum"}:
        return Momentum(learning_rate, **kwargs)
    if key in {"rmsprop", "rms_prop"}:
        return RMSProp(learning_rate, **kwargs)
    if key == "adam":
        return Adam(learning_rate, **kwargs)
    if key in {"adamw", "adam_w"}:
        return AdamW(learning_rate, **kwargs)
    raise ValueError(f"Unknown optimizer {name!r}")


def run_optimizer(
    objective: Objective,
    optimizer: Optimizer,
    *,
    initial: np.ndarray | None = None,
    steps: int = 300,
    schedule: Schedule | None = None,
    grad_tol: float = 1e-9,
) -> OptimizerRun:
    """Run one optimizer for a fixed evaluation budget."""

    if steps < 1:
        raise ValueError("steps must be at least 1")
    theta = np.array(objective.initial if initial is None else initial, dtype=np.float64)
    if theta.shape != np.asarray(objective.initial).shape:
        raise ValueError("initial parameter shape does not match the objective")
    optimizer.reset_state()

    loss, gradient = objective.evaluate(theta)
    trajectory = [theta.copy()]
    losses = [loss]
    grad_norms = [float(np.linalg.norm(gradient))]
    diverged = not (np.isfinite(loss) and np.all(np.isfinite(gradient)))
    completed = 0

    for update_index in range(steps):
        if diverged:
            break
        if schedule is not None:
            optimizer.set_learning_rate(schedule(update_index))
        try:
            optimizer.step([theta], [gradient])
        except FloatingPointError:
            trajectory.append(theta.copy())
            losses.append(float("nan"))
            grad_norms.append(float("nan"))
            diverged = True
            break
        completed += 1
        try:
            loss, gradient = objective.evaluate(theta)
        except ValueError:
            losses.append(float("nan"))
            grad_norms.append(float("nan"))
            trajectory.append(theta.copy())
            diverged = True
            break
        trajectory.append(theta.copy())
        losses.append(loss)
        grad_norms.append(float(np.linalg.norm(gradient)))
        if not (np.isfinite(loss) and np.all(np.isfinite(gradient))):
            diverged = True
            break
        if grad_norms[-1] < grad_tol:
            break

    return OptimizerRun(
        optimizer=optimizer.name,
        objective=objective.name,
        trajectory=np.vstack(trajectory),
        losses=losses,
        grad_norms=grad_norms,
        diverged=diverged,
        steps=completed,
        metadata={
            "learning_rate": optimizer.learning_rate,
            "schedule": None if schedule is None else schedule.name,
            "optimizer_state_keys": sorted(optimizer.state),
        },
    )
