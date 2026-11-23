#!/usr/bin/env python3
"""
Optimizer Example

Demonstrates the custom optimizer implementations:
- SGD, Momentum, RMSProp, Adam, AdamW
- Learning rate schedules
- Controlled comparisons on benchmark objectives
"""

import sys

sys.path.insert(0, "src")

import matplotlib.pyplot as plt
import numpy as np

from from_zero.optim import (
    SGD,
    Adam,
    AdamW,
    ConstantSchedule,
    CosineDecay,
    ExponentialDecay,
    Momentum,
    OptimizerComparisonConfig,
    RMSProp,
    StepDecay,
    compare_optimizers,
    make_objective,
    run_optimizer,
)


def test_optimizer_basics():
    """Test basic optimizer update rules."""
    print("=" * 60)
    print("Optimizers: Basic Update Rules")
    print("=" * 60)

    # SGD
    param = np.array([1.0, -2.0])
    grad = np.array([0.5, -0.25])
    opt = SGD(0.1)
    opt.step([param], [grad])
    print(f"SGD: param={param} (expected: [0.95, -1.975])")

    # Momentum
    param = np.array([0.0])
    opt = Momentum(0.1, beta=0.9)
    opt.step([param], [np.array([1.0])])
    print(f"Momentum step 1: param={param[0]:.4f} (expected: -0.1)")
    opt.step([param], [np.array([1.0])])
    print(f"Momentum step 2: param={param[0]:.4f} (expected: -0.29)")

    # RMSProp
    param = np.array([1.0])
    opt = RMSProp(0.1, rho=0.9, eps=1e-8)
    opt.step([param], [np.array([2.0])])
    expected_sq = 0.1 * 4.0
    expected = 1.0 - 0.1 * 2.0 / (np.sqrt(expected_sq) + 1e-8)
    print(f"RMSProp: param={param[0]:.6f} (expected: {expected:.6f})")

    # Adam
    param = np.array([1.0, -1.0])
    opt = Adam(0.1)
    opt.step([param], [np.array([2.0, -4.0])])
    print(f"Adam: param={param} (expected: [0.9, -0.9])")

    # AdamW
    param = np.array([2.0])
    opt = AdamW(0.1, weight_decay=0.5)
    opt.step([param], [np.array([0.0])])
    print(f"AdamW (decay only): param={param[0]:.4f} (expected: 1.9)")


def test_schedules():
    """Test learning rate schedules."""
    print("\n" + "=" * 60)
    print("Optimizers: Learning Rate Schedules")
    print("=" * 60)

    # Constant
    const = ConstantSchedule(0.1)
    print(f"Constant: {[const(i) for i in range(5)]}")

    # Step decay
    step = StepDecay(0.1, step_size=2, gamma=0.5)
    print(f"Step decay: {[step(i) for i in range(5)]}")

    # Exponential
    exp = ExponentialDecay(0.1, gamma=0.5)
    print(f"Exponential: {[exp(i) for i in range(5)]}")

    # Cosine
    cos = CosineDecay(0.1, total_steps=10, min_learning_rate=0.0)
    print(f"Cosine: {[cos(i) for i in range(0, 12, 2)]}")


def run_objective_comparison():
    """Run optimizer comparison on benchmark objectives."""
    print("\n" + "=" * 60)
    print("Optimizers: Objective Comparison")
    print("=" * 60)

    objectives = ["quadratic", "rosenbrock", "logistic"]
    optimizers = ["sgd", "momentum", "rmsprop", "adam"]

    for obj_name in objectives:
        print(f"\n{obj_name.upper()} Objective:")
        obj = make_objective(obj_name)
        print(f"  Dimension: {obj.dimension}, Optimum: {obj.optimum}")

        for opt_name in optimizers:
            if opt_name == "sgd":
                opt = SGD(
                    0.02 if obj_name == "quadratic" else 0.001 if obj_name == "rosenbrock" else 0.05
                )
            elif opt_name == "momentum":
                opt = Momentum(
                    0.02
                    if obj_name == "quadratic"
                    else 0.001
                    if obj_name == "rosenbrock"
                    else 0.05,
                    beta=0.9,
                )
            elif opt_name == "rmsprop":
                opt = RMSProp(
                    0.02 if obj_name == "quadratic" else 0.001 if obj_name == "rosenbrock" else 0.05
                )
            elif opt_name == "adam":
                opt = Adam(
                    0.02 if obj_name == "quadratic" else 0.001 if obj_name == "rosenbrock" else 0.05
                )

            result = run_optimizer(obj, opt, steps=200)
            print(f"  {opt_name:10s}: final_loss={result.final_loss:.2e}, steps={result.steps}")


def test_quadratic_trajectories():
    """Visualize optimizer trajectories on quadratic."""
    print("\n" + "=" * 60)
    print("Optimizers: Quadratic Trajectories")
    print("=" * 60)

    obj = make_objective("quadratic", curvature=(1.0, 10.0))
    optimizers = {
        "SGD": SGD(0.1),
        "Momentum": Momentum(0.1, beta=0.9),
        "RMSProp": RMSProp(0.1, rho=0.99),
        "Adam": Adam(0.1),
    }

    results = {}
    for name, opt in optimizers.items():
        result = run_optimizer(obj, opt, steps=50)
        results[name] = result
        print(f"{name}: final_loss={result.final_loss:.2e}")

    # Plot trajectories
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * (1.0 * X**2 + 10.0 * Y**2)

    _fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, (name, res) in zip(axes, results.items(), strict=False):
        traj = res.trajectory
        ax.contour(X, Y, Z, levels=np.logspace(-2, 2, 20), alpha=0.5)
        ax.plot(traj[:, 0], traj[:, 1], "o-", markersize=3, label=name)
        ax.set_title(f"{name} (final loss: {res.final_loss:.2e})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("/tmp/optimizer_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Trajectory plot saved to /tmp/optimizer_trajectories.png")


def test_rosenbrock():
    """Test optimizers on Rosenbrock function."""
    print("\n" + "=" * 60)
    print("Optimizers: Rosenbrock Function")
    print("=" * 60)

    obj = make_objective("rosenbrock")
    optimizers = {
        "SGD": SGD(0.001),
        "Momentum": Momentum(0.001, beta=0.9),
        "RMSProp": RMSProp(0.001),
        "Adam": Adam(0.001),
    }

    for name, opt in optimizers.items():
        result = run_optimizer(obj, opt, steps=500)
        print(f"{name:10s}: final_loss={result.final_loss:.4f}, steps={result.steps}")


def test_full_comparison():
    """Run full comparison using the comparison framework."""
    print("\n" + "=" * 60)
    print("Optimizers: Full Comparison Framework")
    print("=" * 60)

    config = OptimizerComparisonConfig(
        name="example_comparison",
        seed=123,
        steps=500,
        optimizers=("sgd", "momentum", "rmsprop", "adam"),
        objectives=("quadratic", "rosenbrock", "logistic"),
        output_dir="artifacts/experiments",
        make_plots=True,
    )

    comparison = compare_optimizers(config)
    print(f"Comparison completed: {comparison.result.directory}")

    for obj_name, runs in comparison.runs.items():
        print(f"\n{obj_name}:")
        sorted_runs = sorted(runs.items(), key=lambda x: x[1].final_loss)
        for name, run in sorted_runs:
            print(f"  {name:10s}: final_loss={run.final_loss:.2e}")


def main():
    test_optimizer_basics()
    test_schedules()
    run_objective_comparison()
    test_quadratic_trajectories()
    test_rosenbrock()
    test_full_comparison()

    print("\n" + "=" * 60)
    print("All optimizer examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
