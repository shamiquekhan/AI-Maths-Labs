"""Reproducible optimizer comparison on shared objectives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..core.experiment import ExperimentRecorder, ExperimentResult, to_jsonable
from ..core.plotting import plot_optimizer_losses, plot_optimizer_trajectories
from ..core.seeds import set_seed
from ..core.validation import ConfigError, check_bool, check_float, check_int, reject_unknown_keys
from .benchmarks import OptimizerRun, make_objective, make_optimizer, run_optimizer

SUPPORTED_OPTIMIZERS = ("sgd", "momentum", "rmsprop", "adam", "adamw")
SUPPORTED_OBJECTIVES = ("quadratic", "rosenbrock", "logistic")


@dataclass
class OptimizerComparisonConfig:
    name: str = "optimizer_comparison"
    seed: int = 42
    steps: int = 1000
    learning_rate: float = 0.02
    learning_rates: dict[str, float] = field(
        default_factory=lambda: {"quadratic": 0.02, "rosenbrock": 0.001, "logistic": 0.05}
    )
    optimizers: tuple[str, ...] = ("sgd", "momentum", "rmsprop", "adam")
    objectives: tuple[str, ...] = ("quadratic", "rosenbrock", "logistic")
    curvature: tuple[float, float] = (1.0, 10.0)
    output_dir: str = "artifacts/experiments"
    make_plots: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigError("name must be a non-empty string")
        check_int("seed", self.seed, minimum=0)
        check_int("steps", self.steps, minimum=1, maximum=10_000)
        check_float("learning_rate", self.learning_rate, minimum=0.0, allow_minimum=False)
        for objective_name, rate in self.learning_rates.items():
            if objective_name not in SUPPORTED_OBJECTIVES:
                raise ConfigError(f"learning_rates contains unknown objective {objective_name!r}")
            check_float(
                f"learning_rates[{objective_name}]",
                rate,
                minimum=0.0,
                allow_minimum=False,
            )
        if not self.optimizers:
            raise ConfigError("At least one optimizer is required")
        unknown = set(self.optimizers) - set(SUPPORTED_OPTIMIZERS)
        if unknown:
            raise ConfigError(
                f"Unknown optimizer(s): {', '.join(sorted(unknown))}. "
                f"Choose from {', '.join(SUPPORTED_OPTIMIZERS)}."
            )
        if not self.objectives:
            raise ConfigError("At least one objective is required")
        unknown = set(self.objectives) - set(SUPPORTED_OBJECTIVES)
        if unknown:
            raise ConfigError(
                f"Unknown objective(s): {', '.join(sorted(unknown))}. "
                f"Choose from {', '.join(SUPPORTED_OBJECTIVES)}."
            )
        if len(self.curvature) != 2 or any(value <= 0 for value in self.curvature):
            raise ConfigError("curvature must contain two positive numbers")
        check_bool("make_plots", self.make_plots)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> OptimizerComparisonConfig:
        if not isinstance(config, dict):
            raise ConfigError("Optimizer comparison config must be a mapping")
        nested = any(key in config for key in ("experiment", "benchmark", "output"))
        if nested:
            reject_unknown_keys(config, ("experiment", "benchmark", "output"))
            experiment = _section(config, "experiment")
            benchmark = _section(config, "benchmark")
            output = _section(config, "output")
            reject_unknown_keys(experiment, ("name", "seed", "notes"), "experiment section")
            reject_unknown_keys(
                benchmark,
                (
                    "steps",
                    "learning_rate",
                    "learning_rates",
                    "optimizers",
                    "objectives",
                    "objective",
                    "curvature",
                ),
                "benchmark section",
            )
            reject_unknown_keys(output, ("directory", "output_dir", "make_plots"), "output section")
            objectives = benchmark.get("objectives")
            if objectives is None:
                objective = benchmark.get("objective", "quadratic")
                objectives = [objective] if isinstance(objective, str) else list(objective)
            optimizers = benchmark.get("optimizers", list(cls.optimizers))
            curvature = benchmark.get("curvature", list(cls.curvature))
            flat = {
                "name": experiment.get("name", cls.name),
                "seed": experiment.get("seed", cls.seed),
                "notes": experiment.get("notes", cls.notes),
                "steps": benchmark.get("steps", cls.steps),
                "learning_rate": benchmark.get("learning_rate", cls.learning_rate),
                "learning_rates": dict(
                    benchmark.get(
                        "learning_rates",
                        {"quadratic": 0.02, "rosenbrock": 0.001, "logistic": 0.05},
                    )
                ),
                "optimizers": tuple(optimizers),
                "objectives": tuple(objectives),
                "curvature": tuple(curvature),
                "output_dir": output.get("directory", output.get("output_dir", cls.output_dir)),
                "make_plots": output.get("make_plots", cls.make_plots),
            }
        else:
            allowed = set(cls.__dataclass_fields__)
            reject_unknown_keys(config, allowed, "optimizer comparison config")
            flat = {**asdict(cls()), **config}
            flat["optimizers"] = tuple(flat["optimizers"])
            flat["objectives"] = tuple(flat["objectives"])
            flat["curvature"] = tuple(flat["curvature"])
            flat["learning_rates"] = dict(flat["learning_rates"])
        return cls(**flat)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass
class OptimizerComparison:
    config: OptimizerComparisonConfig
    runs: dict[str, dict[str, OptimizerRun]]
    metrics: dict[str, Any]
    result: ExperimentResult
    artifacts: dict[str, str] = field(default_factory=dict)


def compare_optimizers(
    config: OptimizerComparisonConfig | dict[str, Any] | None = None,
    *,
    output_root: str | None = None,
    make_plots: bool | None = None,
    notes: str | None = None,
) -> OptimizerComparison:
    """Run every selected optimizer on every selected objective with equal budgets."""

    if config is None:
        config = OptimizerComparisonConfig()
    elif isinstance(config, dict):
        config = OptimizerComparisonConfig.from_dict(config)
    if make_plots is None:
        make_plots = config.make_plots
    if notes is None:
        notes = config.notes or (
            "All optimizers start from the same point and receive the same number of "
            "objective evaluations. The same base learning rate is used for a controlled "
            "comparison; it is not tuned per algorithm."
        )

    set_seed(config.seed)
    objectives = {
        name: make_objective(name, seed=config.seed, curvature=config.curvature)
        for name in config.objectives
    }
    runs: dict[str, dict[str, OptimizerRun]] = {}
    for objective_name, objective in objectives.items():
        runs[objective_name] = {}
        objective_lr = config.learning_rates.get(objective_name, config.learning_rate)
        for optimizer_name in config.optimizers:
            optimizer = make_optimizer(optimizer_name, objective_lr)
            run = run_optimizer(objective, optimizer, steps=config.steps)
            run.metadata["learning_rate"] = objective_lr
            runs[objective_name][optimizer_name] = run

    metrics = _build_metrics(config, objectives, runs)
    recorder = ExperimentRecorder(
        config.name,
        config.to_dict(),
        root=output_root or config.output_dir,
        seed=config.seed,
        notes=notes,
    )
    history_rows: list[dict[str, Any]] = []
    for objective_name, optimizer_runs in runs.items():
        for optimizer_name, run in optimizer_runs.items():
            for step, (loss, grad_norm) in enumerate(zip(run.losses, run.grad_norms, strict=True)):
                history_rows.append(
                    {
                        "objective": objective_name,
                        "optimizer": optimizer_name,
                        "step": step,
                        "loss": _finite_or_none(loss),
                        "grad_norm": _finite_or_none(grad_norm),
                    }
                )
    recorder.log_many(history_rows)
    recorder.metadata["objective_descriptions"] = {
        name: objective.description for name, objective in objectives.items()
    }
    recorder.metadata["known_limitations"] = [
        "A single learning rate is shared across algorithms, so this is a controlled comparison rather than a hyperparameter search.",
        "The objectives are small teaching problems; rankings do not imply universal superiority.",
        "Wall-clock time is not reported because the implementations and hardware vary.",
    ]

    artifacts: dict[str, str] = {}
    if make_plots:
        for objective_name in ("quadratic", "rosenbrock"):
            if objective_name not in objectives:
                continue
            objective = objectives[objective_name]
            trajectories = {name: run.trajectory for name, run in runs[objective_name].items()}
            artifacts[f"{objective_name}_trajectories"] = recorder.save_figure(
                plot_optimizer_trajectories(
                    objective.function,
                    trajectories,
                    bounds=objective.bounds,
                    title=f"{objective_name.title()}: optimizer trajectories",
                ),
                f"{objective_name}_trajectories.png",
            )
        for objective_name, optimizer_runs in runs.items():
            artifacts[f"{objective_name}_losses"] = recorder.save_figure(
                plot_optimizer_losses(
                    {name: run.losses for name, run in optimizer_runs.items()},
                    title=f"{objective_name.title()}: objective by step",
                    log_scale=True,
                ),
                f"{objective_name}_losses.png",
            )

    recorder.artifacts.update(artifacts)
    recorder.save_json(metrics, "metrics.json")
    result = recorder.finish(metrics, notes=notes)
    return OptimizerComparison(
        config=config,
        runs=runs,
        metrics=to_jsonable(metrics),
        result=result,
        artifacts=artifacts,
    )


def _build_metrics(
    config: OptimizerComparisonConfig,
    objectives: dict[str, Any],
    runs: dict[str, dict[str, OptimizerRun]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for objective_name, optimizer_runs in runs.items():
        rows: dict[str, Any] = {}
        for optimizer_name, run in optimizer_runs.items():
            rows[optimizer_name] = {
                "final_loss": _finite_or_none(run.final_loss),
                "best_loss": _finite_or_none(run.best_loss),
                "steps_completed": run.steps,
                "diverged": run.diverged,
                "final_grad_norm": _finite_or_none(run.grad_norms[-1]),
            }
        finite = [
            (name, row["best_loss"])
            for name, row in rows.items()
            if isinstance(row["best_loss"], float) and np.isfinite(row["best_loss"])
        ]
        ranking = [name for name, _ in sorted(finite, key=lambda item: item[1])]
        summary[objective_name] = {
            "description": objectives[objective_name].description,
            "optimizers": rows,
            "best_loss_ranking": ranking,
        }
    return {
        "config": config.to_dict(),
        "objectives": summary,
    }


def _finite_or_none(value: float) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a mapping")
    return value
