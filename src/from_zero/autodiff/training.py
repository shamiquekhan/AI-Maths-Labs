"""End-to-end XOR experiment and artifact recording for the autodiff lab."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..core.experiment import ExperimentRecorder, ExperimentResult, to_jsonable
from ..core.plotting import plot_classification_data, plot_decision_boundary, plot_loss_curves
from ..core.seeds import set_seed
from ..core.validation import (
    ConfigError,
    check_bool,
    check_float,
    check_int,
    reject_unknown_keys,
)
from .examples import XORTrainingResult, train_xor, xor_dataset
from .graph import draw_graph
from .value import Value


@dataclass
class AutodiffConfig:
    name: str = "autodiff_xor"
    seed: int = 42
    hidden_sizes: tuple[int, ...] = (4, 4)
    epochs: int = 5000
    learning_rate: float = 0.2
    target_loss: float = 0.02
    record_every: int = 10
    output_dir: str = "artifacts/experiments"
    make_plots: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigError("name must be a non-empty string")
        check_int("seed", self.seed, minimum=0)
        if not self.hidden_sizes:
            raise ConfigError("hidden_sizes must contain at least one layer")
        for size in self.hidden_sizes:
            check_int("hidden size", size, minimum=1, maximum=256)
        check_int("epochs", self.epochs, minimum=1, maximum=200_000)
        check_float("learning_rate", self.learning_rate, minimum=0.0, allow_minimum=False)
        check_float("target_loss", self.target_loss, minimum=0.0, allow_minimum=False)
        check_int("record_every", self.record_every, minimum=1)
        check_bool("make_plots", self.make_plots)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> AutodiffConfig:
        if not isinstance(config, dict):
            raise ConfigError("Autodiff config must be a mapping")
        nested = any(key in config for key in ("experiment", "model", "output"))
        if nested:
            reject_unknown_keys(config, ("experiment", "model", "output"))
            experiment = _section(config, "experiment")
            model = _section(config, "model")
            output = _section(config, "output")
            reject_unknown_keys(experiment, ("name", "seed", "notes"), "experiment section")
            reject_unknown_keys(
                model,
                (
                    "hidden_sizes",
                    "epochs",
                    "learning_rate",
                    "target_loss",
                    "record_every",
                ),
                "model section",
            )
            reject_unknown_keys(output, ("directory", "output_dir", "make_plots"), "output section")
            flat = {
                "name": experiment.get("name", cls.name),
                "seed": experiment.get("seed", cls.seed),
                "notes": experiment.get("notes", cls.notes),
                "hidden_sizes": tuple(model.get("hidden_sizes", cls.hidden_sizes)),
                "epochs": model.get("epochs", cls.epochs),
                "learning_rate": model.get("learning_rate", cls.learning_rate),
                "target_loss": model.get("target_loss", cls.target_loss),
                "record_every": model.get("record_every", cls.record_every),
                "output_dir": output.get("directory", output.get("output_dir", cls.output_dir)),
                "make_plots": output.get("make_plots", cls.make_plots),
            }
        else:
            allowed = set(cls.__dataclass_fields__)
            reject_unknown_keys(config, allowed, "autodiff config")
            flat = {**asdict(cls()), **config}
            flat["hidden_sizes"] = tuple(flat["hidden_sizes"])
        return cls(**flat)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass
class AutodiffRun:
    config: AutodiffConfig
    training: XORTrainingResult
    metrics: dict[str, Any]
    result: ExperimentResult
    artifacts: dict[str, str] = field(default_factory=dict)


def run_autodiff_experiment(
    config: AutodiffConfig | dict[str, Any] | None = None,
    *,
    output_root: str | None = None,
    make_plots: bool | None = None,
    notes: str | None = None,
) -> AutodiffRun:
    if config is None:
        config = AutodiffConfig()
    elif isinstance(config, dict):
        config = AutodiffConfig.from_dict(config)
    if make_plots is None:
        make_plots = config.make_plots
    if notes is None:
        notes = config.notes or (
            "A scalar reverse-mode autodiff engine trains a tiny MLP on XOR. "
            "The hidden layer is necessary because XOR is not linearly separable."
        )
    set_seed(config.seed)
    training = train_xor(
        hidden_sizes=config.hidden_sizes,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        seed=config.seed,
        target_loss=config.target_loss,
        record_every=config.record_every,
    )

    metrics = {
        "final_loss": float(training.final_loss),
        "accuracy": float(training.accuracy),
        "epochs_run": int(training.epochs_run),
        "solved": bool(training.solved),
        "parameter_count": int(training.model.parameter_count()),
        "hidden_sizes": list(config.hidden_sizes),
        "final_predictions": _xor_predictions(training),
    }
    recorder = ExperimentRecorder(
        config.name,
        config.to_dict(),
        root=output_root or config.output_dir,
        seed=config.seed,
        notes=notes,
    )
    recorder.log_many(training.history)
    recorder.metadata["model_family"] = "scalar-reverse-mode-autodiff-mlp"
    recorder.metadata["known_limitations"] = [
        "The engine supports scalar values only and is not a tensor framework.",
        "XOR is a toy task; it demonstrates representational power, not scale.",
        "Gradient accumulation is explicit; zero_grad is called before each backward pass.",
    ]

    artifacts: dict[str, str] = {}
    if make_plots:
        features = np.array([list(features) for features, _ in xor_dataset()])
        labels = np.array([label for _, label in xor_dataset()])
        artifacts["xor_data"] = recorder.save_figure(
            plot_classification_data(
                features,
                labels,
                title="XOR: no single straight line separates the classes",
                feature_names=("x₁", "x₂"),
            ),
            "xor_data.png",
        )
        artifacts["loss_curve"] = recorder.save_figure(
            plot_loss_curves(
                {
                    "train_loss": [row["loss"] for row in training.history],
                },
                keys=("train_loss",),
                title="Scalar autodiff MLP loss on XOR",
                xlabel="recorded epoch",
            ),
            "xor_loss.png",
        )
        artifacts["decision_boundary"] = recorder.save_figure(
            plot_decision_boundary(
                _xor_probability_function(training),
                features,
                labels,
                title="Learned nonlinear XOR decision boundary",
                grid_steps=160,
            ),
            "xor_decision_boundary.png",
        )
        expression = _demo_graph()
        artifacts["computation_graph"] = recorder.save_figure(
            draw_graph(expression, title="A shared-subexpression computation graph"),
            "autodiff_graph.png",
        )
        artifacts["graph_json"] = recorder.save_json(
            _graph_payload(expression), "autodiff_graph.json"
        )

    artifacts["parameters"] = recorder.save_json(
        _parameter_payload(training), "model_parameters.json"
    )
    recorder.artifacts.update(artifacts)
    recorder.save_json(metrics, "metrics.json")
    result = recorder.finish(metrics, notes=notes)
    return AutodiffRun(
        config=config,
        training=training,
        metrics=to_jsonable(metrics),
        result=result,
        artifacts=artifacts,
    )


def _xor_probability_function(training: XORTrainingResult):
    def predict_proba(grid: np.ndarray) -> np.ndarray:
        probabilities = []
        for row in np.asarray(grid, dtype=float):
            logit = training.model([Value(row[0]), Value(row[1])])
            if not isinstance(logit, Value):
                raise TypeError("XOR model must return a Value")
            z = logit.data
            probabilities.append(
                1.0 / (1.0 + np.exp(-z)) if z >= 0 else np.exp(z) / (1.0 + np.exp(z))
            )
        return np.asarray(probabilities, dtype=float)

    return predict_proba


def _xor_predictions(training: XORTrainingResult) -> list[dict[str, Any]]:
    rows = []
    for features, target in xor_dataset():
        logit = training.model([Value(features[0]), Value(features[1])])
        if not isinstance(logit, Value):
            raise TypeError("XOR model must return a Value")
        rows.append(
            {
                "x1": features[0],
                "x2": features[1],
                "target": target,
                "logit": float(logit.data),
                "prediction": int(logit.data >= 0.0),
            }
        )
    return rows


def _parameter_payload(training: XORTrainingResult) -> dict[str, Any]:
    layers = []
    for layer in training.model.layers:
        layers.append(
            [
                {
                    "weights": [weight.data for weight in neuron.weights],
                    "bias": neuron.bias.data,
                }
                for neuron in layer.neurons
            ]
        )
    return {"layers": layers, "parameter_count": training.model.parameter_count()}


def _demo_graph() -> Value:
    from .examples import demo_expression

    return demo_expression()


def _graph_payload(expression: Value) -> dict[str, Any]:
    from .graph import graph_to_dict

    return graph_to_dict(expression)


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a mapping")
    return value
