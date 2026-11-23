"""Configuration, end-to-end training, evaluation, and artifact recording."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..core.datasets import PreparedClassification, prepare_classification
from ..core.experiment import ExperimentRecorder, ExperimentResult, to_jsonable
from ..core.plotting import (
    plot_calibration,
    plot_classification_data,
    plot_confusion_matrix,
    plot_decision_boundary,
    plot_loss_curves,
    plot_sigmoid,
)
from ..core.seeds import set_seed
from ..core.validation import (
    ConfigError,
    check_bool,
    check_choice,
    check_float,
    check_int,
    reject_unknown_keys,
)
from .metrics import evaluate_classifier, format_confusion_matrix
from .model import LogisticRegression


@dataclass
class LogisticConfig:
    """Flat, validated configuration for one logistic-regression experiment."""

    name: str = "logistic_baseline"
    seed: int = 42
    dataset: str = "linearly_separable"
    n_samples: int = 600
    noise: float = 0.15
    test_fraction: float = 0.2
    val_fraction: float = 0.2
    learning_rate: float = 0.05
    epochs: int = 600
    batch_size: int | None = 64
    regularization: float = 0.001
    threshold: float = 0.5
    early_stopping_patience: int | None = 100
    min_delta: float = 1e-6
    output_dir: str = "artifacts/experiments"
    make_plots: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigError("name must be a non-empty string")
        check_int("seed", self.seed, minimum=0)
        check_choice(
            "dataset",
            self.dataset,
            ("linearly_separable", "moons", "circles", "xor"),
        )
        check_int("n_samples", self.n_samples, minimum=50)
        check_float("noise", self.noise, minimum=0.0)
        check_float(
            "test_fraction", self.test_fraction, minimum=0.0, maximum=0.5, allow_maximum=False
        )
        check_float(
            "val_fraction", self.val_fraction, minimum=0.0, maximum=0.5, allow_maximum=False
        )
        if self.test_fraction + self.val_fraction >= 1.0:
            raise ConfigError("test_fraction + val_fraction must be < 1")
        check_float("learning_rate", self.learning_rate, minimum=0.0, allow_minimum=False)
        check_int("epochs", self.epochs, minimum=1, maximum=100_000)
        if self.batch_size is not None:
            check_int("batch_size", self.batch_size, minimum=1)
        check_float("regularization", self.regularization, minimum=0.0)
        check_float("threshold", self.threshold, minimum=0.0, maximum=1.0)
        if self.early_stopping_patience is not None:
            check_int("early_stopping_patience", self.early_stopping_patience, minimum=1)
        check_float("min_delta", self.min_delta, minimum=0.0)
        check_bool("make_plots", self.make_plots)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> LogisticConfig:
        """Accept either a flat config or the nested YAML layout in ``configs/``."""

        if not isinstance(config, dict):
            raise ConfigError("Logistic config must be a mapping")
        nested = any(key in config for key in ("experiment", "data", "model", "output"))
        if nested:
            reject_unknown_keys(config, ("experiment", "data", "model", "output"))
            experiment = _section(config, "experiment")
            data = _section(config, "data")
            model = _section(config, "model")
            output = _section(config, "output")
            reject_unknown_keys(experiment, ("name", "seed", "notes"), "experiment section")
            reject_unknown_keys(
                data,
                (
                    "dataset",
                    "samples",
                    "n_samples",
                    "noise",
                    "test_fraction",
                    "val_fraction",
                ),
                "data section",
            )
            reject_unknown_keys(
                model,
                (
                    "learning_rate",
                    "epochs",
                    "batch_size",
                    "regularization",
                    "threshold",
                    "early_stopping_patience",
                    "min_delta",
                ),
                "model section",
            )
            reject_unknown_keys(output, ("directory", "output_dir", "make_plots"), "output section")
            flat = {
                "name": experiment.get("name", cls.name),
                "seed": experiment.get("seed", cls.seed),
                "notes": experiment.get("notes", cls.notes),
                "dataset": data.get("dataset", cls.dataset),
                "n_samples": data.get("samples", data.get("n_samples", cls.n_samples)),
                "noise": data.get("noise", cls.noise),
                "test_fraction": data.get("test_fraction", cls.test_fraction),
                "val_fraction": data.get("val_fraction", cls.val_fraction),
                "learning_rate": model.get("learning_rate", cls.learning_rate),
                "epochs": model.get("epochs", cls.epochs),
                "batch_size": model.get("batch_size", cls.batch_size),
                "regularization": model.get("regularization", cls.regularization),
                "threshold": model.get("threshold", cls.threshold),
                "early_stopping_patience": model.get(
                    "early_stopping_patience", cls.early_stopping_patience
                ),
                "min_delta": model.get("min_delta", cls.min_delta),
                "output_dir": output.get("directory", output.get("output_dir", cls.output_dir)),
                "make_plots": output.get("make_plots", cls.make_plots),
            }
        else:
            allowed = set(cls.__dataclass_fields__)
            reject_unknown_keys(config, allowed, "logistic config")
            flat = {**asdict(cls()), **config}
        return cls(**flat)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass
class LogisticRun:
    """Everything needed to inspect a completed logistic experiment."""

    config: LogisticConfig
    model: LogisticRegression
    prepared: PreparedClassification
    metrics: dict[str, Any]
    train_eval: dict[str, Any]
    val_eval: dict[str, Any]
    test_eval: dict[str, Any]
    result: ExperimentResult
    artifacts: dict[str, str] = field(default_factory=dict)


def run_logistic_experiment(
    config: LogisticConfig | dict[str, Any] | None = None,
    *,
    output_root: str | None = None,
    make_plots: bool | None = None,
    notes: str | None = None,
) -> LogisticRun:
    """Run the complete logistic-regression laboratory experiment."""

    if config is None:
        config = LogisticConfig()
    elif isinstance(config, dict):
        config = LogisticConfig.from_dict(config)
    if make_plots is None:
        make_plots = config.make_plots
    if notes is None:
        notes = config.notes or _default_notes(config)

    set_seed(config.seed)
    prepared = prepare_classification(
        name=config.dataset,
        n_samples=config.n_samples,
        noise=config.noise,
        test_fraction=config.test_fraction,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    split = prepared.split
    model = LogisticRegression(
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        regularization=config.regularization,
        seed=config.seed,
        batch_size=config.batch_size,
        threshold=config.threshold,
        early_stopping_patience=config.early_stopping_patience,
        min_delta=config.min_delta,
    )
    model.fit(
        split.X_train,
        split.y_train,
        X_val=split.X_val,
        y_val=split.y_val,
    )

    train_eval = evaluate_classifier(
        model, split.X_train, split.y_train, threshold=config.threshold
    )
    val_eval = evaluate_classifier(model, split.X_val, split.y_val, threshold=config.threshold)
    test_eval = evaluate_classifier(model, split.X_test, split.y_test, threshold=config.threshold)

    metrics: dict[str, Any] = {
        "dataset": prepared.as_metadata(),
        "training": {
            "epochs_requested": config.epochs,
            "epochs_run": model.n_iter_,
            "best_epoch": model.best_epoch_,
            "early_stopped": model.converged_,
            "final_train_loss": model.history_[-1]["train_loss"] if model.history_ else None,
            "final_val_loss": model.history_[-1].get("val_loss") if model.history_ else None,
        },
        "parameters": {
            "weights": model.weights_.tolist() if model.weights_ is not None else None,
            "bias": model.bias_,
        },
        "train": train_eval["metrics"],
        "val": val_eval["metrics"],
        "test": test_eval["metrics"],
        "confusion_matrix_train": format_confusion_matrix(
            train_eval["metrics"]["confusion_matrix"]
        ),
        "confusion_matrix_test": format_confusion_matrix(test_eval["metrics"]["confusion_matrix"]),
    }

    recorder = ExperimentRecorder(
        config.name,
        config.to_dict(),
        root=output_root or config.output_dir,
        seed=config.seed,
        notes=notes,
    )
    for row in model.history_:
        recorder.log(row, step=int(row["epoch"]))
    recorder.metadata["dataset"] = prepared.as_metadata()
    recorder.metadata["model_family"] = "numpy-logistic-regression"
    recorder.metadata["known_limitations"] = [
        "The decision boundary is linear; moons, circles, and XOR are expected to remain partially misclassified.",
        "Metrics come from a small synthetic dataset and are not benchmark results.",
        "Only binary classification is supported.",
    ]

    artifacts: dict[str, str] = {}
    if make_plots:
        artifacts["data"] = recorder.save_figure(
            plot_classification_data(
                split.X_train,
                split.y_train,
                title=f"Training split: {config.dataset} (standardized)",
            ),
            "classification_data.png",
        )
        artifacts["sigmoid"] = recorder.save_figure(plot_sigmoid(), "sigmoid.png")
        artifacts["decision_train"] = recorder.save_figure(
            plot_decision_boundary(
                model.predict_proba,
                split.X_train,
                split.y_train,
                title="Decision boundary on the training split",
            ),
            "decision_boundary_train.png",
        )
        artifacts["decision_test"] = recorder.save_figure(
            plot_decision_boundary(
                model.predict_proba,
                split.X_test,
                split.y_test,
                title="Decision boundary on the held-out test split",
            ),
            "decision_boundary_test.png",
        )
        history = {
            "train_loss": [row["train_loss"] for row in model.history_],
            "val_loss": [row.get("val_loss", float("nan")) for row in model.history_],
        }
        artifacts["loss_curve"] = recorder.save_figure(
            plot_loss_curves(
                history,
                title=f"Logistic regression loss ({config.dataset})",
                xlabel="epoch",
            ),
            "loss_curve.png",
        )
        artifacts["confusion_train"] = recorder.save_figure(
            plot_confusion_matrix(
                train_eval["metrics"]["confusion_matrix"],
                title="Training confusion matrix",
            ),
            "confusion_matrix_train.png",
        )
        artifacts["confusion_test"] = recorder.save_figure(
            plot_confusion_matrix(
                test_eval["metrics"]["confusion_matrix"],
                title="Test confusion matrix",
            ),
            "confusion_matrix_test.png",
        )
        artifacts["calibration_test"] = recorder.save_figure(
            plot_calibration(
                test_eval["metrics"]["calibration"],
                title="Test calibration",
            ),
            "calibration_test.png",
        )

    artifacts["model"] = recorder.save_npz(
        "model.npz",
        weights=model.weights_,
        bias=np.array([model.bias_]),
        mean=prepared.scaler.mean_,
        scale=prepared.scaler.scale_,
    )
    recorder.artifacts.update(artifacts)
    recorder.save_json(metrics, "metrics.json")
    result = recorder.finish(metrics, notes=notes)
    return LogisticRun(
        config=config,
        model=model,
        prepared=prepared,
        metrics=to_jsonable(metrics),
        train_eval=train_eval,
        val_eval=val_eval,
        test_eval=test_eval,
        result=result,
        artifacts=artifacts,
    )


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a mapping")
    return value


def _default_notes(config: LogisticConfig) -> str:
    return (
        f"Logistic regression on the {config.dataset} dataset. "
        "Features were standardized with statistics fitted on the training split only. "
        "The model is linear, so any remaining error on a nonlinear dataset is expected."
    )
