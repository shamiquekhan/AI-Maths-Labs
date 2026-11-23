"""DDPM training, evaluation, sampling, checkpointing, and reports."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
import torch

from ..core.checkpointing import load_checkpoint, save_checkpoint
from ..core.experiment import ExperimentRecorder, ExperimentResult, to_jsonable
from ..core.plotting import (
    plot_forward_diffusion,
    plot_image_grid,
    plot_loss_curves,
    plot_noise_schedule,
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
from .data import ImageData, load_image_dataset
from .losses import diffusion_loss
from .model import UNet, UNetConfig
from .process import q_sample
from .sampling import sample_ddpm
from .schedules import DiffusionSchedule, make_schedule


@dataclass
class DiffusionTrainingConfig:
    name: str = "ddpm_mnist"
    seed: int = 42
    dataset: str = "mnist"
    image_size: int = 28
    max_samples: int | None = 6000
    data_dir: str = "data"
    download: bool = True
    timesteps: int = 1000
    schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    cosine_s: float = 0.008
    model: UNetConfig = field(default_factory=UNetConfig)
    batch_size: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    max_steps: int = 3000
    eval_interval: int = 200
    eval_batches: int = 5
    log_interval: int = 20
    grad_clip: float = 1.0
    sample_count: int = 16
    sample_interval: int = 500
    device: str = "auto"
    output_dir: str = "artifacts/experiments"
    make_plots: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigError("name must be a non-empty string")
        check_int("seed", self.seed, minimum=0)
        check_choice("dataset", self.dataset, ("mnist", "fashion_mnist", "synthetic"))
        check_int("image_size", self.image_size, minimum=8, maximum=128)
        if self.max_samples is not None:
            check_int("max_samples", self.max_samples, minimum=16, maximum=1_000_000)
        check_bool("download", self.download)
        check_int("timesteps", self.timesteps, minimum=2, maximum=2000)
        check_choice("schedule", self.schedule, ("linear", "cosine"))
        check_float(
            "beta_start",
            self.beta_start,
            minimum=0.0,
            maximum=1.0,
            allow_minimum=False,
            allow_maximum=False,
        )
        check_float(
            "beta_end",
            self.beta_end,
            minimum=0.0,
            maximum=1.0,
            allow_minimum=False,
            allow_maximum=False,
        )
        if self.beta_start >= self.beta_end:
            raise ConfigError("beta_start must be less than beta_end")
        check_float("cosine_s", self.cosine_s, minimum=0.0)
        if not isinstance(self.model, UNetConfig):
            raise ConfigError("model must be a UNetConfig")
        check_int("batch_size", self.batch_size, minimum=1, maximum=2048)
        check_float("learning_rate", self.learning_rate, minimum=0.0, allow_minimum=False)
        check_float("weight_decay", self.weight_decay, minimum=0.0)
        check_int("max_steps", self.max_steps, minimum=1, maximum=1_000_000)
        check_int("eval_interval", self.eval_interval, minimum=1)
        check_int("eval_batches", self.eval_batches, minimum=1)
        check_int("log_interval", self.log_interval, minimum=1)
        check_float("grad_clip", self.grad_clip, minimum=0.0, allow_minimum=False)
        check_int("sample_count", self.sample_count, minimum=1, maximum=64)
        check_int("sample_interval", self.sample_interval, minimum=1)
        check_choice("device", self.device, ("auto", "cpu", "cuda"))
        check_bool("make_plots", self.make_plots)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> DiffusionTrainingConfig:
        if not isinstance(config, dict):
            raise ConfigError("Diffusion config must be a mapping")
        nested = any(
            key in config
            for key in (
                "experiment",
                "data",
                "diffusion",
                "model",
                "training",
                "sampling",
                "output",
            )
        )
        if nested:
            reject_unknown_keys(
                config,
                ("experiment", "data", "diffusion", "model", "training", "sampling", "output"),
            )
            experiment = _section(config, "experiment")
            data = _section(config, "data")
            diffusion = _section(config, "diffusion")
            model_section = _section(config, "model")
            training = _section(config, "training")
            sampling = _section(config, "sampling")
            output = _section(config, "output")
            reject_unknown_keys(experiment, ("name", "seed", "notes"), "experiment section")
            reject_unknown_keys(
                data,
                ("dataset", "image_size", "max_samples", "data_dir", "download"),
                "data section",
            )
            reject_unknown_keys(
                diffusion,
                ("timesteps", "schedule", "beta_start", "beta_end", "cosine_s"),
                "diffusion section",
            )
            reject_unknown_keys(
                training,
                (
                    "batch_size",
                    "learning_rate",
                    "weight_decay",
                    "max_steps",
                    "eval_interval",
                    "eval_batches",
                    "log_interval",
                    "grad_clip",
                    "device",
                ),
                "training section",
            )
            reject_unknown_keys(
                sampling,
                ("sample_count", "sample_interval"),
                "sampling section",
            )
            reject_unknown_keys(output, ("directory", "output_dir", "make_plots"), "output section")
            flat = {
                "name": experiment.get("name", cls.name),
                "seed": experiment.get("seed", cls.seed),
                "notes": experiment.get("notes", cls.notes),
                "dataset": data.get("dataset", cls.dataset),
                "image_size": data.get("image_size", cls.image_size),
                "max_samples": data.get("max_samples", cls.max_samples),
                "data_dir": data.get("data_dir", cls.data_dir),
                "download": data.get("download", cls.download),
                "timesteps": diffusion.get("timesteps", cls.timesteps),
                "schedule": diffusion.get("schedule", cls.schedule),
                "beta_start": diffusion.get("beta_start", cls.beta_start),
                "beta_end": diffusion.get("beta_end", cls.beta_end),
                "cosine_s": diffusion.get("cosine_s", cls.cosine_s),
                "model": UNetConfig.from_dict(model_section),
                "batch_size": training.get("batch_size", cls.batch_size),
                "learning_rate": training.get("learning_rate", cls.learning_rate),
                "weight_decay": training.get("weight_decay", cls.weight_decay),
                "max_steps": training.get("max_steps", cls.max_steps),
                "eval_interval": training.get("eval_interval", cls.eval_interval),
                "eval_batches": training.get("eval_batches", cls.eval_batches),
                "log_interval": training.get("log_interval", cls.log_interval),
                "grad_clip": training.get("grad_clip", cls.grad_clip),
                "sample_count": sampling.get("sample_count", cls.sample_count),
                "sample_interval": sampling.get("sample_interval", cls.sample_interval),
                "device": training.get("device", cls.device),
                "output_dir": output.get("directory", output.get("output_dir", cls.output_dir)),
                "make_plots": output.get("make_plots", cls.make_plots),
            }
        else:
            allowed = set(cls.__dataclass_fields__)
            reject_unknown_keys(config, allowed, "diffusion config")
            flat = {**asdict(cls()), **config}
            model_value = flat.get("model", cls().model)
            flat["model"] = (
                UNetConfig.from_dict(model_value) if isinstance(model_value, dict) else model_value
            )
        return cls(**flat)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": {"name": self.name, "seed": self.seed, "notes": self.notes},
            "data": {
                "dataset": self.dataset,
                "image_size": self.image_size,
                "max_samples": self.max_samples,
                "data_dir": self.data_dir,
                "download": self.download,
            },
            "diffusion": {
                "timesteps": self.timesteps,
                "schedule": self.schedule,
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
                "cosine_s": self.cosine_s,
            },
            "model": self.model.to_dict(),
            "training": {
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "max_steps": self.max_steps,
                "eval_interval": self.eval_interval,
                "eval_batches": self.eval_batches,
                "log_interval": self.log_interval,
                "grad_clip": self.grad_clip,
                "device": self.device,
            },
            "sampling": {
                "sample_count": self.sample_count,
                "sample_interval": self.sample_interval,
            },
            "output": {"directory": self.output_dir, "make_plots": self.make_plots},
        }


@dataclass
class DiffusionRun:
    config: DiffusionTrainingConfig
    model: UNet
    schedule: DiffusionSchedule
    data: ImageData
    history: list[dict[str, Any]]
    metrics: dict[str, Any]
    result: ExperimentResult
    artifacts: dict[str, str] = field(default_factory=dict)
    best_val_loss: float = float("inf")
    samples: list[dict[str, Any]] = field(default_factory=list)


def train_diffusion(
    config: DiffusionTrainingConfig | dict[str, Any] | None = None,
    *,
    output_root: str | None = None,
    device: str | None = None,
    max_steps: int | None = None,
    progress_callback: Callable[[int, int, float, float | None], None] | None = None,
    make_plots: bool | None = None,
    notes: str | None = None,
) -> DiffusionRun:
    if config is None:
        config = DiffusionTrainingConfig()
    elif isinstance(config, dict):
        config = DiffusionTrainingConfig.from_dict(config)
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        config = replace(config, max_steps=int(max_steps))
    if make_plots is None:
        make_plots = config.make_plots
    if notes is None:
        notes = config.notes or (
            "A compact DDPM trained to predict Gaussian noise at random timesteps. "
            "Sample quality is limited by the small model, short training budget, and low resolution."
        )
    set_seed(config.seed)

    data = load_image_dataset(
        config.dataset,
        image_size=config.image_size,
        max_samples=config.max_samples,
        data_dir=config.data_dir,
        download=config.download,
        seed=config.seed,
    )
    schedule = make_schedule(
        config.schedule,
        timesteps=config.timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        cosine_s=config.cosine_s,
    )
    target_device = _select_device(device or config.device)
    model = UNet(config.model).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_data = data.train.to(target_device)
    val_data = data.val.to(target_device)
    train_generator = torch.Generator(device=target_device).manual_seed(config.seed)
    eval_generator = torch.Generator(device=target_device).manual_seed(config.seed + 1000)
    sample_generator = torch.Generator(device=target_device).manual_seed(config.seed + 2000)

    recorder = ExperimentRecorder(
        config.name,
        config.to_dict(),
        root=output_root or config.output_dir,
        seed=config.seed,
        notes=notes,
    )
    recorder.metadata["dataset"] = data.metadata
    recorder.metadata["model_family"] = "pytorch-ddpm-unet"
    recorder.metadata["schedule"] = schedule.metadata()
    recorder.metadata["known_limitations"] = [
        "The model is small and the training budget is short; samples are not comparable to large-scale diffusion results.",
        "FID is not reported because a defensible evaluation pipeline is outside this teaching scope.",
        "The reverse process is sensitive to schedule and indexing conventions; see the derivation notes.",
        "MNIST/Fashion-MNIST use is subject to their original dataset terms.",
    ]

    history: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_checkpoint: str | None = None
    final_train_loss = float("nan")
    final_val_loss = float("nan")
    started = time.perf_counter()

    for step in range(1, config.max_steps + 1):
        model.train()
        x0 = _image_batch(train_data, config.batch_size, train_generator)
        loss, _ = diffusion_loss(model, x0, schedule, generator=train_generator)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        final_train_loss = float(loss.item())

        should_evaluate = step % config.eval_interval == 0 or step == config.max_steps
        should_sample = step % config.sample_interval == 0 or step == config.max_steps
        if step == 1 or step % config.log_interval == 0 or should_evaluate:
            history.append({"step": step, "train_loss": final_train_loss})

        if should_evaluate:
            val_loss = _estimate_loss(
                model,
                val_data,
                schedule,
                batch_size=min(config.batch_size, len(val_data)),
                batches=config.eval_batches,
                generator=eval_generator,
            )
            final_val_loss = val_loss
            history.append(
                {
                    "step": step,
                    "train_loss": _estimate_loss(
                        model,
                        train_data,
                        schedule,
                        batch_size=min(config.batch_size, len(train_data)),
                        batches=config.eval_batches,
                        generator=eval_generator,
                    ),
                    "val_loss": val_loss,
                }
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = str(
                    save_checkpoint(
                        recorder.checkpoint_dir / "best.pt",
                        model,
                        optimizer=optimizer,
                        config=config.to_dict(),
                        step=step,
                        extra={
                            "schedule": schedule.metadata(),
                            "val_loss": val_loss,
                            "best_val_loss": best_val_loss,
                        },
                    )
                )
            if progress_callback is not None:
                progress_callback(step, config.max_steps, final_train_loss, val_loss)

        if should_sample:
            sample_images = sample_ddpm(
                model,
                (config.sample_count, 1, config.image_size, config.image_size),
                schedule,
                device=target_device,
                generator=sample_generator,
            )
            sample_path = recorder.save_figure(
                plot_image_grid(
                    sample_images,
                    nrow=4,
                    title=f"DDPM samples at step {step}",
                ),
                f"samples/step_{step:06d}.png",
            )
            samples.append(
                {
                    "step": step,
                    "path": sample_path,
                    "val_loss": final_val_loss if np.isfinite(final_val_loss) else None,
                }
            )

    elapsed = time.perf_counter() - started
    last_checkpoint = save_checkpoint(
        recorder.checkpoint_dir / "last.pt",
        model,
        optimizer=optimizer,
        config=config.to_dict(),
        step=config.max_steps,
        extra={
            "schedule": schedule.metadata(),
            "val_loss": final_val_loss,
            "best_val_loss": best_val_loss,
        },
    )
    final_samples = sample_ddpm(
        model,
        (config.sample_count, 1, config.image_size, config.image_size),
        schedule,
        device=target_device,
        generator=sample_generator,
    )
    final_sample_path = recorder.save_figure(
        plot_image_grid(final_samples, nrow=4, title="Final DDPM samples"),
        "samples/final_samples.png",
    )
    final_npz = recorder.save_npz("samples/final_samples.npz", images=final_samples.cpu().numpy())
    recorder.log_many(history)

    eval_rows = [row for row in history if "val_loss" in row]
    metrics: dict[str, Any] = {
        "dataset": data.metadata,
        "model": {**config.model.to_dict(), "parameters": model.num_parameters()},
        "schedule": schedule.metadata(),
        "training": {
            "steps": config.max_steps,
            "elapsed_seconds": elapsed,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "best_val_loss": best_val_loss,
            "device": str(target_device),
        },
        "samples": samples,
        "checkpoints": {
            "best": best_checkpoint,
            "last": str(last_checkpoint),
        },
        "final_samples": final_npz,
    }

    artifacts: dict[str, str] = {
        "best_checkpoint": best_checkpoint or "",
        "last_checkpoint": str(last_checkpoint),
        "final_samples": final_sample_path,
        "final_samples_npz": final_npz,
    }
    if make_plots:
        artifacts["schedule"] = recorder.save_figure(
            plot_noise_schedule(schedule, title=f"{config.schedule.title()} noise schedule"),
            "noise_schedule.png",
        )
        if eval_rows:
            artifacts["loss_curve"] = recorder.save_figure(
                plot_loss_curves(
                    {
                        "train_loss": [row["train_loss"] for row in eval_rows],
                        "val_loss": [row["val_loss"] for row in eval_rows],
                    },
                    title="DDPM noise-prediction loss",
                    xlabel="evaluation",
                ),
                "loss_curve.png",
            )
        artifacts["forward_process"] = recorder.save_figure(
            _forward_process_figure(val_data, schedule),
            "forward_process.png",
        )
    recorder.artifacts.update(artifacts)
    recorder.save_json(metrics, "metrics.json")
    result = recorder.finish(metrics, notes=notes)
    return DiffusionRun(
        config=config,
        model=model,
        schedule=schedule,
        data=data,
        history=history,
        metrics=to_jsonable(metrics),
        result=result,
        artifacts=artifacts,
        best_val_loss=best_val_loss,
        samples=samples,
    )


def load_diffusion_checkpoint(
    path: str | Any,
    *,
    device: str | torch.device = "cpu",
) -> tuple[UNet, DiffusionSchedule, dict[str, Any]]:
    target_device = torch.device(device)
    payload = load_checkpoint(path, map_location=target_device)
    config_payload = payload.get("config", {})
    if not isinstance(config_payload, dict):
        raise ValueError("Checkpoint config is missing or invalid")
    model_config = UNetConfig.from_dict(config_payload["model"])
    diffusion_config = config_payload["diffusion"]
    schedule = make_schedule(
        diffusion_config.get("schedule", "linear"),
        timesteps=int(diffusion_config.get("timesteps", 1000)),
        beta_start=float(diffusion_config.get("beta_start", 1e-4)),
        beta_end=float(diffusion_config.get("beta_end", 0.02)),
        cosine_s=float(diffusion_config.get("cosine_s", 0.008)),
    )
    model = UNet(model_config)
    load_checkpoint(path, model=model, map_location=target_device)
    model.to(target_device)
    model.eval()
    return model, schedule, payload


def _image_batch(data: torch.Tensor, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    indices = torch.randint(0, len(data), (batch_size,), device=data.device, generator=generator)
    return data[indices]


def _estimate_loss(
    model: UNet,
    data: torch.Tensor,
    schedule: DiffusionSchedule,
    *,
    batch_size: int,
    batches: int,
    generator: torch.Generator,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x0 = _image_batch(data, batch_size, generator)
            loss, _ = diffusion_loss(model, x0, schedule, generator=generator)
            losses.append(float(loss.item()))
    return float(np.mean(losses))


def _forward_process_figure(data: torch.Tensor, schedule: DiffusionSchedule):
    image = data[0:1]
    timesteps = sorted(
        {
            0,
            schedule.timesteps // 4,
            schedule.timesteps // 2,
            (3 * schedule.timesteps) // 4,
            schedule.timesteps - 1,
        }
    )
    noise = torch.randn(image.shape, device=image.device, dtype=image.dtype)
    noisy = torch.stack(
        [
            q_sample(
                image,
                torch.tensor([t], device=image.device, dtype=torch.long),
                noise,
                schedule,
            )[0]
            for t in timesteps
        ]
    )
    return plot_forward_diffusion(
        noisy,
        timesteps,
        title="Forward diffusion on one validation image",
    )


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device {requested!r}; choose auto, cpu, or cuda")


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a mapping")
    return value
