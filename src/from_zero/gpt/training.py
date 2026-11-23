"""Training, evaluation, checkpointing, and reporting for MiniGPT."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
import torch

from ..core.checkpointing import load_checkpoint, save_checkpoint
from ..core.experiment import ExperimentRecorder, ExperimentResult, to_jsonable
from ..core.metrics import perplexity
from ..core.plotting import plot_attention, plot_loss_curves
from ..core.seeds import set_seed
from ..core.validation import (
    ConfigError,
    check_bool,
    check_choice,
    check_float,
    check_int,
    reject_unknown_keys,
)
from .dataset import TextCorpus, get_batch, load_text_corpus
from .generation import generate_text
from .model import GPTModelConfig, MiniGPT
from .tokenizer import CharTokenizer


@dataclass
class GPTTrainingConfig:
    name: str = "mini_gpt"
    seed: int = 42
    data_path: str = "data/corpora/tiny_shakespeare.txt"
    val_fraction: float = 0.1
    model: GPTModelConfig = field(default_factory=GPTModelConfig)
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 1500
    eval_interval: int = 100
    eval_batches: int = 10
    log_interval: int = 10
    grad_clip: float = 1.0
    warmup_steps: int = 0
    device: str = "auto"
    sample_prompt: str = "\n"
    sample_tokens: int = 200
    sample_temperature: float = 0.9
    sample_top_k: int = 40
    output_dir: str = "artifacts/experiments"
    make_plots: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigError("name must be a non-empty string")
        check_int("seed", self.seed, minimum=0)
        check_float(
            "val_fraction",
            self.val_fraction,
            minimum=0.0,
            maximum=0.5,
            allow_minimum=False,
            allow_maximum=False,
        )
        if not isinstance(self.model, GPTModelConfig):
            raise ConfigError("model must be a GPTModelConfig")
        check_int("batch_size", self.batch_size, minimum=1, maximum=4096)
        check_float("learning_rate", self.learning_rate, minimum=0.0, allow_minimum=False)
        check_float("weight_decay", self.weight_decay, minimum=0.0)
        check_int("max_steps", self.max_steps, minimum=1, maximum=1_000_000)
        check_int("eval_interval", self.eval_interval, minimum=1)
        check_int("eval_batches", self.eval_batches, minimum=1)
        check_int("log_interval", self.log_interval, minimum=1)
        check_float("grad_clip", self.grad_clip, minimum=0.0, allow_minimum=False)
        check_int("warmup_steps", self.warmup_steps, minimum=0)
        check_choice("device", self.device, ("auto", "cpu", "cuda"))
        if not isinstance(self.sample_prompt, str):
            raise ConfigError("sample_prompt must be a string")
        check_int("sample_tokens", self.sample_tokens, minimum=0, maximum=5000)
        check_float("sample_temperature", self.sample_temperature, minimum=0.0, allow_minimum=False)
        check_int("sample_top_k", self.sample_top_k, minimum=1)
        check_bool("make_plots", self.make_plots)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> GPTTrainingConfig:
        if not isinstance(config, dict):
            raise ConfigError("GPT config must be a mapping")
        nested = any(
            key in config
            for key in ("experiment", "data", "model", "training", "sampling", "output")
        )
        if nested:
            reject_unknown_keys(
                config, ("experiment", "data", "model", "training", "sampling", "output")
            )
            experiment = _section(config, "experiment")
            data = _section(config, "data")
            model_section = _section(config, "model")
            training = _section(config, "training")
            sampling = _section(config, "sampling")
            output = _section(config, "output")
            reject_unknown_keys(experiment, ("name", "seed", "notes"), "experiment section")
            reject_unknown_keys(data, ("path", "data_path", "file", "val_fraction"), "data section")
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
                    "warmup_steps",
                    "device",
                ),
                "training section",
            )
            reject_unknown_keys(
                sampling,
                (
                    "prompt",
                    "sample_prompt",
                    "max_new_tokens",
                    "sample_tokens",
                    "temperature",
                    "sample_temperature",
                    "top_k",
                    "sample_top_k",
                ),
                "sampling section",
            )
            reject_unknown_keys(output, ("directory", "output_dir", "make_plots"), "output section")
            model_config = GPTModelConfig.from_dict(model_section)
            flat = {
                "name": experiment.get("name", cls.name),
                "seed": experiment.get("seed", cls.seed),
                "notes": experiment.get("notes", cls.notes),
                "data_path": data.get(
                    "path", data.get("data_path", data.get("file", cls.data_path))
                ),
                "val_fraction": data.get("val_fraction", cls.val_fraction),
                "model": model_config,
                "batch_size": training.get("batch_size", cls.batch_size),
                "learning_rate": training.get("learning_rate", cls.learning_rate),
                "weight_decay": training.get("weight_decay", cls.weight_decay),
                "max_steps": training.get("max_steps", cls.max_steps),
                "eval_interval": training.get("eval_interval", cls.eval_interval),
                "eval_batches": training.get("eval_batches", cls.eval_batches),
                "log_interval": training.get("log_interval", cls.log_interval),
                "grad_clip": training.get("grad_clip", cls.grad_clip),
                "warmup_steps": training.get("warmup_steps", cls.warmup_steps),
                "device": training.get("device", cls.device),
                "sample_prompt": sampling.get(
                    "prompt", sampling.get("sample_prompt", cls.sample_prompt)
                ),
                "sample_tokens": sampling.get(
                    "max_new_tokens", sampling.get("sample_tokens", cls.sample_tokens)
                ),
                "sample_temperature": sampling.get(
                    "temperature", sampling.get("sample_temperature", cls.sample_temperature)
                ),
                "sample_top_k": sampling.get(
                    "top_k", sampling.get("sample_top_k", cls.sample_top_k)
                ),
                "output_dir": output.get("directory", output.get("output_dir", cls.output_dir)),
                "make_plots": output.get("make_plots", cls.make_plots),
            }
        else:
            allowed = set(cls.__dataclass_fields__)
            reject_unknown_keys(config, allowed, "GPT config")
            flat = {**asdict(cls()), **config}
            model_value = flat.get("model", cls().model)
            flat["model"] = (
                GPTModelConfig.from_dict(model_value)
                if isinstance(model_value, dict)
                else model_value
            )
        return cls(**flat)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": {"name": self.name, "seed": self.seed, "notes": self.notes},
            "data": {"path": self.data_path, "val_fraction": self.val_fraction},
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
                "warmup_steps": self.warmup_steps,
                "device": self.device,
            },
            "sampling": {
                "prompt": self.sample_prompt,
                "max_new_tokens": self.sample_tokens,
                "temperature": self.sample_temperature,
                "top_k": self.sample_top_k,
            },
            "output": {"directory": self.output_dir, "make_plots": self.make_plots},
        }


@dataclass
class GPTRun:
    config: GPTTrainingConfig
    model: MiniGPT
    tokenizer: CharTokenizer
    corpus: TextCorpus
    history: list[dict[str, Any]]
    metrics: dict[str, Any]
    result: ExperimentResult
    artifacts: dict[str, str] = field(default_factory=dict)
    best_val_loss: float = float("inf")
    samples: list[dict[str, Any]] = field(default_factory=list)


def train_gpt(
    config: GPTTrainingConfig | dict[str, Any] | None = None,
    *,
    output_root: str | None = None,
    device: str | None = None,
    max_steps: int | None = None,
    progress_callback: Callable[[int, int, float, float | None], None] | None = None,
    make_plots: bool | None = None,
    notes: str | None = None,
) -> GPTRun:
    """Train MiniGPT and save checkpoints, metrics, plots, and samples."""

    if config is None:
        config = GPTTrainingConfig()
    elif isinstance(config, dict):
        config = GPTTrainingConfig.from_dict(config)
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        config = replace(config, max_steps=int(max_steps))
    if make_plots is None:
        make_plots = config.make_plots
    if notes is None:
        notes = config.notes or (
            "A small character-level decoder-only transformer trained for next-token "
            "prediction. This is a teaching model, not a general-purpose assistant."
        )
    set_seed(config.seed)

    corpus = load_text_corpus(config.data_path, val_fraction=config.val_fraction)
    if config.model.vocab_size not in (0, corpus.tokenizer.vocab_size):
        raise ValueError(
            f"Configured vocab_size={config.model.vocab_size} does not match corpus "
            f"vocab_size={corpus.tokenizer.vocab_size}"
        )
    model_config = replace(config.model, vocab_size=corpus.tokenizer.vocab_size)
    if len(corpus.train_ids) <= model_config.context_length:
        raise ValueError("Training partition is shorter than context_length")
    if len(corpus.val_ids) <= model_config.context_length:
        raise ValueError("Validation partition is shorter than context_length")

    target_device = _select_device(device or config.device)
    model = MiniGPT(model_config).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    train_data = corpus.train_ids.to(target_device)
    val_data = corpus.val_ids.to(target_device)
    train_generator = torch.Generator(device=target_device).manual_seed(config.seed)
    eval_generator = torch.Generator(device=target_device).manual_seed(config.seed + 1000)

    recorder = ExperimentRecorder(
        config.name,
        config.to_dict(),
        root=output_root or config.output_dir,
        seed=config.seed,
        notes=notes,
    )
    recorder.metadata["dataset"] = corpus.metadata
    recorder.metadata["model_family"] = "pytorch-mini-gpt"
    recorder.metadata["known_limitations"] = [
        "Character-level tokenization is inefficient and the corpus is small.",
        "The model is a toy next-token predictor, not an instruction-following assistant.",
        "Perplexity is comparable only for the same corpus and tokenizer.",
        "Generated text is model output and may be nonsensical.",
    ]

    history: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_checkpoint: str | None = None
    final_train_loss = float("nan")
    final_val_loss = float("nan")
    started = time.perf_counter()

    for step in range(1, config.max_steps + 1):
        learning_rate = _learning_rate_at(config, step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        inputs, targets = get_batch(
            train_data,
            block_size=model_config.context_length,
            batch_size=config.batch_size,
            generator=train_generator,
        )
        _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("MiniGPT returned no loss during training")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        final_train_loss = float(loss.item())

        should_evaluate = step % config.eval_interval == 0 or step == config.max_steps
        if step == 1 or step % config.log_interval == 0 or should_evaluate:
            history.append(
                {
                    "step": step,
                    "train_loss": final_train_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        if should_evaluate:
            train_eval = _estimate_loss(
                model,
                train_data,
                block_size=model_config.context_length,
                batch_size=config.batch_size,
                batches=config.eval_batches,
                generator=eval_generator,
            )
            val_eval = _estimate_loss(
                model,
                val_data,
                block_size=model_config.context_length,
                batch_size=config.batch_size,
                batches=config.eval_batches,
                generator=eval_generator,
            )
            final_val_loss = val_eval
            sample_text = _sample_text(model, corpus.tokenizer, config, target_device)
            samples.append(
                {
                    "step": step,
                    "prompt": config.sample_prompt,
                    "text": sample_text,
                    "val_loss": val_eval,
                }
            )
            history.append(
                {
                    "step": step,
                    "train_loss": train_eval,
                    "val_loss": val_eval,
                    "train_perplexity": perplexity(train_eval),
                    "val_perplexity": perplexity(val_eval),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            if val_eval < best_val_loss:
                best_val_loss = val_eval
                best_checkpoint = str(
                    save_checkpoint(
                        recorder.checkpoint_dir / "best.pt",
                        model,
                        optimizer=optimizer,
                        config=config.to_dict(),
                        step=step,
                        extra={
                            "tokenizer": corpus.tokenizer.to_dict(),
                            "val_loss": val_eval,
                            "best_val_loss": best_val_loss,
                        },
                    )
                )
            if progress_callback is not None:
                progress_callback(step, config.max_steps, train_eval, val_eval)

    elapsed = time.perf_counter() - started
    last_checkpoint = save_checkpoint(
        recorder.checkpoint_dir / "last.pt",
        model,
        optimizer=optimizer,
        config=config.to_dict(),
        step=config.max_steps,
        extra={
            "tokenizer": corpus.tokenizer.to_dict(),
            "val_loss": final_val_loss,
            "best_val_loss": best_val_loss,
        },
    )
    tokenizer_path = corpus.tokenizer.save(recorder.directory / "tokenizer.json")
    final_sample = _sample_text(model, corpus.tokenizer, config, target_device)
    final_sample_path = recorder.save_text(final_sample, "samples/final_sample.txt")
    samples_text_path = recorder.save_json(samples, "samples/samples.json")
    recorder.log_many(history)

    eval_rows = [row for row in history if "val_loss" in row]
    metrics: dict[str, Any] = {
        "dataset": corpus.metadata,
        "model": {
            **model_config.to_dict(),
            "parameters": model.num_parameters(),
            "non_embedding_parameters": model.num_parameters(non_embedding=True),
        },
        "training": {
            "steps": config.max_steps,
            "elapsed_seconds": elapsed,
            "tokens_per_second": config.max_steps
            * config.batch_size
            * model_config.context_length
            / max(elapsed, 1e-9),
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "final_val_perplexity": perplexity(final_val_loss),
            "best_val_loss": best_val_loss,
            "best_val_perplexity": perplexity(best_val_loss),
            "device": str(target_device),
        },
        "checkpoints": {
            "best": best_checkpoint,
            "last": str(last_checkpoint),
            "tokenizer": str(tokenizer_path),
        },
        "samples": samples,
        "final_sample": final_sample,
        "final_sample_path": final_sample_path,
        "samples_path": samples_text_path,
    }

    artifacts: dict[str, str] = {
        "best_checkpoint": best_checkpoint or "",
        "last_checkpoint": str(last_checkpoint),
        "tokenizer": str(tokenizer_path),
        "final_sample": final_sample_path,
        "samples": samples_text_path,
    }
    if make_plots:
        if eval_rows:
            artifacts["loss_curve"] = recorder.save_figure(
                plot_loss_curves(
                    {
                        "train_loss": [row["train_loss"] for row in eval_rows],
                        "val_loss": [row["val_loss"] for row in eval_rows],
                    },
                    title=f"MiniGPT loss on {config.data_path.split('/')[-1]}",
                    xlabel="evaluation",
                ),
                "loss_curve.png",
            )
        artifacts["attention"] = recorder.save_figure(
            _attention_figure(model, corpus.tokenizer, config, target_device),
            "attention.png",
        )
    recorder.artifacts.update(artifacts)
    recorder.save_json(metrics, "metrics.json")
    result = recorder.finish(metrics, notes=notes)
    return GPTRun(
        config=config,
        model=model,
        tokenizer=corpus.tokenizer,
        corpus=corpus,
        history=history,
        metrics=to_jsonable(metrics),
        result=result,
        artifacts=artifacts,
        best_val_loss=best_val_loss,
        samples=samples,
    )


def load_gpt_checkpoint(
    path: str | Any,
    *,
    device: str | torch.device = "cpu",
) -> tuple[MiniGPT, CharTokenizer, dict[str, Any]]:
    """Reconstruct a model and tokenizer from a safe checkpoint."""

    target_device = torch.device(device)
    payload = load_checkpoint(path, map_location=target_device)
    config_payload = payload.get("config", {})
    if not isinstance(config_payload, dict):
        raise ValueError("Checkpoint config is missing or invalid")
    model_section = config_payload.get("model", config_payload)
    tokenizer_payload = payload.get("extra", {}).get("tokenizer")
    if tokenizer_payload is None:
        raise ValueError("Checkpoint does not contain a tokenizer")
    tokenizer = CharTokenizer.from_dict(tokenizer_payload)
    model_config = GPTModelConfig.from_dict(model_section)
    if model_config.vocab_size != tokenizer.vocab_size:
        model_config = replace(model_config, vocab_size=tokenizer.vocab_size)
    model = MiniGPT(model_config)
    load_checkpoint(path, model=model, map_location=target_device)
    model.to(target_device)
    model.eval()
    return model, tokenizer, payload


def _estimate_loss(
    model: MiniGPT,
    data: torch.Tensor,
    *,
    block_size: int,
    batch_size: int,
    batches: int,
    generator: torch.Generator,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            inputs, targets = get_batch(
                data,
                block_size=block_size,
                batch_size=batch_size,
                generator=generator,
            )
            _, loss = model(inputs, targets)
            if loss is None:
                raise RuntimeError("MiniGPT returned no loss during evaluation")
            losses.append(float(loss.item()))
    return float(np.mean(losses))


def _sample_text(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    config: GPTTrainingConfig,
    device: torch.device,
) -> str:
    prompt = tokenizer.filter_text(config.sample_prompt)
    if not prompt:
        prompt = tokenizer.chars[0]
    return generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=config.sample_tokens,
        temperature=config.sample_temperature,
        top_k=config.sample_top_k,
        seed=config.seed,
        device=device,
    )


def _attention_figure(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    config: GPTTrainingConfig,
    device: torch.device,
):
    prompt = tokenizer.filter_text(config.sample_prompt) or tokenizer.chars[0]
    prompt = prompt[-min(len(prompt), model.config.context_length) :]
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        _, _, attentions = model(ids, return_attn=True)
    weights = attentions[0][0, 0].detach().cpu().numpy()
    tokens = list(prompt)
    return plot_attention(
        weights,
        tokens,
        title=f"MiniGPT attention: layer 1, head 1 ({len(tokens)} tokens)",
    )


def _learning_rate_at(config: GPTTrainingConfig, step: int) -> float:
    if config.warmup_steps > 0 and step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    return config.learning_rate


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
