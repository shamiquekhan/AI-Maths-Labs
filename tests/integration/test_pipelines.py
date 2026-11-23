"""End-to-end integration tests for each laboratory pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from from_zero.autodiff.training import AutodiffConfig, run_autodiff_experiment
from from_zero.logistic.training import LogisticConfig, run_logistic_experiment
from from_zero.optim.compare import OptimizerComparisonConfig, compare_optimizers


def test_logistic_pipeline_writes_complete_experiment(tmp_path) -> None:
    config = LogisticConfig(
        name="test_logistic",
        seed=3,
        dataset="linearly_separable",
        n_samples=160,
        noise=0.1,
        epochs=120,
        batch_size=32,
        early_stopping_patience=None,
        output_dir=str(tmp_path),
        make_plots=True,
    )
    run = run_logistic_experiment(config)
    directory = Path(run.result.directory)
    for name in (
        "config.yaml",
        "metadata.json",
        "metrics.json",
        "metrics.csv",
        "notes.md",
        "model.npz",
    ):
        assert (directory / name).exists(), name
    assert (directory / "plots" / "decision_boundary_test.png").exists()
    assert run.metrics["test"]["accuracy"] >= 0.8
    assert run.model.weights_ is not None
    assert run.test_eval["metrics"]["accuracy"] == pytest.approx(run.metrics["test"]["accuracy"])


def test_optimizer_comparison_pipeline(tmp_path) -> None:
    config = OptimizerComparisonConfig(
        name="test_optimizers",
        seed=5,
        steps=25,
        optimizers=("sgd", "adam"),
        objectives=("quadratic", "rosenbrock"),
        output_dir=str(tmp_path),
        make_plots=False,
    )
    comparison = compare_optimizers(config)
    directory = Path(comparison.result.directory)
    assert (directory / "metrics.json").exists()
    assert (directory / "metrics.csv").exists()
    assert set(comparison.runs) == {"quadratic", "rosenbrock"}
    assert set(comparison.runs["quadratic"]) == {"sgd", "adam"}
    for run in comparison.runs["quadratic"].values():
        assert np.isfinite(run.final_loss)
        assert len(run.losses) == run.steps + 1


def test_autodiff_pipeline_learns_xor(tmp_path) -> None:
    config = AutodiffConfig(
        name="test_autodiff",
        seed=42,
        hidden_sizes=(4, 4),
        epochs=4000,
        learning_rate=0.2,
        target_loss=0.02,
        record_every=50,
        output_dir=str(tmp_path),
        make_plots=True,
    )
    run = run_autodiff_experiment(config)
    directory = Path(run.result.directory)
    assert (directory / "metrics.json").exists()
    assert (directory / "plots" / "xor_decision_boundary.png").exists()
    assert (directory / "plots" / "autodiff_graph.png").exists()
    assert (directory / "model_parameters.json").exists()
    assert run.metrics["accuracy"] == 1.0
    assert run.metrics["solved"] is True
    assert run.metrics["parameter_count"] == (2 * 4 + 4) + (4 * 4 + 4) + (4 * 1 + 1)


def test_gpt_pipeline_trains_reloads_and_generates(tmp_path) -> None:
    pytest.importorskip("torch")
    from from_zero.gpt import (
        GPTModelConfig,
        GPTTrainingConfig,
        generate_text,
        load_gpt_checkpoint,
        train_gpt,
    )

    config = GPTTrainingConfig(
        name="test_gpt",
        seed=2,
        model=GPTModelConfig(
            vocab_size=0,
            context_length=16,
            embedding_dim=16,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        ),
        batch_size=2,
        max_steps=2,
        eval_interval=2,
        eval_batches=1,
        log_interval=1,
        sample_prompt="A",
        sample_tokens=8,
        device="cpu",
        output_dir=str(tmp_path),
        make_plots=True,
    )
    run = train_gpt(config)
    directory = Path(run.result.directory)
    assert (directory / "checkpoint" / "last.pt").exists()
    assert (directory / "checkpoint" / "best.pt").exists()
    assert (directory / "tokenizer.json").exists()
    assert (directory / "plots" / "loss_curve.png").exists()
    assert (directory / "plots" / "attention.png").exists()
    assert (directory / "samples" / "final_sample.txt").exists()
    model, tokenizer, payload = load_gpt_checkpoint(directory / "checkpoint" / "best.pt")
    assert tokenizer.vocab_size == run.corpus.tokenizer.vocab_size
    assert payload["step"] == 2
    generated = generate_text(model, tokenizer, "A", max_new_tokens=5, greedy=True)
    assert len(generated) >= 1


def test_diffusion_pipeline_trains_samples_and_reloads(tmp_path) -> None:
    pytest.importorskip("torch")
    from from_zero.diffusion import (
        DiffusionTrainingConfig,
        UNetConfig,
        load_diffusion_checkpoint,
        sample_ddpm,
        train_diffusion,
    )

    config = DiffusionTrainingConfig(
        name="test_diffusion",
        seed=1,
        dataset="synthetic",
        image_size=8,
        max_samples=32,
        timesteps=5,
        model=UNetConfig(
            in_channels=1,
            base_channels=8,
            channel_mults=(1, 2),
            num_res_blocks=1,
            dropout=0.0,
        ),
        batch_size=2,
        max_steps=2,
        eval_interval=2,
        eval_batches=1,
        log_interval=1,
        sample_count=2,
        sample_interval=2,
        device="cpu",
        output_dir=str(tmp_path),
        make_plots=True,
    )
    run = train_diffusion(config)
    directory = Path(run.result.directory)
    assert (directory / "checkpoint" / "last.pt").exists()
    assert (directory / "checkpoint" / "best.pt").exists()
    assert (directory / "plots" / "noise_schedule.png").exists()
    assert (directory / "plots" / "forward_process.png").exists()
    assert (directory / "samples" / "final_samples.npz").exists()
    model, schedule, payload = load_diffusion_checkpoint(directory / "checkpoint" / "best.pt")
    assert schedule.timesteps == 5
    assert payload["step"] == 2
    samples = sample_ddpm(model, (2, 1, 8, 8), schedule, device="cpu")
    assert samples.shape == (2, 1, 8, 8)
    assert np.isfinite(samples.numpy()).all()


def test_experiment_catalog_and_app_services() -> None:
    from from_zero.app import services

    catalog = services.experiment_catalog()
    assert isinstance(catalog, list)
    if catalog:
        detail = services.experiment_detail(catalog[0]["name"])
        assert detail["name"]
        assert isinstance(detail["metrics"], dict)
    checkpoints = services.find_checkpoints("pytorch-mini-gpt")
    assert all(Path(path).exists() for path in checkpoints)


def test_saved_configs_parse() -> None:
    from from_zero.autodiff.training import AutodiffConfig
    from from_zero.core.validation import load_config
    from from_zero.diffusion.training import DiffusionTrainingConfig
    from from_zero.gpt.training import GPTTrainingConfig
    from from_zero.logistic.training import LogisticConfig

    assert LogisticConfig.from_dict(load_config("configs/logistic_regression.yaml"))
    assert AutodiffConfig.from_dict(load_config("configs/micrograd.yaml"))
    assert GPTTrainingConfig.from_dict(load_config("configs/mini_gpt.yaml"))
    assert DiffusionTrainingConfig.from_dict(load_config("configs/ddpm_mnist.yaml"))
    metadata = json.loads(Path("artifacts/experiments/ddpm_mnist/metadata.json").read_text())
    assert metadata["seed"] == 42
