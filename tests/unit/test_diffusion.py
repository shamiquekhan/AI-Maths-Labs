"""Unit tests for the DDPM mathematics, model, and sampler."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from from_zero.core.validation import ConfigError  # noqa: E402
from from_zero.diffusion import (  # noqa: E402
    UNet,
    UNetConfig,
    diffusion_loss,
    load_image_dataset,
    make_schedule,
    make_synthetic_images,
    p_sample,
    posterior_mean,
    posterior_variance,
    predict_x0_from_noise,
    q_sample,
    sample_ddpm,
)
from from_zero.diffusion.training import DiffusionTrainingConfig  # noqa: E402


@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_schedules_are_valid_and_monotone(kind: str) -> None:
    schedule = make_schedule(kind, timesteps=50)
    assert schedule.timesteps == 50
    assert schedule.betas.shape == (50,)
    assert torch.all(schedule.betas > 0) and torch.all(schedule.betas < 1)
    assert torch.all(schedule.alpha_bars > 0)
    assert torch.all(schedule.alpha_bars[1:] <= schedule.alpha_bars[:-1])
    assert schedule.posterior_variance[0] == 0.0
    assert torch.all(schedule.posterior_variance[1:] >= 0)
    metadata = schedule.metadata()
    assert metadata["kind"] == kind
    assert 0.0 < metadata["alpha_bar_last"] < 1.0
    with pytest.raises(ValueError):
        make_schedule("unknown", timesteps=10)


def test_schedule_extract_broadcasts() -> None:
    schedule = make_schedule("linear", timesteps=10)
    timesteps = torch.tensor([0, 9])
    values = schedule.extract(schedule.alpha_bars, timesteps, (2, 1, 4, 4))
    assert values.shape == (2, 1, 4, 4)
    assert torch.allclose(values[0], torch.full((1, 4, 4), schedule.alpha_bars[0]))
    with pytest.raises(ValueError):
        schedule.extract(schedule.alpha_bars, torch.tensor([10]), (1, 1, 4, 4))


def test_q_sample_matches_closed_form_and_inverts() -> None:
    schedule = make_schedule("linear", timesteps=20)
    x0 = torch.randn(3, 1, 8, 8)
    noise = torch.randn_like(x0)
    timesteps = torch.tensor([0, 10, 19])
    xt = q_sample(x0, timesteps, noise, schedule)
    alpha_bar = schedule.alpha_bars[timesteps].view(3, 1, 1, 1)
    expected = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise
    assert torch.allclose(xt, expected)
    recovered = predict_x0_from_noise(xt, timesteps, noise, schedule)
    assert torch.allclose(recovered, x0, atol=1e-5)
    with pytest.raises(ValueError):
        q_sample(x0, torch.tensor([0, 1]), noise, schedule)
    with pytest.raises(TypeError):
        q_sample(x0, torch.tensor([0.0, 1.0, 2.0]), noise, schedule)


def test_posterior_mean_at_zero_is_clean_image() -> None:
    schedule = make_schedule("linear", timesteps=10)
    x0 = torch.randn(2, 1, 4, 4)
    noise = torch.randn_like(x0)
    timesteps = torch.tensor([0, 0])
    xt = q_sample(x0, timesteps, noise, schedule)
    mean = posterior_mean(x0, xt, timesteps, schedule)
    assert torch.allclose(mean, x0, atol=1e-5)
    variance = posterior_variance(timesteps, schedule, shape=x0.shape)
    assert variance.shape == x0.shape
    assert torch.all(variance == 0.0)


def test_posterior_mean_matches_direct_formula() -> None:
    schedule = make_schedule("cosine", timesteps=30)
    x0 = torch.randn(2, 1, 4, 4)
    noise = torch.randn_like(x0)
    timesteps = torch.tensor([5, 17])
    xt = q_sample(x0, timesteps, noise, schedule)
    mean = posterior_mean(x0, xt, timesteps, schedule)
    alpha_bar = schedule.alpha_bars[timesteps].view(2, 1, 1, 1)
    alpha_bar_prev = schedule.previous_alpha_bars[timesteps].view(2, 1, 1, 1)
    beta = schedule.betas[timesteps].view(2, 1, 1, 1)
    alpha = schedule.alphas[timesteps].view(2, 1, 1, 1)
    expected = (
        torch.sqrt(alpha_bar_prev) * beta / (1.0 - alpha_bar) * x0
        + torch.sqrt(alpha) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * xt
    )
    assert torch.allclose(mean, expected)
    assert torch.all(schedule.posterior_variance[1:] < 1.0)


def test_unet_forward_and_backward() -> None:
    config = UNetConfig(
        in_channels=1,
        base_channels=8,
        channel_mults=(1, 2),
        num_res_blocks=1,
        dropout=0.0,
    )
    model = UNet(config)
    images = torch.randn(2, 1, 16, 16)
    timesteps = torch.tensor([0, 9], dtype=torch.long)
    output = model(images, timesteps)
    assert output.shape == images.shape
    loss = output.square().mean()
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert model.num_parameters() > 0
    with pytest.raises(ValueError):
        model(images, torch.tensor([0]))
    with pytest.raises(TypeError):
        model(images, torch.tensor([0.0, 1.0]))
    with pytest.raises(ConfigError):
        UNetConfig(base_channels=4)
    with pytest.raises(ConfigError):
        UNetConfig(channel_mults=())


def test_diffusion_loss_is_finite_and_differentiable() -> None:
    torch.manual_seed(4)
    schedule = make_schedule("linear", timesteps=10)
    model = UNet(UNetConfig(base_channels=8, channel_mults=(1, 2), num_res_blocks=1))
    x0 = torch.randn(2, 1, 8, 8)
    timesteps = torch.tensor([1, 8])
    noise = torch.randn_like(x0)
    loss, info = diffusion_loss(model, x0, schedule, timesteps=timesteps, noise=noise)
    assert torch.isfinite(loss)
    assert info["noisy"].shape == x0.shape
    assert info["prediction"].shape == x0.shape
    loss.backward()
    assert loss.item() >= 0.0
    with pytest.raises(ValueError):
        diffusion_loss(model, torch.zeros(2, 1, 8, 8), schedule, timesteps=torch.tensor([0]))


def test_reverse_step_and_full_sampling() -> None:
    torch.manual_seed(0)
    schedule = make_schedule("linear", timesteps=4)
    model = UNet(UNetConfig(base_channels=8, channel_mults=(1, 2), num_res_blocks=1)).eval()
    x = torch.randn(2, 1, 8, 8)
    mean_only = p_sample(model, x, 0, schedule)
    assert mean_only.shape == x.shape
    # At t=0 there is no added noise: the same input and model must give the same output.
    again = p_sample(model, x, 0, schedule)
    assert torch.allclose(mean_only, again)
    sampled = sample_ddpm(
        model, (2, 1, 8, 8), schedule, device="cpu", generator=torch.Generator().manual_seed(1)
    )
    assert sampled.shape == (2, 1, 8, 8)
    assert torch.all(sampled >= -1.0) and torch.all(sampled <= 1.0)
    with pytest.raises(ValueError):
        p_sample(model, x, 4, schedule)


def test_synthetic_images_are_deterministic_and_bounded() -> None:
    first = make_synthetic_images(4, image_size=16, seed=3)
    second = make_synthetic_images(4, image_size=16, seed=3)
    assert first.shape == (4, 1, 16, 16)
    assert torch.equal(first, second)
    assert torch.all(first >= -1.0) and torch.all(first <= 1.0)
    data = load_image_dataset("synthetic", image_size=16, max_samples=32, seed=3)
    assert data.train.shape[1:] == (1, 16, 16)
    assert len(data.train) + len(data.val) == 32
    assert data.metadata["source"].startswith("synthetic")
    with pytest.raises(ValueError):
        load_image_dataset("synthetic", max_samples=4)
    with pytest.raises(ValueError):
        load_image_dataset("unknown")


def test_diffusion_config_parsing() -> None:
    config = DiffusionTrainingConfig.from_dict(
        {
            "experiment": {"name": "tiny-ddpm", "seed": 2},
            "data": {"dataset": "synthetic", "image_size": 16, "max_samples": 32},
            "diffusion": {"timesteps": 20, "schedule": "cosine"},
            "model": {"base_channels": 8, "channel_mults": [1, 2], "num_res_blocks": 1},
            "training": {"max_steps": 3, "batch_size": 2, "device": "cpu"},
            "sampling": {"sample_count": 2, "sample_interval": 3},
            "output": {"directory": "artifacts/experiments"},
        }
    )
    assert config.name == "tiny-ddpm"
    assert config.schedule == "cosine"
    assert config.model.base_channels == 8
    with pytest.raises(ConfigError):
        DiffusionTrainingConfig.from_dict({"unknown": 1})
    with pytest.raises(ConfigError):
        DiffusionTrainingConfig(beta_start=0.1, beta_end=0.01)
