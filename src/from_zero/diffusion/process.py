"""Forward noising and posterior quantities for DDPM."""

from __future__ import annotations

import torch

from .schedules import DiffusionSchedule


def q_sample(
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    """Sample ``x_t`` directly with the closed-form forward process."""

    _validate_images(x0, "x0")
    if noise.shape != x0.shape:
        raise ValueError(f"noise shape {tuple(noise.shape)} must match x0 shape {tuple(x0.shape)}")
    _validate_timesteps(timesteps, x0.shape[0], schedule)
    sqrt_alpha_bar = schedule.extract(torch.sqrt(schedule.alpha_bars), timesteps, x0.shape)
    sqrt_one_minus = schedule.extract(torch.sqrt(1.0 - schedule.alpha_bars), timesteps, x0.shape)
    return sqrt_alpha_bar * x0 + sqrt_one_minus * noise


def predict_x0_from_noise(
    xt: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    """Recover the implied clean image from a noise prediction."""

    _validate_images(xt, "xt")
    if noise.shape != xt.shape:
        raise ValueError("noise and xt must have the same shape")
    _validate_timesteps(timesteps, xt.shape[0], schedule)
    alpha_bar = schedule.extract(schedule.alpha_bars, timesteps, xt.shape)
    return (xt - torch.sqrt(1.0 - alpha_bar) * noise) / torch.sqrt(alpha_bar)


def predict_noise_from_x0(
    x0: torch.Tensor,
    xt: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    _validate_images(x0, "x0")
    if x0.shape != xt.shape:
        raise ValueError("x0 and xt must have the same shape")
    _validate_timesteps(timesteps, x0.shape[0], schedule)
    alpha_bar = schedule.extract(schedule.alpha_bars, timesteps, x0.shape)
    return (xt - torch.sqrt(alpha_bar) * x0) / torch.sqrt(1.0 - alpha_bar)


def posterior_mean(
    x0: torch.Tensor,
    xt: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    """Mean of ``q(x_{t-1} | x_t, x_0)`` for the DDPM parameterisation."""

    _validate_images(x0, "x0")
    if x0.shape != xt.shape:
        raise ValueError("x0 and xt must have the same shape")
    _validate_timesteps(timesteps, x0.shape[0], schedule)
    previous = schedule.previous_alpha_bars
    alpha_bar_prev = schedule.extract(previous, timesteps, x0.shape)
    alpha_bar = schedule.extract(schedule.alpha_bars, timesteps, x0.shape)
    alpha = schedule.extract(schedule.alphas, timesteps, x0.shape)
    beta = schedule.extract(schedule.betas, timesteps, x0.shape)
    coefficient_x0 = torch.sqrt(alpha_bar_prev) * beta / (1.0 - alpha_bar)
    coefficient_xt = torch.sqrt(alpha) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
    # At t=0 the posterior is exactly x_0.  The formula above is mathematically
    # equivalent, but floating-point cancellation in (1 - alpha_bar) leaves a
    # tiny multiplicative error; special-casing keeps the boundary exact.
    is_zero = (timesteps == 0).view((-1,) + (1,) * (x0.ndim - 1))
    coefficient_x0 = torch.where(is_zero, torch.ones_like(coefficient_x0), coefficient_x0)
    coefficient_xt = torch.where(is_zero, torch.zeros_like(coefficient_xt), coefficient_xt)
    return coefficient_x0 * x0 + coefficient_xt * xt


def posterior_variance(
    timesteps: torch.Tensor,
    schedule: DiffusionSchedule,
    *,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be 1-D")
    values = schedule.posterior_variance.to(timesteps.device)[timesteps]
    if shape is not None:
        values = values.view((len(timesteps),) + (1,) * (len(shape) - 1)).expand(shape)
    return values


def _validate_images(images: torch.Tensor, name: str) -> None:
    if images.ndim != 4:
        raise ValueError(
            f"{name} must be (batch, channels, height, width), got {tuple(images.shape)}"
        )
    if not torch.all(torch.isfinite(images)):
        raise ValueError(f"{name} contains NaN or infinity")


def _validate_timesteps(
    timesteps: torch.Tensor, batch_size: int, schedule: DiffusionSchedule
) -> None:
    if timesteps.ndim != 1 or len(timesteps) != batch_size:
        raise ValueError("timesteps must be 1-D with one entry per batch item")
    if timesteps.dtype != torch.long:
        raise TypeError("timesteps must have dtype torch.long")
    if timesteps.numel() and (timesteps.min() < 0 or timesteps.max() >= schedule.timesteps):
        raise ValueError("timestep outside the schedule")
