"""DDPM reverse sampling."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .model import UNet
from .process import posterior_variance
from .schedules import DiffusionSchedule


@torch.no_grad()
def p_sample(
    model: UNet,
    x: torch.Tensor,
    timestep: int,
    schedule: DiffusionSchedule,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One reverse step from ``x_t`` to ``x_{t-1}``."""

    if x.ndim != 4:
        raise ValueError("x must be a 4-D image batch")
    if not 0 <= timestep < schedule.timesteps:
        raise ValueError("timestep outside the schedule")
    batch = x.shape[0]
    timesteps = torch.full((batch,), timestep, device=x.device, dtype=torch.long)
    predicted_noise = model(x, timesteps)
    alpha = schedule.extract(schedule.alphas, timesteps, x.shape)
    alpha_bar = schedule.extract(schedule.alpha_bars, timesteps, x.shape)
    beta = schedule.extract(schedule.betas, timesteps, x.shape)
    mean = (x - beta / torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
    if timestep > 0:
        variance = posterior_variance(timesteps, schedule, shape=x.shape)
        noise = torch.randn(
            x.shape,
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        return mean + torch.sqrt(variance) * noise
    return mean


@torch.no_grad()
def sample_ddpm(
    model: UNet,
    shape: tuple[int, ...],
    schedule: DiffusionSchedule,
    *,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
    progress: Callable[[int, int], None] | None = None,
    clamp: bool = True,
) -> torch.Tensor:
    """Generate a batch by running the reverse chain from pure Gaussian noise."""

    if len(shape) != 4:
        raise ValueError("shape must be (batch, channels, height, width)")
    target = torch.device(device)
    x = torch.randn(shape, device=target, generator=generator)
    was_training = model.training
    model.eval()
    try:
        for index, timestep in enumerate(reversed(range(schedule.timesteps))):
            x = p_sample(model, x, timestep, schedule, generator=generator)
            if progress is not None:
                progress(index + 1, schedule.timesteps)
    finally:
        model.train(was_training)
    if clamp:
        x = torch.clamp(x, -1.0, 1.0)
    return x
