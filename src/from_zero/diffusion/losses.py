"""The DDPM noise-prediction objective."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from .model import UNet
from .process import q_sample
from .schedules import DiffusionSchedule


def diffusion_loss(
    model: UNet,
    x0: torch.Tensor,
    schedule: DiffusionSchedule,
    *,
    generator: torch.Generator | None = None,
    timesteps: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sample ``t`` and ``ε``, noise the batch, and predict the noise."""

    if x0.ndim != 4:
        raise ValueError(f"x0 must be (batch, channels, height, width), got {tuple(x0.shape)}")
    if not torch.all(torch.isfinite(x0)):
        raise ValueError("x0 contains NaN or infinity")
    batch = x0.shape[0]
    if timesteps is None:
        timesteps = torch.randint(
            0,
            schedule.timesteps,
            (batch,),
            device=x0.device,
            generator=generator,
        )
    if timesteps.shape != (batch,):
        raise ValueError("timesteps must have shape (batch,)")
    if noise is None:
        noise = torch.randn(
            x0.shape,
            device=x0.device,
            dtype=x0.dtype,
            generator=generator,
        )
    if noise.shape != x0.shape:
        raise ValueError("noise must have the same shape as x0")
    noisy = q_sample(x0, timesteps, noise, schedule)
    prediction = model(noisy, timesteps)
    loss = F.mse_loss(prediction, noise)
    return loss, {
        "timesteps": timesteps,
        "noise": noise,
        "noisy": noisy,
        "prediction": prediction,
    }


def describe_loss_batch(info: dict[str, Any]) -> dict[str, float]:
    """Small helper for logs and tests."""

    return {
        "timestep_mean": float(info["timesteps"].float().mean().item()),
        "noise_std": float(info["noise"].std().item()),
        "noisy_std": float(info["noisy"].std().item()),
        "prediction_std": float(info["prediction"].std().item()),
    }
