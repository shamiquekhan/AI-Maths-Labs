"""Variance schedules for the forward diffusion process."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DiffusionSchedule:
    """Precomputed beta, alpha, and cumulative alpha-bar tensors."""

    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    timesteps: int
    kind: str

    def __post_init__(self) -> None:
        if self.betas.ndim != 1:
            raise ValueError("betas must be 1-D")
        if len(self.betas) != self.timesteps:
            raise ValueError("betas length must equal timesteps")
        for name, tensor in (
            ("betas", self.betas),
            ("alphas", self.alphas),
            ("alpha_bars", self.alpha_bars),
        ):
            if tensor.shape != self.betas.shape:
                raise ValueError(f"{name} must have the same shape as betas")
            if not torch.all(torch.isfinite(tensor)):
                raise ValueError(f"{name} contains NaN or infinity")
        if torch.any(self.betas <= 0) or torch.any(self.betas >= 1):
            raise ValueError("Every beta must lie strictly inside (0, 1)")
        if torch.any(self.alpha_bars <= 0):
            raise ValueError("alpha_bars must remain positive")
        if self.timesteps > 1 and torch.any(self.alpha_bars[1:] > self.alpha_bars[:-1] + 1e-6):
            raise ValueError("alpha_bars must be non-increasing")

    @classmethod
    def linear(
        cls,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> DiffusionSchedule:
        if timesteps < 2:
            raise ValueError("timesteps must be at least 2")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("Require 0 < beta_start < beta_end < 1")
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return cls(
            betas=betas, alphas=alphas, alpha_bars=alpha_bars, timesteps=timesteps, kind="linear"
        )

    @classmethod
    def cosine(cls, timesteps: int = 1000, s: float = 0.008) -> DiffusionSchedule:
        if timesteps < 2:
            raise ValueError("timesteps must be at least 2")
        if s < 0:
            raise ValueError("s must be non-negative")
        steps = torch.arange(timesteps + 1, dtype=torch.float32)
        f = torch.cos(((steps / timesteps + s) / (1.0 + s)) * torch.pi / 2.0) ** 2
        alpha_bars = f / f[0]
        betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
        betas = torch.clamp(betas, min=1e-8, max=0.999)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return cls(
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
            timesteps=timesteps,
            kind="cosine",
        )

    @property
    def previous_alpha_bars(self) -> torch.Tensor:
        previous = torch.cat([torch.ones(1, dtype=self.alpha_bars.dtype), self.alpha_bars[:-1]])
        return previous

    @property
    def posterior_variance(self) -> torch.Tensor:
        previous = self.previous_alpha_bars
        variance = self.betas * (1.0 - previous) / (1.0 - self.alpha_bars)
        variance[0] = 0.0
        return variance

    def extract(
        self,
        values: torch.Tensor,
        timesteps: torch.Tensor,
        broadcast_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Gather per-example values and broadcast them to an image shape."""

        if timesteps.ndim != 1:
            raise ValueError("timesteps must be 1-D")
        if values.numel() != self.timesteps:
            raise ValueError("values must have one entry per diffusion timestep")
        indices = timesteps.to(device=values.device, dtype=torch.long)
        if indices.numel() and (indices.min() < 0 or indices.max() >= self.timesteps):
            raise ValueError("timestep index outside the schedule")
        gathered = values.to(timesteps.device)[indices]
        while gathered.ndim < len(broadcast_shape):
            gathered = gathered.unsqueeze(-1)
        return gathered.expand(broadcast_shape)

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "timesteps": self.timesteps,
            "beta_first": float(self.betas[0]),
            "beta_last": float(self.betas[-1]),
            "alpha_bar_first": float(self.alpha_bars[0]),
            "alpha_bar_last": float(self.alpha_bars[-1]),
        }


def make_schedule(
    kind: str = "linear",
    *,
    timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    cosine_s: float = 0.008,
) -> DiffusionSchedule:
    key = kind.lower()
    if key == "linear":
        return DiffusionSchedule.linear(timesteps, beta_start=beta_start, beta_end=beta_end)
    if key == "cosine":
        return DiffusionSchedule.cosine(timesteps, s=cosine_s)
    raise ValueError(f"Unknown diffusion schedule {kind!r}; choose linear or cosine")
