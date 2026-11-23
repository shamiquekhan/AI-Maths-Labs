"""A compact convolutional U-Net noise predictor with timestep conditioning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..core.validation import ConfigError, check_float, check_int, reject_unknown_keys


@dataclass
class UNetConfig:
    in_channels: int = 1
    base_channels: int = 64
    channel_mults: tuple[int, ...] = (1, 2)
    num_res_blocks: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        check_int("in_channels", self.in_channels, minimum=1, maximum=8)
        check_int("base_channels", self.base_channels, minimum=8, maximum=512)
        check_int("num_res_blocks", self.num_res_blocks, minimum=1, maximum=8)
        check_float("dropout", self.dropout, minimum=0.0, maximum=1.0, allow_maximum=False)
        if not self.channel_mults:
            raise ConfigError("channel_mults must contain at least one multiplier")
        for multiplier in self.channel_mults:
            check_int("channel multiplier", multiplier, minimum=1, maximum=16)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> UNetConfig:
        if not isinstance(config, dict):
            raise ConfigError("U-Net config must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        reject_unknown_keys(config, allowed, "U-Net config")
        payload = dict(config)
        if "channel_mults" in payload:
            payload["channel_mults"] = tuple(payload["channel_mults"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_channels": self.in_channels,
            "base_channels": self.base_channels,
            "channel_mults": list(self.channel_mults),
            "num_res_blocks": self.num_res_blocks,
            "dropout": self.dropout,
        }


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 2 or dim % 2 != 0:
            raise ValueError("Sinusoidal embedding dimension must be even and at least 2")
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError("timesteps must be 1-D")
        half = self.dim // 2
        if half == 1:
            frequencies = torch.ones(1, device=timesteps.device)
        else:
            frequencies = torch.exp(
                -math.log(10000.0)
                * torch.arange(half, device=timesteps.device, dtype=torch.float32)
                / (half - 1)
            )
        angles = timesteps.float()[:, None] * frequencies[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def _groups(channels: int) -> int:
    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    """GroupNorm → SiLU → Conv, timestep injection, and a residual path."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_mlp(time_embedding)[:, :, None, None]
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return hidden + self.residual(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """A small U-Net that predicts the noise added at a given timestep."""

    def __init__(self, config: UNetConfig) -> None:
        super().__init__()
        self.config = config
        base = config.base_channels
        time_emb_dim = base * 4
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.init_conv = nn.Conv2d(config.in_channels, base, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channels = base
        for level, multiplier in enumerate(config.channel_mults):
            out_channels = base * multiplier
            blocks = nn.ModuleList()
            for block_index in range(config.num_res_blocks):
                blocks.append(
                    ResidualBlock(
                        channels if block_index == 0 else out_channels,
                        out_channels,
                        time_emb_dim,
                        dropout=config.dropout,
                    )
                )
            self.down_blocks.append(blocks)
            channels = out_channels
            if level < len(config.channel_mults) - 1:
                self.downsamples.append(Downsample(channels))

        self.mid_block1 = ResidualBlock(channels, channels, time_emb_dim, dropout=config.dropout)
        self.mid_block2 = ResidualBlock(channels, channels, time_emb_dim, dropout=config.dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level in reversed(range(len(config.channel_mults))):
            skip_channels = base * config.channel_mults[level]
            out_channels = skip_channels
            self.up_blocks.append(
                ResidualBlock(
                    channels + skip_channels,
                    out_channels,
                    time_emb_dim,
                    dropout=config.dropout,
                )
            )
            channels = out_channels
            if level > 0:
                self.upsamples.append(Upsample(channels))

        self.final_norm = nn.GroupNorm(_groups(base), base)
        self.final_conv = nn.Conv2d(base, config.in_channels, kernel_size=1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must be (batch, channels, height, width), got {tuple(x.shape)}")
        if timesteps.ndim != 1 or len(timesteps) != x.shape[0]:
            raise ValueError("timesteps must be 1-D with one entry per batch item")
        if timesteps.dtype != torch.long:
            raise TypeError("timesteps must have dtype torch.long")
        time_embedding = self.time_mlp(timesteps)
        hidden = self.init_conv(x)
        skips: list[torch.Tensor] = []
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                hidden = block(hidden, time_embedding)
            skips.append(hidden)
            if level < len(self.down_blocks) - 1:
                hidden = self.downsamples[level](hidden)

        hidden = self.mid_block1(hidden, time_embedding)
        hidden = self.mid_block2(hidden, time_embedding)

        for block_index, level in enumerate(reversed(range(len(self.up_blocks)))):
            skip = skips[level]
            if hidden.shape[-2:] != skip.shape[-2:]:
                hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
            hidden = torch.cat([hidden, skip], dim=1)
            hidden = self.up_blocks[block_index](hidden, time_embedding)
            if level > 0:
                hidden = self.upsamples[block_index](hidden)

        hidden = self.final_conv(F.silu(self.final_norm(hidden)))
        return hidden

    def num_parameters(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))
