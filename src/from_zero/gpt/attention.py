"""Scaled dot-product causal self-attention."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with an explicit causal mask.

    ``forward`` returns both the output and the attention probabilities so the
    dashboard can inspect what each head attended to.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.context_length = context_length
        self.c_attn = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        self.c_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_length, context_length, dtype=torch.bool)).view(
                1, 1, context_length, context_length
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"x must be (batch, time, channels), got {tuple(x.shape)}")
        batch, time, channels = x.shape
        if channels != self.embedding_dim:
            raise ValueError(f"expected {self.embedding_dim} channels, got {channels}")
        if time > self.context_length:
            raise ValueError(f"sequence length {time} exceeds context_length={self.context_length}")
        query, key, value = self.c_attn(x).split(self.embedding_dim, dim=2)
        query = query.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)

        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:, :, :time, :time]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        output = weights @ value
        output = output.transpose(1, 2).contiguous().view(batch, time, channels)
        output = self.resid_dropout(self.c_proj(output))
        return output, weights
