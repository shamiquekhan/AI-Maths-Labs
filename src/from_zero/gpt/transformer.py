"""Pre-layer-norm transformer block and feed-forward network."""

from __future__ import annotations

import torch
from torch import nn

from .attention import CausalSelfAttention


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim, bias=bias),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """``x = x + Attention(LN(x)); x = x + FFN(LN(x))``."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(embedding_dim, bias=bias)
        self.attention = CausalSelfAttention(
            embedding_dim,
            num_heads,
            context_length,
            dropout=dropout,
            bias=bias,
        )
        self.ln_2 = nn.LayerNorm(embedding_dim, bias=bias)
        self.feed_forward = FeedForward(embedding_dim, dropout=dropout, bias=bias)

    def forward(
        self, x: torch.Tensor, *, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attention_output, attention_weights = self.attention(self.ln_1(x))
        x = x + attention_output
        x = x + self.feed_forward(self.ln_2(x))
        if return_attention:
            return x, attention_weights
        return x
