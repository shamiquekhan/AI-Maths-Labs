"""Token and learned positional embeddings."""

from __future__ import annotations

import torch
from torch import nn


class GPTEmbedding(nn.Module):
    """Sum token embeddings with learned position embeddings."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        context_length: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if context_length < 2:
            raise ValueError("context_length must be at least 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.token = nn.Embedding(vocab_size, embedding_dim)
        self.position = nn.Embedding(context_length, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 2:
            raise ValueError(f"idx must be (batch, time), got shape {tuple(idx.shape)}")
        _batch, time = idx.shape
        if time > self.context_length:
            raise ValueError(f"sequence length {time} exceeds context_length={self.context_length}")
        if idx.dtype != torch.long:
            raise TypeError("idx must have dtype torch.long")
        if idx.numel() and (idx.min() < 0 or idx.max() >= self.vocab_size):
            raise ValueError("idx contains a token outside the vocabulary")
        positions = torch.arange(time, device=idx.device, dtype=torch.long)
        return self.dropout(self.token(idx) + self.position(positions))
