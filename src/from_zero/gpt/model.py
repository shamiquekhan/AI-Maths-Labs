"""The decoder-only MiniGPT model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..core.experiment import to_jsonable
from ..core.validation import ConfigError, check_bool, check_float, check_int, reject_unknown_keys
from .embeddings import GPTEmbedding
from .transformer import TransformerBlock


@dataclass
class GPTModelConfig:
    """Architecture hyperparameters; ``vocab_size=0`` means infer from the corpus."""

    vocab_size: int = 0
    context_length: int = 64
    embedding_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    bias: bool = False
    tie_weights: bool = True

    def __post_init__(self) -> None:
        check_int("vocab_size", self.vocab_size, minimum=0)
        check_int("context_length", self.context_length, minimum=2, maximum=4096)
        check_int("embedding_dim", self.embedding_dim, minimum=8, maximum=4096)
        check_int("num_heads", self.num_heads, minimum=1, maximum=128)
        check_int("num_layers", self.num_layers, minimum=1, maximum=64)
        check_float("dropout", self.dropout, minimum=0.0, maximum=1.0, allow_maximum=False)
        check_bool("bias", self.bias)
        check_bool("tie_weights", self.tie_weights)
        if self.embedding_dim % self.num_heads != 0:
            raise ConfigError("embedding_dim must be divisible by num_heads")

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> GPTModelConfig:
        if not isinstance(config, dict):
            raise ConfigError("Model config must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        reject_unknown_keys(config, allowed, "model config")
        return cls(**config)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


class MiniGPT(nn.Module):
    """A small pre-layer-norm decoder-only transformer."""

    def __init__(self, config: GPTModelConfig) -> None:
        super().__init__()
        if config.vocab_size < 1:
            raise ValueError("MiniGPT requires a concrete positive vocab_size")
        self.config = config
        self.embedding = GPTEmbedding(
            config.vocab_size,
            config.embedding_dim,
            config.context_length,
            dropout=config.dropout,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.embedding_dim,
                    config.num_heads,
                    config.context_length,
                    dropout=config.dropout,
                    bias=config.bias,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.embedding_dim, bias=config.bias)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.embedding.token.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=0.02 / (2.0 * config.num_layers) ** 0.5,
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        return_attn: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor | None]
        | tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]
    ):
        if idx.ndim != 2:
            raise ValueError(f"idx must be (batch, time), got shape {tuple(idx.shape)}")
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError(
                    f"targets shape {tuple(targets.shape)} must match idx shape {tuple(idx.shape)}"
                )
            if targets.dtype != torch.long:
                raise TypeError("targets must have dtype torch.long")
        x = self.embedding(idx)
        attentions: list[torch.Tensor] = []
        for block in self.blocks:
            if return_attn:
                x, weights = block(x, return_attention=True)
                attentions.append(weights)
            else:
                x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        if return_attn:
            return logits, loss, attentions
        return logits, loss

    def num_parameters(self, *, non_embedding: bool = False) -> int:
        total = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            total -= self.embedding.token.weight.numel()
            total -= self.embedding.position.weight.numel()
        return int(total)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
