"""Text corpus loading and next-token batch construction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..core.paths import resolve_path
from .tokenizer import CharTokenizer


@dataclass
class TextCorpus:
    """A character-tokenized train/validation corpus."""

    text: str
    train_text: str
    val_text: str
    train_ids: torch.Tensor
    val_ids: torch.Tensor
    tokenizer: CharTokenizer
    metadata: dict[str, Any]


def load_text_corpus(
    path: str | Path,
    *,
    val_fraction: float = 0.1,
    tokenizer: CharTokenizer | None = None,
) -> TextCorpus:
    """Read a UTF-8 text file, split it, and encode both partitions."""

    if not 0.0 < val_fraction < 0.9:
        raise ValueError("val_fraction must be in (0, 0.9)")
    source = resolve_path(path)
    if not source.exists():
        raise FileNotFoundError(f"Text corpus not found: {source}")
    text = source.read_text(encoding="utf-8")
    if len(text) < 200:
        raise ValueError("Corpus is too small for a meaningful character-level experiment")
    split = int(len(text) * (1.0 - val_fraction))
    train_text, val_text = text[:split], text[split:]
    tokenizer = tokenizer or CharTokenizer.from_text(train_text)
    train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    unknown_val = sorted({char for char in val_text if char not in tokenizer._stoi})
    val_ids = torch.tensor(tokenizer.encode(val_text, strict=False), dtype=torch.long)
    if len(train_ids) < 2 or len(val_ids) < 2:
        raise ValueError("Train and validation partitions must each contain at least two tokens")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    metadata = {
        "path": str(source),
        "sha256": digest,
        "characters": len(text),
        "train_characters": len(train_text),
        "val_characters": len(val_text),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "vocab_size": tokenizer.vocab_size,
        "val_fraction": float(val_fraction),
        "val_unknown_characters": unknown_val,
        "tokenizer": "character",
    }
    return TextCorpus(
        text=text,
        train_text=train_text,
        val_text=val_text,
        train_ids=train_ids,
        val_ids=val_ids,
        tokenizer=tokenizer,
        metadata=metadata,
    )


def get_batch(
    data: torch.Tensor,
    *,
    block_size: int,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample fixed-length next-token batches.

    Every target row is the input row shifted by one position, so position
    ``t`` learns to predict token ``t + 1``.
    """

    if data.ndim != 1:
        raise ValueError(f"data must be 1-D, got shape {tuple(data.shape)}")
    if block_size < 1 or batch_size < 1:
        raise ValueError("block_size and batch_size must be positive")
    if len(data) <= block_size:
        raise ValueError(
            f"data has {len(data)} tokens, which is not longer than block_size={block_size}"
        )
    starts = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,),
        generator=generator,
        device=data.device,
    )
    inputs = torch.stack([data[start : start + block_size] for start in starts])
    targets = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return inputs, targets


def corpus_summary(corpus: TextCorpus) -> dict[str, Any]:
    return dict(corpus.metadata)
