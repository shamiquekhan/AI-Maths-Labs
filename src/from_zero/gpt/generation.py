"""Autoregressive decoding with greedy, temperature, and top-k sampling."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .model import MiniGPT
from .tokenizer import CharTokenizer


@torch.no_grad()
def generate(
    model: MiniGPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
    seed: int | None = None,
) -> torch.Tensor:
    """Append tokens to ``idx``, cropping the conditioning window each step."""

    if idx.ndim != 2:
        raise ValueError(f"idx must be (batch, time), got shape {tuple(idx.shape)}")
    if idx.dtype != torch.long:
        raise TypeError("idx must have dtype torch.long")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not greedy and temperature <= 0:
        raise ValueError("temperature must be positive when sampling")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive or None")
    if idx.shape[1] == 0:
        raise ValueError("idx must contain at least one conditioning token")

    generator = None
    if seed is not None:
        generator = torch.Generator(device=idx.device)
        generator.manual_seed(int(seed))

    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            context = model.config.context_length
            idx_cond = idx[:, -context:]
            logits, _ = model(idx_cond)
            next_logits = logits[:, -1, :].float()
            if greedy:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / float(temperature)
                if top_k is not None:
                    k = min(top_k, next_logits.size(-1))
                    values, _ = torch.topk(next_logits, k)
                    cutoff = values[:, [-1]]
                    next_logits = next_logits.masked_fill(next_logits < cutoff, float("-inf"))
                probabilities = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1, generator=generator)
            idx = torch.cat([idx, next_token], dim=1)
    finally:
        model.train(was_training)
    return idx


def generate_text(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 200,
    temperature: float = 0.9,
    top_k: int | None = 40,
    greedy: bool = False,
    seed: int | None = None,
    device: str | torch.device | None = None,
) -> str:
    """Encode a prompt, sample, and decode the full sequence."""

    filtered = tokenizer.filter_text(prompt)
    if not filtered:
        raise ValueError("Prompt contains no characters from the tokenizer vocabulary")
    target_device = torch.device(device) if device is not None else model.device
    ids = torch.tensor([tokenizer.encode(filtered)], dtype=torch.long, device=target_device)
    output = generate(
        model,
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        greedy=greedy,
        seed=seed,
    )
    return tokenizer.decode(output[0].tolist())
