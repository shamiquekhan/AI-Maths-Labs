#!/usr/bin/env python3
"""Strict character-level GPT: causality, training, and sampling."""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from from_zero.strict import GPTConfig, MiniGPT, generate_text, train_gpt
from from_zero.strict.gpt import CharTokenizer
from from_zero.tensor import Tensor, no_grad
from from_zero.tensor.storage import RNG


def main() -> None:
    print("=" * 66)
    print("FROM ZERO - strict character GPT")
    print("=" * 66)

    corpus = "the quick brown fox jumps over the lazy dog. " * 3
    tokenizer = CharTokenizer.from_text(corpus)
    config = GPTConfig(block_size=8, n_embd=8, n_head=2, n_layer=1, dropout=0.0, seed=0)

    print("\n1. Configuration")
    print("-" * 40)
    print(f"vocabulary  {tokenizer.vocab_size} characters")
    print(f"context     {config.block_size} tokens")
    print(f"heads       {config.n_head} (head dimension {config.head_dim})")

    model = MiniGPT(config, tokenizer.vocab_size, generator=RNG(0))
    print(f"parameters  {sum(parameter.size for parameter in model.parameters()):,}")

    print("\n2. Causal mask check")
    print("-" * 40)
    with no_grad():
        first = model(Tensor([tokenizer.encode("the quic")]))[0].tolist()
        second = model(Tensor([tokenizer.encode("the quia")]))[0].tolist()
    earlier = max(
        abs(left - right)
        for row_a, row_b in zip(first[:7], second[:7], strict=True)
        for left, right in zip(row_a, row_b, strict=True)
    )
    last = max(abs(left - right) for left, right in zip(first[7], second[7], strict=True))
    print(f"max change before the edited token: {earlier:.2e}")
    print(f"max change at the edited token:     {last:.2e}")

    print("\n3. Training")
    print("-" * 40)
    model, tokenizer, losses = train_gpt(
        corpus,
        config=config,
        steps=40,
        batch_size=4,
        learning_rate=5e-3,
        seed=0,
        verbose=True,
    )
    print(f"first loss  {losses[0]:.4f}")
    print(f"final loss  {losses[-1]:.4f}")

    print("\n4. Sampling")
    print("-" * 40)
    for temperature in (0.0, 0.8):
        sample = generate_text(
            model,
            tokenizer,
            "the ",
            max_new_tokens=30,
            temperature=temperature,
            seed=1,
        )
        print(f"temperature {temperature:.1f}: {sample!r}")

    print("\nStrict GPT example complete.")


if __name__ == "__main__":
    main()
