"""Character-level MiniGPT implemented on the strict tensor core."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..nn_tensor import (
    GELU,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    Module,
)
from ..optim_tensor import Adam, clip_grad_norm_
from ..tensor import Tensor, cross_entropy, no_grad
from ..tensor.storage import RNG


class CharTokenizer:
    """Character-level tokenizer with a deterministic vocabulary."""

    def __init__(self, chars: Sequence[str]) -> None:
        unique = sorted(set(chars))
        if not unique:
            raise ValueError("Cannot build a tokenizer from an empty vocabulary")
        self.chars = unique
        self.stoi = {char: index for index, char in enumerate(unique)}
        self.itos = {index: char for char, index in self.stoi.items()}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        if not text:
            raise ValueError("Cannot build a tokenizer from empty text")
        return cls(text)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[char] for char in text]
        except KeyError as error:
            raise ValueError(f"Character {error.args[0]!r} is not in the vocabulary") from None

    def decode(self, token_ids: Sequence[int]) -> str:
        try:
            return "".join(self.itos[int(token)] for token in token_ids)
        except KeyError as error:
            raise ValueError(f"Token id {error.args[0]!r} is not in the vocabulary") from None


@dataclass
class GPTConfig:
    block_size: int = 16
    n_embd: int = 16
    n_head: int = 2
    n_layer: int = 1
    dropout: float = 0.0
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.n_embd <= 0:
            raise ValueError("n_embd must be positive")
        if self.n_head <= 0 or self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


def make_lm_batch(
    token_ids: Sequence[int],
    block_size: int,
    batch_size: int,
    rng: RNG,
) -> tuple[Tensor, list[int]]:
    """Sample a batch of shifted input/target sequences."""
    if len(token_ids) <= block_size:
        raise ValueError("Corpus is shorter than block_size + 1")
    inputs: list[list[int]] = []
    targets: list[int] = []
    for _ in range(batch_size):
        start = rng.randint(0, len(token_ids) - block_size - 1)
        chunk = token_ids[start : start + block_size + 1]
        inputs.append(chunk[:-1])
        targets.extend(chunk[1:])
    return Tensor(inputs), targets


class CausalSelfAttention(Module):
    """Multi-head causal self-attention with a cached lower-triangular mask."""

    def __init__(self, config: GPTConfig, generator: RNG | None = None) -> None:
        super().__init__()
        self.config = config
        self.q_proj = Linear(config.n_embd, config.n_embd, bias=False, generator=generator)
        self.k_proj = Linear(config.n_embd, config.n_embd, bias=False, generator=generator)
        self.v_proj = Linear(config.n_embd, config.n_embd, bias=False, generator=generator)
        self.out_proj = Linear(config.n_embd, config.n_embd, generator=generator)
        self.attention_dropout = Dropout(config.dropout, generator=generator)
        self.residual_dropout = Dropout(config.dropout, generator=generator)
        self.last_attention: Tensor | None = None
        self._mask_cache: dict[int, Tensor] = {}

    def _mask(self, length: int) -> Tensor:
        if length not in self._mask_cache:
            values = []
            for row in range(length):
                for column in range(length):
                    values.append(0.0 if column <= row else -1e9)
            self._mask_cache[length] = Tensor._from_flat(values, (1, 1, length, length))
        return self._mask_cache[length]

    def forward(self, x: Tensor) -> Tensor:
        batch, length, channels = x.shape
        if length > self.config.block_size:
            raise ValueError(
                f"Sequence length {length} exceeds block_size {self.config.block_size}"
            )
        heads = self.config.n_head
        head_dim = self.config.head_dim

        q = self.q_proj(x).reshape(batch, length, heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, length, heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, length, heads, head_dim).transpose(1, 2)

        scores = q.matmul(k.transpose(-2, -1)) / math.sqrt(head_dim)
        scores = scores + self._mask(length)
        weights = scores.softmax(dim=-1)
        self.last_attention = weights.detach()

        attended = weights.matmul(v)
        attended = attended.transpose(1, 2).reshape(batch, length, channels)
        return self.residual_dropout(self.out_proj(attended))


class MLP(Module):
    def __init__(self, config: GPTConfig, generator: RNG | None = None) -> None:
        super().__init__()
        self.fc1 = Linear(config.n_embd, 4 * config.n_embd, generator=generator)
        self.activation = GELU()
        self.fc2 = Linear(4 * config.n_embd, config.n_embd, generator=generator)
        self.dropout = Dropout(config.dropout, generator=generator)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.fc2(self.activation(self.fc1(x))))


class Block(Module):
    """Pre-layer-normalised Transformer block."""

    def __init__(self, config: GPTConfig, generator: RNG | None = None) -> None:
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, generator=generator)
        self.ln2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config, generator=generator)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(Module):
    """Decoder-only Transformer language model."""

    def __init__(
        self,
        config: GPTConfig,
        vocab_size: int,
        generator: RNG | None = None,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.config = config
        self.vocab_size = vocab_size
        rng = generator or RNG(config.seed)
        self.token_embedding = Embedding(vocab_size, config.n_embd, generator=rng)
        self.position_embedding = Embedding(config.block_size, config.n_embd, generator=rng)
        self.block_list: list[Block] = []
        for index in range(config.n_layer):
            block = Block(config, generator=rng)
            self.add_module(f"block_{index}", block)
            self.block_list.append(block)
        self.ln_f = LayerNorm(config.n_embd)
        self.head = Linear(config.n_embd, vocab_size, bias=False, generator=rng)

    def forward(self, token_ids) -> Tensor:
        if not isinstance(token_ids, Tensor):
            token_ids = Tensor(token_ids)
        if token_ids.ndim != 2:
            raise ValueError(f"MiniGPT expects (batch, time) token ids, got {token_ids.shape}")
        _batch, length = token_ids.shape
        if length > self.config.block_size:
            raise ValueError(
                f"Sequence length {length} exceeds block_size {self.config.block_size}"
            )
        token_vectors = self.token_embedding(token_ids)
        positions = self.position_embedding(list(range(length)))
        x = token_vectors + positions
        for block in self.block_list:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def loss(self, token_ids: Tensor, targets: Sequence[int]) -> Tensor:
        logits = self.forward(token_ids)
        vocabulary = logits.shape[-1]
        return cross_entropy(logits.reshape(-1, vocabulary), targets)

    def generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: RNG | None = None,
    ) -> list[int]:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive or None")
        rng = generator or RNG(0)
        ids = [int(token) for token in prompt_ids]
        if not ids:
            ids = [0]
        self.eval()
        with no_grad():
            for _ in range(max_new_tokens):
                context = ids[-self.config.block_size :]
                logits = self.forward(Tensor([context]))
                last = logits[0, len(context) - 1]
                values = last._values()
                if temperature == 0.0:
                    next_id = values.index(max(values))
                else:
                    scaled = [value / temperature for value in values]
                    if top_k is not None and top_k < len(scaled):
                        threshold = sorted(scaled, reverse=True)[top_k - 1]
                        scaled = [value if value >= threshold else -1e9 for value in scaled]
                    maximum = max(scaled)
                    exponentials = [math.exp(value - maximum) for value in scaled]
                    total = sum(exponentials)
                    probabilities = [value / total for value in exponentials]
                    draw = rng.uniform(0.0, 1.0)
                    cumulative = 0.0
                    next_id = len(probabilities) - 1
                    for index, probability in enumerate(probabilities):
                        cumulative += probability
                        if draw <= cumulative:
                            next_id = index
                            break
                ids.append(next_id)
        return ids

    def attention_maps(self) -> list[list[list[list[float]]]]:
        maps = []
        for block in self.block_list:
            attention = block.attn.last_attention
            maps.append(attention.tolist() if attention is not None else [])
        return maps


def train_gpt(
    corpus: str,
    config: GPTConfig | None = None,
    steps: int = 100,
    batch_size: int = 4,
    learning_rate: float = 3e-3,
    seed: int = 0,
    grad_clip: float = 1.0,
    verbose: bool = False,
) -> tuple[MiniGPT, CharTokenizer, list[float]]:
    """Train a tiny character-level GPT; intended for correctness demos."""
    config = config or GPTConfig(seed=seed)
    tokenizer = CharTokenizer.from_text(corpus)
    token_ids = tokenizer.encode(corpus)
    if len(token_ids) <= config.block_size:
        raise ValueError("Corpus must contain more than block_size + 1 characters")

    model = MiniGPT(config, tokenizer.vocab_size, generator=RNG(seed))
    optimizer = Adam(model.parameters(), lr=learning_rate)
    rng = RNG(seed + 1)
    losses: list[float] = []
    model.train()
    for step in range(steps):
        inputs, targets = make_lm_batch(token_ids, config.block_size, batch_size, rng)
        optimizer.zero_grad()
        loss = model.loss(inputs, targets)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
        if verbose and (step % max(1, steps // 10) == 0):
            print(f"step {step:4d} loss {losses[-1]:.4f}")
    return model, tokenizer, losses


def generate_text(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 30,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int = 0,
) -> str:
    prompt_ids = tokenizer.encode(prompt)
    ids = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        generator=RNG(seed),
    )
    return tokenizer.decode(ids)


__all__ = [
    "Block",
    "CausalSelfAttention",
    "CharTokenizer",
    "GPTConfig",
    "MiniGPT",
    "generate_text",
    "make_lm_batch",
    "train_gpt",
]
