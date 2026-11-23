#!/usr/bin/env python3
"""Numerical and model-behavior checks for the strict from-scratch core.

Run from the repository root:

    python scripts/verify_release_checks.py

The script uses only the standard library and the strict packages. It exits
nonzero on the first failed check, so it can be used as a release gate.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from from_zero.autograd_tensor import gradcheck
from from_zero.nn_tensor import BCEWithLogitsLoss
from from_zero.optim_tensor import Adam
from from_zero.strict import (
    GPTConfig,
    LogisticRegression,
    MiniGPT,
    analytical_gradients,
    generate_text,
    load_checkpoint,
    make_pattern_images,
    sample_diffusion,
    save_checkpoint,
    train_diffusion,
    train_gpt,
)
from from_zero.strict.gpt import CharTokenizer
from from_zero.tensor import Tensor, cross_entropy, no_grad, tensor
from from_zero.tensor.storage import RNG


def _check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        raise SystemExit(1)


def gradient_checks() -> None:
    print("\n== Gradient checks (central finite differences, h=1e-6) ==")
    x = Tensor._from_flat([0.3, -0.7, 1.1, 0.0], (4,), requires_grad=True)
    cases = [
        ("x^2 + x", lambda: (x * x + x).sum()),
        ("tanh + sigmoid", lambda: (x.tanh() + x.sigmoid()).sum()),
        ("softplus (includes 0)", lambda: x.softplus().sum()),
        ("exp / (1 + x^2)", lambda: (x.exp() / (x * x + 1.5)).sum()),
        (
            "softmax weighted",
            lambda: (x.softmax() * tensor([1.0, 2.0, 3.0, 4.0])).sum(),
        ),
        (
            "BCE with logits at 0",
            lambda: BCEWithLogitsLoss()(x, tensor([1.0, 0.0, 1.0, 0.0])),
        ),
        ("MSE", lambda: ((x - tensor([0.1, 0.2, 0.3, 0.4])) ** 2).mean()),
    ]
    for name, build in cases:
        _check(name, gradcheck(build, [x]))

    matrix = Tensor._from_flat([0.2, -0.4, 0.6, 1.0, -1.2, 0.3], (2, 3), requires_grad=True)
    other = Tensor._from_flat([0.5, -0.1, 0.7, 0.2, -0.3, 0.9], (3, 2), requires_grad=True)
    _check(
        "broadcast add",
        gradcheck(lambda: (matrix + tensor([1.0, 2.0, 3.0])).sum(), [matrix]),
    )
    _check("matmul", gradcheck(lambda: (matrix @ other).sum(), [matrix, other]))

    logits = Tensor._from_flat([0.2, -0.1, 0.7, 1.1, -0.8, 0.0], (2, 3), requires_grad=True)
    _check(
        "cross-entropy",
        gradcheck(lambda: cross_entropy(logits, [2, 0]), [logits]),
    )


def logistic_checks() -> None:
    print("\n== Strict logistic: analytical vs. autograd ==")
    features = [[0.5, -1.0], [1.5, 0.0], [-0.5, 2.0], [0.0, 0.5]]
    targets = [1.0, 1.0, 0.0, 0.0]
    weight = [0.3, -0.2]
    bias = 0.1
    l2 = 0.2
    n_samples = len(features)

    analytical_weight, analytical_bias = analytical_gradients(
        features, targets, weight, bias, l2=l2
    )
    model = LogisticRegression(2)
    model.weight._data[:] = weight
    model.bias._data[:] = [bias]
    loss = BCEWithLogitsLoss()(model(Tensor(features)), Tensor([[value] for value in targets]))
    loss = loss + (model.weight * model.weight).sum() * (l2 / (2.0 * n_samples))
    loss.backward()

    autograd_weight = [row[0] for row in model.weight.grad.tolist()]
    autograd_bias = model.bias.grad.tolist()[0]
    error = max(
        abs(left - right)
        for left, right in zip(
            [*analytical_weight, analytical_bias],
            [*autograd_weight, autograd_bias],
            strict=True,
        )
    )
    _check("analytical == autograd", error < 1e-12, f"max error {error:.2e}")


def gpt_checks() -> None:
    print("\n== Strict character GPT ==")
    corpus = "the quick brown fox jumps over the lazy dog. " * 3
    tokenizer = CharTokenizer.from_text(corpus)
    config = GPTConfig(block_size=8, n_embd=8, n_head=2, n_layer=1, dropout=0.0, seed=0)
    model = MiniGPT(config, tokenizer.vocab_size, generator=RNG(0))

    with no_grad():
        first = model(Tensor([tokenizer.encode("the quic")]))[0].tolist()
        second = model(Tensor([tokenizer.encode("the quia")]))[0].tolist()
    earlier = max(
        abs(left - right)
        for row_a, row_b in zip(first[:7], second[:7], strict=True)
        for left, right in zip(row_a, row_b, strict=True)
    )
    changed = max(abs(left - right) for left, right in zip(first[7], second[7], strict=True))
    _check("causal mask: earlier logits unchanged", earlier < 1e-12, f"{earlier:.2e}")
    _check("causal mask: edited position changes", changed > 1e-6, f"{changed:.2e}")

    model, tokenizer, losses = train_gpt(
        corpus,
        config=config,
        steps=20,
        batch_size=4,
        learning_rate=5e-3,
        seed=0,
        verbose=False,
    )
    _check(
        "loss decreases",
        min(losses[-5:]) < losses[0],
        f"{losses[0]:.4f} -> {min(losses[-5:]):.4f}",
    )
    sample = generate_text(model, tokenizer, "the ", max_new_tokens=16, temperature=0.8, seed=1)
    _check("generation returns text", len(sample) >= 4 and isinstance(sample, str), repr(sample))


def diffusion_checks() -> None:
    print("\n== Strict diffusion ==")
    images = make_pattern_images(n_images=6, size=4, seed=2)
    model, schedule, losses = train_diffusion(
        images,
        steps=60,
        batch_size=6,
        learning_rate=0.01,
        schedule_steps=12,
        time_dim=8,
        hidden_dim=16,
        seed=2,
        verbose=False,
    )
    _check(
        "loss decreases",
        min(losses[-5:]) < losses[0],
        f"{losses[0]:.4f} -> {min(losses[-5:]):.4f}",
    )
    sample, frames = sample_diffusion(model, schedule, image_dim=16, seed=2)
    _check("reverse chain length", len(frames) == schedule.steps + 1, f"{len(frames)} frames")
    _check("sample is finite", all(abs(value) < 1e6 for value in sample))


def checkpoint_checks() -> None:
    print("\n== Strict JSON checkpoint round-trip ==")
    model = LogisticRegression(2)
    model.weight._data[:] = [0.25, -0.5]
    model.bias._data[:] = [0.125]
    optimizer = Adam(model.parameters(), lr=0.05)
    loss = BCEWithLogitsLoss()(model(tensor([[1.0, 0.0], [0.0, 1.0]])), tensor([[1.0], [0.0]]))
    loss.backward()
    optimizer.step()

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.json"
        save_checkpoint(path, model, optimizer=optimizer, metadata={"epoch": 4})
        restored = LogisticRegression(2)
        restored_optimizer = Adam(restored.parameters(), lr=0.05)
        metadata = load_checkpoint(path, restored, optimizer=restored_optimizer)
        same = all(
            original.tolist() == copy.tolist()
            for original, copy in zip(model.parameters(), restored.parameters(), strict=True)
        )
        _check("parameters restored", same)
        _check("metadata restored", metadata == {"epoch": 4})
        _check("optimizer step restored", restored_optimizer.step_count == 1)


def main() -> None:
    print("FROM ZERO — strict release checks")
    gradient_checks()
    logistic_checks()
    gpt_checks()
    diffusion_checks()
    checkpoint_checks()
    print("\nAll strict numerical and model-behavior checks passed.")


if __name__ == "__main__":
    main()
