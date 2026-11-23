"""End-to-end pipelines for the strict from-scratch engine."""

from __future__ import annotations

import math
import os
import tempfile

import pytest

from from_zero.nn_tensor import BCEWithLogitsLoss
from from_zero.strict import (
    DiffusionSchedule,
    GPTConfig,
    LogisticRegression,
    MLPDenoiser,
    analytical_gradients,
    evaluate_logistic,
    fit_logistic_regression,
    generate_text,
    load_checkpoint,
    make_classification,
    make_pattern_images,
    sample_diffusion,
    save_checkpoint,
    train_diffusion,
    train_gpt,
    train_valid_test_experiment,
)
from from_zero.strict.gpt import CharTokenizer, MiniGPT
from from_zero.tensor import Tensor, no_grad
from from_zero.tensor.storage import RNG


def test_logistic_regression_learns_separable_data() -> None:
    features, targets = make_classification(
        n_samples=160, n_features=2, separation=3.0, noise=0.4, seed=5
    )
    model, history = fit_logistic_regression(
        features, targets, epochs=150, learning_rate=0.5, seed=5
    )
    assert history.train_loss[-1] < history.train_loss[0]
    assert history.train_loss[-1] < 0.1
    assert history.train_accuracy[-1] >= 0.95
    report = evaluate_logistic(model, features, targets)
    assert report["accuracy"] >= 0.95
    assert len(report["probabilities"]) == len(features)


def test_logistic_analytical_gradients_match_autograd() -> None:
    features = [[0.5, -1.0], [1.5, 0.0], [-0.5, 2.0], [0.0, 0.5]]
    targets = [1.0, 1.0, 0.0, 0.0]
    weight = [0.3, -0.2]
    bias = 0.1
    l2 = 0.2
    analytical_weight, analytical_bias = analytical_gradients(
        features, targets, weight, bias, l2=l2
    )

    model = LogisticRegression(2)
    model.weight._data[:] = weight
    model.bias._data[:] = [bias]
    loss = BCEWithLogitsLoss()(model(Tensor(features)), Tensor([[value] for value in targets]))
    loss = loss + (model.weight * model.weight).sum() * (l2 / (2.0 * len(features)))
    loss.backward()

    assert [row[0] for row in model.weight.grad.tolist()] == pytest.approx(
        analytical_weight, abs=1e-10
    )
    assert model.bias.grad.tolist()[0] == pytest.approx(analytical_bias, abs=1e-10)


def test_logistic_train_valid_test_split() -> None:
    result = train_valid_test_experiment(
        n_samples=180, epochs=100, learning_rate=0.2, noise=0.5, seed=11
    )
    assert sum(len(indices) for indices in result["splits"].values()) == 180
    assert result["history"].valid_loss[-1] < result["history"].valid_loss[0]
    assert result["test"]["accuracy"] >= 0.9
    assert all(0.0 <= probability <= 1.0 for probability in result["test_probabilities"])
    assert len(result["means"]) == 2
    assert len(result["stds"]) == 2


def test_gpt_overfits_and_generates_valid_characters() -> None:
    corpus = "the quick brown fox jumps over the lazy dog. " * 3
    tokenizer = CharTokenizer.from_text(corpus)
    config = GPTConfig(block_size=6, n_embd=8, n_head=2, n_layer=1, dropout=0.0, seed=0)
    model, tokenizer, losses = train_gpt(
        corpus,
        config=config,
        steps=30,
        batch_size=4,
        learning_rate=5e-3,
        seed=0,
    )
    assert len(losses) == 30
    assert losses[-1] < losses[0] - 0.2

    generated = generate_text(model, tokenizer, "the ", max_new_tokens=15, temperature=0.7, seed=1)
    assert generated.startswith("the ")
    assert len(generated) == len("the ") + 15
    assert all(character in tokenizer.stoi for character in generated)


def test_gpt_causality_is_exact() -> None:
    corpus = "abcdefghij" * 3
    tokenizer = CharTokenizer.from_text(corpus)
    config = GPTConfig(block_size=6, n_embd=8, n_head=2, n_layer=1, seed=0)
    model = MiniGPT(config, tokenizer.vocab_size, generator=RNG(0))

    with no_grad():
        first = model(Tensor([tokenizer.encode("abcde")]))[0].tolist()
        second = model(Tensor([tokenizer.encode("abcda")]))[0].tolist()

    for position in range(4):
        assert first[position] == pytest.approx(second[position], abs=1e-12)
    assert any(abs(left - right) > 1e-9 for left, right in zip(first[4], second[4], strict=False))


def test_gpt_checkpoint_roundtrip_preserves_logits() -> None:
    corpus = "abcabcabcabcabc"
    tokenizer = CharTokenizer.from_text(corpus)
    config = GPTConfig(block_size=4, n_embd=8, n_head=2, n_layer=1, seed=0)
    model, _, _ = train_gpt(
        corpus, config=config, steps=5, batch_size=2, learning_rate=1e-2, seed=0
    )
    inputs = Tensor([tokenizer.encode("abca")])
    with no_grad():
        expected = model(inputs).tolist()

    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "gpt.json")
    save_checkpoint(path, model, metadata={"vocab_size": tokenizer.vocab_size})

    restored = MiniGPT(config, tokenizer.vocab_size, generator=RNG(99))
    metadata = load_checkpoint(path, restored)
    assert metadata == {"vocab_size": tokenizer.vocab_size}
    with no_grad():
        actual = restored(inputs).tolist()
    for row_expected, row_actual in zip(expected, actual, strict=False):
        for value_expected, value_actual in zip(row_expected, row_actual, strict=False):
            assert value_expected == pytest.approx(value_actual, abs=1e-12)


def test_diffusion_training_reduces_loss_and_samples_frames() -> None:
    images = make_pattern_images(n_images=6, size=4, seed=0)
    model, schedule, losses = train_diffusion(
        images,
        steps=80,
        batch_size=6,
        learning_rate=0.01,
        schedule_steps=15,
        time_dim=8,
        hidden_dim=16,
        seed=0,
    )
    assert isinstance(schedule, DiffusionSchedule)
    assert len(losses) == 80
    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10

    image, frames = sample_diffusion(model, schedule, image_dim=16, seed=0)
    assert len(image) == 16
    assert len(frames) == schedule.steps + 1
    assert all(len(frame) == 16 for frame in frames)
    assert all(math.isfinite(value) for value in image)


def test_diffusion_schedule_and_denoiser_are_deterministic() -> None:
    images = make_pattern_images(n_images=4, size=3, seed=1)
    first = train_diffusion(
        images, steps=10, batch_size=4, schedule_steps=10, seed=3, verbose=False
    )
    second = train_diffusion(
        images, steps=10, batch_size=4, schedule_steps=10, seed=3, verbose=False
    )
    assert first[2] == pytest.approx(second[2])
    assert isinstance(first[0], MLPDenoiser)
