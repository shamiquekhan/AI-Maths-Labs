"""Unit tests for strict data utilities, metrics, schedules, and checkpoints."""

from __future__ import annotations

import json
import math
import os
import tempfile

import pytest

from from_zero.nn_tensor import Linear, Sequential
from from_zero.optim_tensor import Adam
from from_zero.strict import (
    DataLoader,
    DiffusionSchedule,
    GPTConfig,
    MLPDenoiser,
    SequenceDataset,
    TensorDataset,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    load_checkpoint,
    precision_score,
    recall_score,
    save_checkpoint,
    sinusoidal_embedding,
    split_indices,
    standardize,
    threshold_predictions,
)
from from_zero.strict.gpt import CharTokenizer, make_lm_batch
from from_zero.tensor import Tensor
from from_zero.tensor.storage import RNG


def test_split_indices_are_disjoint_and_cover_everything() -> None:
    train, valid, test = split_indices(100, 0.6, 0.2, seed=3)
    assert (len(train), len(valid), len(test)) == (60, 20, 20)
    assert sorted(train + valid + test) == list(range(100))
    assert not set(train) & set(valid)
    assert not set(train) & set(test)
    assert not set(valid) & set(test)
    assert split_indices(100, seed=3) == (train, valid, test)


def test_split_indices_validation() -> None:
    with pytest.raises(ValueError):
        split_indices(0)
    with pytest.raises(ValueError):
        split_indices(10, train_fraction=1.0)
    with pytest.raises(ValueError):
        split_indices(10, train_fraction=0.8, valid_fraction=0.3)


def test_standardize_uses_training_statistics() -> None:
    train = [[1.0, 10.0], [3.0, 20.0]]
    test = [[5.0, 30.0]]
    means, stds, (train_scaled, test_scaled) = standardize(train, test)
    assert means == [2.0, 15.0]
    assert stds == [1.0, 5.0]
    assert train_scaled == [[-1.0, -1.0], [1.0, 1.0]]
    assert test_scaled == [[3.0, 3.0]]


def test_standardize_handles_constant_features() -> None:
    means, stds, (scaled,) = standardize([[1.0], [1.0]])
    assert means == [1.0]
    assert stds == [1.0]
    assert scaled == [[0.0], [0.0]]


def test_dataloader_batches_and_shuffle_determinism() -> None:
    dataset = TensorDataset([[float(i)] for i in range(10)], [float(i) for i in range(10)])
    batches = list(DataLoader(dataset, batch_size=4, shuffle=True, seed=7))
    assert [batch[0].shape for batch in batches] == [(4, 1), (4, 1), (2, 1)]
    assert [batch[1].shape for batch in batches] == [(4, 1), (4, 1), (2, 1)]
    repeated = list(DataLoader(dataset, batch_size=4, shuffle=True, seed=7))
    assert [batch[0].tolist() for batch in batches] == [batch[0].tolist() for batch in repeated]
    ordered = list(DataLoader(dataset, batch_size=5, shuffle=False))
    assert ordered[0][1].tolist() == [[0.0], [1.0], [2.0], [3.0], [4.0]]
    dropped = list(DataLoader(dataset, batch_size=4, shuffle=False, drop_last=True))
    assert len(dropped) == 2


def test_sequence_dataset_shifted_pairs() -> None:
    dataset = SequenceDataset([1, 2, 3, 4, 5], block_size=3)
    assert len(dataset) == 2
    inputs, targets = dataset[0]
    assert inputs == [1, 2, 3]
    assert targets == [2, 3, 4]
    with pytest.raises(IndexError):
        dataset[2]


def test_metrics_hand_computed() -> None:
    truth = [0, 1, 1, 0]
    predictions = [0, 1, 0, 0]
    assert confusion_matrix(truth, predictions) == [[2, 0], [1, 1]]
    assert accuracy_score(truth, predictions) == 0.75
    assert precision_score(truth, predictions) == 1.0
    assert recall_score(truth, predictions) == 0.5
    assert f1_score(truth, predictions) == pytest.approx(2.0 / 3.0)
    report = classification_report(truth, predictions)
    assert report["support"] == 4
    assert report["confusion_matrix"] == [[2, 0], [1, 1]]
    assert threshold_predictions([0.49, 0.5, 0.9]) == [0, 1, 1]
    with pytest.raises(ValueError):
        confusion_matrix([0, 2], [0, 1])


def test_tokenizer_roundtrip() -> None:
    tokenizer = CharTokenizer.from_text("cabba")
    assert tokenizer.vocab_size == 3
    ids = tokenizer.encode("abc")
    assert tokenizer.decode(ids) == "abc"
    with pytest.raises(ValueError):
        tokenizer.encode("z")
    with pytest.raises(ValueError):
        tokenizer.decode([99])
    with pytest.raises(ValueError):
        CharTokenizer.from_text("")


def test_gpt_config_validation() -> None:
    assert GPTConfig(block_size=8, n_embd=16, n_head=4).head_dim == 4
    with pytest.raises(ValueError):
        GPTConfig(n_embd=15, n_head=4)
    with pytest.raises(ValueError):
        GPTConfig(block_size=0)
    with pytest.raises(ValueError):
        GPTConfig(dropout=1.0)


def test_make_lm_batch_shapes() -> None:
    rng = RNG(0)
    inputs, targets = make_lm_batch(list(range(20)), block_size=5, batch_size=3, rng=rng)
    assert inputs.shape == (3, 5)
    assert len(targets) == 15
    values = inputs.tolist()
    for row, index in zip(values, range(0, 15, 5), strict=False):
        assert targets[index : index + 5] == [int(value) + 1 for value in row]
    with pytest.raises(ValueError):
        make_lm_batch(list(range(3)), block_size=5, batch_size=1, rng=RNG(0))


def test_diffusion_schedule_monotone() -> None:
    schedule = DiffusionSchedule.linear(10)
    assert schedule.steps == 10
    assert all(0.0 < beta < 1.0 for beta in schedule.betas)
    assert all(
        later < earlier
        for earlier, later in zip(schedule.alpha_bars, schedule.alpha_bars[1:], strict=False)
    )
    assert schedule.alpha_bars[0] == pytest.approx(1.0 - schedule.betas[0])
    assert schedule.posterior_variance[0] == 0.0
    assert all(variance > 0.0 for variance in schedule.posterior_variance[1:])


def test_diffusion_forward_closed_form() -> None:
    schedule = DiffusionSchedule.linear(4, beta_start=0.1, beta_end=0.4)
    clean = Tensor([[1.0, -1.0]])
    noise = Tensor([[0.5, 0.5]])
    noisy = schedule.add_noise(clean, [2], noise)
    alpha_bar = schedule.alpha_bars[2]
    expected = [
        1.0 * math.sqrt(alpha_bar) + 0.5 * math.sqrt(1.0 - alpha_bar),
        -1.0 * math.sqrt(alpha_bar) + 0.5 * math.sqrt(1.0 - alpha_bar),
    ]
    assert noisy.tolist()[0] == pytest.approx(expected)
    with pytest.raises(IndexError):
        schedule.add_noise(clean, [4], noise)


def test_diffusion_reverse_mean_at_final_step() -> None:
    schedule = DiffusionSchedule.linear(5, beta_start=0.1, beta_end=0.2)
    x_t = [1.0, 2.0]
    noise = [0.3, -0.4]
    t = 0
    mean = schedule.reverse_mean(x_t, t, noise)
    beta = schedule.betas[t]
    alpha = schedule.alphas[t]
    factor = beta / math.sqrt(1.0 - schedule.alpha_bars[t])
    expected = [
        (value - factor * epsilon) / math.sqrt(alpha)
        for value, epsilon in zip(x_t, noise, strict=False)
    ]
    assert mean == pytest.approx(expected)


def test_sinusoidal_embedding() -> None:
    values = sinusoidal_embedding(3, 8)
    assert len(values) == 8
    assert values[0] == pytest.approx(math.sin(3.0))
    assert values[1] == pytest.approx(math.cos(3.0))
    with pytest.raises(ValueError):
        sinusoidal_embedding(0, 7)
    with pytest.raises(ValueError):
        sinusoidal_embedding(0, 0)


def test_mlp_denoiser_shapes() -> None:
    model = MLPDenoiser(image_dim=16, time_dim=8, hidden_dim=8, generator=RNG(0))
    output = model(Tensor([[0.5] * 16, [0.1] * 16]), [0, 9])
    assert output.shape == (2, 16)
    with pytest.raises(ValueError):
        model(Tensor([[0.5] * 16]), [0, 1])


def test_checkpoint_roundtrip_and_validation() -> None:
    model = Sequential(Linear(2, 3, generator=RNG(0)), Linear(3, 1, generator=RNG(1)))
    optimizer = Adam(model.parameters(), lr=0.01)
    loss = model(Tensor([[1.0, 2.0]])).sum()
    loss.backward()
    optimizer.step()

    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "checkpoint.json")
    save_checkpoint(path, model, optimizer, {"epoch": 4})
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["format"] == "from-zero-strict-1"
    assert payload["metadata"] == {"epoch": 4}

    restored = Sequential(Linear(2, 3, generator=RNG(2)), Linear(3, 1, generator=RNG(3)))
    restored_optimizer = Adam(restored.parameters(), lr=0.01)
    metadata = load_checkpoint(path, restored, restored_optimizer)
    assert metadata == {"epoch": 4}
    assert restored_optimizer.step_count == 1
    for original, copy in zip(model.parameters(), restored.parameters(), strict=False):
        assert original.tolist() == copy.tolist()

    bad = os.path.join(directory, "bad.json")
    with open(bad, "w", encoding="utf-8") as handle:
        handle.write('{"format": "something-else", "model": {}}')
    with pytest.raises(ValueError):
        load_checkpoint(bad, restored)

    with pytest.raises(ValueError):
        load_checkpoint(path, restored, max_bytes=10)
    with pytest.raises(FileNotFoundError):
        load_checkpoint(os.path.join(directory, "missing.json"), restored)
