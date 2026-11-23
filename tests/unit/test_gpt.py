"""Unit tests for the PyTorch MiniGPT components."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from from_zero.core.validation import ConfigError  # noqa: E402
from from_zero.gpt import (  # noqa: E402
    CharTokenizer,
    GPTModelConfig,
    MiniGPT,
    generate,
    generate_text,
    get_batch,
    load_text_corpus,
)
from from_zero.gpt.training import GPTTrainingConfig  # noqa: E402


def test_tokenizer_round_trip_and_validation(tmp_path) -> None:
    tokenizer = CharTokenizer.from_text("cabba")
    assert tokenizer.chars == ("a", "b", "c")
    assert tokenizer.vocab_size == 3
    encoded = tokenizer.encode("abc")
    assert tokenizer.decode(encoded) == "abc"
    with pytest.raises(ValueError):
        tokenizer.encode("z")
    assert tokenizer.encode("azb", strict=False) == [0, 1]
    assert tokenizer.filter_text("azb") == "ab"
    with pytest.raises(ValueError):
        tokenizer.decode([3])
    path = tokenizer.save(tmp_path / "tokenizer.json")
    loaded = CharTokenizer.load(path)
    assert loaded.chars == tokenizer.chars
    with pytest.raises(ValueError):
        CharTokenizer.from_text("")


def test_corpus_loading_and_batch_shift() -> None:
    corpus = load_text_corpus("data/corpora/tiny_shakespeare.txt", val_fraction=0.1)
    assert corpus.train_ids.ndim == 1
    assert corpus.val_ids.ndim == 1
    assert corpus.metadata["vocab_size"] == corpus.tokenizer.vocab_size
    assert corpus.metadata["characters"] > 1_000_000
    generator = torch.Generator().manual_seed(1)
    inputs, targets = get_batch(
        corpus.train_ids,
        block_size=16,
        batch_size=4,
        generator=generator,
    )
    assert inputs.shape == targets.shape == (4, 16)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])
    with pytest.raises(ValueError):
        get_batch(torch.arange(5), block_size=8, batch_size=1)


def test_attention_is_causal() -> None:
    config = GPTModelConfig(
        vocab_size=11,
        context_length=8,
        embedding_dim=16,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )
    model = MiniGPT(config).eval()
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6]])
    logits, loss, attentions = model(inputs, return_attn=True)
    assert loss is None
    assert logits.shape == (1, 6, 11)
    weights = attentions[0]
    assert weights.shape == (1, 2, 6, 6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))
    upper = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
    assert torch.all(weights[..., upper] == 0.0)
    changed = inputs.clone()
    changed[0, 4] = 10
    changed_logits, _ = model(changed)
    assert torch.allclose(logits[:, :4], changed_logits[:, :4], atol=0.0)
    assert not torch.allclose(logits[:, 4:], changed_logits[:, 4:])


def test_model_loss_and_backward() -> None:
    config = GPTModelConfig(
        vocab_size=12,
        context_length=8,
        embedding_dim=16,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
    )
    model = MiniGPT(config)
    inputs = torch.randint(0, 12, (2, 8))
    targets = torch.randint(0, 12, (2, 8))
    logits, loss = model(inputs, targets)
    assert logits.shape == (2, 8, 12)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert model.num_parameters() > 0
    assert model.num_parameters(non_embedding=True) < model.num_parameters()


def test_model_rejects_bad_shapes_and_configs() -> None:
    config = GPTModelConfig(vocab_size=5, context_length=4, embedding_dim=8, num_heads=2)
    model = MiniGPT(config)
    with pytest.raises(ValueError):
        model(torch.zeros(2, 5, dtype=torch.long))
    with pytest.raises(ValueError):
        model(torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 4, dtype=torch.long))
    with pytest.raises(ValueError):
        GPTModelConfig(embedding_dim=10, num_heads=3)
    with pytest.raises(ConfigError):
        GPTModelConfig(context_length=1)
    with pytest.raises(ValueError):
        MiniGPT(GPTModelConfig(vocab_size=0))


def test_generation_greedy_is_deterministic_and_respects_length() -> None:
    torch.manual_seed(0)
    config = GPTModelConfig(
        vocab_size=7,
        context_length=8,
        embedding_dim=16,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )
    model = MiniGPT(config).eval()
    prompt = torch.tensor([[1, 2, 3]])
    first = generate(model, prompt, max_new_tokens=5, greedy=True)
    second = generate(model, prompt, max_new_tokens=5, greedy=True)
    assert first.shape == (1, 8)
    assert torch.equal(first, second)
    top1 = generate(model, prompt, max_new_tokens=5, top_k=1, temperature=1.0, seed=3)
    assert torch.equal(first, top1)
    same = generate(model, prompt, max_new_tokens=0)
    assert torch.equal(same, prompt)
    with pytest.raises(ValueError):
        generate(model, prompt, max_new_tokens=1, temperature=0.0)


def test_generate_text_filters_prompt_and_validates() -> None:
    config = GPTModelConfig(
        vocab_size=5,
        context_length=8,
        embedding_dim=8,
        num_heads=1,
        num_layers=1,
        dropout=0.0,
    )
    tokenizer = CharTokenizer(["a", "b", "c", " ", "\n"])
    model = MiniGPT(config).eval()
    text = generate_text(model, tokenizer, "abcXYZ", max_new_tokens=4, greedy=True)
    assert len(text) >= 3
    with pytest.raises(ValueError):
        generate_text(model, tokenizer, "XYZ", max_new_tokens=2)


def test_training_config_parsing() -> None:
    config = GPTTrainingConfig.from_dict(
        {
            "experiment": {"name": "tiny", "seed": 3},
            "data": {"path": "data/corpora/tiny_shakespeare.txt", "val_fraction": 0.05},
            "model": {
                "vocab_size": 0,
                "context_length": 16,
                "embedding_dim": 16,
                "num_heads": 2,
                "num_layers": 1,
            },
            "training": {"max_steps": 3, "batch_size": 2, "device": "cpu"},
            "sampling": {"prompt": "A", "max_new_tokens": 4},
            "output": {"directory": "artifacts/experiments"},
        }
    )
    assert config.name == "tiny"
    assert config.model.context_length == 16
    assert config.sample_tokens == 4
    with pytest.raises(ConfigError):
        GPTTrainingConfig.from_dict({"unknown": 1})
