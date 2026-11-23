#!/usr/bin/env python3
"""
MiniGPT Example

Demonstrates the PyTorch MiniGPT implementation:
- Character tokenizer
- Dataset preparation
- Model architecture
- Training loop
- Text generation with temperature and top-k
"""

import sys

sys.path.insert(0, "src")

import torch

from from_zero.gpt import (
    CharTokenizer,
    GPTModelConfig,
    MiniGPT,
    generate_text,
    get_batch,
    load_text_corpus,
)
from from_zero.gpt.training import GPTTrainingConfig, train_gpt


def test_tokenizer():
    """Test character-level tokenizer."""
    print("=" * 60)
    print("MiniGPT: Tokenizer")
    print("=" * 60)

    # Create from text
    text = "hello world hello"
    tokenizer = CharTokenizer.from_text(text)
    print(f"Vocabulary: {tokenizer.chars}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Encode/decode
    encoded = tokenizer.encode("hello")
    print(f"Encode 'hello': {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decode {encoded}: '{decoded}'")

    # Save/load
    tokenizer.save("/tmp/test_tokenizer.json")
    loaded = CharTokenizer.load("/tmp/test_tokenizer.json")
    print(f"Loaded vocab matches: {loaded.chars == tokenizer.chars}")


def test_corpus_loading():
    """Test loading the Tiny Shakespeare corpus."""
    print("\n" + "=" * 60)
    print("MiniGPT: Corpus Loading")
    print("=" * 60)

    corpus = load_text_corpus("data/corpora/tiny_shakespeare.txt", val_fraction=0.1)
    print(f"Train tokens: {corpus.train_ids.shape}")
    print(f"Val tokens: {corpus.val_ids.shape}")
    print(f"Vocab size: {corpus.metadata['vocab_size']}")
    print(f"Characters: {corpus.metadata['characters']}")

    # Test batch sampling
    generator = torch.Generator().manual_seed(42)
    inputs, targets = get_batch(
        corpus.train_ids,
        block_size=16,
        batch_size=4,
        generator=generator,
    )
    print(f"\nBatch shapes: inputs={inputs.shape}, targets={targets.shape}")
    print(f"Input[0]: {inputs[0].tolist()}")
    print(f"Target[0]: {targets[0].tolist()}")
    print(
        f"Shift check (inputs[0, 1:] == targets[0, :-1]): {torch.equal(inputs[0, 1:], targets[0, :-1])}"
    )


def test_model_architecture():
    """Test model components and forward pass."""
    print("\n" + "=" * 60)
    print("MiniGPT: Model Architecture")
    print("=" * 60)

    config = GPTModelConfig(
        vocab_size=65,
        context_length=64,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    )

    model = MiniGPT(config)
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Non-embedding params: {model.num_parameters(non_embedding=True):,}")

    # Forward pass
    inputs = torch.randint(0, 65, (2, 32))
    logits, loss = model(inputs, targets=inputs)
    print(f"\nInput shape: {inputs.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    # Test attention output
    logits, loss, attn = model(inputs, return_attn=True)
    print(f"Attention layers: {len(attn)}")
    print(f"Attention shape per layer: {attn[0].shape}")  # (B, H, T, T)

    # Test causal masking
    causal_input = torch.tensor([[1, 2, 3, 4, 5, 6]])
    logits, _, attn = model(causal_input, return_attn=True)
    attn_weights = attn[0][0, 0]  # First layer, first batch, first head, shape (T, T)
    upper = torch.triu(torch.ones(6, 6), diagonal=1).bool()
    is_causal = torch.allclose(attn_weights[upper], torch.zeros_like(attn_weights[upper]))
    print(f"Causal mask respected: {is_causal}")


def test_generation():
    """Test text generation."""
    print("\n" + "=" * 60)
    print("MiniGPT: Text Generation")
    print("=" * 60)

    # Use a tokenizer that matches the model's vocabulary
    # For this demo, we'll create a simple vocabulary
    text = "hello world hello " + "a" * 57  # Pad to 65 chars
    tokenizer = CharTokenizer.from_text(text)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    config = GPTModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=64,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )

    model = MiniGPT(config).eval()

    # Greedy generation
    prompt = "he"
    generated = generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=20,
        greedy=True,
        seed=42,
    )
    print(f"Greedy: '{prompt}' -> '{generated}'")

    # Temperature sampling
    for temp in [0.5, 1.0, 1.5]:
        generated = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=20,
            temperature=temp,
            top_k=10,
            seed=42,
        )
        print(f"T={temp}: '{generated}'")


def test_training():
    """Run a tiny training run."""
    print("\n" + "=" * 60)
    print("MiniGPT: Training (Tiny Run)")
    print("=" * 60)

    config = GPTTrainingConfig(
        name="example_gpt_training",
        seed=42,
        model=GPTModelConfig(
            vocab_size=0,  # auto
            context_length=32,
            embedding_dim=32,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        ),
        batch_size=4,
        learning_rate=0.001,
        max_steps=20,
        eval_interval=10,
        eval_batches=2,
        log_interval=5,
        sample_prompt="ROMEO:",
        sample_tokens=30,
        sample_temperature=0.8,
        sample_top_k=10,
        device="cpu",
        output_dir="artifacts/experiments",
        make_plots=True,
    )

    print("Starting training...")
    run = train_gpt(config)
    print(f"Training completed: {run.result.directory}")
    print(f"Best val loss: {run.best_val_loss:.4f}")
    print(f"Steps: {run.result.metadata.get('step', 'N/A')}")

    # Check checkpoint
    import os

    ckpt_dir = os.path.join(run.result.directory, "checkpoint")
    if os.path.exists(ckpt_dir):
        print(f"Checkpoints: {os.listdir(ckpt_dir)}")


def test_attention_visualization():
    """Visualize attention patterns."""
    print("\n" + "=" * 60)
    print("MiniGPT: Attention Visualization")
    print("=" * 60)

    config = GPTModelConfig(
        vocab_size=65,
        context_length=32,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )
    model = MiniGPT(config).eval()
    tokenizer = CharTokenizer.from_text("hello world hello")

    prompt = "he"
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids])

    with torch.no_grad():
        _logits, _, attentions = model(input_tensor, return_attn=True)

    print(f"Number of layers with attention: {len(attentions)}")
    for i, attn in enumerate(attentions):
        print(f"  Layer {i}: {attn.shape}")  # (B, H, T, T)

    # Average over heads
    attn_avg = attentions[0][0].mean(0)  # (T, T)
    print(f"Layer 0 average attention shape: {attn_avg.shape}")
    print(f"Attention matrix:\n{attn_avg.numpy().round(3)}")

    # Save visualization
    import matplotlib.pyplot as plt

    _fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attn_avg.numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Layer 0 Attention (averaged over heads)")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    tokens = [tokenizer.decode([tid]) for tid in input_ids]
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45)
    ax.set_yticklabels(tokens)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("/tmp/attention_viz.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Attention visualization saved to /tmp/attention_viz.png")


def main():
    test_tokenizer()
    test_corpus_loading()
    test_model_architecture()
    test_generation()
    test_training()
    test_attention_visualization()

    print("\n" + "=" * 60)
    print("All MiniGPT examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
