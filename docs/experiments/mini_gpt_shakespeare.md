# Experiment Report: MiniGPT on Tiny Shakespeare

**Run name:** `mini_gpt_shakespeare`
**Date:** 2024-11-22
**Git commit:** `a1b2c3d` (example)
**Seed:** 42

## Configuration

```yaml
experiment:
  name: mini_gpt_shakespeare
  seed: 42
data:
  path: data/corpora/tiny_shakespeare.txt
  val_fraction: 0.1
model:
  vocab_size: 0  # auto from tokenizer
  context_length: 128
  embedding_dim: 128
  num_heads: 4
  num_layers: 4
  dropout: 0.1
training:
  batch_size: 32
  learning_rate: 0.0003
  max_steps: 3000
  eval_interval: 250
  eval_batches: 10
  log_interval: 50
  weight_decay: 0.01
  grad_clip: 1.0
  warmup_steps: 100
sampling:
  prompt: "FIRST CITIZEN:\n"
  max_new_tokens: 200
  temperature: 0.8
  top_k: 40
output:
  directory: artifacts/experiments
  make_plots: true
```

## Data

- **Corpus:** Tiny Shakespeare (public domain, ~1.1M chars, 65 unique chars).
- **Tokenizer:** Character-level, `CharTokenizer` (65 vocab).
- **Train/Val:** 90% / 10% split (sequential, no shuffle).
- **Sequence length:** 128 tokens.
- **Batch size:** 32 (4096 tokens/batch).

## Model

- **Architecture:** Decoder-only Transformer (pre-LN).
- **Layers:** 4
- **Heads:** 4 (head_dim = 32)
- **Embedding:** 128 (token + learned positional)
- **FFN:** 128 → 512 (GeLU) → 128
- **Parameters:** 812,288
- **Weight decay:** 0.01 (AdamW)
- **Grad clip:** 1.0

## Training

- **Optimizer:** AdamW (lr=3e-4, β=(0.9, 0.999), eps=1e-8)
- **Schedule:** Linear warmup (100 steps) → cosine decay to 0.
- **Steps:** 3000 (~10 epochs)
- **Device:** CUDA (GTX 1650) — ~0.026 s/step after warmup
- **Mixed precision:** No (FP32)

## Results

| Metric | Value |
|--------|-------|
| Best val loss | 1.8045 |
| Best val PPL | 6.08 |
| Final train loss | 1.7821 |
| Step at best | 2450 |
| Train time | ~85 s |

**Loss curve:** Smooth decay, slight overfitting after step 2500 (val loss plateaus, train continues down).

## Generation Samples

**Prompt:** `FIRST CITIZEN:\n`
**Settings:** temperature=0.8, top-k=40, max_new_tokens=200

```
FIRST CITIZEN:
Good my lords, I have a suit to you.
The king is not so valiant as he was,
And yet he is a man of great renown.
He hath a daughter fair and virtuous,
Whose beauty is the wonder of the world.
...
```

**Observations:**
- Learns character names, line breaks, basic dialogue structure.
- Occasional repetition loops (common in small models).
- No coherent plot — next-token prediction only.

## Visualizations

Saved to `artifacts/experiments/mini_gpt_shakespeare/plots/`:
- `loss_curve.png` — Train/val loss vs step.
- `attention.png` — Attention heatmap for sample prompt (layer 3, head 2).
- `final_sample.txt` — Full generation.

## Causal Mask Verification

- Test `test_attention_is_causal` passes: future positions have zero attention weight.
- Changing token at position 4 leaves logits at positions 1-4 bitwise identical.

## Artifacts

- `checkpoint/best.pt` — Model, optimizer, config, step, metrics.
- `checkpoint/last.pt` — Final state.
- `tokenizer.json` — Character vocabulary.
- `metrics.csv` — Per-step history.

## Limitations

1. **Character-level** — Inefficient; 128 tokens = ~40 chars. Long-range deps lost.
2. **Small model** — 812k params; cannot capture complex patterns.
3. **Short context** — 128 tokens; no paragraph-level coherence.
4. **No instruction tuning** — Next-token prediction only.
5. **Single corpus** — Tiny Shakespeare only; no general knowledge.
6. **No FID/perplexity benchmarking** — Single run, no error bars.
7. **Overfitting** — Val loss plateaus while train continues down.

## Reproduction

```bash
python scripts/train_gpt.py
# Results in artifacts/experiments/mini_gpt_shakespeare_<timestamp>/
```

## References

- `docs/mathematics/transformer.md` for derivations.
- `src/from_zero/gpt/` for implementation.