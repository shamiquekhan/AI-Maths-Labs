# Transformer Language Model: Mathematics and Implementation

## Notation

| Symbol | Meaning |
|--------|---------|
| $V$ | Vocabulary size |
| $L$ | Context length (block size) |
| $d$ | Embedding dimension (`n_embd`) |
| $h$ | Number of attention heads |
| $d_k = d/h$ | Head dimension |
| $B$ | Batch size |
| $T$ | Sequence length ($T \le L$) |
| $x_t$ | Token ID at position $t$ |
| $E_{tok} \in \mathbb{R}^{V \times d}$ | Token embedding matrix |
| $E_{pos} \in \mathbb{R}^{L \times d}$ | Position embedding matrix |

---

## Tokenisation (Character-Level)

For the baseline model, each unique character maps to an integer ID:
- Vocabulary: sorted unique characters in the corpus.
- `encode(str) -> List[int]`
- `decode(List[int]) -> str`
- No subword merging, no BPE. Simple but creates a large sequence length for a given text.

**Why character-level?**
- Tiny vocabulary (65 for Tiny Shakespeare).
- No external tokenizer dependency.
- Transparent mapping between text and IDs.
- **Trade-off:** Sequences are ~4× longer than subword tokenisation, requiring more compute for the same context.

---

## Input Representation

Given token IDs $x = [x_1, \dots, x_T]$:

$$X_0 = E_{tok}[x] + E_{pos}[:T] \in \mathbb{R}^{T \times d}$$

- Token embeddings: learned lookup table.
- Position embeddings: learned (not sinusoidal), allowing the model to learn position interactions.
- **No dropout** on embeddings in the minimal implementation.

---

## Scaled Dot-Product Attention

For a single head, given input $X \in \mathbb{R}^{T \times d}$:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$
where $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$.

Attention scores:

$$S = \frac{Q K^\top}{\sqrt{d_k}} \in \mathbb{R}^{T \times T}$$

**Causal mask** $M$: $M_{ij} = 0$ for $j \le i$, $M_{ij} = -\infty$ (large negative) for $j > i$.

$$\tilde{S} = S + M$$

Weights:

$$A = \operatorname{softmax}(\tilde{S}, \text{dim}=-1) \in \mathbb{R}^{T \times T}$$

Output:

$$\text{head} = A V \in \mathbb{R}^{T \times d_k}$$

**Why scale by $\sqrt{d_k}$?**
Without scaling, dot products grow with $d_k$, pushing softmax into regions with vanishing gradients. Scaling keeps variance $\approx 1$ at initialization.

**Why causal mask?**
Language modeling predicts $x_{t+1}$ given $x_{\le t}$. The mask ensures position $t$ cannot attend to positions $> t$. Without it, the model would "cheat" by seeing future tokens during training.

---

## Multi-Head Attention

Concatenate $h$ heads, then project:

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O$$
where $W_O \in \mathbb{R}^{d \times d}$.

All heads share the same input $X$ but have independent $W_Q, W_K, W_V$ projections.

**Implementation detail:** The code reshapes $Q,K,V$ to $(B, h, T, d_k)$, computes batched attention, then reshapes back.

---

## Transformer Block (Pre-LN)

Each block applies two sublayers with residual connections and **pre-layer-normalization**:

$$X' = X + \text{MultiHead}(\text{LayerNorm}(X))$$
$$X'' = X' + \text{FFN}(\text{LayerNorm}(X'))$$

**Why pre-LN?** (vs. post-LN: `LayerNorm(X + Sublayer(X))`)
- Better gradient flow at initialization.
- More stable training, especially at larger depths.
- Original Transformer used post-LN; pre-LN is now standard (GPT-2, etc.).

---

## Feed-Forward Network (FFN)

Per-token, position-wise MLP:

$$\text{FFN}(x) = \text{GeLU}(x W_1 + b_1) W_2 + b_2$$

- Expansion factor: $d \to 4d \to d$.
- GeLU activation (Gaussian Error Linear Unit).
- Applied identically to each token position.

---

## Full Model

$$X_0 = E_{tok}[x] + E_{pos}[:T]$$
For $i = 1$ to $n\_layers$:
  $$X_i = \text{Block}_i(X_{i-1})$$
$$X_{final} = \text{LayerNorm}(X_n)$$
$$\text{logits} = X_{final} W_{out}^\top \in \mathbb{R}^{T \times V}$$
where $W_{out} \in \mathbb{R}^{V \times d}$ (tied with token embeddings in some implementations; separate here).

---

## Language Modeling Objective

Given input sequence $x_{1:T}$, targets are $y_t = x_{t+1}$ (shifted by one).

Cross-entropy loss (mean over positions and batch):

$$\mathcal{L} = -\frac{1}{BT} \sum_{b=1}^B \sum_{t=1}^T \log \frac{\exp(\text{logits}_{b,t,y_{b,t}})}{\sum_{v=1}^V \exp(\text{logits}_{b,t,v})}$$

**Perplexity:**
$$\text{PPL} = \exp(\mathcal{L})$$

Perplexity is comparable **only** for the same tokenizer and evaluation split. Character-level PPL $\neq$ subword PPL.

---

## Generation (Autoregressive)

At inference, generate token by token:

1. Start with prompt tokens $x_{1:t}$.
2. For $t+1, \dots, t+N$:
   a. Crop context to last $L$ tokens.
   b. Forward pass to get logits at last position.
   c. Apply temperature $\tau$: $\text{logits} \leftarrow \text{logits} / \tau$.
   d. Optional top-$k$: zero out all but top-$k$ logits.
   e. Sample $x_{next} \sim \text{Categorical}(\text{softmax}(\text{logits}))$.
   f. Append to sequence.

**Temperature:**
- $\tau \to 0$: greedy (argmax).
- $\tau = 1$: standard softmax.
- $\tau > 1$: flatter distribution, more diversity.

**Top-$k$:** Removes tail of distribution before sampling, avoiding nonsense tokens.

---

## Causal Mask Verification

The test (`tests/unit/test_gpt.py::test_attention_is_causal`):
1. Forward pass with input `[1,2,3,4,5,6]`, get attention weights.
2. Assert upper triangle (future positions) is zero.
3. Change token at position 4 → recompute.
4. Assert logits at positions 1-4 are **identical** (bitwise).
5. Assert logits at position 5+ **changed**.

This proves no information leaks from future to past.

---

## Configuration (Documented Run)

| Parameter | Value |
|-----------|-------|
| `vocab_size` | 65 (Tiny Shakespeare chars) |
| `context_length` | 128 |
| `embedding_dim` | 128 |
| `num_heads` | 4 |
| `num_layers` | 4 |
| `dropout` | 0.1 |
| `batch_size` | 32 |
| `learning_rate` | 3e-4 |
| `max_steps` | 3000 |
| `eval_interval` | 250 |
| Parameters | 812,288 |

---

## Limitations

- **Character-level:** Inefficient vocabulary; long-range dependencies harder.
- **No instruction tuning:** Next-token prediction only.
- **No alignment:** No RLHF, no safety filters.
- **Context 128:** Very short; cannot maintain coherence over paragraphs.
- **Single corpus:** Tiny Shakespeare only; no general knowledge.
- **Perplexity ~1.8:** Low absolute value (character-level), but not comparable to word-level models.

---

## References

- Vaswani et al., *Attention Is All You Need* (2017)
- Radford et al., *Improving Language Understanding by Generative Pre-Training* (2018)
- Karpathy, *nanoGPT* (https://github.com/karpathy/nanoGPT)
- The Annotated Transformer: https://nlp.seas.harvard.edu/annotated-transformer/