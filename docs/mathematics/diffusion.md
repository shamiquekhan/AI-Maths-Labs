# Denoising Diffusion Probabilistic Models: Mathematics and Implementation

## Notation

| Symbol | Meaning |
|--------|---------|
| $x_0 \in \mathbb{R}^{C \times H \times W}$ | Clean image (data sample) |
| $x_t$ | Noisy image at timestep $t$ |
| $T$ | Total timesteps (e.g., 1000) |
| $\beta_t \in (0,1)$ | Variance schedule |
| $\alpha_t = 1 - \beta_t$ | Signal retention at step $t$ |
| $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ | Cumulative signal retention |
| $\epsilon \sim \mathcal{N}(0, I)$ | Standard Gaussian noise |
| $\epsilon_\theta(x_t, t)$ | Noise prediction network |

---

## Forward Process (Fixed, No Learning)

The forward process gradually adds Gaussian noise:

$$q(x_t | x_{t-1}) = \mathcal{N}\bigl(\sqrt{1-\beta_t} x_{t-1}, \beta_t I\bigr)$$

By the Markov property and Gaussian composition, we can sample $x_t$ **directly** from $x_0$:

$$q(x_t | x_0) = \mathcal{N}\bigl(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I\bigr)$$

**Reparameterization trick:**
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

This allows training to sample any timestep $t$ without simulating the full chain.

---

## Noise Schedules

### Linear Schedule

$$\beta_t = \beta_{start} + \frac{t-1}{T-1}(\beta_{end} - \beta_{start})$$

Typical: $\beta_{start} = 10^{-4}, \beta_{end} = 0.02$.

### Cosine Schedule (Improved DDPM)

Defines $\bar{\alpha}_t$ via a cosine curve:

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos^2\Bigl(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\Bigr)$$

Then $\beta_t = 1 - \bar{\alpha}_t / \bar{\alpha}_{t-1}$ (clamped to $[10^{-8}, 0.999]$).

Parameter $s$ (offset) controls how quickly noise increases at the start. Typical $s = 0.008$.

**Properties checked:**
- $\beta_t \in (0, 1)$
- $\bar{\alpha}_t \in (0, 1]$
- $\bar{\alpha}_t$ is non-increasing
- $\bar{\alpha}_0 = 1$

---

## Training Objective (Noise Prediction)

Instead of predicting $x_0$, the model predicts the **noise** $\epsilon$ added to get $x_t$. This is a reparameterization of the variational lower bound; the "simple" objective is:

$$\mathcal{L}_{simple} = \mathbb{E}_{t, x_0, \epsilon} \bigl[ \| \epsilon - \epsilon_\theta(x_t, t) \|_2^2 \bigr]$$

**Why predict noise?**
- Equivalent to predicting $x_0$ or the score $\nabla_{x_t} \log q(x_t)$.
- Noise has stationary distribution $\mathcal{N}(0,I)$ regardless of $t$.
- Empirically more stable.

**Training algorithm:**
1. Sample $x_0$ from data.
2. Sample $t \sim \text{Uniform}\{1, \dots, T\}$.
3. Sample $\epsilon \sim \mathcal{N}(0, I)$.
4. Compute $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$.
5. Predict $\hat{\epsilon} = \epsilon_\theta(x_t, t)$.
6. Loss = MSE($\epsilon, \hat{\epsilon}$).

---

## Reverse Process (Learned)

The true reverse posterior $q(x_{t-1} | x_t, x_0)$ is Gaussian:

$$q(x_{t-1} | x_t, x_0) = \mathcal{N}\bigl(\tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I\bigr)$$

where
$$\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t$$
$$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t$$

Since $x_0$ is unknown at sampling time, substitute the model's prediction:
$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

Then the **sampling mean** is:
$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \Bigl( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \Bigr)$$

**Reverse step ($t > 0$):**
$$x_{t-1} = \mu_\theta(x_t, t) + \sqrt{\tilde{\beta}_t} z, \quad z \sim \mathcal{N}(0, I)$$

**Final step ($t = 1 \to 0$):**
$$x_0 = \mu_\theta(x_1, 1) \quad \text{(no noise added)}$$

---

## Timestep Embedding

The model needs to know $t$ to predict $\epsilon$. A sinusoidal embedding (like Transformer positional encodings) is used:

$$\text{emb}_t[2i] = \sin(t / 10000^{2i/d})$$
$$\text{emb}_t[2i+1] = \cos(t / 10000^{2i/d})$$

This is passed through a small MLP (2-layer, SiLU activation) and added to the U-Net blocks via AdaGN (adaptive group norm) or simple addition.

---

## Denoising Network: Compact U-Net

For MNIST (28×28, 1 channel), a small U-Net suffices:

```
Input: (B, 1, 28, 28) + timestep embedding
|
|-- Down 1: ResBlock(1 -> 32) -> ResBlock(32 -> 32) -> Downsample
|-- Down 2: ResBlock(32 -> 64) -> ResBlock(64 -> 64) -> Downsample
|-- Mid:    ResBlock(64 -> 64) -> Attention(64) -> ResBlock(64 -> 64)
|-- Up 2:   Upsample -> Concat(skip) -> ResBlock(128 -> 32) -> ResBlock(32 -> 32)
|-- Up 1:   Upsample -> Concat(skip) -> ResBlock(64 -> 32) -> ResBlock(32 -> 32)
|
Output: (B, 1, 28, 28)  # predicted noise
```

- **ResBlock:** Conv -> GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU (+ residual if channels match).
- **Attention:** Single-head spatial self-attention at bottleneck.
- **Timestep injection:** Projected embedding added to GroupNorm scale/shift (AdaGN) or added to feature maps.

Total parameters: ~540k (for the documented run).

---

## Sampling Algorithm

```python
def sample_ddpm(model, shape, schedule, device):
    x = torch.randn(shape, device=device)  # x_T ~ N(0, I)
    for t in reversed(range(schedule.timesteps)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps_pred = model(x, t_batch)
        mu = posterior_mean_from_noise_pred(x, t_batch, eps_pred, schedule)
        if t > 0:
            z = torch.randn_like(x)
            x = mu + schedule.posterior_std[t] * z
        else:
            x = mu  # t=0: no noise
    return x.clamp(-1, 1)  # images in [-1, 1]
```

Key detail: **At t=0, no noise is added.** The implementation special-cases this to avoid numerical issues where $\tilde{\beta}_0 = 0$ but the formula might produce NaN due to $1-\bar{\alpha}_0 = 0$ in the denominator.

---

## Posterior Mean Identities (Numerical Tests)

The test `tests/numerical/test_diffusion_math.py` verifies:

1. **$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$** (cumulative product).
2. **Posterior variance** matches closed form:
   $$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$$
3. **Forward process moments:** Empirical mean/variance of $x_t$ match $\sqrt{\bar{\alpha}_t}x_0$ and $1-\bar{\alpha}_t$.
4. **Noise/x0 recovery:** Given $x_t$ and $\epsilon$, recovering $x_0$ and $\epsilon$ is consistent (within float32 tolerance).
5. **Posterior mean** is a convex combination of $x_0$ and $x_t$ (coefficients sum to $\approx 1$).
6. **t=0 boundary:** Posterior mean = $x_0$, variance = 0.

---

## Configuration (Documented MNIST Run)

| Parameter | Value |
|-----------|-------|
| Dataset | MNIST (60k train, 10k val subset) |
| Image size | 28×28, 1 channel |
| Timesteps | 1000 |
| Schedule | Linear ($\beta_1=10^{-4}, \beta_T=0.02$) |
| Model | U-Net (base_ch=32, mults=[1,2,4], 2 res blocks) |
| Parameters | 540,961 |
| Batch size | 128 |
| Learning rate | 2e-4 (AdamW) |
| Max steps | 3000 |
| Eval interval | 250 |
| Sample interval | 500 |
| Device | CUDA (GTX 1650, ~0.045 s/step) |

---

## Limitations

- **No FID evaluation:** Requires a reference batch, Inception features, multiple seeds. Not implemented.
- **Small model/data:** 3k steps on 6k MNIST images. Not comparable to published DDPM results (100M+ params, 1M+ images, millions of steps).
- **No classifier-free guidance:** Conditional generation not implemented.
- **Float32 only:** No mixed precision or EMA.
- **Sample quality:** Recognizable digits but blurry; demonstrates the reverse process works, not SOTA generation.

---

## References

- Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models* (2020)
- Nichol & Dhariwal, *Improved Denoising Diffusion Probabilistic Models* (2021)
- Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations* (2021)
- Hugging Face Annotated Diffusion: https://huggingface.co/blog/annotated-diffusion
- Hugging Face Diffusion Course: https://huggingface.co/learn/diffusion-course/