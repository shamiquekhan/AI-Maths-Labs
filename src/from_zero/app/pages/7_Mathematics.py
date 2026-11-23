"""Mathematics reference page."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st  # noqa: E402

from from_zero.app import common  # noqa: E402

common.setup_page("Mathematics", "∑")
common.sidebar()

common.hero(
    "The equations behind the code.",
    "This page is the compact reference: what each model computes, what its objective is, "
    "and where the implementation makes a numerical choice. Full derivations and references "
    "live in `docs/mathematics/`.",
    ("Derivations", "Numerical stability", "Notation", "References"),
)

topic = st.selectbox(
    "Choose a topic",
    [
        "Strict tensor engine",
        "Logistic regression",
        "Optimizers",
        "Reverse-mode autodiff",
        "Transformer language model",
        "Denoising diffusion",
        "Evaluation and uncertainty",
    ],
)

if topic == "Strict tensor engine":
    st.markdown("## Tensors as storage plus a map")
    st.markdown(
        "A tensor is one flat list of floats, a shape, row-major strides, and an offset. "
        "The logical index maps to storage through"
    )
    st.latex(r"\mathrm{offset}(i_0,\dots,i_{n-1})=o+\sum_{k=0}^{n-1} i_k s_k")
    st.markdown(
        "For a contiguous row-major tensor, $s_{n-1}=1$ and $s_k=s_{k+1}d_{k+1}$. "
        "Transpose swaps shape and stride entries without moving data:"
    )
    st.latex(r"\mathrm{transpose}(d,s)\mapsto(d^{\mathrm{rev}},s^{\mathrm{rev}})")
    st.markdown("### Broadcasting")
    st.markdown(
        "Shapes align from the right; a dimension of size one stretches. The forward pass "
        "maps each output index back to the input index. The backward pass sums over the "
        "stretched axes:"
    )
    st.latex(r"\bar x_{i}=\sum_{j:\,\mathrm{proj}(j)=i}\bar y_j")
    st.markdown("### Reverse-mode autograd")
    st.markdown(
        "Every operation stores its parents and a local backward closure. `backward()` "
        "builds a reverse topological order and accumulates gradients, so a value used "
        "on several paths receives the sum of contributions:"
    )
    st.latex(
        r"\frac{\partial\mathcal{L}}{\partial v}"
        r"=\sum_{c\in\mathrm{children}(v)}"
        r"\frac{\partial\mathcal{L}}{\partial c}\frac{\partial c}{\partial v}"
    )
    st.markdown("### Matrix multiplication")
    st.latex(r"C_{ij}=\sum_k A_{ik}B_{kj},\qquad \bar A=\bar C B^\top,\qquad \bar B=A^\top\bar C")
    st.markdown(
        "Batched matmul broadcasts leading dimensions. Attention uses "
        "$(B,H,T,D)@(B,H,D,T)\\to(B,H,T,T)$; the batch and head axes are broadcast and the "
        "last two axes follow the rule above."
    )
    st.markdown("### Optimizers")
    st.markdown(
        "The strict optimizers update parameter storage in place and keep state in plain "
        "lists, so checkpoints are JSON. Adam is"
    )
    st.latex(
        r"m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad"
        r"v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2"
    )
    st.latex(r"w_t=w_{t-1}-\eta\,\frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)}+\epsilon}")
    st.markdown("### Verification")
    st.markdown(
        "Every differentiable operation is checked against a central finite difference. "
        "The import audit rejects NumPy, SciPy, PyTorch, TensorFlow, Keras, JAX, and "
        "scikit-learn inside the strict packages. The `logit = 0` BCE case has its own "
        "regression test because a naive `abs`/`relu` decomposition has the wrong "
        "derivative there."
    )
    st.markdown("### Limitations")
    st.markdown(
        "Pure Python loops are slow; non-contiguous reshape is rejected; advanced indexing "
        "is not implemented; everything is CPU double precision. The engine is a teaching "
        "tool, not a production framework."
    )

elif topic == "Logistic regression":
    st.markdown("## Binary logistic regression")
    st.markdown(
        "Let $x\\in\\mathbb{R}^d$ be a feature vector, $y\\in\\{0,1\\}$ a label, "
        "$w\\in\\mathbb{R}^d$ weights, and $b$ a bias. The linear score is "
        "$z = w^\\top x + b$."
    )
    st.latex(r"\sigma(z)=\frac{1}{1+e^{-z}},\qquad p(y=1\mid x)=\sigma(z)")
    st.markdown("### Loss")
    st.latex(
        r"\mathcal{L}(w,b)=-\frac{1}{N}\sum_{i=1}^N"
        r"\left[y_i\log\sigma(z_i)+(1-y_i)\log(1-\sigma(z_i))\right]"
        r"+\frac{\lambda}{2}\|w\|_2^2"
    )
    st.markdown(
        "Using $\\log\\sigma(z)=-\\log(1+e^{-z})$ and $\\log(1-\\sigma(z))=-z-\\log(1+e^{-z})$, the numerically stable form is"
    )
    st.latex(
        r"\mathcal{L}=\frac{1}{N}\sum_{i=1}^N"
        r"\left[\log(1+e^{z_i})-y_i z_i\right]+\frac{\lambda}{2}\|w\|_2^2"
    )
    st.markdown("### Gradient")
    st.markdown(
        "Differentiate with respect to the logit. The sigmoid derivative cancels the denominator:"
    )
    st.latex(r"\frac{\partial \mathcal{L}}{\partial z_i}=\frac{1}{N}(\sigma(z_i)-y_i)")
    st.latex(
        r"\nabla_w\mathcal{L}=X^\top\frac{\partial\mathcal{L}}{\partial z}+\lambda w,"
        r"\qquad \frac{\partial\mathcal{L}}{\partial b}=\sum_i\frac{\partial\mathcal{L}}{\partial z_i}"
    )
    st.markdown("### Numerical choices in this repository")
    st.markdown(
        "- The sigmoid is evaluated piecewise to avoid overflow.\n"
        "- The loss uses `softplus(z) - y z` rather than clipping probabilities.\n"
        "- Features are standardized with statistics fitted on the training split only.\n"
        "- The finite-difference check uses central differences with a relative error tolerance."
    )
    st.markdown("### Limitations")
    st.markdown(
        "The boundary is linear. Nonlinear datasets such as circles and XOR are expected to "
        "retain errors; that is a property of the model class, not a bug in the optimizer."
    )

elif topic == "Optimizers":
    st.markdown("## Gradient-based update rules")
    st.markdown(
        "All methods consume the same gradient $g_t=\\nabla_w\\mathcal{L}(w_t)$ and differ only in the state they maintain."
    )
    st.markdown("### SGD")
    st.latex(r"w_{t+1}=w_t-\eta g_t")
    st.markdown("### Momentum")
    st.latex(r"v_{t+1}=\beta v_t+g_t,\qquad w_{t+1}=w_t-\eta v_{t+1}")
    st.markdown("### RMSProp")
    st.latex(
        r"s_{t+1}=\rho s_t+(1-\rho)g_t^2,\qquad"
        r"w_{t+1}=w_t-\frac{\eta}{\sqrt{s_{t+1}}+\epsilon}g_t"
    )
    st.markdown("### Adam")
    st.latex(r"m_{t+1}=\beta_1m_t+(1-\beta_1)g_t,\qquad v_{t+1}=\beta_2v_t+(1-\beta_2)g_t^2")
    st.latex(
        r"\hat m_{t+1}=\frac{m_{t+1}}{1-\beta_1^{t+1}},\qquad"
        r"\hat v_{t+1}=\frac{v_{t+1}}{1-\beta_2^{t+1}}"
    )
    st.latex(r"w_{t+1}=w_t-\eta\frac{\hat m_{t+1}}{\sqrt{\hat v_{t+1}}+\epsilon}")
    st.markdown("### AdamW")
    st.markdown("AdamW decouples the weight decay from the adaptive gradient step:")
    st.latex(
        r"w_{t+1}=(1-\eta\lambda)w_t-\eta"
        r"\frac{\hat m_{t+1}}{\sqrt{\hat v_{t+1}}+\epsilon}"
    )
    st.markdown("### Schedules")
    st.markdown(
        "The library provides constant, step, exponential, and cosine schedules. A schedule "
        "changes $\\eta_t$; it does not change the state equations above."
    )
    st.markdown("### What a comparison can and cannot say")
    st.markdown(
        "The comparison uses one learning rate per objective and a fixed step budget. It "
        "answers “what happens under this controlled protocol?” It does not answer “which "
        "optimizer is best?” A proper answer requires tuning, multiple seeds, and error bars."
    )

elif topic == "Reverse-mode autodiff":
    st.markdown("## Reverse-mode differentiation on a scalar graph")
    st.markdown(
        "A computation is a directed acyclic graph. Each node $v$ has a forward value and a "
        "local partial derivative with respect to each parent. The chain rule gives"
    )
    st.latex(
        r"\frac{\partial \mathcal{L}}{\partial v}"
        r"=\sum_{c\in\mathrm{children}(v)}"
        r"\frac{\partial \mathcal{L}}{\partial c}\frac{\partial c}{\partial v}"
    )
    st.markdown(
        "If a node feeds several children, every contribution is added. That is why the "
        "backward closures use `+=`."
    )
    st.markdown("### Algorithm")
    st.markdown(
        "1. Build a topological order with parents before children.\n"
        "2. Set the root gradient to 1.\n"
        "3. Traverse the order in reverse; each node applies its local backward rule.\n"
        "4. Leaf gradients accumulate across calls until `zero_grad()` is invoked."
    )
    st.markdown("### Local rules used in this engine")
    st.markdown(
        "| node | forward | backward contribution |\n"
        "|---|---|---|\n"
        "| add | $a+b$ | $\\bar a += \\bar c$, $\\bar b += \\bar c$ |\n"
        "| multiply | $ab$ | $\\bar a += b\\bar c$, $\\bar b += a\\bar c$ |\n"
        "| power | $a^p$ | $\\bar a += p a^{p-1}\\bar c$ |\n"
        "| exp | $e^a$ | $\\bar a += e^a\\bar c$ |\n"
        "| log | $\\log a$ | $\\bar a += \\bar c/a$ |\n"
        "| tanh | $\\tanh a$ | $\\bar a += (1-\\tanh^2 a)\\bar c$ |\n"
        "| sigmoid | $\\sigma(a)$ | $\\bar a += \\sigma(a)(1-\\sigma(a))\\bar c$ |"
    )
    st.markdown("### Stability")
    st.markdown(
        "The binary cross-entropy is built from a stable softplus, not from `log(sigmoid(z))`. "
        "The engine rejects `log` of non-positive values and undefined powers at zero rather "
        "than silently returning NaN."
    )
    st.markdown("### Cost")
    st.markdown(
        "Every scalar operation allocates a Python object. The scalar engine is a teaching "
        "tool. Tensor workloads use the strict tensor engine (flat lists and the same "
        "reverse-mode idea) or PyTorch for the larger hybrid laboratories, where the same "
        "algorithm is implemented in C++ over multidimensional arrays."
    )

elif topic == "Transformer language model":
    st.markdown("## A decoder-only transformer")
    st.markdown(
        "Tokens are embedded and added to learned position embeddings. Each block applies "
        "layer normalization, causal self-attention, another normalization, and a feed-forward "
        "network, with residual connections."
    )
    st.latex(r"x_0 = E_{\mathrm{tok}}[x] + E_{\mathrm{pos}}")
    st.latex(
        r"x \leftarrow x+\mathrm{Attn}(\mathrm{LN}(x)),\qquad x\leftarrow x+\mathrm{FFN}(\mathrm{LN}(x))"
    )
    st.markdown("### Scaled dot-product attention")
    st.latex(
        r"Q=XW_Q,\quad K=XW_K,\quad V=XW_V,\quad"
        r"\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V"
    )
    st.markdown(
        "The causal mask has $M_{ij}=0$ for $j\\le i$ and a large negative value for $j>i$. "
        "It is applied before the softmax, so future positions receive zero weight."
    )
    st.markdown("### Objective")
    st.latex(r"\mathcal{L}=-\frac{1}{T}\sum_{t=1}^T\log p_\theta(x_{t+1}\mid x_{1:t})")
    st.markdown("Perplexity is the exponential of mean negative log-likelihood:")
    st.latex(r"\mathrm{PPL}=\exp\left(\frac{1}{T}\sum_t -\log p_\theta(x_{t+1}\mid x_{1:t})\right)")
    st.markdown(
        "Perplexity is comparable only for the same tokenizer and evaluation split. "
        "A character-level model and a subword model do not have comparable perplexities."
    )
    st.markdown("### Sampling")
    st.markdown(
        "Temperature divides logits before softmax; top-k removes all but the $k$ largest "
        "logits; greedy decoding takes the argmax. All three are implemented and tested for "
        "shape and determinism."
    )
    st.markdown("### Limitations")
    st.markdown(
        "The saved hybrid model has four layers, 128-dimensional embeddings, a 128-token "
        "context, and a 65-character vocabulary. It is trained for next-token prediction on "
        "Tiny Shakespeare. The strict variant is a one-block, 8-dimensional model trained "
        "for a few dozen steps on a tiny corpus; it exists to verify the custom engine and "
        "the causal mask. Neither model follows instructions or retrieves external knowledge."
    )

elif topic == "Denoising diffusion":
    st.markdown("## Denoising diffusion probabilistic models")
    st.markdown("### Forward process")
    st.latex(r"q(x_t\mid x_{t-1})=\mathcal{N}\left(\sqrt{1-\beta_t}x_{t-1},\beta_t I\right)")
    st.markdown("With $\\alpha_t=1-\\beta_t$ and $\\bar\\alpha_t=\\prod_{s=1}^t\\alpha_s$:")
    st.latex(r"q(x_t\mid x_0)=\mathcal{N}\left(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I\right)")
    st.latex(
        r"x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I)"
    )
    st.markdown("### Training objective")
    st.markdown(
        "The model predicts the noise $\\epsilon$ rather than the clean image. This is a "
        "reparameterization of the variational lower bound; the simple MSE objective is the "
        "one used here."
    )
    st.latex(
        r"\mathcal{L}=\mathbb{E}_{t,x_0,\epsilon}"
        r"\left[\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|_2^2\right]"
    )
    st.markdown("### Reverse posterior")
    st.markdown("Given $x_t$ and $x_0$, the posterior over $x_{t-1}$ is Gaussian:")
    st.latex(r"q(x_{t-1}\mid x_t,x_0)=\mathcal{N}\left(\tilde\mu_t(x_t,x_0),\tilde\beta_t I\right)")
    st.latex(
        r"\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0"
        r"+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t"
    )
    st.latex(r"\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t")
    st.markdown("Substituting the noise prediction for $x_0$ gives the sampler used in the code:")
    st.latex(
        r"\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}"
        r"\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)"
    )
    st.markdown(
        "At $t=0$ the sampler returns the mean without adding noise. At $t>0$ it adds "
        "$\\sqrt{\\tilde\beta_t}\\,z$ with $z\\sim\\mathcal{N}(0,I)$."
    )
    st.markdown("### Schedules")
    st.markdown(
        "The linear schedule sets $\\beta_t$ uniformly between `beta_start` and `beta_end`. "
        "The cosine schedule defines $\\bar\\alpha_t$ through a cosine curve and derives the "
        "betas from consecutive ratios. Both are checked for positivity and monotonicity."
    )
    st.markdown("### Limitations")
    st.markdown(
        "The saved hybrid model is a 540,961-parameter U-Net trained for 3,000 steps on 6,000 "
        "MNIST images. It does not report FID. The strict variant is an MLP denoiser on 4×4 "
        "patterns, used to verify the schedule and reverse chain without PyTorch. Neither is "
        "evidence of state-of-the-art image generation."
    )

else:
    st.markdown("## Evaluation and uncertainty")
    st.markdown("### Binary classification")
    st.latex(r"\mathrm{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}")
    st.latex(r"\mathrm{Precision}=\frac{TP}{TP+FP},\qquad \mathrm{Recall}=\frac{TP}{TP+FN}")
    st.latex(
        r"F_1=\frac{2\,\mathrm{Precision}\,\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}"
    )
    st.markdown(
        "ROC AUC summarizes ranking across thresholds. Calibration compares predicted "
        "probabilities with observed frequencies; a model can be accurate and poorly calibrated."
    )
    st.markdown("### Language modeling")
    st.latex(
        r"\mathrm{Perplexity}=\exp\left(-\frac{1}{N}\sum_{i=1}^N\log p_\theta(x_i\mid x_{<i})\right)"
    )
    st.markdown(
        "Perplexity depends on the tokenizer and the evaluation text. It is not a measure of "
        "factual correctness, helpfulness, or safety."
    )
    st.markdown("### Generative images")
    st.markdown(
        "This repository does not implement FID or a human evaluation protocol. Visual "
        "inspection is useful for debugging but is not a quantitative claim. A defensible "
        "evaluation would require a fixed reference set, a documented feature extractor, "
        "multiple seeds, and uncertainty estimates."
    )
    st.markdown("### Uncertainty and seeds")
    st.markdown(
        "Every experiment records its seed and environment. The optimizer comparison is a "
        "single seed, so its ranking should be read as a demonstration of dynamics, not a "
        "statistically stable conclusion. The test suite checks numerical correctness, not "
        "generalization."
    )

st.divider()
st.markdown("### References")
st.markdown(
    "- Rumelhart, Hinton & Williams (1986), *Learning representations by back-propagating errors*.\n"
    "- Kingma & Ba (2015), *Adam: A Method for Stochastic Optimization*.\n"
    "- Loshchilov & Hutter (2019), *Decoupled Weight Decay Regularization*.\n"
    "- Vaswani et al. (2017), *Attention Is All You Need*.\n"
    "- Radford et al. (2018), *Improving Language Understanding by Generative Pre-Training*.\n"
    "- Ho, Jain & Abbeel (2020), *Denoising Diffusion Probabilistic Models*.\n"
    "- Nichol & Dhariwal (2021), *Improved Denoising Diffusion Probabilistic Models*.\n"
    "- Goodfellow, Bengio & Courville (2016), *Deep Learning*.\n"
    "- Prince (2023), *Understanding Deep Learning*.\n"
    "- Karpathy, *nanoGPT* and *micrograd* (reference implementations and teaching material)."
)

common.footer()
