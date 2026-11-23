# Mathematical Foundations

This document establishes the mathematical notation, conventions, and foundational concepts used throughout the FROM ZERO AI Laboratory.

## Notation Conventions

### Scalars, Vectors, Matrices, Tensors

| Symbol | Meaning | Example |
|--------|---------|---------|
| $x, y, z$ | Scalars | $x = 3.14$ |
| $\mathbf{x}, \mathbf{y}$ | Vectors (column) | $\mathbf{x} = [1, 2, 3]^\top$ |
| $X, Y, Z$ | Matrices | $X \in \mathbb{R}^{m \times n}$ |
| $\mathcal{X}, \mathcal{Y}$ | Higher-order tensors | $\mathcal{X} \in \mathbb{R}^{B \times T \times C}$ |

### Indices and Dimensions

| Symbol | Meaning |
|--------|---------|
| $i, j, k$ | Indices (0-based) |
| $m, n, k$ | Matrix dimensions (rows, cols, inner) |
| $B, T, C$ | Batch, time/sequence, channels |
| $H, W$ | Height, width (images) |
| $D, d$ | Embedding dimension, head dimension |

### Sets and Spaces

| Symbol | Meaning |
|--------|---------|
| $\mathbb{R}$ | Real numbers |
| $\mathbb{N}$ | Natural numbers |
| $\mathbb{R}^{m \times n}$ | Space of $m \times n$ real matrices |
| $\mathcal{N}(\mu, \sigma^2)$ | Gaussian distribution |
| $\mathcal{U}(a, b)$ | Uniform distribution |

## Linear Algebra

### Vectors

A vector $\mathbf{x} \in \mathbb{R}^n$ is an ordered list of $n$ scalars.

**Operations:**

- **Dot product**: $\mathbf{x} \cdot \mathbf{y} = \sum_{i=1}^n x_i y_i = \mathbf{x}^\top \mathbf{y}$
- **Norm**: $\|\mathbf{x}\|_2 = \sqrt{\mathbf{x} \cdot \mathbf{x}} = \sqrt{\sum_i x_i^2}$
- **Elementwise**: $\mathbf{x} \odot \mathbf{y} = [x_1 y_1, \dots, x_n y_n]$

### Matrices

A matrix $X \in \mathbb{R}^{m \times n}$ has $m$ rows and $n$ columns.

**Operations:**

- **Transpose**: $X^\top \in \mathbb{R}^{n \times m}$, $(X^\top)_{ij} = X_{ji}$
- **Matrix-vector**: $X\mathbf{y} \in \mathbb{R}^m$, $(X\mathbf{y})_i = \sum_j X_{ij} y_j$
- **Matrix-matrix**: $AB \in \mathbb{R}^{m \times p}$, $(AB)_{ik} = \sum_j A_{ij} B_{jk}$
- **Trace**: $\text{tr}(X) = \sum_i X_{ii}$ (square matrices)
- **Determinant**: $\det(X)$ (square matrices)
- **Inverse**: $X^{-1}$ where $X X^{-1} = I$ (if invertible)

### Special Matrices

| Type | Notation | Property |
|------|----------|----------|
| Identity | $I$ | $I_{ij} = \delta_{ij}$ |
| Diagonal | $\text{diag}(\mathbf{d})$ | $D_{ij} = d_i \delta_{ij}$ |
| Symmetric | $X = X^\top$ | $X_{ij} = X_{ji}$ |
| Orthogonal | $Q^\top Q = I$ | $Q^{-1} = Q^\top$ |
| Positive definite | $X \succ 0$ | $\mathbf{v}^\top X \mathbf{v} > 0$ for all $\mathbf{v} \neq 0$ |

### Matrix Decompositions

- **Eigendecomposition**: $X = V \Lambda V^{-1}$ (symmetric: $V^\top = V^{-1}$)
- **SVD**: $X = U \Sigma V^\top$, $\Sigma$ diagonal
- **Cholesky**: $X = LL^\top$ (positive definite)
- **QR**: $X = QR$ ($Q$ orthogonal, $R$ upper triangular)

## Calculus

### Derivatives

**Scalar function $f: \mathbb{R} \to \mathbb{R}$:**

$$f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Multivariable $f: \mathbb{R}^n \to \mathbb{R}$:**

- **Gradient**: $\nabla f(\mathbf{x}) = \left[\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n}\right]^\top$
- **Hessian**: $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$

**Vector-valued $f: \mathbb{R}^n \to \mathbb{R}^m$:**

- **Jacobian**: $J_{ij} = \frac{\partial f_i}{\partial x_j} \in \mathbb{R}^{m \times n}$

### Chain Rule

**Scalar composition $z = f(g(x))$:**

$$\frac{dz}{dx} = \frac{df}{dg} \frac{dg}{dx} = f'(g(x)) g'(x)$$

**Multivariable $z = f(\mathbf{g}(\mathbf{x}))$:**

$$\frac{\partial z}{\partial x_i} = \sum_j \frac{\partial z}{\partial g_j} \frac{\partial g_j}{\partial x_i}$$

**Matrix form:**

$$\nabla_\mathbf{x} z = J_{\mathbf{g}}(\mathbf{x})^\top \nabla_{\mathbf{g}} z$$

### Matrix Calculus

**Key identities:**

| Expression | Derivative |
|------------|------------|
| $\frac{\partial}{\partial X} \text{tr}(AX)$ | $A^\top$ |
| $\frac{\partial}{\partial X} \text{tr}(X^\top A)$ | $A$ |
| $\frac{\partial}{\partial X} \text{tr}(XAX^\top)$ | $X(A + A^\top)$ |
| $\frac{\partial}{\partial X} \log \det(X)$ | $X^{-\top}$ |
| $\frac{\partial}{\partial X} X^{-1}$ | $-X^{-1} \otimes X^{-\top}$ |

**Vector-by-matrix:**

| Expression | Derivative |
|------------|------------|
| $\frac{\partial}{\partial \mathbf{x}} \mathbf{a}^\top \mathbf{x}$ | $\mathbf{a}$ |
| $\frac{\partial}{\partial \mathbf{x}} \mathbf{x}^\top A \mathbf{x}$ | $(A + A^\top)\mathbf{x}$ |
| $\frac{\partial}{\partial X} \text{tr}(X^\top B)$ | $B$ |

## Probability and Statistics

### Distributions

**Bernoulli:** $X \sim \text{Bernoulli}(p)$
- $P(X=1) = p$, $P(X=0) = 1-p$
- $\mathbb{E}[X] = p$, $\text{Var}(X) = p(1-p)$

**Categorical:** $\mathbf{x} \sim \text{Cat}(\mathbf{p})$
- $P(x_i = 1) = p_i$, $\sum_i p_i = 1$

**Gaussian:** $X \sim \mathcal{N}(\mu, \sigma^2)$
- $f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
- $\mathbb{E}[X] = \mu$, $\text{Var}(X) = \sigma^2$

**Multivariate Gaussian:** $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$
- $f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$

### Expectation and Variance

$$\mathbb{E}[X] = \sum_x x P(X=x) \quad \text{(discrete)}$$
$$\mathbb{E}[X] = \int x f(x) dx \quad \text{(continuous)}$$

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$$

**Linearity:** $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$

**Covariance:** $\text{Cov}(X, Y) = \mathbb{E}[(X-\mu_X)(Y-\mu_Y)]$

**Correlation:** $\rho_{XY} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$

### Sampling

- **Inverse transform**: $X = F^{-1}(U)$, $U \sim \mathcal{U}(0,1)$
- **Box-Muller** (Gaussian): $Z_0 = \sqrt{-2\ln U_1} \cos(2\pi U_2)$
- **Reparameterization**: $\mathbf{x} = \boldsymbol{\mu} + \Sigma^{1/2} \boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$

## Information Theory

### Entropy

**Discrete:** $H(X) = -\sum_x P(x) \log P(x)$

**Continuous (differential):** $h(X) = -\int f(x) \log f(x) dx$

**Cross-entropy:** $H(p, q) = -\sum_x p(x) \log q(x)$

**KL Divergence:** $D_{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} \ge 0$

### Mutual Information

$$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = D_{KL}(P_{XY} \| P_X P_Y)$$

## Optimization Theory

### Gradient Descent

For $f: \mathbb{R}^n \to \mathbb{R}$:

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)$$

**Convergence (convex, L-smooth):**

$$f(\mathbf{x}_T) - f(\mathbf{x}^*) \le \frac{\|\mathbf{x}_0 - \mathbf{x}^*\|^2}{2\eta T}$$

### Stochastic Gradient Descent

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta_t \mathbf{g}_t, \quad \mathbb{E}[\mathbf{g}_t] = \nabla f(\mathbf{x}_t)$$

**Convergence (non-convex):** $\min_{t < T} \mathbb{E}[\|\nabla f(\mathbf{x}_t)\|^2] \le \mathcal{O}(1/\sqrt{T})$

### Momentum Methods

**Classical momentum:** $\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \nabla f(\mathbf{x}_t)$, $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \mathbf{v}_{t+1}$

**Nesterov momentum:** $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t + \beta \mathbf{v}_t)$

### Adaptive Methods

**AdaGrad:** $\mathbf{r}_t = \mathbf{r}_{t-1} + \mathbf{g}_t^2$, $\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{r}_t} + \epsilon} \mathbf{g}_t$

**RMSProp:** $\mathbf{r}_t = \rho \mathbf{r}_{t-1} + (1-\rho)\mathbf{g}_t^2$

**Adam:** $\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t$, $\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2$

## Numerical Analysis

### Floating Point

- **Machine epsilon**: $\epsilon \approx 2.22 \times 10^{-16}$ (double), $1.19 \times 10^{-7}$ (float32)
- **Underflow**: $x \approx 0$ when $|x| < 10^{-308}$ (double)
- **Overflow**: $x = \infty$ when $|x| > 10^{308}$ (double)

### Condition Number

$$\kappa(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)} = \|A\| \|A^{-1}\|$$

- **Well-conditioned**: $\kappa \approx 1$
- **Ill-conditioned**: $\kappa \gg 1$

### Numerical Stability

**Catastrophic cancellation**: $x - y \approx 0$ for $x \approx y$

**Stable formulas:**

- $\log(1+x) \approx x - x^2/2 + x^3/3$ for small $x$
- $\log(1+e^x) = \max(0, x) + \log(1+e^{-|x|})$
- Softmax: $\text{softmax}(x_i) = \frac{e^{x_i - \max_j x_j}}{\sum_j e^{x_j - \max_j x_j}}$

## Notation Summary

| Category | Symbols |
|----------|---------|
| Indices | $i, j, k, l, r, s, t$ |
| Dimensions | $m, n, k, d, D, B, T, C$ |
| Data | $x, y, z$ (scalars), $\mathbf{x}, \mathbf{y}$ (vectors), $X, Y, Z$ (matrices) |
| Parameters | $\theta, \phi, w, W, b$ |
| Gradients | $\nabla, \frac{\partial}{\partial x}, g$ |
| Learning rate | $\eta, \alpha, \gamma$ |
| Regularization | $\lambda, \lambda_2$ |
| Probability | $p, q, P, Q$ |
| Expectation | $\mathbb{E}[\cdot]$ |
| Variance | $\text{Var}(\cdot)$ |
| Norm | $\|\cdot\|$ (default L2) |
| Inner product | $\langle \cdot, \cdot \rangle$ or $\cdot$ |
| Elementwise | $\odot$ |

---

*This document serves as a reference for all mathematical notation used in the FROM ZERO documentation and implementation.*