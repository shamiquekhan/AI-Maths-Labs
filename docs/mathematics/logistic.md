# Binary Logistic Regression: Mathematics and Implementation

## Notation

| Symbol | Meaning |
|--------|---------|
| $x \in \mathbb{R}^d$ | Feature vector (one example) |
| $X \in \mathbb{R}^{N \times d}$ | Design matrix ($N$ examples) |
| $y \in \{0,1\}^N$ | Binary labels |
| $w \in \mathbb{R}^d$ | Weight vector |
| $b \in \mathbb{R}$ | Bias scalar |
| $z = Xw + b$ | Logits (linear scores) |
| $\hat{y} = \sigma(z)$ | Predicted probabilities |
| $\lambda \ge 0$ | L2 regularization strength |

## Sigmoid

The logistic sigmoid maps any real logit to a probability in $(0,1)$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Numerically stable evaluation** (used in the implementation):

```
if z >= 0:
    return 1 / (1 + exp(-z))
else:
    exp_z = exp(z)
    return exp_z / (1 + exp_z)
```

This avoids overflow for large negative $z$ (where $e^{-z}$ would overflow) and large positive $z$ (where $e^{-z} \to 0$ and $1/(1+\epsilon) \approx 1 - \epsilon$ loses precision).

Derivative:
$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

## Binary Cross-Entropy

For a single example with target $y \in \{0,1\}$ and predicted probability $\hat{y} = \sigma(z)$:

$$\ell(\hat{y}, y) = - \left[ y \log \hat{y} + (1-y) \log(1-\hat{y}) \right]$$

Using the logits $z$ directly avoids evaluating $\log \sigma(z)$ and $\log(1-\sigma(z))$ separately, which can underflow. The stable form uses **softplus**:

$$\log \sigma(z) = -\operatorname{softplus}(-z), \quad \log(1-\sigma(z)) = -\operatorname{softplus}(z) - z$$

where $\operatorname{softplus}(x) = \log(1+e^x)$ is itself evaluated stably:

```
if x >= 0:
    return x + log1p(exp(-x))
else:
    return log1p(exp(x))
```

Mean loss over $N$ examples with L2 regularization on weights only:

$$\mathcal{L}(w,b) = \frac{1}{N} \sum_{i=1}^N \Bigl[ \operatorname{softplus}(z_i) - y_i z_i \Bigr] + \frac{\lambda}{2} \|w\|_2^2$$

## Gradients

The derivative of the loss with respect to the logit is remarkably simple:

$$\frac{\partial \mathcal{L}}{\partial z_i} = \frac{1}{N} (\hat{y}_i - y_i)$$

This follows from:
- $\frac{\partial}{\partial z} [-\log \sigma(z)] = 1 - \sigma(z)$
- $\frac{\partial}{\partial z} [-\log(1-\sigma(z))] = \sigma(z)$
- Combined with the chain rule and $z_i = x_i^\top w + b$

Then by the chain rule:

$$\nabla_w \mathcal{L} = X^\top \frac{\partial \mathcal{L}}{\partial z} + \lambda w$$

$$\frac{\partial \mathcal{L}}{\partial b} = \sum_{i=1}^N \frac{\partial \mathcal{L}}{\partial z_i} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$

**Key insight:** The gradient does not contain the sigmoid derivative explicitly—it cancels out. This is why the implementation computes `probs - y` once and reuses it.

## Finite-Difference Gradient Check

For a scalar function $f(\theta)$, the central difference approximation is:

$$f'(\theta_i) \approx \frac{f(\theta_i + h) - f(\theta_i - h)}{2h}$$

The implementation uses $h = 10^{-6}$ by default and checks:

$$\text{rel\_error} = \frac{|\nabla_\text{analytic} - \nabla_\text{numeric}|}{\max(|\nabla_\text{analytic}|, |\nabla_\text{numeric}|, 1)}$$

A check passes if $\text{rel\_error} < 10^{-5}$ for all components.

## Decision Boundary

For a 2D dataset, the decision boundary $p(y=1|x) = 0.5$ corresponds to:

$$w_1 x_1 + w_2 x_2 + b = 0 \quad \Longrightarrow \quad x_2 = -\frac{w_1}{w_2} x_1 - \frac{b}{w_2}$$

This is a line. **Limitation:** A linear boundary cannot separate non-linearly-separable data (e.g., concentric circles, XOR). The `moons` dataset in the demo demonstrates this—it achieves ~85% accuracy with a linear model, and the remaining errors are inherent to the model class, not optimizer failure.

## Regularization

L2 regularization $\frac{\lambda}{2}\|w\|_2^2$ adds $\lambda w$ to the weight gradient. It shrinks weights toward zero, reducing variance at the cost of bias. The bias $b$ is **not** regularized (standard practice).

## Implementation Notes

| Component | File | Key function |
|-----------|------|--------------|
| Sigmoid | `logistic/losses.py` | `sigmoid()` |
| BCE from logits | `logistic/losses.py` | `binary_cross_entropy_with_logits()` |
| Gradients | `logistic/gradients.py` | `compute_gradients()` |
| Gradient check | `logistic/gradients.py` | `check_gradients()` |
| Classifier | `logistic/model.py` | `LogisticRegression` class |

The `LogisticRegression.fit()` method uses mini-batch SGD with optional early stopping on validation loss. Features are standardized using `StandardScaler` fitted on the **training split only**—never on the full dataset before splitting.

## References

- Bishop, *Pattern Recognition and Machine Learning*, §4.3.2
- Goodfellow et al., *Deep Learning*, §6.2.2
- scikit-learn `LogisticRegression` source (for comparison only)