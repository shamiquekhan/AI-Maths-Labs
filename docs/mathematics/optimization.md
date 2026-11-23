# Optimization Algorithms: Mathematics and Implementation

## Common Setting

All optimizers in this library consume a gradient $g_t = \nabla_\theta \mathcal{L}(\theta_t)$ and update parameters $\theta_t$. They differ only in the **state** they maintain and the **update rule**.

Parameters are represented as a list of NumPy arrays (matching the neural network convention). The optimizer stores its state in a public `state` dictionary keyed by array identity.

---

## Vanilla SGD

$$\theta_{t+1} = \theta_t - \eta g_t$$

- No state.
- Simplest baseline.
- Sensitive to learning rate and curvature.

**Implementation:** `optim/sgd.py`

---

## Momentum (Classical)

Convention (matching PyTorch's `momentum` argument):

$$v_{t+1} = \beta v_t + g_t$$
$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$

**Note:** Some texts use $v_{t+1} = \beta v_t + (1-\beta)g_t$. Both are valid but mixing them changes the effective learning rate. This project uses the PyTorch convention.

- State: velocity $v$ (same shape as parameters).
- $\beta \in [0, 1)$ typical: 0.9.
- Dampens oscillations in ravines; accelerates along consistent gradients.

**Implementation:** `optim/momentum.py`

---

## RMSProp

$$s_{t+1} = \rho s_t + (1-\rho) g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_{t+1}} + \epsilon} g_t$$

- State: moving average of squared gradients $s$.
- $\rho \in [0, 1)$ typical: 0.99.
- $\epsilon = 10^{-8}$ prevents division by zero.
- Adapts per-parameter learning rates based on gradient magnitude history.

**Implementation:** `optim/rmsprop.py`

---

## Adam

Combines momentum and RMSProp with bias correction.

$$m_{t+1} = \beta_1 m_t + (1-\beta_1) g_t$$
$$v_{t+1} = \beta_2 v_t + (1-\beta_2) g_t^2$$

Bias-corrected estimates (because $m,v$ start at 0):

$$\hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^{t+1}}$$
$$\hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^{t+1}}$$

Update:

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_{t+1}}{\sqrt{\hat{v}_{t+1}} + \epsilon}$$

- State: $m$ (first moment), $v$ (second moment), $t$ (step counter).
- Defaults: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$.
- Widely used default; works well across many problems.

**Implementation:** `optim/adam.py` (`Adam` class)

---

## AdamW (Decoupled Weight Decay)

Standard L2 regularization adds $\lambda \theta$ to the gradient, which interacts with the adaptive learning rate. AdamW applies weight decay **directly to the parameters** before the adaptive step:

$$\theta_{t+1} = (1 - \eta \lambda) \theta_t - \eta \frac{\hat{m}_{t+1}}{\sqrt{\hat{v}_{t+1}} + \epsilon}$$

This is mathematically cleaner and often generalizes better.

**Implementation:** `optim/adam.py` (`AdamW` class, inherits from `Adam`)

---

## Learning-Rate Schedules

A schedule returns $\eta_t$ given step $t$. It does **not** change the optimizer's state equations.

| Schedule | Formula |
|----------|---------|
| Constant | $\eta_t = \eta_0$ |
| Step decay | $\eta_t = \eta_0 \gamma^{\lfloor t / s \rfloor}$ |
| Exponential | $\eta_t = \eta_0 \gamma^t$ |
| Cosine | $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})(1 + \cos(\pi t / T))$ |

**Implementation:** `optim/schedules.py`

---

## Controlled Comparison Protocol

To compare optimizers fairly:

1. **Same initial point** $\theta_0$ for each optimizer.
2. **Same objective function** (quadratic, Rosenbrock, or logistic loss).
3. **Objective-specific learning rate** chosen to make the comparison meaningful:
   - Quadratic: $\eta = 0.02$
   - Rosenbrock: $\eta = 0.001$
   - Logistic: $\eta = 0.05$
4. **Fixed step budget** (1000 steps in the documented run).
5. **Single seed** (results demonstrate dynamics, not statistical stability).

The comparison does **not** answer "which optimizer is best?"—that requires tuning, multiple seeds, and error bars. It answers "what happens under this controlled protocol?"

---

## Objectives Used

### Quadratic (Convex, Axis-Aligned)

$$f(\theta) = \frac{1}{2} \theta^\top A \theta, \quad A = \operatorname{diag}(c_1, c_2)$$

- Curvature $c_1, c_2$ control condition number.
- Optimum at $\theta = 0$.
- Analytical gradient: $A\theta$.

### Rosenbrock (Non-Convex, Narrow Valley)

$$f(x, y) = (a - x)^2 + b(y - x^2)^2$$

- Default $a=1, b=100$.
- Global minimum at $(1, 1)$ with $f=0$.
- Classic test for valley navigation.

### Logistic Loss (Non-Convex in Parameters, Convex in Logits)

$$\mathcal{L}(w,b) = \frac{1}{N}\sum_i \log(1+e^{-y_i(w^\top x_i + b)})$$

- Small 2D dataset.
- Represents a real ML objective.

---

## Implementation Interface

```python
class Optimizer:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        self.state: dict[str, Any] = {}

    def step(self, parameters: list[np.ndarray], gradients: list[np.ndarray]) -> None:
        """In-place update of parameters."""
        raise NotImplementedError

    def reset_state(self) -> None:
        self.state.clear()
```

- `step()` validates: non-empty lists, matching shapes, float64 dtype, finite values, writable arrays.
- State is stored per-parameter-array using `id(parameter)` as key.

---

## Numerical Tests

Each optimizer is tested for:
1. **First step** from a known scalar gradient.
2. **State updates** (momentum velocity, RMSProp square_avg, Adam m/v).
3. **Adam bias correction** at $t=1,2$.
4. **Zero gradient** behavior (should decay AdamW, not move SGD).
5. **Shape mismatch** errors.
6. **Invalid hyperparameters** (NaN, inf, negative LR, beta out of range).
7. **Reinitialization** via `reset_state()`.

---

## Limitations

- The comparison uses **one learning rate per objective**, not a tuned grid.
- **Single seed**—ranking is not statistically stable.
- **No batch stochasticity**—objectives use full gradients.
- **Quadratic and Rosenbrock** are low-dimensional; high-dimensional behavior differs.
- **No wall-clock time** measured (Python overhead dominates for tiny arrays).

---

## References

- Kingma & Ba, *Adam: A Method for Stochastic Optimization* (2015)
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (2019)
- PyTorch `torch.optim` source code (for convention matching)
- Dive into Deep Learning, Ch. 11 *Optimization Algorithms*