# Tensor Engine: Storage, Views, Broadcasting, and Autograd

This document describes the **strict tensor core** in `src/from_zero/tensor/`.
The core imports only the Python standard library. It is deliberately small and
readable: flat lists for storage, row-major strides for views, and closures for
local backward rules.

The NumPy-based laboratories (`from_zero.logistic`, `from_zero.optim`) and the
PyTorch laboratories (`from_zero.gpt`, `from_zero.diffusion`) are separate and
are labelled as such in the dashboard.

---

## 1. Design goals

| Goal | Consequence |
|---|---|
| No third-party numerical library | `math`, `random`, `itertools`, and `builtins` only |
| Visible memory layout | one flat `list[float]` per tensor |
| Cheap views | transpose/permute swap strides; no data copy |
| Explicit autograd | each op stores parents and a local backward closure |
| Testable boundary | import audit + finite-difference checks |
| CPU-safe deployment | no CUDA, no large framework install |

The engine is educational, not a performance library. Python loops dominate the
cost; strict models stay tiny on purpose.

---

## 2. Storage and views

### 2.1 Flat storage

A `Tensor` has:

```python
_data:    list[float]   # flat storage (possibly shared by views)
_shape:   tuple[int, ...]
_strides: tuple[int, ...]  # row-major
_offset:  int
requires_grad: bool
grad:     Tensor | None
_prev:    tuple[Tensor, ...]
_backward: Callable[[], None] | None
_op:      str
```

The constructor accepts nested Python lists, a flat list plus an explicit
shape, or a scalar. The public factories are `tensor`, `zeros`, `ones`, `full`,
`randn`, `rand`, `uniform`, `eye`, and `arange`.

```python
from from_zero.tensor import tensor, zeros, randn

x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# shape (2, 3), strides (3, 1), flat storage [1, 2, 3, 4, 5, 6]

y = zeros((2, 2))
z = randn((3, 4), seed=0)
```

`Tensor.data` returns a **copy** of the logical row-major values, not the
underlying buffer. Internal code uses `_values()` for the same traversal.

### 2.2 Strides

For shape `(d₀, d₁, …, dₙ₋₁)`, row-major strides are:

```
sₙ₋₁ = 1
sₖ   = sₖ₊₁ · dₖ₊₁
```

The storage offset for logical index `(i₀, …, iₙ₋₁)` is:

```
offset = base + Σₖ iₖ · sₖ
```

`transpose` and `permute` return views with permuted shapes and strides. A
transpose of `(2, 3)` has shape `(3, 2)` and strides `(1, 3)`; the same six
floats are read in a different order.

```python
x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
x.T.tolist()        # [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
x.T.strides         # (1, 3)
```

`reshape` accepts `-1` for one inferred dimension. It is only valid when the
requested layout is compatible with the current strides; the implementation
raises rather than silently copying. `contiguous()` materializes a row-major
copy when needed.

### 2.3 Indexing

`__getitem__` supports integers, slices, and `None` (a new axis). Negative
indices follow Python conventions. The result is a view when possible; scalar
selection returns a 0-d tensor.

```python
x[0]        # shape (3,)
x[:, 1]     # shape (2,)
x[0, 1]     # 0-d tensor
x[None]     # shape (1, 2, 3)
```

`__setitem__` performs in-place scalar or cyclic tensor assignment and is not
differentiable. It exists for parameter initialization and data loading, not
for model math.

---

## 3. Broadcasting

Shapes are aligned from the right. Dimensions are compatible when they are
equal or one of them is 1:

| A | B | Result |
|---|---|---|
| `(3, 4)` | `(4,)` | `(3, 4)` |
| `(3, 1)` | `(1, 4)` | `(3, 4)` |
| `(5, 1, 3)` | `(4, 3)` | `(5, 4, 3)` |
| `(2, 3)` | `(2, 4)` | error |

The forward pass does not materialize the broadcast tensor. For each output
flat index, `broadcast_flat_index` maps back into the smaller input's logical
index. The backward pass must undo the expansion: `sum_to_shape` sums the
upstream gradient over every axis that was broadcast, preserving dimensions of
size one.

```python
bias = tensor([10.0, 20.0, 30.0])       # shape (3,)
out = x + bias                          # shape (2, 3)
# dL/dbias has shape (3,), not (2, 3)
```

This is the most common source of silent autograd bugs in hand-written
engines, so it has dedicated unit and numerical tests.

---

## 4. Autograd

### 4.1 Graph construction

Each differentiable operation:

1. Computes output values with gradient tracking disabled internally.
2. Creates an output tensor whose `_prev` contains its parents.
3. Stores a `_backward` closure that reads `output.grad` and accumulates into
   each parent's `grad`.

The graph is a DAG. A tensor used twice receives the sum of both contributions.

```python
x = tensor([2.0], requires_grad=True)
y = x * x + x          # x appears on two paths
y.backward()
x.grad.tolist()        # [5.0] = 2x + 1
```

### 4.2 Backward traversal

`backward()`:

1. Seeds the output gradient with ones (or accepts an explicit upstream tensor
   for non-scalar outputs).
2. Builds a post-order DFS over `_prev`.
3. Visits nodes in reverse order.
4. Calls each `_backward` closure with gradient tracking disabled, so the
   backward pass itself does not build a graph.

```python
loss.backward()          # scalar output
loss.backward(upstream)  # non-scalar output, same shape
```

Gradients accumulate in `.grad`; call `zero_grad()` (or an optimizer's
`zero_grad()`) between steps.

### 4.3 `no_grad`

```python
from from_zero.tensor import no_grad

with no_grad():
    logits = model(tokens)   # no graph is recorded
```

`no_grad` is a context manager backed by a module-level flag. Operations check
the flag before attaching `_prev`/`_backward`.

---

## 5. Operation reference

### 5.1 Elementwise unary ops

| Operation | Forward | Local derivative |
|---|---|---|
| `exp` | `eˣ` | `eˣ` |
| `log` | `ln x` | `1/x` |
| `log1p` | `ln(1+x)` | `1/(1+x)` |
| `sqrt` | `√x` | `1/(2√x)` |
| `abs` | `|x|` | `sign(x)` (0 at 0) |
| `tanh` | `tanh x` | `1−tanh²x` |
| `sigmoid` | `σ(x)` | `σ(x)(1−σ(x))` |
| `relu` | `max(x,0)` | `1` if `x>0`, else `0` |
| `softplus` | `ln(1+eˣ)` | `σ(x)` |
| `clamp(lo,hi)` | clipped value | `1` inside, `0` outside |
| `pow(p)` | `xᵖ` | `p·xᵖ⁻¹` |

`softplus` has an explicit backward rule rather than being composed from
`relu` and `abs`. The decomposition has derivative 0 at `x = 0`, while the true
derivative is `σ(0) = 0.5`. This matters for BCE-with-logits on zero logits and
is covered by a regression test.

### 5.2 Binary ops

All binary ops broadcast:

| Operation | `∂L/∂a` | `∂L/∂b` |
|---|---|---|
| `a + b` | `g` | `g` |
| `a - b` | `g` | `−g` |
| `a * b` | `g·b` | `g·a` |
| `a / b` | `g/b` | `−g·a/b²` |
| `a ** p` | `g·p·aᵖ⁻¹` | — |

The local gradients are computed in the broadcast shape, then reduced with
`sum_to_shape`.

### 5.3 Matrix multiplication

For `A ∈ ℝ^{m×k}` and `B ∈ ℝ^{k×n}`:

```
Cᵢⱼ = Σₖ Aᵢₖ Bₖⱼ
```

`matmul` supports:

- 2-D matrices,
- vectors (promoted to 1×k or k×1 and squeezed),
- batched operands with broadcast batch dimensions, e.g.
  `(B, H, T, D) @ (B, H, D, T) → (B, H, T, T)`.

Gradients are:

```
∂L/∂A = (∂L/∂C) @ Bᵀ
∂L/∂B = Aᵀ @ (∂L/∂C)
```

with the same batch broadcasting and reduction rules as elementwise ops.

### 5.4 Reductions

`sum`, `mean`, and `max` accept `dim` and `keepdim`. `argmax` returns Python
indices. The backward rules are:

- `sum`: broadcast the upstream gradient to the input shape.
- `mean`: broadcast and divide by the reduced element count.
- `max`: route the upstream gradient to the argmax position only (ties go to
  the first maximum).

`softmax` and `log_softmax` subtract the detached row maximum before
exponentiating, which prevents overflow without changing the gradient.

### 5.5 Sequence helpers

| Helper | Shape | Use |
|---|---|---|
| `embedding_lookup(weight, indices)` | `(...,) → (..., D)` | token embeddings |
| `gather_last(values, targets)` | `(N, C), (N,) → (N,)` | cross-entropy |
| `cat_last(a, b)` | `(..., D₁), (..., D₂) → (..., D₁+D₂)` | attention heads |

### 5.6 Losses

| Loss | Definition | Notes |
|---|---|---|
| `mse_loss` | `mean((p−y)²)` | regression / diffusion noise |
| `binary_cross_entropy_with_logits` | `mean(y·softplus(−z) + (1−y)·softplus(z))` | stable; correct at `z=0` |
| `cross_entropy` | `mean(−log_softmax(z)[target])` | language modeling |

All losses return a scalar tensor and participate in autograd.

---

## 6. Layers, initializers, and optimizers

The `nn_tensor` package builds on the tensor API:

- `Parameter` is a `Tensor` registered on a `Module`.
- `Linear`, `Embedding`, `LayerNorm`, `Dropout`, `Flatten`, and `Sequential`
  compose with `__call__`.
- `state_dict()` / `load_state_dict()` use plain Python values, not pickle.
- Initializers include Xavier/Glorot, He, LeCun, uniform, normal, zeros, and
  ones.

The `optim_tensor` package updates parameter storage **in place** and keeps
per-parameter state in plain lists:

| Optimizer | Update |
|---|---|
| SGD | `w ← w − ηg` |
| Momentum | `v ← μv + g; w ← w − ηv` |
| AdaGrad | `s ← s + g²; w ← w − ηg/(√s+ε)` |
| RMSprop | `s ← ρs + (1−ρ)g²; w ← w − ηg/(√s+ε)` |
| Adam | bias-corrected first/second moments |
| AdamW | Adam with decoupled weight decay |

Schedules include constant, step, exponential, cosine, polynomial, linear
warmup, and cosine warmup restarts. Gradient clipping is available as
`clip_grad_norm_` and `clip_grad_value_`.

---

## 7. Worked example

```python
from from_zero.tensor import tensor, no_grad
from from_zero.nn_tensor import Linear, ReLU, Sequential
from from_zero.optim_tensor import Adam

model = Sequential(Linear(2, 8), ReLU(), Linear(8, 1))
optimizer = Adam(model.parameters(), lr=0.05)

x = tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = tensor([[0.0], [1.0], [1.0], [0.0]])

for step in range(200):
    optimizer.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    optimizer.step()

with no_grad():
    print(model(x).tolist())
```

The same engine runs the strict logistic, character GPT, and diffusion
pipelines in `from_zero.strict`.

---

## 8. Verification

| Test file | Coverage |
|---|---|
| `tests/unit/test_tensor_engine.py` | construction, strides, views, indexing, broadcasting, matmul, softmax, embeddings, losses |
| `tests/numerical/test_strict_gradients.py` | central finite differences for every differentiable op, including BCE at zero logits |
| `tests/unit/test_nn_optim.py` | module registration, state dicts, layers, hand-checked first optimizer updates |
| `tests/unit/test_strict_support.py` | schedules, metrics, checkpoint round-trip |
| `tests/unit/test_strict_import_audit.py` | static import policy and subprocess import check |
| `tests/integration/test_strict_pipelines.py` | analytical vs. autograd logistic gradients, GPT causality/training, diffusion training/sampling |

Gradient checks use central differences with `h = 1e-6` and compare the maximum
absolute error against a tolerance appropriate to the expression. The strict
logistic analytical oracle is also compared directly against autograd.

---

## 9. Limitations

- **Speed:** Python loops make this orders of magnitude slower than NumPy.
- **Reshape:** non-contiguous reshape is rejected rather than copied silently.
- **Indexing:** advanced/fancy indexing is not implemented.
- **Precision:** all values are Python floats (IEEE-754 double).
- **Devices:** CPU only; there is no CUDA or distributed backend.
- **Scope:** enough operations for the strict models, not a general framework.

These limits are intentional. The hybrid NumPy and PyTorch laboratories remain
available for larger experiments and are labelled accordingly.
