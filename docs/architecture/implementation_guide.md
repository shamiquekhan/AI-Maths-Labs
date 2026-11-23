# Implementation Guide

This guide provides detailed implementation patterns and conventions for building the FROM ZERO AI Engine.

## Project Structure

```
src/from_zero/
├── core/              # Tensor engine, autograd, ops
├── nn/                # Neural network modules
├── optim/             # Optimizers
├── models/            # Model implementations
├── data/              # Data loading
├── training/          # Training loop
├── experiments/       # Experiment tracking
└── utils/             # Utilities
```

## Coding Standards

### Python Version

- Target: Python 3.11+
- Type hints required for all public APIs
- Use `from __future__ import annotations` for forward references

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Modules | `snake_case` | `tensor.py` |
| Classes | `PascalCase` | `Tensor`, `Linear` |
| Functions | `snake_case` | `matmul`, `softmax` |
| Methods | `snake_case` | `backward`, `zero_grad` |
| Constants | `UPPER_SNAKE` | `DEFAULT_LR` |
| Private | `_leading_underscore` | `_internal_method` |

### Type Hints

```python
from __future__ import annotations
from typing import Optional, Tuple, List, Union, Callable
from from_zero.core.tensor import Tensor


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication."""
    ...


def broadcast_shape(shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute broadcast shape."""
    ...
```

### Docstrings

Use NumPy-style docstrings:

```python
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication of two tensors.

    Args:
        a: Left tensor with shape (..., M, K).
        b: Right tensor with shape (..., K, N).

    Returns:
        Output tensor with shape (..., M, N).

    Raises:
        ValueError: If inner dimensions don't match.

    Example:
        >>> a = tensor([[1., 2.], [3., 4.]])
        >>> b = tensor([[5., 6.], [7., 8.]])
        >>> matmul(a, b)
        tensor([[19., 22.], [43., 50.]])
    """
```

## Tensor Engine Patterns

### Tensor Creation

```python
# Preferred constructors
x = tensor([1.0, 2.0, 3.0])  # From list
x = zeros((3, 4))  # Zeros
x = ones((2, 2))  # Ones
x = randn((3, 3))  # Normal distribution
x = eye(3)  # Identity

# With gradient tracking
x = tensor([1.0, 2.0], requires_grad=True)
```

### Shape Manipulation

```python
# Always validate shapes
assert a.shape[-1] == b.shape[-2], f"Incompatible: {a.shape} @ {b.shape}"

# Use properties, not methods
x.shape
x.ndim
x.size  # total elements

# Reshape returns new tensor
y = x.reshape(3, 4)
z = x.transpose(-2, -1)
```

### Broadcasting

```python
# Check compatibility
if not broadcastable(a.shape, b.shape):
    raise ValueError(f"Cannot broadcast {a.shape} with {b.shape}")

# Use virtual indexing
out_shape = broadcast_shape(a.shape, b.shape)


# In backward pass
def unbroadcast_gradient(grad, target_shape):
    while grad.ndim > target.ndim:
        grad = grad.sum(0)
    for dim, (g, t) in enumerate(zip(grad.shape, target_shape)):
        if g != t:
            grad = grad.sum(dim=dim, keepdim=True)
    return grad
```

## Autograd Patterns

### Gradient Function Structure

```python
class Add(Function):
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        return a + b

    @staticmethod
    def backward(ctx, grad: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = ctx.saved_tensors
        # Unbroadcast to original shapes
        grad_a = unbroadcast(grad, a.shape)
        grad_b = unbroadcast(grad, b.shape)
        return grad_a, grad_b
```

### Gradient Accumulation

```python
def accumulate_grad(param: Tensor, grad: Tensor):
    """Add gradient to parameter's accumulated gradient."""
    if param.grad is None:
        param.grad = grad
    else:
        param.grad = param.grad + grad


def zero_grad(param: Tensor):
    """Reset parameter gradient."""
    param.grad = None
```

### Topological Sort

```python
def topological_sort(output: Tensor) -> List[Tensor]:
    """Return nodes in topological order (parents before children)."""
    visited = set()
    order = []

    def visit(node):
        if id(node) in visited:
            return
        visited.add(id(node))
        for parent in node.parents:
            visit(parent)
        order.append(node)

    visit(output)
    return order  # Parents first


def backward(output: Tensor, grad: Optional[Tensor] = None):
    if grad is None:
        assert output.size == 1, "Non-scalar output requires grad argument"
        grad = ones(())

    output.grad = grad
    for node in reversed(topological_sort(output)):
        if node.grad_fn:
            parent_grads = node.grad_fn(node.grad)
            for parent, g in zip(node.parents, parent_grads):
                if parent.requires_grad:
                    accumulate_grad(parent, g)
```

## Neural Network Patterns

### Module Base Class

```python
class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True

    def forward(self, *inputs) -> Tensor:
        raise NotImplementedError

    def __call__(self, *inputs) -> Tensor:
        return self.forward(*inputs)

    def parameters(self) -> List[Tensor]:
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def state_dict(self) -> Dict[str, Tensor]:
        return {name: p.data.clone() for name, p in self.named_parameters()}

    def load_state_dict(self, state: Dict[str, Tensor]):
        for name, param in self.named_parameters():
            if name in state:
                param.data.copy_(state[name])
```

### Parameter Registration

```python
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(randn((out_features, in_features)))
        self.register_parameter("weight", self.weight)

        if bias:
            self.bias = Parameter(zeros((out_features,)))
            self.register_parameter("bias", self.bias)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = matmul(x, self.weight.transpose())
        if self.bias is not None:
            out = out + self.bias
        return out
```

### Sequential Container

```python
class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        for i, module in enumerate(modules):
            self.add_module(str(i), module)

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x
```

## Optimizer Patterns

### Base Optimizer

```python
class Optimizer:
    def __init__(self, parameters: Iterable[Tensor], lr: float):
        self.parameters = list(parameters)
        self.lr = lr
        self.state = {}

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None

    def step(self):
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {"lr": self.lr, "state": {id(p): s for p, s in self.state.items()}}

    def load_state_dict(self, state: dict):
        self.lr = state["lr"]
        for p in self.parameters:
            if id(p) in state["state"]:
                self.state[id(p)] = state["state"][id(p)]
```

### SGD with Momentum

```python
class SGD(Optimizer):
    def __init__(
        self, parameters, lr=0.01, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False
    ):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum > 0 and dampening = 0")

    def step(self):
        for p in self.parameters:
            if p.grad is None:
                continue

            grad = p.grad

            # Weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data

            # Momentum
            if self.momentum != 0:
                state = self.state.setdefault(id(p), {})
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = grad.clone()
                    state["momentum_buffer"] = buf
                else:
                    buf.mul_(self.momentum).add_(grad, alpha=1 - self.dampening)

                if self.nesterov:
                    grad = grad + self.momentum * buf
                else:
                    grad = buf

            # Update
            p.data = p.data - self.lr * grad
```

## Training Loop Patterns

### Basic Trainer

```python
class Trainer:
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_fn: Callable,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.history = {"train_loss": [], "val_loss": []}

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            inputs, targets = batch
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with no_grad():
            for batch in self.val_loader:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def fit(self, epochs: int) -> Dict[str, List[float]]:
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate() if self.val_loader else None

            self.history["train_loss"].append(train_loss)
            if val_loss is not None:
                self.history["val_loss"].append(val_loss)

            print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        return self.history
```

## Data Loading Patterns

### Dataset

```python
class Dataset:
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError
```

### DataLoader

```python
class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = random.Random(seed)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            self.generator.shuffle(indices)

        if self.drop_last and len(indices) % self.batch_size != 0:
            indices = indices[: -(len(indices) % self.batch_size)]

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            inputs = stack([x for x, _ in batch])
            targets = stack([y for _, y in batch])
            yield inputs, targets

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
```

## Experiment Tracking Patterns

### Checkpoint Format

```python
checkpoint = {
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "epoch": epoch,
    "step": step,
    "rng_state": {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state() if has_torch else None,
    },
    "config": config_dict,
    "metrics": metrics_dict,
    "version": "1.0",
}
```

### Saving/Loading

```python
def save_checkpoint(path: str, checkpoint: dict):
    """Save checkpoint using safe serialization."""
    import pickle

    with open(path, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path: str, model: Module, optimizer: Optimizer) -> dict:
    import pickle

    with open(path, "rb") as f:
        checkpoint = pickle.load(f)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    # Restore RNG states
    random.setstate(checkpoint["rng_state"]["python"])
    np.random.set_state(checkpoint["rng_state"]["numpy"])

    return checkpoint
```

## Testing Patterns

### Unit Test Structure

```python
import pytest
from from_zero.core import Tensor


class TestTensorOperations:
    def test_addition(self):
        a = Tensor([1.0, 2.0])
        b = Tensor([3.0, 4.0])
        c = a + b
        assert c.data == [4.0, 6.0]

    def test_broadcasting(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([10, 20])
        c = a + b
        assert c.shape == (2, 2)
        assert c.data == [11, 22, 23, 24]

    def test_matmul(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[1, 0], [0, 1]])
        c = a @ b
        assert c.data == a.data

    def test_gradient(self):
        x = Tensor([2.0], requires_grad=True)
        y = x * 3.0
        y.backward()
        assert x.grad == 3.0
```

### Gradient Checking

```python
def finite_diff_gradient(fn, x, h=1e-6):
    """Central difference gradient."""
    grad = []
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((fn(x_plus) - fn(x_minus)) / (2 * h))
    return grad


def test_gradient_finite_diff():
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = (x[0] ** 2 + x[1] ** 2).sum()
    y.backward()

    numeric = finite_diff_gradient(lambda x: (x[0] ** 2 + x[1] ** 2).sum(), [2.0, 3.0])
    assert_allclose(x.grad, numeric, atol=1e-6)
```

## Error Handling

### Validation Helpers

```python
def validate_shape(tensor: Tensor, expected: Tuple[int, ...], name: str = "tensor"):
    if tensor.shape != expected:
        raise ValueError(f"{name} shape mismatch: expected {expected}, got {tensor.shape}")


def validate_dim(tensor: Tensor, min_dim: int, max_dim: int = None, name: str = "tensor"):
    if tensor.ndim < min_dim or (max_dim and tensor.ndim > max_dim):
        raise ValueError(f"{name} must have {min_dim}-{max_dim} dims, got {tensor.ndim}")


def check_finite(tensor: Tensor, name: str = "tensor"):
    if not all(math.isfinite(v) for v in tensor.data):
        raise ValueError(f"{name} contains NaN or Inf")
```

## Performance Profiling

```python
import time
from contextlib import contextmanager


@contextmanager
def timer(name: str):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.4f}s")


# Usage
with timer("matmul"):
    C = A @ B

# Profile memory
import tracemalloc

tracemalloc.start()
# ... operations ...
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1e6:.1f} MB, Peak: {peak / 1e6:.1f} MB")
tracemalloc.stop()
```

## Configuration Management

```python
from dataclasses import dataclass, asdict
import yaml


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    embedding_dim: int
    num_heads: int
    num_layers: int
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(**data)

    def save(self, path: str):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))
```

## Summary

This guide establishes consistent patterns for:
- Tensor operations and autograd
- Neural network modules
- Optimizers and training loops
- Data loading and experiment tracking
- Testing and debugging

Follow these patterns to maintain consistency across the codebase.