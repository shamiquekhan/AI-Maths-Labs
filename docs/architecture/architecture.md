# System Architecture

## Overview

FROM ZERO is a layered AI engineering laboratory built on a custom tensor and autograd engine. The architecture follows a strict dependency hierarchy where each layer builds only on the layers below it.

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                         │
│  Tensor Studio │ Autograd Studio │ Optimizer Lab             │
│  Logistic Lab  │ Mini GPT Lab    │ Diffusion Lab             │
│  Experiment Lab                                              │
└──────────────────────────┬────────────────────────────────────┘
                           │
┌──────────────────────────┴────────────────────────────────────┐
│                    MODEL LAYER                                 │
│  Logistic Regression │ Mini GPT │ Diffusion Model             │
└──────────────────────────┬────────────────────────────────────┘
                           │
┌──────────────────────────┴────────────────────────────────────┐
│                 NEURAL NETWORK ENGINE                          │
│  Modules │ Layers │ Activations │ Losses │ Initializers       │
└──────────────────────────┬────────────────────────────────────┘
                           │
┌──────────────────────────┴────────────────────────────────────┐
│                 OPTIMIZATION ENGINE                            │
│  SGD │ Momentum │ AdaGrad │ RMSProp │ Adam                     │
└──────────────────────────┬────────────────────────────────────┘
                           │
┌──────────────────────────┴────────────────────────────────────┐
│                   AUTOGRAD ENGINE                              │
│  Computational Graph │ Reverse-Mode AD │ Gradient Checking    │
└──────────────────────────┬────────────────────────────────────┘
                           │
┌──────────────────────────┴────────────────────────────────────┐
│                    TENSOR ENGINE                               │
│  Storage │ Indexing │ Broadcasting │ Ops │ Reductions │ MatMul │
└──────────────────────────┬────────────────────────────────────┘
                           │
┌──────────────────────────┴────────────────────────────────────┐
│                      PYTHON RUNTIME                            │
└────────────────────────────────────────────────────────────────┘
```

## Core Design Principles

### 1. Strict Layer Separation

Each layer only depends on layers below it:

- **Tensor Engine**: No dependencies on autograd or higher
- **Autograd**: Depends only on Tensor Engine
- **Neural Network**: Depends on Tensor + Autograd
- **Optimizers**: Depend on Tensor + Autograd (for gradients)
- **Models**: Depend on NN Engine + Optimizers
- **Application**: Uses all layers via public APIs only

### 2. No Circular Dependencies

```
Tensor Engine → Autograd → NN Engine → Optimizers → Models → App
     ↑                                                       │
     └─────────────────── No backward deps ──────────────────┘
```

### 3. Explicit Interfaces

Each layer exposes a clean public API. Internal implementation details are private.

## Layer Details

### Tensor Engine (`src/from_zero/core/`)

**Responsibilities:**
- Flat array storage with shape/stride metadata
- Elementwise operations with broadcasting
- Reductions (sum, mean, max, min)
- Matrix multiplication (batched)
- Stable mathematical functions (sigmoid, softmax, etc.)
- Shape manipulation (reshape, transpose, squeeze)

**Key Classes:**
- `Tensor`: Core data structure
- `Storage`: Flat array management
- `Shape`: Shape validation and manipulation
- `Broadcasting`: Shape compatibility and indexing

**Dependencies:** Python standard library only (`math`, `random`, `itertools`)

### Autograd Engine (`src/from_zero/autograd/`)

**Responsibilities:**
- Computational graph construction
- Reverse-mode automatic differentiation
- Gradient accumulation for shared subexpressions
- Finite-difference gradient checking
- Graph visualization and serialization

**Key Classes:**
- `Value` / `Tensor`: Autograd-enabled tensor
- `Function`: Base class for differentiable operations
- `Graph`: Topological sort and traversal
- `GradCheck`: Finite-difference verification

**Dependencies:** Tensor Engine only

### Neural Network Engine (`src/from_zero/nn/`)

**Responsibilities:**
- Parameter management
- Module abstraction (layers, containers)
- Standard layers (Linear, Embedding, LayerNorm, etc.)
- Activation functions
- Weight initialization
- Loss functions
- Dropout and regularization

**Key Classes:**
- `Module`: Base class for all layers
- `Parameter`: Trainable tensor
- `Linear`, `Embedding`, `LayerNorm`, `Dropout`
- `Sequential`, `ModuleList`
- `CrossEntropyLoss`, `MSELoss`, `BCEWithLogitsLoss`

**Dependencies:** Tensor Engine + Autograd Engine

### Optimization Engine (`src/from_zero/optim/`)

**Responsibilities:**
- Gradient descent variants
- Learning rate scheduling
- State management (momentum buffers, etc.)
- Gradient clipping
- Checkpoint save/load

**Key Classes:**
- `Optimizer`: Base class
- `SGD`, `Momentum`, `AdaGrad`, `RMSProp`, `Adam`, `AdamW`
- `ConstantSchedule`, `StepDecay`, `ExponentialDecay`, `CosineDecay`

**Dependencies:** Tensor Engine + Autograd (for gradient access)

### Model Implementations (`src/from_zero/models/`)

**Responsibilities:**
- Logistic Regression
- Mini GPT (Transformer)
- Diffusion Model (DDPM)

**Dependencies:** All lower layers

### Training & Experiment Infrastructure (`src/from_zero/training/`, `src/from_zero/experiments/`)

**Responsibilities:**
- Training loops with validation
- Metrics computation
- Checkpointing
- Experiment registry
- Report generation

**Dependencies:** All model layers

### Application Layer (`app/`)

**Responsibilities:**
- Streamlit-based interactive UI
- Tensor Studio, Autograd Studio, Optimizer Lab
- Logistic Regression Lab
- Mini GPT Lab
- Diffusion Lab
- Experiment Browser

**Dependencies:** All layers via public APIs only

## Data Flow

### Training Pipeline

```
Data Loader → Model Forward → Loss → Autograd Backward → Optimizer Step
     │              │            │           │               │
     ▼              ▼            ▼           ▼               ▼
Dataset         Model         Loss Fn    Autograd       Optimizer
Split           Forward       Computation Backward       Update
```

### Inference Pipeline

```
Input → Tokenizer → Model Forward → Post-process → Output
```

### Experiment Flow

```
Config → Seed → Data → Model → Train → Metrics → Checkpoint → Report
              │
              └── RNG State (Python, NumPy, custom)
```

## Configuration System

### Hierarchy

```
Global Config (YAML)
    ├── Model Config
    ├── Optimizer Config
    ├── Data Config
    ├── Training Config
    └── Output Config
```

### Validation

Each config class validates:
- Required fields present
- Type correctness
- Value ranges
- Cross-field constraints

```python
@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float

    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
```

## Memory Management

### Tensor Lifecycle

```
Creation → Operations → Gradients → Backward → Release
    │                                    │
    └────────────────────────────────────┘
    (Automatic after backward by default)
```

### Gradient Accumulation

```python
def backward(self):
    # Topological order
    for node in reversed(topological_sort(self)):
        if node.grad_fn:
            parent_grads = node.grad_fn(node.grad)
            for parent, grad in zip(node.parents, parent_grads):
                if parent.requires_grad:
                    # Accumulate (not replace)
                    if parent.grad is None:
                        parent.grad = grad
                    else:
                        parent.grad = parent.grad + grad
```

### Graph Release

After backward pass, graph references are released to allow garbage collection:

```python
def backward(self, retain_graph=False):
    # ... compute gradients ...
    if not retain_graph:
        # Clear references
        for node in self._all_nodes():
            node.parents = ()
            node.grad_fn = None
```

## Error Handling

### Exception Hierarchy

```
FromZeroError
    ├── ConfigurationError
    ├── ShapeError
    ├── GradientError
    ├── NumericalError
    └── CheckpointError
```

### Validation Points

1. **Tensor creation**: Shape validation, data type
2. **Operations**: Shape compatibility, broadcasting
3. **Autograd**: Gradient finiteness, graph integrity
4. **Optimizers**: Parameter/gradient matching, state consistency
5. **Checkpoints**: Version compatibility, state completeness

## Testing Strategy

### Layer Isolation

Each layer tested independently:

```python
# Tensor tests don't need autograd
def test_tensor_matmul():
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[1, 0], [0, 1]])
    c = a @ b
    assert c.data == a.data


# Autograd tests use tensor primitives
def test_autograd_matmul():
    a = Value([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Value([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    c = a @ b
    c.backward()
    # Check gradients
```

### Integration Tests

```python
def test_full_pipeline():
    # Data → Model → Loss → Backward → Optimizer
    model = Linear(10, 5)
    optimizer = SGD(model.parameters(), lr=0.01)
    x = randn((32, 10))
    y = randn((32, 5))

    for _ in range(10):
        pred = model(x)
        loss = mse_loss(pred, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

    # Loss should decrease
    assert loss.data < initial_loss
```

## Deployment Considerations

### Serialization Safety

- **No pickle in core**: Use explicit dict serialization
- **Version tracking**: Every checkpoint has format version
- **Safe loading**: Validate structure before deserializing

### Reproducibility

```python
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    # Custom RNG states for each subsystem
    tensor_rng.seed(seed)
    autograd_rng.seed(seed)
    data_rng.seed(seed)
```

### Performance Monitoring

```python
# Built-in profiling
with profile("matmul"):
    C = A @ B

# Memory tracking
tracemalloc.start()
# ... operations ...
current, peak = tracemalloc.get_traced_memory()
```

## Extension Points

### Adding New Operations

1. Implement forward in `operations.py`
2. Implement backward in `autograd/functions.py`
3. Add gradient check in `tests/numerical/`
4. Document in `docs/mathematics/`

### Adding New Layers

1. Subclass `Module` in `nn/layers.py`
2. Implement `forward()`
3. Register parameters in `__init__`
4. Add tests in `tests/nn/`

### Adding New Models

1. Create model file in `models/`
2. Use existing NN building blocks
3. Implement training logic in `training/`
3. Add config class
4. Add UI page in `app/pages/`

## Summary

The architecture enforces:
- **Separation of concerns** through strict layering
- **Testability** through isolated layers
- **Extensibility** through clean interfaces
- **Correctness** through mathematical validation
- **Reproducibility** through explicit state management