# Broadcasting: Semantics, Implementation, and Gradient Handling

## Introduction

Broadcasting is a mechanism that allows tensors with different shapes to participate in elementwise operations. It follows NumPy-style semantics and is essential for writing concise, efficient numerical code.

## Broadcasting Rules

### Definition

Two tensors are **broadcast-compatible** if, for each dimension starting from the rightmost (trailing) dimension:
1. The dimensions are equal, OR
2. One of the dimensions is 1

### Algorithm

```python
def broadcast_shapes(shape_a, shape_b):
    """Return the broadcasted shape or raise ValueError."""
    # Reverse for right-alignment
    a_rev = list(reversed(shape_a))
    b_rev = list(reversed(shape_b))

    # Pad shorter shape with 1s
    max_len = max(len(a_rev), len(b_rev))
    a_rev += [1] * (max_len - len(a_rev))
    b_rev += [1] * (max_len - len(b_rev))

    result_rev = []
    for a, b in zip(a_rev, b_rev):
        if a == b:
            result_rev.append(a)
        elif a == 1:
            result_rev.append(b)
        elif b == 1:
            result_rev.append(a)
        else:
            raise ValueError(f"Shapes {shape_a} and {shape_b} cannot be broadcast together")

    return tuple(reversed(result_rev))
```

### Examples

| Shape A | Shape B | Broadcast Result | Explanation |
|---------|---------|------------------|-------------|
| `(3, 4)` | `(4,)` | `(3, 4)` | B's dims prepended with 1s: `(1, 4)` |
| `(3, 1)` | `(1, 4)` | `(3, 4)` | Both dims broadcast |
| `(5, 1, 3)` | `(4, 3)` | `(5, 4, 3)` | A's middle dim broadcast |
| `(2, 3)` | `(2, 4)` | **Error** | Last dim 3 ≠ 4, neither is 1 |
| `(1,)` | `(3, 4)` | `(3, 4)` | Scalar broadcasts to matrix |

### Visual Example

```
Matrix (2, 3)  +  Vector (3,)
┌─────┐         ┌─────┐
│ 1 2 3│         │10 20 30│
│ 4 5 6│   +     └─────┘
└─────┘
   │           broadcast
   ▼
┌─────────┐
│ 11 22 33│
│ 14 25 36│
└─────────┘
```

## Implementation

### Virtual Replication

During the forward pass, broadcasting does **not** actually replicate data. Instead:

1. Compute output shape
2. Create output tensor with correct shape
3. Use **virtual indexing** to map output indices to input indices

```python
def broadcast_index(output_idx, original_shape, broadcast_shape):
    """Map output index to original tensor index."""
    # Align from right
    orig_rev = list(reversed(original_shape))
    out_rev = list(reversed(broadcast_shape))

    idx_rev = list(reversed(output_idx))

    result_rev = []
    for i in range(len(out_rev)):
        if orig_rev[i] == 1:
            # This dimension was broadcast, use index 0
            result_rev.append(0)
        else:
            result_rev.append(idx_rev[i])

    return tuple(reversed(result_rev))
```

### Forward Pass Implementation

```python
def broadcast_elementwise(a, b, op):
    """Elementwise operation with broadcasting."""
    out_shape = broadcast_shape(a.shape, b.shape)
    out_data = []

    # Iterate over output shape
    for flat_idx in range(prod(out_shape)):
        out_idx = unflatten_index(flat_idx, out_shape)

        # Map to a's indices
        a_idx = broadcast_index(out_idx, a.shape, out_shape)
        a_val = a.data[flatten_index(a_idx, a.shape)]

        # Map to b's indices
        b_idx = broadcast_index(out_idx, b.shape, out_shape)
        b_val = b.data[flatten_index(b_idx, b.shape)]

        out_data.append(op(a_val, b_val))

    return Tensor(out_data, out_shape)
```

## Gradient Handling: Unbroadcasting

### The Challenge

When a tensor is broadcast during the forward pass, its gradient during backpropagation must be **reduced** back to the original shape.

### Key Principle

If a dimension was broadcast (size 1 → size N), the gradient along that dimension must be **summed**.

### Unbroadcasting Algorithm

```python
def unbroadcast_gradient(gradient, original_shape):
    """
    Reduce gradient from broadcast shape back to original shape.

    Args:
        gradient: Tensor with broadcast shape
        original_shape: Tuple of ints

    Returns:
        Tensor with original_shape
    """
    grad = gradient
    grad_shape = grad.shape
    orig_shape = original_shape

    # 1. Reduce leading dimensions (if gradient has more dims)
    while len(grad_shape) > len(orig_shape):
        grad = grad.sum(axis=0)
        grad_shape = grad.shape

    # 2. Sum over dimensions that were broadcast (size 1 → size > 1)
    for axis in range(len(grad_shape)):
        if grad_shape[axis] != orig_shape[axis] and orig_shape[axis] == 1:
            grad = grad.sum(axis=axis, keepdims=True)

    # 3. Reshape to exact original shape
    return grad.reshape(orig_shape)
```

### Examples

| Original Shape | Broadcast Shape | Gradient Reduction |
|----------------|-----------------|-------------------|
| `(3,)` | `(2, 3)` | Sum over axis 0 |
| `(1, 4)` | `(3, 4)` | Sum over axis 0 |
| `(1, 1)` | `(3, 4)` | Sum over axes 0, 1 |
| `(5, 1, 3)` | `(5, 4, 3)` | Sum over axis 1 |

### Mathematical Derivation

For operation `z = x + y` where `x` has shape `(3, 1)` and `y` has shape `(1, 4)`:

Forward: `zᵢⱼ = xᵢ + yⱼ` (broadcast to `(3, 4)`)

Loss `L` depends on `z`. Chain rule:
```
∂L/∂xᵢ = Σⱼ ∂L/∂zᵢⱼ × ∂zᵢⱼ/∂xᵢ = Σⱼ ∂L/∂zᵢⱼ
∂L/∂yⱼ = Σᵢ ∂L/∂zᵢⱼ × ∂zᵢⱼ/∂yⱼ = Σᵢ ∂L/∂zᵢⱼ
```

The gradients are summed over the broadcast dimensions!

## Implementation in Autograd

### Gradient Function for Addition with Broadcasting

```python
def add_grad_fn(grad):
    # grad has broadcast shape
    # Return gradients for both parents
    return (unbroadcast_gradient(grad, self.shape), unbroadcast_gradient(grad, other.shape))
```

### Generic Pattern

```python
def make_broadcast_grad(grad_fn):
    """Wrap a gradient function to handle broadcasting."""

    def wrapped(grad):
        # grad has broadcast shape
        parent_grads = grad_fn(grad)
        return tuple(unbroadcast_gradient(g, p.shape) for g, p in zip(parent_grads, self.parents))

    return wrapped
```

## Edge Cases

### 1. Scalar Broadcasting

A scalar (shape `()`) broadcasts to any shape. Its gradient is the sum of all elements in the upstream gradient.

```python
x = Tensor(2.0)  # shape ()
y = Tensor([1, 2, 3])  # shape (3,)
z = x + y  # shape (3,)

# dz/dx = sum(dz/dxᵢ) = 3 * 1 = 3
```

### 2. Multiple Broadcast Dimensions

```python
x = Tensor([[1]])  # shape (1, 1)
y = Tensor([[1, 2], [3, 4]])  # shape (2, 2)
z = x + y  # shape (2, 2)

# dz/dx = sum over all 4 elements
```

### 3. Non-Contiguous Broadcasting

Broadcasting may create non-contiguous virtual views. The unbroadcast algorithm handles this correctly by working on the logical shape.

## Testing Strategy

### Unit Tests

```python
def test_broadcast_shapes():
    assert broadcast_shape((3, 4), (4,)) == (3, 4)
    assert broadcast_shape((3, 1), (1, 4)) == (3, 4)
    with pytest.raises(ValueError):
        broadcast_shape((2, 3), (2, 4))


def test_broadcast_forward():
    a = Tensor([1, 2, 3])  # (3,)
    b = Tensor([[1], [2]])  # (2, 1)
    c = a + b  # (2, 3)
    assert c.shape == (2, 3)
    assert c.data == [2, 3, 4, 3, 4, 5]


def test_unbroadcast_gradient():
    # x: (3,), y: (1, 4) -> z: (3, 4)
    grad_z = Tensor([[1] * 4] * 3)  # all ones
    grad_x = unbroadcast_gradient(grad_z, (3,))
    grad_y = unbroadcast_gradient(grad_z, (1, 4))
    assert grad_x.shape == (3,)
    assert grad_y.shape == (1, 4)
    # Each element of x contributes to 4 outputs -> gradient = 4
    assert all(v == 4 for v in grad_x.data)
    # Each element of y contributes to 3 outputs -> gradient = 3
    assert all(v == 3 for v in grad_y.data)
```

### Finite-Difference Verification

```python
def test_broadcast_gradients_finite_diff():
    for shape_a, shape_b in [
        ((3,), (1,)),
        ((3, 1), (1, 4)),
        ((1, 4), (3, 1)),
        ((2, 1, 3), (2, 4, 1)),
    ]:
        x = Tensor.randn(shape_a, requires_grad=True)
        y = Tensor.randn(shape_b, requires_grad=True)
        z = (x + y).sum()  # Scalar output
        z.backward()

        # Check with finite differences
        numeric_x = finite_diff_gradient(lambda x: (Tensor(x) + y).sum(), x.data)
        assert_allclose(x.grad, numeric_x, atol=1e-6)
```

## Performance Considerations

### Virtual vs Materialized Broadcasting

| Approach | Memory | Speed | Complexity |
|----------|--------|-------|------------|
| Virtual (index mapping) | Low | Slower | Higher |
| Materialized (replication) | High | Faster | Lower |

The current implementation uses **virtual indexing** to minimize memory usage, at the cost of some computation overhead during indexing.

### Optimization Opportunities

1. **Contiguous fast path**: If one tensor is contiguous and the other is broadcast, use optimized loops
2. **Kernel fusion**: Combine broadcast + op + reduction
3. **Cache strides**: Precompute mapping tables for repeated broadcasts

## Summary

Broadcasting enables expressive, concise numerical code by automatically aligning tensor shapes. The key implementation challenges are:

1. **Shape compatibility checking** - Right-aligned dimension matching
2. **Forward pass** - Virtual indexing without materialization
3. **Backward pass** - Gradient reduction (unbroadcasting) via summation
4. **Edge cases** - Scalars, multiple broadcast dims, non-contiguous views

The gradient reduction rule is mathematically derived from the chain rule and is the most critical component for correct autograd integration.