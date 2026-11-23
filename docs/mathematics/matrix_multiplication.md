# Matrix Multiplication: Mathematics, Implementation, and Gradients

## Mathematical Definition

### Basic Matrix Multiplication

For matrices $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$:

$$C = AB \in \mathbb{R}^{m \times n}$$

where each element is:

$$C_{ij} = \sum_{r=0}^{k-1} A_{ir} B_{rj} \quad \text{for } i=0,\dots,m-1,\; j=0,\dots,n-1$$

### Properties

- **Associative**: $(AB)C = A(BC)$
- **Distributive**: $A(B+C) = AB + AC$ and $(A+B)C = AC + BC$
- **Not commutative**: $AB \neq BA$ in general
- **Identity**: $AI = IA = A$ where $I$ is the identity matrix
- **Transpose**: $(AB)^\top = B^\top A^\top$

## Batched Matrix Multiplication

### Definition

For batched tensors, the last two dimensions are matrix dimensions; preceding dimensions are batch dimensions:

```python
# Shapes
A: (..., m, k)
B: (..., k, n)
C = A @ B: (..., m, n)
```

Batch dimensions are broadcast according to standard broadcasting rules.

### Use Cases

- **Multi-head attention**: $(B, H, T, D) @ (B, H, D, T) \to (B, H, T, T)$
- **Batch processing**: $(B, M, K) @ (B, K, N) \to (B, M, N)$

## Implementation Strategies

### Naive Triple Loop

```python
def matmul_naive(a, b):
    """O(m*k*n) time, O(m*n) space."""
    m, k = a.shape[-2], a.shape[-1]
    k2, n = b.shape[-2], b.shape[-1]

    assert k == k2, "Inner dimensions must match"

    out = zeros((m, n))
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for r in range(k):
                acc += a[i, r] * b[r, j]
            out[i, j] = acc
    return out
```

### Optimized: Loop Reordering

Cache-friendly loop ordering:

```python
def matmul_optimized(a, b):
    """Better cache utilization."""
    m, k = a.shape[-2], a.shape[-1]
    k2, n = b.shape[-2], b.shape[-1]

    out = zeros((m, n))

    # Loop order: i, k, j (or j, k, i)
    # Accesses a[i,k] with stride 1, b[k,j] with stride 1
    for i in range(m):
        for r in range(k):
            a_ir = a[i, r]
            if a_ir == 0:
                continue
            b_r = b[r]  # Cache row of b
            for j in range(n):
                out[i, j] += a_ir * b_r[j]
    return out
```

### Tiled/Blocked Multiplication

For large matrices, block into submatrices to improve cache utilization:

```python
def matmul_blocked(a, b, block_size=32):
    """Blocked matrix multiplication for cache efficiency."""
    m, k = a.shape[-2], a.shape[-1]
    k2, n = b.shape[-2], b.shape[-1]

    out = zeros((m, n))

    for i_block in range(0, m, block_size):
        for j_block in range(0, n, block_size):
            for r_block in range(0, k, block_size):
                # Compute block of output
                for i in range(i_block, min(i_block + block_size, m)):
                    for r in range(r_block, min(r_block + block_size, k)):
                        a_ir = a[i, r]
                        if a_ir == 0:
                            continue
                        for j in range(j_block, min(j_block + block_size, n)):
                            out[i, j] += a_ir * b[r, j]
    return out
```

## Strided Implementation

### General Formula with Strides

For arbitrary strides (non-contiguous tensors):

```python
def matmul_strided(a, b):
    """Matrix multiplication with arbitrary strides."""
    # Get shapes and strides
    a_shape = a.shape[-2:]
    b_shape = b.shape[-2:]
    a_strides = a.strides[-2:]
    b_strides = b.strides[-2:]

    m, k = a_shape
    k2, n = b_shape

    out = zeros((m, n))

    for i in range(m):
        a_i_offset = i * a_strides[0]
        for j in range(n):
            acc = 0.0
            for r in range(k):
                a_val = a.data[a_i_offset + r * a_strides[1]]
                b_val = b.data[r * b_strides[0] + j * b_strides[1]]
                acc += a_val * b_val
            out[i, j] = acc

    return out
```

## Gradients (Backpropagation)

### Mathematical Derivation

Given $C = AB$ and scalar loss $L$:

We want $\frac{\partial L}{\partial A}$ and $\frac{\partial L}{\partial B}$ given $\frac{\partial L}{\partial C}$.

Using the chain rule and the fact that $C_{ij} = \sum_k A_{ik} B_{kj}$:

$$\frac{\partial L}{\partial A_{ik}} = \sum_j \frac{\partial L}{\partial C_{ij}} \frac{\partial C_{ij}}{\partial A_{ik}} = \sum_j \frac{\partial L}{\partial C_{ij}} B_{kj}$$

In matrix form:

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} B^\top$$

$$\frac{\partial L}{\partial B} = A^\top \frac{\partial L}{\partial C}$$

### Proof

$$
\frac{\partial C_{ij}}{\partial A_{pq}} = \frac{\partial}{\partial A_{pq}} \sum_k A_{ik} B_{kj} = B_{qj} \delta_{ip} = B_{qj} \text{ if } i=p \text{ else } 0
$$

$$
\frac{\partial L}{\partial A_{pq}} = \sum_{ij} \frac{\partial L}{\partial C_{ij}} \frac{\partial C_{ij}}{\partial A_{pq}} = \sum_j \frac{\partial L}{\partial C_{pj}} B_{qj} = \sum_j \frac{\partial L}{\partial C_{pj}} B_{jq}^\top = \left(\frac{\partial L}{\partial C} B^\top\right)_{pq}
$$

Similarly for $B$:

$$
\frac{\partial L}{\partial B_{pq}} = \sum_{ij} \frac{\partial L}{\partial C_{ij}} A_{ip} \delta_{jq} = \sum_i \frac{\partial L}{\partial C_{iq}} A_{ip} = \sum_i A_{ip} \frac{\partial L}{\partial C_{iq}} = \left(A^\top \frac{\partial L}{\partial C}\right)_{pq}
$$

## Batched Gradients

### Independent Batch Processing

For batched matmul $C_b = A_b @ B_b$:

$$\frac{\partial L}{\partial A_b} = \frac{\partial L}{\partial C_b} B_b^\top$$
$$\frac{\partial L}{\partial B_b} = A_b^\top \frac{\partial L}{\partial C_b}$$

Each batch is independent; gradients don't cross batch boundaries.

### Implementation

```python
def matmul_backward(grad_out, a, b):
    """Compute gradients for A @ B = C."""
    # grad_out: (..., m, n)
    # a: (..., m, k)
    # b: (..., k, n)

    # grad_a = grad_out @ b^T
    grad_a = batched_matmul(grad_out, b.transpose(-2, -1))

    # grad_b = a^T @ grad_out
    grad_b = batched_matmul(a.transpose(-2, -1), grad_out)

    return grad_a, grad_b
```

## Broadcasting in Batched Matmul

### Rules

Batch dimensions follow standard broadcasting:

```python
def matmul_broadcast_shapes(a_shape, b_shape):
    """Compute output shape for batched matmul with broadcasting."""
    a_batch = a_shape[:-2]
    b_batch = b_shape[:-2]

    # Broadcast batch dimensions
    batch_shape = broadcast_shape(a_batch, b_batch)

    m, k = a_shape[-2:]
    k2, n = b_shape[-2:]

    return batch_shape + (m, n)
```

### Gradient Broadcasting

If batch dimensions were broadcast, gradients must be reduced:

```python
def matmul_backward_broadcast(grad_out, a, b, a_shape, b_shape, out_shape):
    grad_a = batched_matmul(grad_out, b.transpose(-2, -1))
    grad_b = batched_matmul(a.transpose(-2, -1), grad_out)

    # Unbroadcast to original shapes
    grad_a = unbroadcast_gradient(grad_a, a_shape)
    grad_b = unbroadcast_gradient(grad_b, b_shape)

    return grad_a, grad_b
```

## Numerical Stability

### Overflow/Underflow

For large matrices, element values can grow large. Mitigations:

1. **Float64 accumulation**: Use double precision for the inner product
2. **Blocked computation**: Smaller intermediate values
3. **Scaling**: Scale inputs if values are extreme

```python
def matmul_stable(a, b):
    """Use float64 for accumulation."""
    # Convert to float64 for inner product
    # Result back to float32/float16 if needed
```

### Special Cases

- **Identity**: `A @ I = A`, gradients pass through
- **Zero matrix**: Gradients are zero
- **Orthogonal**: `Q @ Q.T = I`, gradients simplified

## Testing Strategy

### Forward Pass Tests

```python
def test_matmul_basic():
    A = Tensor([[1, 2], [3, 4]])
    B = Tensor([[5, 6], [7, 8]])
    C = A @ B
    assert C.data == [19, 22, 43, 50]


def test_matmul_batched():
    A = Tensor([[[1, 2], [3, 4]], [[1, 0], [0, 1]]])
    B = Tensor([[[5, 6], [7, 8]], [[2, 0], [0, 2]]])
    C = A @ B
    # Check each batch


def test_matmul_broadcast():
    # (2, 3, 4) @ (3, 4, 5) -> (2, 3, 5)
    A = Tensor.randn(2, 3, 4)
    B = Tensor.randn(3, 4, 5)
    C = A @ B
    assert C.shape == (2, 3, 5)
```

### Gradient Tests

```python
def test_matmul_gradient():
    A = Tensor([[1, 2], [3, 4]], requires_grad=True)
    B = Tensor([[5, 6], [7, 8]], requires_grad=True)
    C = A @ B
    L = C.sum()
    L.backward()

    # Check A.grad = 1 @ B^T
    expected_a = B.T @ torch.ones((2, 2))
    assert_allclose(A.grad, expected_a)

    # Check B.grad = A^T @ 1
    expected_b = A.T @ torch.ones((2, 2))
    assert_allclose(B.grad, expected_b)


def test_matmul_gradient_finite_diff():
    for m, k, n in [(2, 3, 4), (5, 2, 3), (3, 5, 3)]:
        A = Tensor.randn(m, k, requires_grad=True)
        B = Tensor.randn(k, n, requires_grad=True)
        C = A @ B
        L = C.sum()
        L.backward()

        # Finite difference check
        numeric_A = finite_diff(lambda x: (x @ B).sum(), A.data)
        numeric_B = finite_diff(lambda x: (A @ x).sum(), B.data)
        assert_allclose(A.grad, numeric_A, atol=1e-6)
        assert_allclose(B.grad, numeric_B, atol=1e-6)


def test_batched_matmul_gradient():
    A = Tensor.randn(2, 3, 4, requires_grad=True)
    B = Tensor.randn(2, 4, 5, requires_grad=True)
    C = A @ B
    L = C.sum()
    L.backward()
    # Check shapes
    assert A.grad.shape == (2, 3, 4)
    assert B.grad.shape == (2, 4, 5)
```

### Broadcasting Gradient Tests

```python
def test_matmul_broadcast_gradient():
    # A: (2, 3, 4), B: (4, 5) -> broadcast B to (2, 3, 4, 5)
    A = Tensor.randn(2, 3, 4, requires_grad=True)
    B = Tensor.randn(4, 5, requires_grad=True)
    C = A @ B  # Shape: (2, 3, 5)
    L = C.sum()
    L.backward()

    assert A.grad.shape == (2, 3, 4)
    assert B.grad.shape == (4, 5)
```

## Performance Comparison

### Complexity

| Algorithm | Time | Space | Best For |
|-----------|------|-------|----------|
| Naive | O(mkn) | O(mn) | Small matrices |
| Optimized loop | O(mkn) | O(mn) | General purpose |
| Blocked | O(mkn) | O(mn) | Large matrices |
| Strassen | O(n^2.81) | O(n²) | Very large |
| GPU (BLAS) | O(mkn) | O(mn) | Production |

### When to Use Each

- **Educational/Prototyping**: Naive/optimized loops
- **Small matrices (< 100)**: Optimized loops
- **Medium (100-1000)**: Blocked/Strassen
- **Large (> 1000)**: BLAS/GPU

## Usage in FROM ZERO

The tensor engine implements:
1. **Naive triple loop** for clarity and correctness
2. **Batched matmul** with broadcasting for attention
3. **Exact gradient formulas** verified against finite differences
4. **Shape validation** with clear error messages

```python
from from_zero.core import Tensor, matmul

A = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
B = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

C = matmul(A, B)  # or A @ B
loss = C.sum()
loss.backward()

print(A.grad)  # Gradient wrt A
print(B.grad)  # Gradient wrt B
```

## Limitations

- **No GPU acceleration**: Pure Python loops
- **No sparsity exploitation**: Dense only
- **No quantization**: Full precision only
- **Single-threaded**: No parallel execution
- **No fused kernels**: Separate operations

These are acceptable for educational purposes. Production systems should use optimized BLAS libraries.