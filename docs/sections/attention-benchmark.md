### Attention Benchmark Results

#### Forward Pass Timing (ms)

| seq_len \ d_model | 16    | 32    | 64    | 128   |
| ----------------- | ----- | ----- | ----- | ----- |
| 256               | 0.61  | 0.61  | 0.62  | 0.61  |
| 1024              | 0.85  | 0.87  | 0.91  | 0.98  |
| 4096              | 7.29  | 7.53  | 7.99  | 8.98  |
| 8192              | 26. 4 | 27.3  | 29.3  | 33.3  |
| 16384             | 105.4 | 109.5 | 117.2 | 132.1 |

#### Backward Pass Timing (ms)

| seq_len \ d_model | 16    | 32    | 64    | 128   |
| ----------------- | ----- | ----- | ----- | ----- |
| 256               | 1.03  | 0.97  | 0.96  | 0.97  |
| 1024              | 1.85  | 1.87  | 1.92  | 2.03  |
| 4096              | 19.8  | 20.0  | 20.5  | 21.6  |
| 8192              | 72.8  | 73.6  | 75.7  | 79.9  |
| 16384             | 290.6 | 294.2 | 301.9 | 317.1 |

#### Memory Before Backward (MB)

| seq_len \ d_model | 16      | 32      | 64      | 128     |
| ----------------- | ------- | ------- | ------- | ------- |
| 256               | 23.1    | 23.8    | 25.4    | 28.6    |
| 1024              | 116.3   | 119.3   | 125.4   | 137.6   |
| 4096              | 1580.5  | 1592.5  | 1616.6  | 1664.8  |
| 8192              | 6248.8  | 6272.8  | 6320.8  | 6417.0  |
| 16384             | 24897.3 | 24945.3 | 25041.3 | 25233.5 |

Memory scales as **O(n²)** with sequence length (4x memory for 2x seq_len).

#### OOM Analysis

All configurations ran successfully on an 80GB A800 GPU. Doubling seq_len to 32768 would require ~100GB (4x increase), exceeding the 80GB limit.

#### Memory Accounting

Taking **seq_len=16384, d_model=16, batch_size=8, num_heads=1** as an example:

| Component                          | Formula                 | Memory  |
| ---------------------------------- | ----------------------- | ------- |
| Input data                         | B × S × D × 4 bytes     | 8 MB    |
| Q, K, V matrices                   | 3 × B × S × D × 4 bytes | 24 MB   |
| Attention scores (Q @ K^T)         | B × H × S × S × 4 bytes | 8192 MB |
| Softmax intermediate (exp_x)       | B × H × S × S × 4 bytes | 8192 MB |
| Attention weights (softmax output) | B × H × S × S × 4 bytes | 8192 MB |
| Attention output                   | B × S × D × 4 bytes     | 8 MB    |

**Theoretical total**: ~24.6 GB | **Measured**: 24.9 GB ✓

#### Memory Optimization Analysis

The custom implementation stores 3 O(n²) tensors, but backward only needs **attention weights**:

- **Attention scores**: softmax backward only needs output, not input
- **exp_x**: unnecessary intermediate from custom softmax implementation

Softmax backward formula: `∂L/∂x_i = y_i × (∂L/∂y_i - Σ_j y_j × ∂L/∂y_j)` — only requires output `y`.

#### Memory Comparison

| Implementation                                                             | O(n²) Tensors | Memory (seq=16384) |
| -------------------------------------------------------------------------- | ------------- | ------------------ |
| Custom softmax (current)                                                   | 3             | ~24 GB             |
| PyTorch built-in softmax                                                   | 2             | ~16 GB             |
| Custom autograd (weights only)                                             | 1             | ~8 GB              |
| FlashAttention (`F.scaled_dot_product_attention(q, k, v, is_causal=True)`) | 0             | ~32 MB             |

#### Summary

Standard attention has **O(n²) time complexity** (computing n×n dot products) and **O(n²) memory complexity** (storing attention matrix). The custom implementation is computationally correct but stores unnecessary tensors for backward.

FlashAttention maintains O(n²) time complexity but reduces **memory complexity to O(n)** through two key techniques: **(1) tiling** — computing attention in small blocks that fit in fast SRAM without storing the full matrix, and **(2) recomputation** — recalculating attention weights during backward instead of storing them. This also improves speed by reducing slow HBM memory access.
