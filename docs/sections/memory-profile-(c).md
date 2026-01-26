#### (c) Table of autocast peak memory usage

| Model Size | Sequence Length | Forward Only Peak Memory (GB) | Forward + Backward Peak Memory (GB) |
| --- | --- | --- | --- |
| 2.7B | 128 | 22.76 GB | 61.39 GB |
| 2.7B | 256 | 27.51 GB | 66.72 GB |
| 2.7B | 512 | 39.23 GB | N/A |

##### Comparison: Baseline vs Autocast

| Model Size | Seq Len | Baseline FWD | Autocast FWD | Δ FWD | Baseline FWD+BWD | Autocast FWD+BWD | Δ FWD+BWD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2.7B | 128 | 18.51 GB | 22.76 GB | +4.26 GB (+23.0%) | 57.18 GB | 61.39 GB | +4.21 GB (+7.4%) |
| 2.7B | 256 | 25.34 GB | 27.51 GB | +2.16 GB (+8.5%) | 66.09 GB | 66.72 GB | +0.62 GB (+0.9%) |
| 2.7B | 512 | 44.17 GB | 39.23 GB | -4.95 GB (-11.2%) | N/A | N/A | N/A |


Autocast increases memory usage in most cases, contrary to expectations.

Root Cause Analysis: The custom implementations in `cs336_basics/blocks.py`
prevent autocast from providing benefits:

1. **Custom `Linear` using `einsum`** (blocks.py:48-50)
   - `einsum` cannot trigger cuBLAS Tensor Core optimizations
   - PyTorch's `torch.matmul` or `F.linear` would automatically use
     Tensor Cores under autocast

2. **Custom `scaled_dot_product_attention`** (blocks.py:289-321)
   - Uses manual `einsum` for Q·K^T and attention·V computations
   - Does not use `F.scaled_dot_product_attention` which provides
     FlashAttention/Memory-Efficient Attention

3. **Custom `softmax`** (blocks.py:281-286)
   - Manual implementation without optimized CUDA kernels
   - `F.softmax` has fused CUDA implementations

4. **Type Conversion Overhead**
   - Since custom ops are not recognized as "autocast-eligible",
     frequent fp32 ↔ bf16 conversions occur
   - Each `einsum` call may trigger unnecessary dtype casts
   - Extra memory is allocated for intermediate tensors in different
     precisions

Expected vs Actual Flow:

```
Expected (with native PyTorch ops):
  Input(fp32) → auto-cast to bf16 → Tensor Core compute → Output(bf16)
  Memory: reduced by ~50% for activations

Actual (with custom einsum ops):
  Input(fp32) → cast bf16 → einsum(no Tensor Core) → cast fp32 → ...
  Memory: increased due to duplicate tensors in both precisions
```

Conclusion:

To benefit from autocast, the model should use PyTorch native operations:
- Replace `einsum` in `Linear` with `F.linear` or `x @ weight.T`
- Replace custom attention with `F.scaled_dot_product_attention`
- Replace custom `softmax` with `F.softmax`

However, since this is a course assignment (cs336), the custom
implementations are intentional for educational purposes.
