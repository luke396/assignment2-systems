# CS336 Spring 2025 Assignment 2: Systems

This repo uses a git submodule for the Assignment 1 basics implementation.

- The original assignment README is preserved in `README.original.md`.
- The `cs336-basics` directory is a submodule pointing to
  [assignment1-basics repository](https://github.com/luke396/assignment1-basics).

## Setup

After cloning, initialize the submodule:

```sh
git submodule update --init --recursive
```

Then follow the rest of the assignment instructions in `README.original.md`.

## Answering Questions

### benchmarking_script

[benchmark_basic_results](result/benchmark_results_rtx5090.md) is the output from a single RTX 5090 32G run; `nan` indicates OOM.

> The forward pass times including data generation, theroretically bigger than below nsys's only profile `model(data)`.

Using 5 warmup steps and 10 measured steps, at seq_len=512 the small/medium configs average 0.018s/0.050s forward and 0.061s/0.178s forward+backward (std <= 0.0038s), so the backward component is roughly 0.043s/0.128s. Large OOM at seq_len=512; at seq_len=256 large averages 0.056s forward and 0.208s forward+backward (std ~= 0.002s), while xl/2.7B only ran forward at seq_len=128 (0.065s/0.084s).

Without warm-up, variance is huge (e.g., small seq_len=512 has std 0.212s FWD / 0.121s FWD+BWD), but with 5 warmup steps it drops to 0.0038s / 0.0016s. With only 1 warmup step it still differs (0.0029s / 0.0020s at small seq_len=512), likely because GPU clocks, caches, and kernel autotuning/allocator state are still settling.

### nsys_profile

> When the `.nsysrep` file is generated under WSL2, opening it in the Windows UI can cause the `Stats System` view to fail. Copy the `.nsysrep` file to the Windows filesystem before opening it.

(a) Forward timing from NVTX

| Model  | seq128 | seq256 | seq512 | seq1024 |
| ------ | ------ | ------ | ------ | ------- |
| small  | 75.04  | 82.42  | 81.12  | 108.34  |
| medium | 93.19  | 98.42  | 117.96 | OOM     |
| large  | 113.67 | 128.50 | OOM    | OOM     |
| xl     | 141.89 | OOM    | OOM    | OOM     |
| 2.7B   | 154.74 | OOM    | OOM    | OOM     |

The time of nsys is a little slower than python version. A reasonable explanation is profiling overhead and run-to-run variance; the NVTX ranges are measured under the profiler and can be slower even if the benchmark uses `torch.cuda.synchronize()`.

(b) Dominant kernels

> iteration 6

small Top Kernel per Pass

| Seq Len | Inference Forward                    | Training Forward                     | Training Backward                    |
| ------- | ------------------------------------ | ------------------------------------ | ------------------------------------ |
| seq128  | void cutlass::Kernel2<cut... (58.2%) | void cutlass::Kernel2<cut... (58.3%) | void cutlass::Kernel2<cut... (17.0%) |
| seq256  | void cutlass::Kernel2<cut... (52.5%) | void cutlass::Kernel2<cut... (52.6%) | void cutlass::Kernel2<cut... (24.0%) |
| seq512  | void cutlass::Kernel2<cut... (72.1%) | void cutlass::Kernel2<cut... (71.0%) | void cutlass::Kernel2<cut... (25.6%) |
| seq1024 | void cutlass::Kernel2<cut... (40.8%) | void cutlass::Kernel2<cut... (41.3%) | void cutlass::Kernel2<cut... (15.2%) |

medium Top Kernel per Pass

| Seq Len | Inference Forward                    | Training Forward                     | Training Backward                    |
| ------- | ------------------------------------ | ------------------------------------ | ------------------------------------ |
| seq128  | void cutlass::Kernel2<cut... (48.6%) | void cutlass::Kernel2<cut... (49.2%) | void cutlass::Kernel2<cut... (29.4%) |
| seq256  | void cutlass::Kernel2<cut... (56.6%) | void cutlass::Kernel2<cut... (56.2%) | void cutlass::Kernel2<cut... (25.7%) |
| seq512  | void cutlass::Kernel2<cut... (50.5%) | void cutlass::Kernel2<cut... (48.8%) | void cutlass::Kernel2<cut... (26.9%) |
| seq1024 | OOM                                  | OOM                                  | OOM                                  |

large Top Kernel per Pass

| Seq Len | Inference Forward                    | Training Forward                     | Training Backward                    |
| ------- | ------------------------------------ | ------------------------------------ | ------------------------------------ |
| seq128  | void cutlass::Kernel2<cut... (56.1%) | void cutlass::Kernel2<cut... (55.5%) | void cutlass::Kernel2<cut... (29.6%) |
| seq256  | void cutlass::Kernel2<cut... (60.0%) | void cutlass::Kernel2<cut... (59.6%) | void cutlass::Kernel2<cut... (25.2%) |
| seq512  | OOM                                  | OOM                                  | OOM                                  |
| seq1024 | OOM                                  | OOM                                  | OOM                                  |

xl Top Kernel per Pass

| Seq Len | Inference Forward                    | Training Forward | Training Backward |
| ------- | ------------------------------------ | ---------------- | ----------------- |
| seq128  | void cutlass::Kernel2<cut... (60.4%) | OOM              | OOM               |
| seq256  | OOM                                  | OOM              | OOM               |
| seq512  | OOM                                  | OOM              | OOM               |
| seq1024 | OOM                                  | OOM              | OOM               |

2.7B Top Kernel per Pass

| Seq Len | Inference Forward                    | Training Forward | Training Backward |
| ------- | ------------------------------------ | ---------------- | ----------------- |
| seq128  | void cutlass::Kernel2<cut... (93.2%) | OOM              | OOM               |
| seq256  | OOM                                  | OOM              | OOM               |
| seq512  | OOM                                  | OOM              | OOM               |
| seq1024 | OOM                                  | OOM              | OOM               |

The domain kernal is always GEMM for matrix multiplication. Forward's GEMM radio is larger than backward's because backward has more non-GEMM kernels (e.g., elementwise ops for gradients). As seq_len increases, non-GEMM kernels (e.g., elementwise ops) become more prominent due to more activations and intermediate results.

(c) Non-GEMM kernels

small Top Non-GEMM Kernel per Sequence Length

| Seq Len | Top Non-GEMM Kernel                         | Time% | Total    | Inst |
| ------- | ------------------------------------------- | ----- | -------- | ---- |
| seq128  | void at::native::elementwise_kernel<(int... | 1.9%  | 1.74 ms  | 720  |
| seq256  | void at::native::elementwise_kernel<(int... | 2.3%  | 2.86 ms  | 720  |
| seq512  | void at::native::elementwise_kernel<(int... | 4.0%  | 9.28 ms  | 360  |
| seq1024 | void at::native::elementwise_kernel<(int... | 13.1% | 90.05 ms | 360  |

medium Top Non-GEMM Kernel per Sequence Length

| Seq Len | Top Non-GEMM Kernel                         | Time% | Total    | Inst |
| ------- | ------------------------------------------- | ----- | -------- | ---- |
| seq128  | void at::native::elementwise_kernel<(int... | 2.1%  | 4.24 ms  | 1440 |
| seq256  | void at::native::elementwise_kernel<(int... | 2.2%  | 7.19 ms  | 1440 |
| seq512  | void at::native::elementwise_kernel<(int... | 5.3%  | 37.36 ms | 720  |
| seq1024 | OOM                                         | -     | -        | -    |

large Top Non-GEMM Kernel per Sequence Length

| Seq Len | Top Non-GEMM Kernel                         | Time% | Total    | Inst |
| ------- | ------------------------------------------- | ----- | -------- | ---- |
| seq128  | void at::native::elementwise_kernel<(int... | 1.7%  | 7.41 ms  | 2160 |
| seq256  | void at::native::elementwise_kernel<(int... | 2.0%  | 14.63 ms | 2160 |
| seq512  | OOM                                         | -     | -        | -    |
| seq1024 | OOM                                         | -     | -        | -    |

xl Top Non-GEMM Kernel per Sequence Length

| Seq Len | Top Non-GEMM Kernel                         | Time% | Total    | Inst |
| ------- | ------------------------------------------- | ----- | -------- | ---- |
| seq128  | void at::native::elementwise_kernel<(int... | 1.5%  | 11.42 ms | 2880 |
| seq256  | OOM                                         | -     | -        | -    |
| seq512  | OOM                                         | -     | -        | -    |
| seq1024 | OOM                                         | -     | -        | -    |

2.7B Top Non-GEMM Kernel per Sequence Length

| Seq Len | Top Non-GEMM Kernel                         | Time% | Total    | Inst |
| ------- | ------------------------------------------- | ----- | -------- | ---- |
| seq128  | void at::native::vectorized*elementwise*... | 1.0%  | 11.76 ms | 960  |
| seq256  | OOM                                         | -     | -        | -    |
| seq512  | OOM                                         | -     | -        | -    |
| seq1024 | OOM                                         | -     | -        | -    |

The most time-consuming non-GEMM kernels are element wise copy, as seq length increases, elementwise kernels (activations, copy, etc.) take a larger proportion of time due to more data being processed.

(d) Top kernels by total GPU time

small GEMM Fraction

| Seq Len | Inference | Training |
| ------- | --------- | -------- |
| seq128  | 88.0%     | 59.3%    |
| seq256  | 84.6%     | 61.1%    |
| seq512  | 80.0%     | 53.5%    |
| seq1024 | 53.7%     | 31.0%    |

medium GEMM Fraction

| Seq Len | Inference | Training |
| ------- | --------- | -------- |
| seq128  | 87.0%     | 58.9%    |
| seq256  | 85.4%     | 63.5%    |
| seq512  | 77.1%     | 55.1%    |
| seq1024 | OOM       | OOM      |

large GEMM Fraction

| Seq Len | Inference | Training |
| ------- | --------- | -------- |
| seq128  | 89.6%     | 59.4%    |
| seq256  | 87.8%     | 64.7%    |
| seq512  | OOM       | OOM      |
| seq1024 | OOM       | OOM      |

xl GEMM Fraction

| Seq Len | Inference | Training |
| ------- | --------- | -------- |
| seq128  | 91.2%     | OOM      |
| seq256  | OOM       | OOM      |
| seq512  | OOM       | OOM      |
| seq1024 | OOM       | OOM      |

2.7B GEMM Fraction

| Seq Len | Inference | Training |
| ------- | --------- | -------- |
| seq128  | 94.4%     | OOM      |
| seq256  | OOM       | OOM      |
| seq512  | OOM       | OOM      |
| seq1024 | OOM       | OOM      |

Non-GEMM kernels become more significant as sequence length increases.

Training has a lower GEMM fraction than inference due to additional non-GEMM operations in the backward pass (e.g., elementwise ops for gradients).

As model size increases, GEMM fraction tends to increase because larger models have more compute-intensive matrix multiplications relative to non-GEMM ops.

(e) Attention sub-ranges (NVTX GPU projection)
Softmax Time (ms)
Softmax Time (ms)

| Model  | seq128 | seq256 | seq512 | seq1024 |
| ------ | ------ | ------ | ------ | ------- |
| small  | 0.70   | 0.75   | 1.20   | 11.95   |
| medium | 1.47   | 1.63   | 4.65   | OOM     |
| large  | 2.23   | 2.42   | OOM    | OOM     |
| xl     | 3.10   | OOM    | OOM    | OOM     |
| 2.7B   | 2.01   | OOM    | OOM    | OOM     |

Matmul Time (ms)

| Model  | seq128 | seq256 | seq512 | seq1024 |
| ------ | ------ | ------ | ------ | ------- |
| small  | 1.53   | 1.41   | 2.40   | 8.69    |
| medium | 3.07   | 3.20   | 6.30   | OOM     |
| large  | 3.60   | 4.75   | OOM    | OOM     |
| xl     | 5.81   | OOM    | OOM    | OOM     |
| 2.7B   | 3.67   | OOM    | OOM    | OOM     |

Softmax/Matmul Ratio

| Model  | seq128 | seq256 | seq512 | seq1024 |
| ------ | ------ | ------ | ------ | ------- |
| small  | 45.9%  | 52.9%  | 49.8%  | 137.6%  |
| medium | 48.0%  | 50.9%  | 73.9%  | OOM     |
| large  | 62.0%  | 50.9%  | OOM    | OOM     |
| xl     | 53.3%  | OOM    | OOM    | OOM     |
| 2.7B   | 55.0%  | OOM    | OOM    | OOM     |

Bigger seq length, softmax time increases faster than matmul time, leading to a higher softmax/matmul ratio. This is because softmax involves more elementwise operations and reductions that scale quadratically with sequence length, while matmul benefits from optimized GPU kernels.

For one head and one batch, softmax FLOPs per row is 5mn; across attention this is 5 x seq x seq. Computing attention scores is 2 x seq x head_dim x seq, and the final matmul is the same, so total matmul FLOPs is 4 x seq x head_dim x seq. The FLOPs ratio is (5 x seq x seq) : (4 x seq x head_dim x seq) = 5 : (4 x head_dim). In our medium config, head_dim = 64, so the ratio is 5 : 256, about 1.95%.

```shell
m x (n-1)  get row max
m x n      minus max
m x n      exp
m x (n-1)  get sum
m x n      divide
```

The time spent computing softmax is much higher than its FLOPs ratio, likely because softmax is elementwise and memory-bound (more memory traffic), while GEMM kernels are highly optimized and more compute-bound. A possible improvement is to use a fused kernel to avoid intermediate softmax stores/loads, trading a bit more compute for less memory access.

### mixed_precision_accumulation

```python
import torch

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(s)  # tensor(10.0001)

s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)  # tensor(9.9531, dtype=torch.float16)

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)  # tensor(10.0021)

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print(s)  # tensor(10.0021)
```

### benchmarking_mixed_precision

(a)

Auto cast with `torch.float16`:

```shell
Paramater's dtype in autocast: torch.float32
Output of fc1 dtype : torch.float16
Output of fc2 dtype : torch.float16
Output of relu dtype : torch.float16
Output of ln dtype : torch.float32
Model's logits dtype : torch.float16
Loss dtype : torch.float32
Gradient dtype of first layer weights: torch.float32
```

Auto cast with `torch.bfloat16`:

```shell
Paramater's dtype in autocast: torch.float32
Output of fc1 dtype : torch.bfloat16
Output of fc2 dtype : torch.bfloat16
Output of relu dtype : torch.bfloat16
Output of ln dtype : torch.float32
Model's logits dtype : torch.bfloat16
Loss dtype : torch.float32
Gradient dtype of first layer weights: torch.float32
```

(b)

The reason layerNorm remaing float32 is that layerNorm involves reductions (mean/stddev) which can lose precision in float16; PyTorch's autocast keeps such ops in float32 to maintain numerical stability.

Although BF16 has sufficient dynamic range (same 8 exponent bits as FP32), PyTorch autocast conservatively keeps LayerNorm in float32 for both FP16 and BF16. Theoretically BF16 could be used for LayerNorm, but the precision loss (7 mantissa bits vs 23) may affect training stability.

(c)

[analyze_mixed_precision.py](scripts/analyze_mixed_precision.py)

> Note: autocast only wraps the forward pass and loss computation; backward pass runs outside autocast (gradients are computed in the appropriate dtype automatically).

```shell
## Forward Pass Comparison (Median Time in ms)

| Model | Full Precision | BF16 Mixed | Speedup |
| --- | --- | --- | --- |
| small | 75.40 | 79.35 | 0.95x |
| medium | 95.23 | 103.34 | 0.92x |
| large | 120.83 | 123.42 | 0.98x |
| xl | 149.19 | 146.84 | 1.02x |
| 2.7B | 156.09 | 629.84 (OOM) | - |

## Forward+Backward Pass Comparison (Median Time in ms)

| Model | Full Precision | BF16 Mixed | Speedup |
| --- | --- | --- | --- |
| small | 96.08 | 105.35 | 0.91x |
| medium | 132.39 | 141.50 | 0.94x |
| large | 167.90 | 180.92 | 0.93x |
| xl | 817.06 | 925.99 | 0.88x |
| 2.7B | 809.29 | 855.10 | 0.95x |

## Attention Score Computation (Median Time in ms)

| Model | Full Precision | BF16 Mixed | Speedup |
| --- | --- | --- | --- |
| small | 0.30 | 0.23 | 1.34x |
| medium | 0.23 | 0.19 | 1.20x |
| large | 0.33 | 0.17 | 1.91x |
| xl | 0.53 | 0.18 | 2.89x |
| 2.7B | 1.67 | 0.50 | 3.31x |

## Conclusions

Top 5 kernels in BF16 autocast F+B (medium model):

| Time | Name |
| --- | --- |
| 12.1% | elementwise_kernel (direct_copy_kernel_cuda) - type conversion |
| 9.0% | vectorized_elementwise_kernel (add) |
| 8.2% | vectorized_elementwise_kernel (mul) |
| 4.9% | cutlass wmma_tensorop_bf16 gemm (nt) |
| 4.7% | cutlass wmma_tensorop_bf16 gemm (nn) |

1. BF16 mixed precision is slower (5-12%) on small-medium models
   due to autocast type conversion overhead (copy kernels dominate)
2. Attention score computation shows 1.2x-3.3x speedup with BF16
   larger models benefit more from reduced precision matmul
3. As model size increases, BF16 disadvantage decreases
   because compute time dominates conversion overhead
```

```

```
