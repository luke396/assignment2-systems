### Nsys Profile Analysis

>based on nsys reports with run tag `new_base`

#### (a) Forward Pass Timing

Forward Pass Avg Time (ms)

| Model  | seq128 | seq256 | seq512 | seq1024 |
| ------ | ------ | ------ | ------ | ------- |
| small  | 75.04  | 82.42  | 81.12  | 108.34  |
| medium | 93.19  | 98.42  | 117.96 | OOM     |
| large  | 113.67 | 128.50 | OOM    | OOM     |
| xl     | 141.89 | OOM    | OOM    | OOM     |
| 2.7B   | 154.74 | OOM    | OOM    | OOM     |

The time of nsys is a little slower than python version. A reasonable explanation is profiling overhead and run-to-run variance; the NVTX ranges are measured under the profiler and can be slower even if the benchmark uses torch.cuda.synchronize().

#### (b) Dominant Kernels (iteration 6)

small Top Kernel per Pass

| Seq Len | Inference Forward                    | Training Forward                     | Training Backward                    |
| ------- | ------------------------------------ | ------------------------------------ | ------------------------------------ |
| seq128  | void cutlass::Kernel2<cut... (58.2%) | void cutlass::Kernel2<cut... (58.3%) | void cutlass::Kernel2<cut... (17.0%) |
| seq256  | void cutlass::Kernel2<cut... (52.5%) | void cutlass::Kernel2<cut... (52.6%) | void cutlass::Kernel2<cut... (24.0%) |
| seq512  | void cutlass::Kernel2<cut... (72.1%) | void cutlass::Kernel2<cut... (71.0%) | N/A                                  |
| seq1024 | void cutlass::Kernel2<cut... (40.8%) | N/A                                  | void cutlass::Kernel2<cut... (15.2%) |

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

The dominant kernel is always GEMM for matrix multiplication. Forward's GEMM ratio is larger than backward's because backward has more non-GEMM kernels (e.g., elementwise ops for gradients). As seq_len increases, non-GEMM kernels (e.g., elementwise ops) become more prominent due to more activations and intermediate results.

#### (c) Non-GEMM Kernels in Forward Pass

small Top Non-GEMM Kernel per Sequence Length

| Seq Len | Top Non-GEMM Kernel                         | Time% | Total    | Inst |
| ------- | ------------------------------------------- | ----- | -------- | ---- |
| seq128  | void at::native::elementwise_kernel<(int... | 1.9%  | 1.74 ms  | 720  |
| seq256  | void at::native::elementwise_kernel<(int... | 2.3%  | 2.86 ms  | 720  |
| seq512  | void at::native::elementwise_kernel<(int... | 4.0%  | 9.28 ms  | 360  |
| seq1024 | void at::native::elementwise_kernel<(int... | 13.1% | 90.05 ms | 360  |

The most time-consuming non-GEMM kernels are element wise copy, as seq length increases, elementwise kernels (activations, copy, etc.) take a larger proportion of time due to more data being processed.

medium Top Non-GEMM Kernel per Sequence Length

| Seq Len | Top Non-GEMM Kernel                         | Time% | Total    | Inst |
| ------- | ------------------------------------------- | ----- | -------- | ---- |
| seq128  | void at::native::elementwise_kernel<(int... | 2.1%  | 4.24 ms  | 1440 |
| seq256  | void at::native::elementwise_kernel<(int... | 2.2%  | 7.19 ms  | 1440 |
| seq512  | void at::native::elementwise_kernel<(int... | 5.3%  | 37.36 ms | 720  |
| seq1024 | OOM                                         | -     | -        | -    |

The most time-consuming non-GEMM kernels are element wise copy, as seq length increases, elementwise kernels (activations, copy, etc.) take a larger proportion of time due to more data being processed.

large Top Non-GEMM Kernel per Sequence Length

| Seq Len | Top Non-GEMM Kernel                         | Time% | Total    | Inst |
| ------- | ------------------------------------------- | ----- | -------- | ---- |
| seq128  | void at::native::elementwise_kernel<(int... | 1.7%  | 7.41 ms  | 2160 |
| seq256  | void at::native::elementwise_kernel<(int... | 2.0%  | 14.63 ms | 2160 |
| seq512  | OOM                                         | -     | -        | -    |
| seq1024 | OOM                                         | -     | -        | -    |

The most time-consuming non-GEMM kernels are element wise copy, as seq length increases, elementwise kernels (activations, copy, etc.) take a larger proportion of time due to more data being processed.

xl Top Non-GEMM Kernel per Sequence Length

| Seq Len | Top Non-GEMM Kernel                         | Time% | Total    | Inst |
| ------- | ------------------------------------------- | ----- | -------- | ---- |
| seq128  | void at::native::elementwise_kernel<(int... | 1.5%  | 11.42 ms | 2880 |
| seq256  | OOM                                         | -     | -        | -    |
| seq512  | OOM                                         | -     | -        | -    |
| seq1024 | OOM                                         | -     | -        | -    |

The most time-consuming non-GEMM kernels are element wise copy, as seq length increases, elementwise kernels (activations, copy, etc.) take a larger proportion of time due to more data being processed.

2.7B Top Non-GEMM Kernel per Sequence Length

| Seq Len | Top Non-GEMM Kernel                         | Time% | Total    | Inst |
| ------- | ------------------------------------------- | ----- | -------- | ---- |
| seq128  | void at::native::vectorized_elementwise_... | 1.0%  | 11.76 ms | 960  |
| seq256  | OOM                                         | -     | -        | -    |
| seq512  | OOM                                         | -     | -        | -    |
| seq1024 | OOM                                         | -     | -        | -    |

The most time-consuming non-GEMM kernels are element wise copy, as seq length increases, elementwise kernels (activations, copy, etc.) take a larger proportion of time due to more data being processed.

#### (d) GEMM Fraction: Inference vs Training

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


#### (e) Softmax vs Matmul in Attention (iteration 6)

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


