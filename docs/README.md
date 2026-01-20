# CS336 Spring 2025 Assignment 2: Systems - Analysis Report

## Profiling

### Basic benchmarking Results

>based on `benchmark_results__5090.md`

| Config | Seq Len | Warmup Steps | d_model |  d_ff | n_layers | num_heads | Status FWD | Status FWD+BWD | Total(s) FWD | Total(s) FWD+BWD | Warmup(s) FWD | Warmup(s) FWD+BWD | Avg/Step(s) FWD | Avg/Step(s) FWD+BWD | Std/Step(s) FWD | Std/Step(s) FWD+BWD |
| :----- | ------: | -----------: | ------: | ----: | -------: | --------: | :--------- | :------------- | -----------: | ---------------: | ------------: | ----------------: | --------------: | ------------------: | --------------: | ------------------: |
| small  |     128 |            0 |     768 |  3072 |       12 |        12 | OK         | OK             |       1.0023 |           1.2453 |             0 |                 0 |          0.0668 |               0.083 |        0.216775 |            0.124395 |
| small  |     128 |            1 |     768 |  3072 |       12 |        12 | OK         | OK             |        0.975 |            1.248 |        0.8252 |            0.5291 |          0.0107 |              0.0513 |         0.00181 |             0.00799 |
| small  |     128 |            5 |     768 |  3072 |       12 |        12 | OK         | OK             |       0.9851 |           1.2351 |        0.8761 |            0.6769 |          0.0109 |              0.0558 |        0.002056 |            0.004872 |
| small  |     256 |            0 |     768 |  3072 |       12 |        12 | OK         | OK             |       1.0414 |           1.2672 |             0 |                 0 |          0.0694 |              0.0845 |        0.215732 |            0.127523 |
| small  |     256 |            1 |     768 |  3072 |       12 |        12 | OK         | OK             |       1.0085 |           1.2514 |         0.825 |            0.5286 |          0.0131 |              0.0516 |        0.003709 |            0.006786 |
| small  |     256 |            5 |     768 |  3072 |       12 |        12 | OK         | OK             |       1.0072 |           1.3504 |        0.8887 |            0.7437 |          0.0119 |              0.0607 |         3.4e-05 |             0.00593 |
| small  |     512 |            0 |     768 |  3072 |       12 |        12 | OK         | OK             |       1.1365 |            1.392 |             0 |                 0 |          0.0758 |              0.0928 |        0.212369 |            0.121347 |
| small  |     512 |            1 |     768 |  3072 |       12 |        12 | OK         | OK             |       1.0731 |           1.3859 |        0.8273 |            0.5344 |          0.0176 |              0.0608 |        0.002892 |            0.002008 |
| small  |     512 |            5 |     768 |  3072 |       12 |        12 | OK         | OK             |        1.101 |           1.3767 |        0.9204 |            0.7638 |          0.0181 |              0.0613 |        0.003814 |            0.001623 |
| small  |    1024 |            0 |     768 |  3072 |       12 |        12 | OK         | OK             |        1.509 |             2.84 |             0 |                 0 |          0.1006 |              0.1893 |        0.205508 |            0.120335 |
| small  |    1024 |            1 |     768 |  3072 |       12 |        12 | OK         | OK             |       1.5142 |           2.8371 |        0.8497 |            0.6225 |          0.0475 |              0.1582 |        0.000623 |            0.000174 |
| small  |    1024 |            5 |     768 |  3072 |       12 |        12 | OK         | OK             |       1.5418 |           2.8312 |        1.0676 |             1.248 |          0.0474 |              0.1583 |        0.000495 |            0.000485 |
| medium |     128 |            0 |    1024 |  4096 |       24 |        16 | OK         | OK             |       1.1948 |           1.9575 |             0 |                 0 |          0.0796 |              0.1305 |        0.221662 |            0.128232 |
| medium |     128 |            1 |    1024 |  4096 |       24 |        16 | OK         | OK             |        1.195 |           1.9319 |        0.8836 |            0.5972 |          0.0222 |              0.0953 |        0.003978 |            0.012665 |
| medium |     128 |            5 |    1024 |  4096 |       24 |        16 | OK         | OK             |       1.2178 |            1.949 |        0.9998 |            0.9369 |          0.0218 |              0.1012 |        0.001429 |             0.00528 |
| medium |     256 |            0 |    1024 |  4096 |       24 |        16 | OK         | OK             |       1.3428 |           2.0675 |             0 |                 0 |          0.0895 |              0.1378 |        0.229779 |            0.126324 |
| medium |     256 |            1 |    1024 |  4096 |       24 |        16 | OK         | OK             |       1.3478 |           2.0783 |        0.9265 |             0.604 |          0.0301 |              0.1053 |        0.006274 |            0.005216 |
| medium |     256 |            5 |    1024 |  4096 |       24 |        16 | OK         | OK             |       1.3606 |            2.112 |        1.0808 |            1.0427 |           0.028 |              0.1069 |        0.001225 |            0.004758 |
| medium |     512 |            0 |    1024 |  4096 |       24 |        16 | OK         | OK             |       1.6291 |           3.1863 |             0 |                 0 |          0.1086 |              0.2124 |        0.225688 |            0.127486 |
| medium |     512 |            1 |    1024 |  4096 |       24 |        16 | OK         | OK             |       1.6152 |           3.1583 |        0.9113 |            0.6598 |          0.0503 |              0.1785 |        0.003004 |            0.002339 |
| medium |     512 |            5 |    1024 |  4096 |       24 |        16 | OK         | OK             |       1.6232 |           3.1381 |        1.1262 |            1.3567 |          0.0497 |              0.1781 |        0.000231 |            0.000944 |
| medium |    1024 |            0 |    1024 |  4096 |       24 |        16 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| medium |    1024 |            1 |    1024 |  4096 |       24 |        16 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| medium |    1024 |            5 |    1024 |  4096 |       24 |        16 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| large  |     128 |            0 |    1280 |  5120 |       36 |        20 | OK         | OK             |       1.5116 |            2.993 |             0 |                 0 |          0.1008 |              0.1995 |        0.229969 |            0.144448 |
| large  |     128 |            1 |    1280 |  5120 |       36 |        20 | OK         | OK             |       1.4998 |           2.9996 |        0.9195 |            0.7097 |          0.0414 |              0.1636 |        0.007115 |            0.005697 |
| large  |     128 |            5 |    1280 |  5120 |       36 |        20 | OK         | OK             |       1.5041 |           2.9977 |        1.1123 |            1.3507 |          0.0392 |              0.1647 |        0.000193 |               0.001 |
| large  |     256 |            0 |    1280 |  5120 |       36 |        20 | OK         | OK             |       1.7686 |           3.6186 |             0 |                 0 |          0.1179 |              0.2412 |        0.231367 |            0.133437 |
| large  |     256 |            1 |    1280 |  5120 |       36 |        20 | OK         | OK             |       1.7756 |           3.6202 |        0.9612 |            0.7216 |          0.0582 |               0.207 |        0.009429 |            0.001389 |
| large  |     256 |            5 |    1280 |  5120 |       36 |        20 | OK         | OK             |       1.7699 |           3.6449 |        1.2078 |            1.5692 |          0.0562 |              0.2076 |        0.001835 |            0.001967 |
| large  |     512 |            0 |    1280 |  5120 |       36 |        20 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| large  |     512 |            1 |    1280 |  5120 |       36 |        20 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| large  |     512 |            5 |    1280 |  5120 |       36 |        20 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| large  |    1024 |            0 |    1280 |  5120 |       36 |        20 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| large  |    1024 |            1 |    1280 |  5120 |       36 |        20 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| large  |    1024 |            5 |    1280 |  5120 |       36 |        20 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| xl     |     128 |            0 |    1600 |  6400 |       48 |        25 | OK         | OOM            |       1.9182 |              nan |             0 |               nan |          0.1279 |                 nan |        0.231533 |                 nan |
| xl     |     128 |            1 |    1600 |  6400 |       48 |        25 | OK         | OOM            |       1.9049 |              nan |        0.9652 |               nan |          0.0671 |                 nan |        0.008872 |                 nan |
| xl     |     128 |            5 |    1600 |  6400 |       48 |        25 | OK         | OOM            |       1.9103 |              nan |        1.2565 |               nan |          0.0654 |                 nan |        0.001968 |                 nan |
| xl     |     256 |            0 |    1600 |  6400 |       48 |        25 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| xl     |     256 |            1 |    1600 |  6400 |       48 |        25 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| xl     |     256 |            5 |    1600 |  6400 |       48 |        25 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| xl     |     512 |            0 |    1600 |  6400 |       48 |        25 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| xl     |     512 |            1 |    1600 |  6400 |       48 |        25 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| xl     |     512 |            5 |    1600 |  6400 |       48 |        25 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| xl     |    1024 |            0 |    1600 |  6400 |       48 |        25 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| xl     |    1024 |            1 |    1600 |  6400 |       48 |        25 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| xl     |    1024 |            5 |    1600 |  6400 |       48 |        25 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| 2.7B   |     128 |            0 |    2560 | 10240 |       32 |        32 | OK         | OOM            |       2.1486 |              nan |             0 |               nan |          0.1432 |                 nan |         0.22732 |                 nan |
| 2.7B   |     128 |            1 |    2560 | 10240 |       32 |        32 | OK         | OOM            |        2.136 |              nan |         0.955 |               nan |          0.0844 |                 nan |        0.002824 |                 nan |
| 2.7B   |     128 |            5 |    2560 | 10240 |       32 |        32 | OK         | OOM            |       2.1625 |              nan |        1.3247 |               nan |          0.0838 |                 nan |        0.000205 |                 nan |
| 2.7B   |     256 |            0 |    2560 | 10240 |       32 |        32 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| 2.7B   |     256 |            1 |    2560 | 10240 |       32 |        32 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| 2.7B   |     256 |            5 |    2560 | 10240 |       32 |        32 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| 2.7B   |     512 |            0 |    2560 | 10240 |       32 |        32 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| 2.7B   |     512 |            1 |    2560 | 10240 |       32 |        32 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| 2.7B   |     512 |            5 |    2560 | 10240 |       32 |        32 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| 2.7B   |    1024 |            0 |    2560 | 10240 |       32 |        32 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| 2.7B   |    1024 |            1 |    2560 | 10240 |       32 |        32 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |
| 2.7B   |    1024 |            5 |    2560 | 10240 |       32 |        32 | OOM        | OOM            |          nan |              nan |           nan |               nan |             nan |                 nan |             nan |                 nan |

#### Analysis

The table above shows benchmark results for different model configurations across various sequence lengths. Results include both forward-only (inference) and forward+backward (training) passes. Time is recorded and measured with python's time module.

#### Warm up 

The first step is cost longer time than the rest. With warm up, the time of train or inference are stable.

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



### Mix precision

#### Mixed_precision_accumulation

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

#### Parameters and outputs dtypes under different autocast types

Auto cast with torch.float16:

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

Auto cast with torch.bfloat16:

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

#### LayerNorm

The reason layerNorm remaing float32 is that layerNorm involves reductions (mean/stddev) which can lose precision in float16; PyTorch's autocast keeps such ops in float32 to maintain numerical stability.

Although BF16 has sufficient dynamic range (same 8 exponent bits as FP32), PyTorch autocast conservatively keeps LayerNorm in float32 for both FP16 and BF16. Theoretically BF16 could be used for LayerNorm, but the precision loss (7 mantissa bits vs 23) may affect training stability.


#### Mixed Precision Analysis

##### Forward Pass Comparison (Median Time in ms)

| Model | Full Precision | BF16 Mixed | Speedup |
| --- | --- | --- | --- |
| small | 75.89 | 79.35 | 0.96x |
| medium | 94.93 | 103.34 | 0.92x |
| large | 115.40 | 123.42 | 0.94x |
| xl | 144.16 | 146.84 | 0.98x |
| 2.7B | 155.02 | 629.84 | 0.25x |

##### Forward+Backward Pass Comparison (Median Time in ms)

| Model | Full Precision | BF16 Mixed | Speedup |
| --- | --- | --- | --- |
| small | 85.71 | 105.35 | 0.81x |
| medium | 119.69 | 141.50 | 0.85x |
| large | 156.81 | 180.92 | 0.87x |

##### Attention Score Computation (Median Time in ms)

| Model | Full Precision | BF16 Mixed | Speedup |
| --- | --- | --- | --- |
| small | 0.31 | 0.23 | 1.35x |
| medium | 0.24 | 0.19 | 1.27x |
| large | 0.35 | 0.17 | 2.02x |
| xl | 0.56 | 0.18 | 3.07x |
| 2.7B | 1.68 | 0.50 | 3.34x |

##### Conclusions

1. BF16 mixed precision is slower (5-12%) on small-medium models
   due to autocast type conversion overhead (copy kernels dominate)
2. Attention score computation shows 1.2x-3.3x speedup with BF16
   larger models benefit more from reduced precision matmul
3. As model size increases, BF16 disadvantage decreases
   because compute time dominates conversion overhead
