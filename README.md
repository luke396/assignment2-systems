# CS336 Spring 2025 Assignment 2: Systems

This repo uses a git submodule for the Assignment 1 basics implementation.

- The original assignment README is preserved in `README.original.md`.
- The `cs336-basics` directory is a submodule pointing to
  https://github.com/luke396/assignment1-basics.

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

Using the medium config with seq_len=128 and 5 warmup steps as an example, [benchmark_nsys_profile_results](result/benchmark_results_nsys.md) shows total F+B cost 2.3811s and warmup F+B cost 1.1835s. That implies a steady-state F+B cost of about 1.1976s over 10 steps, or 0.1198s/step. (source: `result/benchmark_results_nsys.md`)

In benchmark_basic_results, the same config shows total F+B cost 1.949s and warmup F+B cost 0.9369s, so the steady average is 1.0121s over 10 steps, or 0.1012s/step. (source: `result/benchmark_results_rtx5090.md`)

The nsys run is about 18% slower. This can be profiling overhead and/or run-to-run variance.

(a) Forward timing from NVTX

From the nsys profile (medium + seq_len=128 + 5 warmup), the total forward time based on NVTX is 761.613ms across 15 instances. The maximum is 366.601ms at the first step; the remaining steps are stable. Excluding that first outlier, the average forward time is (761.613 - 366.601) / 14 = 28ms. (source: `output/nsys/benchmark__20260113_105512__medium__FWD_BWD__seq128__warm5.nsys-rep`, `nsys stats --report nvtx_sum`)

Comparing to the benchmark forward-only result for the same config (which may differ slightly because it includes data generation), the average forward time is 21.8ms, which is lower than the nsys result. A reasonable explanation is profiling overhead and run-to-run variance; the NVTX ranges are measured under the profiler and can be slower even if the benchmark uses `torch.cuda.synchronize()`.

(b) Dominant kernels

During the full F+B process, `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nt_align1>(T1::Params)` totals 113.704ms across 2535 instances (12.1%). (source: `output/nsys/benchmark__20260113_105512__medium__FWD_BWD__seq128__warm5.nsys-rep`, `nsys stats --report cuda_gpu_kern_sum`)

Within a single forward pass, `cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_tn_align1>(T1::Params)` has 120 instances totaling 6.408ms (48.8%), the largest cumulative GPU time in forward. (source: `output/nsys/benchmark__20260113_105512__medium__FWD_BWD__seq128__warm5.nsys-rep`, `nsys stats --report cuda_gpu_kern_sum --filter-nvtx forward`)

Within a single backward pass, `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nt_align1>(T1::Params)` totals 7.570ms (28.9%) across 169 instances; this is the same kernel that dominates the full process. (source: `output/nsys/benchmark__20260113_105512__medium__FWD_BWD__seq128__warm5.nsys-rep`, `nsys stats --report cuda_gpu_kern_sum --filter-nvtx backward`)

(c) Non-GEMM kernels

The kernels above are GEMM (matrix multiplication) variants. In the forward pass, the first non-GEMM kernel is `at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)`, totaling 277 μs (2.1%) across 96 instances; this is an elementwise copy kernel. (source: `output/nsys/benchmark__20260113_105512__medium__FWD_BWD__seq128__warm5.nsys-rep`, `nsys stats --report cuda_gpu_kern_sum --filter-nvtx forward`)

In the backward pass, the first non-GEMM kernel is the same `elementwise_kernel`, totaling 4.342ms (16.6%) across 265 instances. (source: `output/nsys/benchmark__20260113_105512__medium__FWD_BWD__seq128__warm5.nsys-rep`, `nsys stats --report cuda_gpu_kern_sum --filter-nvtx backward`)

(d) Top kernels by total GPU time

Inference (forward only), top 5. The top 2 (GEMM) account for 83% of total GPU time. (source: `output/nsys/benchmark__forward_only__medium__FWD__seq128__warm5.nsys-rep`, `nsys stats --report cuda_gpu_kern_sum`)

| Time  | Total Time | Instances | Avg       | Med       | Min       | Max        | StdDev    | Name                                                                                                                                                                                                                                                                                                                                                               |
| ----- | ---------- | --------- | --------- | --------- | --------- | ---------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 48.7% | 96.056 ms  | 1800      | 53.364 μs | 39.584 μs | 38.880 μs | 113.343 μs | 27.704 μs | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_tn_align1>(T1::Params)                                                                                                                                                                                                                                                                                     |
| 34.6% | 68.171 ms  | 720       | 94.682 μs | 94.048 μs | 92.640 μs | 102.400 μs | 1.385 μs  | void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_tn_align1>(T1::Params)                                                                                                                                                                                                                                                                                     |
| 2.1%  | 4.139 ms   | 1440      | 2.874 μs  | 1.984 μs  | 1.824 μs  | 5.888 μs   | 1.639 μs  | void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3) |
| 1.8%  | 3.602 ms   | 720       | 5.002 μs  | 4.992 μs  | 4.640 μs  | 5.568 μs   | 108 ns    | void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char \*, (unsigned long)3>>(int, T2, T3)                                                                                                                                                         |
| 1.5%  | 2.927 ms   | 1470      | 1.991 μs  | 2.048 μs  | 1.760 μs  | 2.624 μs   | 184 ns    | void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)                                                                                          |

Training (forward + backward), top 5. GEMM kernels (rows 1-3 and 5) account for 41% of total GPU time. (source: `output/nsys/benchmark__20260113_105512__medium__FWD_BWD__seq128__warm5.nsys-rep`, `nsys stats --report cuda_gpu_kern_sum`)

| Time  | Total Time | Instances | Avg       | Med       | Min       | Max        | StdDev    | Name                                                                                                                                                                                                                                                                                                                                                               |
| ----- | ---------- | --------- | --------- | --------- | --------- | ---------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 12.1% | 113.704 ms | 2535      | 44.853 μs | 23.520 μs | 22.880 μs | 180.577 μs | 25.793 μs | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nt_align1>(T1::Params)                                                                                                                                                                                                                                                                                      |
| 11.3% | 105.873 ms | 2160      | 49.015 μs | 33.344 μs | 32.448 μs | 87.905 μs  | 22.463 μs | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(T1::Params)                                                                                                                                                                                                                                                                                      |
| 10.3% | 96.209 ms  | 1800      | 53.449 μs | 39.808 μs | 38.977 μs | 111.424 μs | 27.381 μs | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_tn_align1>(T1::Params)                                                                                                                                                                                                                                                                                     |
| 7.4%  | 69.243 ms  | 5415      | 12.787 μs | 5.760 μs  | 1.760 μs  | 99.745 μs  | 15.275 μs | void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3) |
| 7.3%  | 68.356 ms  | 720       | 94.938 μs | 94.624 μs | 91.969 μs | 99.168 μs  | 1.381 μs  | void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_tn_align1>(T1::Params)                                                                                                                                                                                                                                                                                     |

Inference (forward only): kernels account for 53.8% of total time. (source: `output/nsys/benchmark__forward_only__medium__FWD__seq128__warm5.nsys-rep`, `nsys stats --report cuda_gpu_sum`)

```
85.6% Kernel2 (GEMM)           → matrix multiplication for forward/backward
5.8%  elementwise_kernel       → activations, copy, etc.
5.0%  vectorized_elementwise   → vectorized elementwise ops
```

Training (forward + backward): kernels account for 85.1% of total time. (source: `output/nsys/benchmark__20260113_105512__medium__FWD_BWD__seq128__warm5.nsys-rep`, `nsys stats --report cuda_gpu_sum`)

```
46.0% Kernel2 (GEMM)           → matrix multiplication for forward/backward
37.2% multi_tensor_apply       → optimizer parameter updates
10.2% elementwise_kernel       → activations, copy, etc.
4.5%  vectorized_elementwise   → vectorized elementwise ops
```

- `multi_tensor_apply` relates to `optimizer.step()` and only appears in training.
- Training is more compute-intensive, so non-kernel time (100% - 85.1%) is smaller than inference (100% - 53.8%).

(e) Attention sub-ranges (NVTX GPU projection)

All ranges use the `PushPop` style.

Timing

| Range                         | Total Proj Time | Total Range Time | Instances | Proj Avg   | Proj Med   | Proj Min   | Proj Max   | Proj StdDev |
| ----------------------------- | --------------- | ---------------- | --------- | ---------- | ---------- | ---------- | ---------- | ----------- |
| :forward                      | 1.043 s         | 1.054 s          | 1         | 1.043 s    | 1.043 s    | 1.043 s    | 1.043 s    | 0 ns        |
| :scaled dot product attention | 82.794 ms       | 101.292 ms       | 24        | 3.450 ms   | 546.130 μs | 486.082 μs | 69.771 ms  | 14.127 ms   |
| :computing attention scores   | 22.329 ms       | 40.620 ms        | 24        | 930.365 μs | 273.105 μs | 120.448 μs | 16.722 ms  | 3.365 ms    |
| :computing softmax            | 17.560 ms       | 32.895 ms        | 24        | 731.664 μs | 74.032 μs  | 63.649 μs  | 14.344 ms  | 2.901 ms    |
| :final matmul                 | 1.079 ms        | 2.552 ms         | 24        | 44.957 μs  | 38.272 μs  | 31.872 μs  | 210.977 μs | 35.465 μs   |

Ops

| Range                         | Total GPU Ops | Avg GPU Ops | Avg Range Lvl | Avg Num Child |
| ----------------------------- | ------------- | ----------- | ------------- | ------------- |
| :forward                      | 1184          | 1184        | 0             | 24            |
| :scaled dot product attention | 360           | 15          | 1             | 3             |
| :computing attention scores   | 144           | 6           | 2             | 0             |
| :computing softmax            | 120           | 5           | 2             | 0             |
| :final matmul                 | 48            | 2           | 2             | 0             |

(source: `output/nsys/benchmark__fix_annotation__medium__FWD__seq128__warm5.nsys-rep`, `nsys stats --report nvtx_gpu_proj_sum --filter-nvtx forward`)

In a single forward pass, softmax time is 17.56ms vs matrix multiplications at (22.329ms + 1.079ms) = 23.408ms, so softmax is about 75% of matmul time.

For one head and one batch, softmax FLOPs per row is 5mn; across attention this is 5 x seq x seq. Computing attention scores is 2 x seq x head_dim x seq, and the final matmul is the same, so total matmul FLOPs is 4 x seq x head_dim x seq. The FLOPs ratio is (5 x seq x seq) : (4 x seq x head_dim x seq) = 5 : (4 x head_dim). In our medium config, head_dim = 64, so the ratio is 5 : 256, about 1.95%.

The time spent computing softmax is much higher than its FLOPs ratio, likely because softmax is elementwise and memory-bound (more memory traffic), while GEMM kernels are highly optimized and more compute-bound. A possible improvement is to use a fused kernel to avoid intermediate softmax stores/loads, trading a bit more compute for less memory access.

```
m x (n-1)  get row max
m x n      minus max
m x n      exp
m x (n-1)  get sum
m x n      divide
```
