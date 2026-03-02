### DDP Nsight Systems Profiling (Naive vs Overlapped)

Below are Nsight Systems timeline traces comparing naive DDP (per-parameter synchronous all-reduce) with overlapped DDP (async all-reduce via backward hooks), both on the XL model with 1 node x 2 GPUs, seq_len=64.

**Naive DDP**

![Naive DDP timeline](images/ddp-native.png)

| Phase          | Duration (ms) |
| -------------- | ------------- |
| forward        | 79.4          |
| backward       | 128.4         |
| grad_sync      | 25.9          |
| optimizer_step | 11.2          |

In the naive trace, NCCL communication kernels appear only during the `grad_sync` phase, after backward computation has fully completed. The compute stream and NCCL stream are active sequentially — no overlap.

**Overlapped DDP**

![Overlapped DDP timeline](images/ddp-overlap.png)

| Phase          | Duration (ms) |
| -------------- | ------------- |
| forward        | 78.2          |
| backward       | 183.5         |
| grad_sync      | 8.4           |
| optimizer_step | 11.2          |

In the overlapped trace, NCCL all-reduce operations are launched during backward via `register_post_accumulate_grad_hook`. The `pt_autograd_1` thread shows NCCL activity concurrent with backward compute kernels. The `grad_sync` phase (which only calls `finish_gradient_synchronization`) drops from 25.9 ms to 8.4 ms, confirming that most communication has already completed by the time backward finishes.
