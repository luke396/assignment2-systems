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

Using 5 warmup steps and 10 measured steps, at seq_len=512 the small/medium configs average 0.018s/0.050s forward and 0.061s/0.178s forward+backward (std <= 0.0038s), so the backward component is roughly 0.043s/0.128s. Large OOM at seq_len=512; at seq_len=256 large averages 0.056s forward and 0.208s forward+backward (std ~= 0.002s), while xl/2.7B only ran forward at seq_len=128 (0.065s/0.084s).

Without warm-up, variance is huge (e.g., small seq_len=512 has std 0.212s FWD / 0.121s FWD+BWD), but with 5 warmup steps it drops to 0.0038s / 0.0016s. With only 1 warmup step it still differs (0.0029s / 0.0020s at small seq_len=512), likely because GPU clocks, caches, and kernel autotuning/allocator state are still settling.
