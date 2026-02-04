"""For benchmarking Multihead Self-Attention module."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from cs336_basics.blocks import MultiheadSelfAttention

BATCH_SIZE = 8
HEAD_EMBD_DIM = [16, 32, 64, 128]
SEQ_E_LEN = [256, 1024, 4096, 8192, 16384]
VOCAB_SIZE = 10000
SEED = 42

torch.set_float32_matmul_precision("high")


@dataclass
class TimingStats:
    """Statistics for timing measurements."""

    avg_ms: float
    std_ms: float
    max_ms: float
    min_ms: float


def compute_timing_stats(times: list[float]) -> TimingStats:
    """Compute timing statistics from a list of measurements."""
    return TimingStats(
        avg_ms=sum(times) / len(times),
        std_ms=float(torch.std(torch.tensor(times))),
        max_ms=max(times),
        min_ms=min(times),
    )


def generate_random_data(
    seq_len: int,
    d_model: int,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> torch.Tensor:
    """Generate random data shape of (batch_size, seq_len, d_model)."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        size=(batch_size, seq_len, d_model),
        generator=generator,
        device=device,
    )


def _get_oom_info() -> dict:
    """Get memory info when OOM occurs."""
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "free_mb": (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_reserved()
        )
        / (1024**2),
        "total_mb": torch.cuda.get_device_properties(0).total_memory / (1024**2),
    }


def benchmark_attention(
    seq_len: int,
    embed_dim: int,
    device: torch.device,
    num_heads: int = 1,
    *,
    forward_only: bool = True,
    jit: bool = False,
) -> dict:
    """Benchmark Multihead Self-Attention module."""
    if jit:
        torch._dynamo.reset()  # noqa: SLF001 # Reset compile cache to avoid recompile_limit
    try:
        return _benchmark_attention_impl(
            seq_len, embed_dim, device, num_heads, forward_only=forward_only, jit=jit
        )
    except torch.cuda.OutOfMemoryError:
        oom_info = _get_oom_info()
        print(f"seq={seq_len}, d={embed_dim}, fwd={forward_only}: OOM")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return {
            "seq_len": seq_len,
            "embed_dim": embed_dim,
            "forward_only": forward_only,
            "status": "OOM",
            **oom_info,
        }


def _benchmark_attention_impl(
    seq_len: int,
    embed_dim: int,
    device: torch.device,
    num_heads: int = 1,
    *,
    forward_only: bool = True,
    jit: bool = False,
) -> dict:
    """Run the benchmark implementation."""
    data = generate_random_data(
        seq_len=seq_len,
        d_model=embed_dim,
        device=device,
    )  # shape: (bat, seq, d_model)

    module = MultiheadSelfAttention(
        d_model=embed_dim, num_heads=num_heads, rope=None, device=device
    )
    model = torch.compile(module) if jit else module
    iters = 100
    warmup_iters = 10

    for _ in range(warmup_iters):
        output = model(data)
        if not forward_only:
            loss = output.sum()
            loss.backward()
            module.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # using cuda events timing on GPU
    fwd_time, bwd_time = [], []
    memory_before_backward_lst = []
    for _ in range(iters):
        fwd_start, fwd_end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        fwd_start.record()
        output = model(data)
        fwd_end.record()

        torch.cuda.synchronize()
        fwd_time.append(fwd_start.elapsed_time(fwd_end))

        if not forward_only:
            memory_before_backward_lst.append(torch.cuda.memory_allocated())
            bwd_start, bwd_end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            bwd_start.record()
            loss = output.sum()
            loss.backward()
            bwd_end.record()
            module.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            bwd_time.append(bwd_start.elapsed_time(bwd_end))
    # Compute statistics once
    fwd_stats = compute_timing_stats(fwd_time)
    mode = "fwd" if forward_only else "fwd+bwd"
    print(f"seq={seq_len}, d={embed_dim}, {mode}: {fwd_stats.avg_ms:.3f} ms", end="")

    result = {
        "seq_len": seq_len,
        "embed_dim": embed_dim,
        "forward_only": forward_only,
        "fwd_avg_ms": fwd_stats.avg_ms,
        "fwd_std_ms": fwd_stats.std_ms,
        "fwd_max_ms": fwd_stats.max_ms,
        "fwd_min_ms": fwd_stats.min_ms,
    }

    if not forward_only:
        bwd_stats = compute_timing_stats(bwd_time)
        mem_max = max(memory_before_backward_lst) / (1024**2)
        mem_min = min(memory_before_backward_lst) / (1024**2)
        mem_avg = (
            sum(memory_before_backward_lst)
            / len(memory_before_backward_lst)
            / (1024**2)
        )
        print(f" (bwd: {bwd_stats.avg_ms:.3f} ms, mem: {mem_avg:.1f} MB)")

        result.update(
            {
                "bwd_avg_ms": bwd_stats.avg_ms,
                "bwd_std_ms": bwd_stats.std_ms,
                "bwd_max_ms": bwd_stats.max_ms,
                "bwd_min_ms": bwd_stats.min_ms,
                "mem_before_bwd_max_mb": mem_max,
                "mem_before_bwd_min_mb": mem_min,
                "mem_before_bwd_avg_mb": mem_avg,
            }
        )
    else:
        print()

    return result


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    jit = True
    for seq_len in SEQ_E_LEN:
        for embed_dim in HEAD_EMBD_DIM:
            result_fwd = benchmark_attention(
                seq_len=seq_len,
                embed_dim=embed_dim,
                device=device,
                forward_only=True,
                jit=jit,
            )
            results.append(result_fwd)
            result_bwd = benchmark_attention(
                seq_len=seq_len,
                embed_dim=embed_dim,
                device=device,
                forward_only=False,
                jit=jit,
            )
            results.append(result_bwd)

    # Save results to output directory
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = (
        output_dir / "benchmark_attention_results.json"
        if not jit
        else output_dir / "benchmark_attention_results_jit.json"
    )
    with output_file.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
