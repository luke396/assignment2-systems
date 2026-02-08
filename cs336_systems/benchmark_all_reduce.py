"""Benchmark all-reduce across backends, data sizes, and process counts."""

import argparse
import json
import os
import time
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ITER = 5
DATA_SIZE_MB = [1, 10, 100, 1024]
N_PROC = [2, 4, 6]


def _setup(rank: int, world_size: int, backend: str = "gloo") -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def _bench_all_reduce(
    rank: int,
    world_size: int,
    backend: str,
    data_size_mb: int,
    result_queue: Queue[dict[str, Any]],
) -> None:
    """Run all-reduce benchmark in a spawned process."""
    device = f"cuda:{rank}" if backend == "nccl" else "cpu"
    if backend == "nccl":
        torch.cuda.set_device(rank)
    _setup(rank, world_size, backend=backend)

    n_elements = (data_size_mb * 1024 * 1024) // 4
    tensor = torch.randn(n_elements, dtype=torch.float32, device=device)

    for _ in range(5):
        dist.all_reduce(tensor, async_op=False)

    if backend == "nccl":
        torch.cuda.synchronize()
    dist.barrier()

    start = time.perf_counter()
    for _ in range(ITER):
        dist.all_reduce(tensor, async_op=False)
    if backend == "nccl":
        torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / ITER * 1000
    if rank == 0:
        result_queue.put(
            {
                "backend": backend,
                "data_size_mb": data_size_mb,
                "n_proc": world_size,
                "avg_ms": round(avg_ms, 3),
            }
        )
        print(
            f"  {backend} | {data_size_mb:>5}MB | {world_size} procs | {avg_ms:.3f}ms",
        )

    dist.destroy_process_group()


def _run_benchmarks(
    backends: list[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    result_queue: Queue[dict[str, Any]] = ctx.Queue()
    for backend in backends:
        for data_size in DATA_SIZE_MB:
            for n_proc in N_PROC:
                mp.spawn(
                    _bench_all_reduce,
                    args=(n_proc, backend, data_size, result_queue),
                    nprocs=n_proc,
                    join=True,
                )
                while not result_queue.empty():
                    results.append(result_queue.get())
    return results


def _plot_results(
    results: list[dict[str, Any]],
    output_dir: str,
) -> None:
    backends = sorted({r["backend"] for r in results})
    _, axes = plt.subplots(
        1,
        len(backends),
        figsize=(7 * len(backends), 5),
        sharey=True,
    )
    if len(backends) == 1:
        axes = [axes]

    for ax, backend in zip(axes, backends, strict=True):
        subset = [r for r in results if r["backend"] == backend]
        procs = sorted({r["n_proc"] for r in subset})
        for n_proc in procs:
            data = sorted(
                [r for r in subset if r["n_proc"] == n_proc],
                key=lambda r: r["data_size_mb"],
            )
            sizes = [r["data_size_mb"] for r in data]
            times = [r["avg_ms"] for r in data]
            ax.plot(sizes, times, marker="o", label=f"{n_proc} procs")
        device_label = "GPU" if backend == "nccl" else "CPU"
        ax.set_title(f"{backend.upper()} + {device_label}")
        ax.set_xlabel("Data Size (MB)")
        ax.set_ylabel("Avg Latency (ms)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(visible=True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(output_dir) / "all_reduce_benchmark.png"
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.close()


def main() -> None:
    """Entry point for all-reduce benchmarking."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        nargs="+",
        default=["gloo", "nccl"],
        choices=["gloo", "nccl"],
    )
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "all_reduce_results.json"

    if not args.plot_only:
        new_results = _run_benchmarks(args.backend)
        if json_path.exists():
            existing = json.loads(json_path.read_text())
            ran_backends = set(args.backend)
            merged = [r for r in existing if r["backend"] not in ran_backends]
            merged.extend(new_results)
        else:
            merged = new_results
        json_path.write_text(json.dumps(merged, indent=2))
        results = merged
    else:
        results = json.loads(json_path.read_text())

    _plot_results(results, args.output_dir)


if __name__ == "__main__":
    main()
