"""Run DDP benchmarks on XL model and save results as markdown."""

from __future__ import annotations

import argparse
from pathlib import Path

from cs336_systems.ddp import benchmark_ddp_xl, naive_ddp

SEQ_LENS = (64, 96, 128, 256, 512, 1024)
OUT_DIR = Path("docs/sections")


def format_results(
    all_results: list[dict],
    n_procs: int,
    label: str = "Naive",
    oom_seq_lens: list[int] | None = None,
) -> str:
    """Format benchmark results as markdown table."""
    seq_lens = sorted({r["seq_len"] for r in all_results})
    rows = []
    for sl in seq_lens:
        group = [r for r in all_results if r["seq_len"] == sl]
        n = len(group)
        avg_t = sum(r["training_time"] for r in group) / n * 1000
        avg_c = sum(r["allreduce_time"] for r in group) / n * 1000
        avg_o = sum(r["optimizer_time"] for r in group) / n * 1000
        avg_total = avg_t + avg_c + avg_o
        pct = avg_c / avg_total * 100
        rows.append(
            f"| {sl} | {avg_t:.2f} | {avg_c:.2f} "
            f"| {avg_o:.2f} | {avg_total:.2f} | {pct:.1f}% |"
        )
    rows.extend(
        f"| {sl} | OOM | OOM | OOM | OOM | OOM |"
        for sl in sorted(oom_seq_lens or [])
    )
    table = "\n".join(rows)

    return (
        f"### {label} DDP Benchmark (XL, 1 Node x {n_procs} GPUs)\n\n"
        "| Seq Len | Fwd+Bwd (ms) | AllReduce (ms) "
        "| Optimizer (ms) | Total (ms) | Comm % |\n"
        "|---------|-------------|----------------"
        "|----------------|------------|--------|\n"
        f"{table}\n"
    )


def run_benchmarks(
    modes: list[tuple[bool, str, str]],
    n_procs: int = 2,
    seq_lens: tuple[int, ...] = SEQ_LENS,
) -> None:
    """Run DDP benchmarks for given modes and save markdown results."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for use_flat, label, filename in modes:
        all_res: list[dict] = []
        oom_seqs: list[int] = []
        for seq in seq_lens:
            print(f"[{label}] Benchmarking seq_len={seq}...")
            try:
                res = benchmark_ddp_xl(
                    n_procs=n_procs, backend="nccl", seq_len=seq, flat_grad=use_flat
                )
            except Exception as e:
                if "out of memory" in str(e).lower():
                    print(f"[{label}] OOM at seq_len={seq}, skipping rest.")
                    oom_seqs.extend(
                        s for s in seq_lens if s >= seq and s not in oom_seqs
                    )
                    break
                raise
            all_res.extend(res)

        if not all_res and not oom_seqs:
            print(f"[{label}] No results collected, skipping output.")
            continue
        md = format_results(
            all_res, n_procs=n_procs, label=label, oom_seq_lens=oom_seqs
        )
        print(md)
        out = OUT_DIR / filename
        out.write_text(md)
        print(f"Saved: {out}")


def main() -> None:
    """Parse args and run DDP benchmarks."""
    parser = argparse.ArgumentParser(description="Run DDP benchmarks on XL model")
    parser.add_argument(
        "--verify", action="store_true", help="Run correctness verification first"
    )
    parser.add_argument("--n-procs", type=int, default=2, help="Number of GPUs")
    args = parser.parse_args()

    if args.verify:
        print("Running correctness verification...")
        naive_ddp(batch_size=32, n_procs=8, backend="gloo")
        print("Verification passed.")

    modes: list[tuple[bool, str, str]] = [
        (False, "Naive", "ddp-xl-naive.md"),
        (True, "Flat-Grad-Naive", "ddp-xl-naive-flat-grad.md"),
    ]
    run_benchmarks(modes, n_procs=args.n_procs)


if __name__ == "__main__":
    main()
