"""Run DDP benchmarks on XL model and save results as markdown."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from cs336_systems.ddp import benchmark_ddp_xl, naive_ddp

SEQ_LENS = (64, 96, 128, 256, 512, 1024)
OUT_DIR = Path("docs/sections")
BUCKET_SIZE_MB = [1, 10, 100, 1000]


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
        avg_f = sum(r["forward_time"] for r in group) / n * 1000
        avg_b = sum(r["backward_time"] for r in group) / n * 1000
        avg_c = sum(r["allreduce_time"] for r in group) / n * 1000
        avg_o = sum(r["optimizer_time"] for r in group) / n * 1000
        avg_total = avg_f + avg_b + avg_c + avg_o
        pct = avg_c / avg_total * 100
        rows.append(
            f"| {sl} | {avg_f:.2f} | {avg_b:.2f} | {avg_c:.2f} "
            f"| {avg_o:.2f} | {avg_total:.2f} | {pct:.1f}% |"
        )
    rows.extend(
        f"| {sl} | OOM | OOM | OOM | OOM | OOM | OOM |"
        for sl in sorted(oom_seq_lens or [])
    )
    table = "\n".join(rows)

    return (
        f"### {label} DDP Benchmark (XL, 1 Node x {n_procs} GPUs)\n\n"
        "| Seq Len | Fwd (ms) | Bwd (ms) | AllReduce (ms) "
        "| Optimizer (ms) | Total (ms) | Comm % |\n"
        "|---------|---------|---------|----------------"
        "|----------------|------------|--------|\n"
        f"{table}\n"
    )


def parse_total_from_md(path: Path) -> dict[int, float | str]:
    """Extract {seq_len: total_ms} from an existing result markdown file."""
    lines = path.read_text().splitlines()
    # Find "Total" column index from header row.
    total_idx = 5  # default for old format (Fwd+Bwd combined)
    for line in lines:
        if "Total" in line and line.startswith("|"):
            cols = [c.strip() for c in line.split("|")]
            for i, c in enumerate(cols):
                if "Total" in c:
                    total_idx = i
                    break
            break

    out: dict[int, float | str] = {}
    for line in lines:
        m = re.match(r"\|\s*(\d+)\s*\|", line)
        if not m:
            continue
        cols = [c.strip() for c in line.split("|")]
        sl = int(cols[1])
        val = cols[total_idx]
        out[sl] = val if val == "OOM" else float(val)
    return out


def format_combined_table(
    data: dict[str, dict[int, float | str]],
    n_procs: int,
) -> str:
    """Format a combined comparison table across multiple DDP modes."""
    labels = list(data.keys())
    all_seqs = sorted({sl for d in data.values() for sl in d})

    header = "| Seq Len | " + " | ".join(f"{lb} (ms)" for lb in labels) + " |"
    sep = "|---------|" + "|".join("-" * (len(lb) + 6) for lb in labels) + "|"
    rows = []
    for sl in all_seqs:
        cells = []
        for lb in labels:
            val = data[lb].get(sl)
            cells.append("OOM" if val == "OOM" or val is None else f"{val:.2f}")
        rows.append(f"| {sl} | " + " | ".join(cells) + " |")

    return (
        f"### Bucketed DDP Comparison (XL, 1 Node x {n_procs} GPUs)\n\n"
        f"{header}\n{sep}\n" + "\n".join(rows) + "\n"
    )


def run_benchmarks(
    modes: list[tuple[bool, bool, float | None, str, str]],
    n_procs: int = 2,
    seq_lens: tuple[int, ...] = SEQ_LENS,
    profile_dir: str | None = None,
) -> None:
    """Run DDP benchmarks for given modes and save per-mode markdown files."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for use_flat, use_overlap, bucket_size_mb, label, filename in modes:
        all_res: list[dict] = []
        oom_seqs: list[int] = []
        for seq in seq_lens:
            print(f"[{label}] Benchmarking seq_len={seq}...")
            try:
                res = benchmark_ddp_xl(
                    n_procs=n_procs,
                    backend="nccl",
                    seq_len=seq,
                    flat_grad=use_flat,
                    overlapped=use_overlap,
                    bucket_size_mb=bucket_size_mb,
                    profile_dir=profile_dir,
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


ALL_MODES: dict[str, tuple[bool, bool, float | None, str, str]] = {
    "naive": (False, False, None, "Naive", "ddp-xl-naive.md"),
    "flat-grad": (True, False, None, "Flat-Grad", "ddp-xl-naive-flat-grad.md"),
    "overlapped": (False, True, None, "Overlapped", "ddp-xl-overlapped.md"),
    **{
        f"bucketed-{bs}mb": (
            False,
            False,
            bs,
            f"Bucketed ({bs}MB)",
            f"ddp-xl-bucketed-{bs}mb.md",
        )
        for bs in BUCKET_SIZE_MB
    },
}

# Files to include in the bucketed comparison table (label → filename).
COMBINED_TABLE_FILES: dict[str, str] = {
    "Naive": "ddp-xl-naive.md",
    "Overlapped": "ddp-xl-overlapped.md",
    **{f"Bucketed ({bs}MB)": f"ddp-xl-bucketed-{bs}mb.md" for bs in BUCKET_SIZE_MB},
}


def generate_combined_table(n_procs: int = 2) -> str:
    """Read per-mode result files and generate a combined comparison table.

    Raises FileNotFoundError if any required result file is missing.
    """
    data: dict[str, dict[int, float | str]] = {}
    for label, filename in COMBINED_TABLE_FILES.items():
        path = OUT_DIR / filename
        if not path.exists():
            msg = f"Missing result file: {path}"
            raise FileNotFoundError(msg)
        data[label] = parse_total_from_md(path)
    return format_combined_table(data, n_procs=n_procs)


def main() -> None:
    """Parse args and run DDP benchmarks."""
    parser = argparse.ArgumentParser(description="Run DDP benchmarks on XL model")
    parser.add_argument(
        "--verify", action="store_true", help="Run correctness verification first"
    )
    parser.add_argument("--n-procs", type=int, default=2, help="Number of GPUs")
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=list(ALL_MODES),
        help="Run only specified modes (default: all)",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Only generate combined table from existing result files",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        metavar="DIR",
        help="Enable torch.profiler and save traces to DIR",
    )
    args = parser.parse_args()

    if args.verify:
        print("Running correctness verification...")
        naive_ddp(batch_size=32, n_procs=8, backend="gloo")
        print("Verification passed.")

    if not args.combine_only:
        selected = args.mode or list(ALL_MODES)
        modes = [ALL_MODES[m] for m in selected]
        run_benchmarks(
            modes,
            n_procs=args.n_procs,
            profile_dir=args.profile,
        )

    md = generate_combined_table(n_procs=args.n_procs)
    print(md)
    out = OUT_DIR / "ddp-xl-bucketed.md"
    out.write_text(md)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
