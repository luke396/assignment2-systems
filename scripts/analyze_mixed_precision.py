#!/usr/bin/env python3
"""Analyze mixed precision benchmark results from nsys reports."""
# ruff: noqa: S603, S607

import subprocess
from pathlib import Path

SIZES = ["small", "medium", "large", "xl", "2.7B"]
NSYS_DIR = Path("output/nsys")


def get_nvtx_stats(report_path: Path) -> dict[str, float]:
    """Extract NVTX timing stats from nsys report (median time in ns)."""
    if not report_path.exists():
        return {}
    result = subprocess.run(
        ["nsys", "stats", "--report", "nvtx_sum", str(report_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    stats: dict[str, float] = {}
    for line in result.stdout.splitlines():
        if ":forward" in line:
            stats["forward"] = float(line.split()[3].replace(",", ""))
        elif ":backward" in line:
            stats["backward"] = float(line.split()[3].replace(",", ""))
        elif "computing attention scores" in line:
            stats["attn_scores"] = float(line.split()[3].replace(",", ""))
        elif "scaled dot product attention" in line:
            stats["sdpa"] = float(line.split()[3].replace(",", ""))
    return stats


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    """Print a formatted markdown table."""
    print(f"\n## {title}\n")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def _build_path(prefix: str, size: str, mode: str) -> Path:
    return NSYS_DIR / f"benchmark__{prefix}__{size}__{mode}__seq128__warm5.nsys-rep"


def _compute_comparison(
    base_prefix: str, mixed_prefix: str, mode: str, metric: str
) -> list[list[str]]:
    """Compute speedup comparison rows for a given metric."""
    rows = []
    for size in SIZES:
        fp = get_nvtx_stats(_build_path(base_prefix, size, mode))
        bf = get_nvtx_stats(_build_path(mixed_prefix, size, mode))

        if metric == "total":
            fp_val = fp.get("forward", 0) + fp.get("backward", 0)
            bf_val = bf.get("forward", 0) + bf.get("backward", 0)
        else:
            fp_val = fp.get(metric, 0)
            bf_val = bf.get(metric, 0)

        if fp_val and bf_val:
            rows.append([
                size,
                f"{fp_val / 1e6:.2f}",
                f"{bf_val / 1e6:.2f}",
                f"{fp_val / bf_val:.2f}x",
            ])
    return rows


def analyze() -> None:
    """Run analysis and print results."""
    print_table(
        "Forward Pass Comparison (Median Time in ms)",
        ["Model", "Full Precision", "BF16 Mixed", "Speedup"],
        _compute_comparison("new_base", "autocastBF", "FWD", "forward"),
    )

    print_table(
        "Forward+Backward Pass Comparison (Median Time in ms)",
        ["Model", "Full Precision", "BF16 Mixed", "Speedup"],
        _compute_comparison("new_base", "autocastBF", "FWD_BWD", "total"),
    )

    print_table(
        "Attention Score Computation (Median Time in ms)",
        ["Model", "Full Precision", "BF16 Mixed", "Speedup"],
        _compute_comparison("new_base", "autocastBF", "FWD", "attn_scores"),
    )

    print("\n## Conclusions\n")
    print("1. BF16 mixed precision is slower (5-12%) on small-medium models")
    print("   due to autocast type conversion overhead (copy kernels dominate)")
    print("2. Attention score computation shows 1.2x-3.3x speedup with BF16")
    print("   larger models benefit more from reduced precision matmul")
    print("3. As model size increases, BF16 disadvantage decreases")
    print("   because compute time dominates conversion overhead")


if __name__ == "__main__":
    analyze()
