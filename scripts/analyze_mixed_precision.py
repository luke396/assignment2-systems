#!/usr/bin/env python3
"""Analyze mixed precision benchmark results from nsys reports."""

import argparse
import subprocess
from pathlib import Path
from textwrap import dedent

from scripts.constants import NSYS_DIR, SIZES
from scripts.table_utils import format_table


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
            rows.append(
                [
                    size,
                    f"{fp_val / 1e6:.2f}",
                    f"{bf_val / 1e6:.2f}",
                    f"{fp_val / bf_val:.2f}x",
                ]
            )
    return rows


CONCLUSIONS = dedent("""\

##### Conclusions

1. BF16 mixed precision is slower (5-12%) on small-medium models \
due to autocast type conversion overhead (copy kernels dominate)
2. Attention score computation shows 1.2x-3.3x speedup with BF16 \
larger models benefit more from reduced precision matmul
3. As model size increases, BF16 disadvantage decreases \
because compute time dominates conversion overhead
    """)


def _generate_analysis(
    base_prefix: str = "new_base", mixed_prefix: str = "autocastBF"
) -> str:
    """Generate full analysis as markdown string."""
    headers = ["Model", "Full Precision", "BF16 Mixed", "Speedup"]
    sections = [
        ("Forward Pass Comparison (Median Time in ms)", "FWD", "forward"),
        ("Forward+Backward Pass Comparison (Median Time in ms)", "FWD_BWD", "total"),
        ("Attention Score Computation (Median Time in ms)", "FWD", "attn_scores"),
    ]
    parts = []
    for title, mode, metric in sections:
        rows = _compute_comparison(base_prefix, mixed_prefix, mode, metric)
        parts.append(format_table(f"\n##### {title}", headers, rows))
    parts.append(CONCLUSIONS)
    return "\n".join(parts)


def analyze(base_prefix: str = "new_base", mixed_prefix: str = "autocastBF") -> str:
    """Run analysis and return results as string."""
    result = "\n#### Mixed Precision Analysis\n" + _generate_analysis(
        base_prefix, mixed_prefix
    )
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze mixed precision benchmarks")
    parser.add_argument(
        "--base-prefix",
        type=str,
        default="new_base",
        help="Base run prefix (default: new_base)",
    )
    parser.add_argument(
        "--mixed-prefix",
        type=str,
        default="autocastBF",
        help="Mixed precision run prefix (default: autocastBF)",
    )
    args = parser.parse_args()

    result = analyze(args.base_prefix, args.mixed_prefix)

    output_path = Path("docs/sections/mixed-precision.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result)
    print(f"\nMarkdown saved: {output_path}")
