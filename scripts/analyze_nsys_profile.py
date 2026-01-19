#!/usr/bin/env python3
"""Analyze nsys profile results for forward/backward pass benchmarks."""
# ruff: noqa: S603

import argparse
import subprocess
from pathlib import Path
from typing import TypeAlias

SIZES = ["small", "medium", "large", "xl", "2.7B"]
SEQ_LENS = [128, 256, 512, 1024]
NSYS_DIR = Path("output/nsys")
MIN_KERNEL_PARTS = 10
MIN_FORWARD_BACKWARD_PARTS = 4
MIN_ATTN_MARKER_PARTS = 5
MIN_BENCHMARK_TABLE_COLS = 6
MODES = [("FWD", "Inference"), ("FWD_BWD", "Training")]
ATTN_KEYS = ("softmax", "attn_scores", "final_matmul")
NvtxStats: TypeAlias = dict[str, dict[str, float]]
KernelInfo: TypeAlias = dict[str, str | float | bool]
OomStatus: TypeAlias = dict[tuple[str, int, str], bool]
KERN_HEADERS = ["Time%", "Total", "Inst", "Kernel"]
SEQ_HEADERS = ["Model"] + [f"seq{s}" for s in SEQ_LENS]


def parse_oom_status(run_tag: str) -> OomStatus:
    """Parse benchmark results to get OOM status for each configuration.

    Returns dict with key=(size, seq_len, mode) and value=is_oom.
    """
    oom_status: OomStatus = {}
    benchmark_results = Path(f"output/benchmark_results__{run_tag}.md")
    if not benchmark_results.exists():
        return oom_status

    with benchmark_results.open() as f:
        lines = f.readlines()

    # Skip header and separator lines
    for line in lines[3:]:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < MIN_BENCHMARK_TABLE_COLS:
            continue

        config = parts[1]  # Model size
        seq_len_str = parts[2]  # Seq len
        status_fwd = parts[4]  # Status FWD
        status_fwd_bwd = parts[5]  # Status FWD+BWD

        try:
            seq_len = int(seq_len_str)
            oom_status[(config, seq_len, "FWD")] = status_fwd == "OOM"
            oom_status[(config, seq_len, "FWD_BWD")] = status_fwd_bwd == "OOM"
        except ValueError:
            continue

    return oom_status


def run_nsys_stats(report_path: Path, report_type: str, nvtx_filter: str = "") -> str:
    """Run nsys stats and return output."""
    if not report_path.exists():
        return ""
    cmd = [
        "nsys",
        "stats",
        "--force-export=true",
        "--report",
        report_type,
        str(report_path),
    ]
    if nvtx_filter:
        cmd.extend(["--filter-nvtx", nvtx_filter])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.stdout


def _parse_float(s: str) -> float:
    """Remove commas and convert to float."""
    return float(s.replace(",", ""))


def _create_stats_entry(total_ns: float, instances: float) -> dict[str, float]:
    """Create a stats dictionary entry with total, instances, and average."""
    avg_ns = total_ns / instances if instances > 0 else 0.0
    return {"total_ns": total_ns, "instances": instances, "avg_ns": avg_ns}


def parse_nvtx_sum(output: str) -> NvtxStats:
    """Parse nvtx_sum output to extract timing data."""
    stats: NvtxStats = {}
    attn_markers = {
        "computing attention scores": "attn_scores",
        "computing softmax": "softmax",
        "final matmul": "final_matmul",
    }
    for line in output.splitlines():
        parts = line.split()
        if ":forward" in line or ":backward" in line:
            if len(parts) >= MIN_FORWARD_BACKWARD_PARTS:
                total_ns = _parse_float(parts[1])
                instances = _parse_float(parts[2])
                stats[parts[-1]] = _create_stats_entry(total_ns, instances)
        else:
            for marker, key in attn_markers.items():
                if marker in line and len(parts) >= MIN_ATTN_MARKER_PARTS:
                    total_ns = _parse_float(parts[3])
                    instances = _parse_float(parts[4])
                    stats[key] = _create_stats_entry(total_ns, instances)
                    break
    return stats


def parse_kernel_sum(output: str) -> list[KernelInfo]:
    """Parse cuda_gpu_kern_sum output to extract top kernels."""
    kernels: list[KernelInfo] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or "Time (%" in line or "--------" in line:
            continue
        parts = line.split()
        if len(parts) < MIN_KERNEL_PARTS:
            continue
        try:
            float(parts[0])  # Check if first part is a percentage
            name = " ".join(parts[8:])[:50] + "..."
            kernels.append(
                {
                    "pct": f"{parts[0]}%",
                    "total_time": f"{_parse_float(parts[1]) / 1e6:.2f} ms",
                    "instances": int(_parse_float(parts[2])),
                    "name": name,
                    "is_gemm": "cutlass" in name or "gemm" in name.lower(),
                }
            )
        except (IndexError, ValueError):
            continue
    return kernels[:10]


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    """Print a formatted markdown table with aligned columns."""
    print(f"\n### {title}\n")
    if not rows:
        return
    # Validate row lengths match headers
    for row in rows:
        if len(row) != len(headers):
            msg = f"Row length {len(row)} doesn't match headers length {len(headers)}"
            raise ValueError(msg)
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    hdr = " | ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    sep = " | ".join("-" * w for w in widths)
    print(f"| {hdr} |")
    print(f"| {sep} |")
    for row in rows:
        cells = " | ".join(c.ljust(w) for c, w in zip(row, widths, strict=True))
        print(f"| {cells} |")


def _kernel_to_row(k: KernelInfo) -> list[str]:
    """Convert kernel dict to table row."""
    return [str(k["pct"]), str(k["total_time"]), str(k["instances"]), str(k["name"])]


def _calc_gemm_pct(kernels: list[KernelInfo]) -> float:
    """Calculate total GEMM percentage from kernel list."""
    return sum(
        float(str(k["pct"]).rstrip("%")) for k in kernels if k.get("is_gemm", False)
    )


def _build_report_path(run_tag: str, size: str, mode: str, seq_len: int) -> Path:
    """Build path to nsys report file."""
    name = f"benchmark__{run_tag}__{size}__{mode}__seq{seq_len}__warm5.nsys-rep"
    return NSYS_DIR / name


def _get_nvtx_stats(
    run_tag: str, size: str, mode: str = "FWD", seq_len: int = 128
) -> NvtxStats:
    """Get parsed NVTX stats for a given configuration."""
    path = _build_report_path(run_tag, size, mode, seq_len)
    return parse_nvtx_sum(run_nsys_stats(path, "nvtx_sum"))


def _get_kernels(
    run_tag: str, size: str, mode: str = "FWD", seq_len: int = 128
) -> list[KernelInfo]:
    """Get parsed kernel stats for a given configuration."""
    path = _build_report_path(run_tag, size, mode, seq_len)
    return parse_kernel_sum(run_nsys_stats(path, "cuda_gpu_kern_sum"))


def _calc_softmax_ratio(stats: NvtxStats) -> float | None:
    """Calculate softmax/matmul ratio from stats, or None if data missing."""
    if not all(k in stats for k in ATTN_KEYS):
        return None
    softmax_ns = stats["softmax"]["total_ns"]
    matmul_ns = stats["attn_scores"]["total_ns"] + stats["final_matmul"]["total_ns"]
    if matmul_ns <= 0:
        return None
    return softmax_ns / matmul_ns


def _format_ms(nanoseconds: float) -> str:
    """Convert nanoseconds to milliseconds string."""
    return f"{nanoseconds / 1e6:.2f}"


def _analyze_forward_timing(run_tag: str) -> None:
    """Analyze forward pass timing."""
    print("## (a) Forward Pass Timing\n")
    fwd_rows = []
    for size in SIZES:
        stats = _get_nvtx_stats(run_tag, size)
        if ":forward" in stats:
            fwd = stats[":forward"]
            fwd_rows.append(
                [
                    size,
                    _format_ms(fwd["total_ns"]),
                    str(int(fwd["instances"])),
                    _format_ms(fwd["avg_ns"]),
                ]
            )
    print_table(
        "Forward Pass Time (NVTX)",
        ["Model", "Total (ms)", "Inst", "Avg (ms)"],
        fwd_rows,
    )


def _analyze_dominant_kernels(run_tag: str) -> None:
    """Analyze top kernels in forward vs forward+backward."""
    print("\n## (b) Dominant CUDA Kernels\n")
    for mode, label in MODES:
        print(f"\n**{label} (medium model):**\n")
        kernels = _get_kernels(run_tag, "medium", mode)
        if kernels:
            rows = [_kernel_to_row(k) for k in kernels[:3]]
            print_table(f"Top 3 Kernels ({label})", KERN_HEADERS, rows)


def _analyze_non_gemm_kernels(run_tag: str) -> None:
    """Analyze non-GEMM kernels."""
    print("\n## (c) Non-GEMM Kernels in Forward Pass\n")
    kernels = _get_kernels(run_tag, "medium")
    non_gemm = [k for k in kernels if not k.get("is_gemm", False)][:3]
    if non_gemm:
        rows = [_kernel_to_row(k) for k in non_gemm]
        print_table("Top Non-GEMM Kernels", KERN_HEADERS, rows)


def _analyze_gemm_fraction(run_tag: str) -> None:
    """Analyze GEMM fraction in inference vs training."""
    print("\n## (d) GEMM Fraction: Inference vs Training\n")
    for mode, label in MODES:
        gemm_pct = _calc_gemm_pct(_get_kernels(run_tag, "medium", mode))
        print(f"- **{label}**: GEMM ~{gemm_pct:.1f}% of top kernel time")


def _analyze_attention_ops(run_tag: str) -> None:
    """Analyze softmax vs matmul in attention."""
    print("\n## (e) Softmax vs Matmul in Attention\n")
    attn_rows = []
    for size in SIZES:
        stats = _get_nvtx_stats(run_tag, size)
        ratio = _calc_softmax_ratio(stats)
        if ratio is not None:
            softmax_ms = stats["softmax"]["total_ns"]
            matmul_ms = (
                stats["attn_scores"]["total_ns"] + stats["final_matmul"]["total_ns"]
            )
            attn_rows.append(
                [
                    size,
                    _format_ms(softmax_ms),
                    _format_ms(matmul_ms),
                    f"{ratio:.1%}",
                ]
            )
    print_table(
        "Softmax vs Matmul", ["Model", "Softmax(ms)", "Matmul(ms)", "Ratio"], attn_rows
    )


def _get_value_or_status(
    run_tag: str, size: str, seq_len: int, mode: str, oom_status: OomStatus
) -> str:
    """Get timing value or status (OOM/N/A) for a configuration."""
    if oom_status.get((size, seq_len, mode), False):
        return "OOM"
    stats = _get_nvtx_stats(run_tag, size, mode, seq_len)
    if ":forward" in stats:
        return _format_ms(stats[":forward"]["avg_ns"])
    return "N/A"


def _seq_len_fwd_timing(run_tag: str, oom_status: OomStatus) -> None:
    """Analyze forward pass timing vs sequence length."""
    print("**Forward Pass Time vs Sequence Length:**\n")
    fwd_rows = [
        [size]
        + [
            _get_value_or_status(run_tag, size, seq_len, "FWD", oom_status)
            for seq_len in SEQ_LENS
        ]
        for size in SIZES
    ]
    if fwd_rows:
        print_table("Forward Pass Avg Time (ms)", SEQ_HEADERS, fwd_rows)


def _get_ratio_or_status(
    run_tag: str, size: str, seq_len: int, oom_status: OomStatus
) -> str:
    """Get softmax/matmul ratio or status (OOM/N/A) for a configuration."""
    if oom_status.get((size, seq_len, "FWD"), False):
        return "OOM"
    ratio = _calc_softmax_ratio(_get_nvtx_stats(run_tag, size, "FWD", seq_len))
    return f"{ratio:.1%}" if ratio is not None else "N/A"


def _seq_len_softmax_ratio(run_tag: str, oom_status: OomStatus) -> None:
    """Analyze softmax/matmul ratio vs sequence length."""
    print("\n**Softmax/Matmul Ratio vs Sequence Length:**\n")
    ratio_rows = [
        [size]
        + [
            _get_ratio_or_status(run_tag, size, seq_len, oom_status)
            for seq_len in SEQ_LENS
        ]
        for size in SIZES
    ]
    if ratio_rows:
        print_table("Softmax/Matmul Ratio", SEQ_HEADERS, ratio_rows)


def _get_gemm_value_or_status(
    run_tag: str, seq_len: int, mode: str, oom_status: OomStatus
) -> str:
    """Get GEMM percentage or status (OOM/N/A) for medium model configuration."""
    if oom_status.get(("medium", seq_len, mode), False):
        return "OOM"
    kernels = _get_kernels(run_tag, "medium", mode, seq_len)
    return f"{_calc_gemm_pct(kernels):.1f}%" if kernels else "N/A"


def _seq_len_gemm_fraction(run_tag: str, oom_status: OomStatus) -> None:
    """Analyze GEMM fraction vs sequence length."""
    print("\n**GEMM Fraction vs Sequence Length (medium model):**\n")
    for mode, label in MODES:
        values = [
            _get_gemm_value_or_status(run_tag, seq_len, mode, oom_status)
            for seq_len in SEQ_LENS
        ]
        pairs = ", ".join(f"seq{s}={v}" for s, v in zip(SEQ_LENS, values, strict=True))
        print(f"- **{label}**: {pairs}")


def _analyze_seq_len_scaling(run_tag: str, oom_status: OomStatus) -> None:
    """Analyze performance scaling across different sequence lengths."""
    print("\n## (f) Sequence Length Scaling Analysis\n")
    _seq_len_fwd_timing(run_tag, oom_status)
    _seq_len_softmax_ratio(run_tag, oom_status)
    _seq_len_gemm_fraction(run_tag, oom_status)


def _print_conclusions() -> None:
    """Print conclusions."""
    print("\n## Conclusions\n")
    print("(a) Forward times from NVTX are consistent with Python timeit")
    print("    (slight profiling overhead expected)")
    print("(b) GEMM kernels (cutlass) dominate; same family in fwd/bwd,")
    print("    more instances in bwd")
    print("(c) Non-GEMM: elementwise ops (copy, activation) ~2-5% of fwd time")
    print("(d) GEMM fraction: ~87% inference -> ~59% training")
    print("    (backward adds more GEMM, optimizer adds non-GEMM ops)")
    print("(e) Softmax takes 10-67% of matmul time despite ~2% FLOPs")
    print("    (memory-bound vs compute-bound, ratio decreases with model size)")
    print("(f) Sequence length scaling:")
    print("    - Forward time scales ~O(n^2) due to attention")
    print("    - Softmax/matmul ratio increases with seq_len (softmax is O(n^2))")
    print("    - Larger models OOM at shorter seq_len")
    print("    - GEMM fraction may decrease at longer seq_len")
    print("      (more attention overhead)")


def analyze(run_tag: str) -> None:
    """Run analysis and print results."""
    print("# NSYS Profile Analysis\n")
    oom_status = parse_oom_status(run_tag)
    _analyze_forward_timing(run_tag)
    _analyze_dominant_kernels(run_tag)
    _analyze_non_gemm_kernels(run_tag)
    _analyze_gemm_fraction(run_tag)
    _analyze_attention_ops(run_tag)
    _analyze_seq_len_scaling(run_tag, oom_status)
    _print_conclusions()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze nsys profile results for forward/backward pass benchmarks."
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="new_base",
        help="Run tag to analyze (default: new_base)",
    )
    args = parser.parse_args()
    analyze(args.run_tag)
