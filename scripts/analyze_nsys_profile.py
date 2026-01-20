"""Analyze nsys profile results for forward/backward pass benchmarks."""

import argparse
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

from scripts.constants import NSYS_DIR, NVTX_INSTANCE, SEQ_LENS, SIZES

MIN_KERNEL_PARTS = 10
MIN_BENCHMARK_TABLE_COLS = 6
ATTN_KEYS = ("softmax", "attn_scores", "final_matmul")
ATTN_MARKERS = {
    "computing attention scores": "attn_scores",
    "computing softmax": "softmax",
    "final matmul": "final_matmul",
}
NvtxStats: TypeAlias = dict[str, dict[str, float]]
KernelInfo: TypeAlias = dict[str, str | float | bool]
OomStatus: TypeAlias = dict[tuple[str, int, str], bool]
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

    for line in lines[3:]:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < MIN_BENCHMARK_TABLE_COLS:
            continue

        try:
            config = parts[1]
            seq_len = int(parts[2])
            oom_status[(config, seq_len, "FWD")] = parts[4] == "OOM"
            oom_status[(config, seq_len, "FWD_BWD")] = parts[5] == "OOM"
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


def parse_nvtx_gpu_proj_sum(output: str) -> NvtxStats:
    """Parse nvtx_gpu_proj_sum output to extract GPU timing data.

    Expected format:
    Range  Style  Total Proj Time (ns)  Total Range Time (ns)  Range Instances  ...
    :forward  PushPop  1,397,875,298  1,423,929,186  15  ...
    :computing softmax  PushPop  39,831,840  61,604,838  360  ...
    """
    stats: NvtxStats = {}
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or not stripped.startswith(":"):
            continue

        parts = stripped.split()

        try:
            pushpop_idx = parts.index("PushPop")
        except ValueError:
            continue

        if len(parts) < pushpop_idx + 4:
            continue

        total_ns = _parse_float(parts[pushpop_idx + 1])
        instances = _parse_float(parts[pushpop_idx + 3])

        if ":forward" in stripped or ":backward" in stripped:
            stats[parts[0]] = _create_stats_entry(total_ns, instances)
        else:
            for marker, key in ATTN_MARKERS.items():
                if marker in stripped:
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
            float(parts[0])
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


def _format_table_aligned(
    title: str, headers: list[str], rows: list[list[str]]
) -> str:
    """Format a markdown table with aligned columns."""
    if not rows:
        return f"{title}\n\nNo data available.\n\n"

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    hdr = " | ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    sep = " | ".join("-" * w for w in widths)

    lines = [f"{title}\n", f"| {hdr} |", f"| {sep} |"]
    lines.extend(
        f"| {' | '.join(c.ljust(w) for c, w in zip(row, widths, strict=True))} |"
        for row in rows
    )
    lines.append("")
    return "\n".join(lines)


def _calc_gemm_pct(kernels: list[KernelInfo]) -> float:
    """Calculate total GEMM percentage from kernel list."""
    return sum(
        float(str(k["pct"]).rstrip("%")) for k in kernels if k.get("is_gemm", False)
    )


def _build_report_path(run_tag: str, size: str, mode: str, seq_len: int) -> Path:
    """Build path to nsys report file."""
    name = f"benchmark__{run_tag}__{size}__{mode}__seq{seq_len}__warm5.nsys-rep"
    return NSYS_DIR / name


def _get_stats_with_filter(  # noqa: PLR0913 - All parameters needed for report generation
    run_tag: str,
    size: str,
    mode: str,
    seq_len: int,
    report_type: str,
    nvtx_range: str = "",
) -> str:
    """Get nsys stats output with optional NVTX filter."""
    path = _build_report_path(run_tag, size, mode, seq_len)
    nvtx_filter = f"{nvtx_range}/{NVTX_INSTANCE - 1}" if nvtx_range else ""
    return run_nsys_stats(path, report_type, nvtx_filter)


def _get_nvtx_stats(
    run_tag: str,
    size: str,
    mode: str = "FWD",
    seq_len: int = 128,
    nvtx_range: str = "",
) -> NvtxStats:
    """Get parsed NVTX GPU projection stats for a given configuration."""
    output = _get_stats_with_filter(
        run_tag, size, mode, seq_len, "nvtx_gpu_proj_sum", nvtx_range
    )
    return parse_nvtx_gpu_proj_sum(output)


def _get_kernels(
    run_tag: str,
    size: str,
    mode: str = "FWD",
    seq_len: int = 128,
    nvtx_range: str = "",
) -> list[KernelInfo]:
    """Get parsed kernel stats for a given configuration."""
    output = _get_stats_with_filter(
        run_tag, size, mode, seq_len, "cuda_gpu_kern_sum", nvtx_range
    )
    return parse_kernel_sum(output)


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


def _build_metric_table(
    run_tag: str,
    oom_status: OomStatus,
    metric_fn: Callable[[str, str, int, OomStatus], str],
) -> list[list[str]]:
    """Build a table of metrics for all model sizes and sequence lengths."""
    return [
        [size] + [metric_fn(run_tag, size, seq_len, oom_status) for seq_len in SEQ_LENS]
        for size in SIZES
    ]


def _analyze_forward_timing(run_tag: str, oom_status: OomStatus) -> str:
    """Analyze forward pass timing."""
    parts = ["#### (a) Forward Pass Timing\n"]
    rows = _build_metric_table(
        run_tag,
        oom_status,
        lambda rt, sz, sl, oom: _get_value_or_status(rt, sz, sl, "FWD", oom),
    )
    parts.append(_format_table_aligned("Forward Pass Avg Time (ms)", SEQ_HEADERS, rows))
    conclusions = (
        "The time of nsys is a "
        "little slower than python version. A reasonable explanation is profiling "
        "overhead and run-to-run variance; the NVTX ranges are measured under "
        "the profiler and can be slower even if the benchmark "
        "uses torch.cuda.synchronize().\n"
    )
    parts.append(conclusions)
    return "\n".join(parts)


def _analyze_non_gemm_kernels(run_tag: str, oom_status: OomStatus) -> str:
    """Analyze non-GEMM kernels for all models in compact table format."""
    parts = ["#### (c) Non-GEMM Kernels in Forward Pass\n"]

    for size in SIZES:
        headers = ["Seq Len", "Top Non-GEMM Kernel", "Time%", "Total", "Inst"]
        rows = []

        for seq_len in SEQ_LENS:
            if oom_status.get((size, seq_len, "FWD"), False):
                rows.append([f"seq{seq_len}", "OOM", "-", "-", "-"])
                continue

            kernels = _get_kernels(run_tag, size, "FWD", seq_len)
            non_gemm = [k for k in kernels if not k.get("is_gemm", False)]

            if non_gemm:
                top = non_gemm[0]
                kernel_name = str(top["name"])[:40] + "..."
                rows.append(
                    [
                        f"seq{seq_len}",
                        kernel_name,
                        str(top["pct"]),
                        str(top["total_time"]),
                        str(top["instances"]),
                    ]
                )
            else:
                rows.append([f"seq{seq_len}", "N/A", "-", "-", "-"])

        if rows:
            title = f"{size} Top Non-GEMM Kernel per Sequence Length"
            parts.append(_format_table_aligned(title, headers, rows))

    conclusion = (
        "The most time-consuming "
        "non-GEMM kernels are element wise copy, as seq length "
        "increases, elementwise kernels (activations, copy, etc.) take "
        "a larger proportion of time due to more data being processed.\n"
    )
    parts.append(conclusion)
    return "\n".join(parts)


def _analyze_gemm_fraction(run_tag: str, oom_status: OomStatus) -> str:
    """Analyze GEMM fraction in inference vs training for all models."""
    parts = ["#### (d) GEMM Fraction: Inference vs Training\n"]

    for size in SIZES:
        headers = ["Seq Len", "Inference", "Training"]
        rows = []

        for seq_len in SEQ_LENS:
            row = [f"seq{seq_len}"]
            # Inference (FWD mode)
            row.append(
                _get_gemm_value_or_status(run_tag, size, seq_len, "FWD", oom_status)
            )
            # Training (FWD_BWD mode)
            row.append(
                _get_gemm_value_or_status(run_tag, size, seq_len, "FWD_BWD", oom_status)
            )
            rows.append(row)

        if rows:
            parts.append(_format_table_aligned(f"{size} GEMM Fraction", headers, rows))

    conclusion1 = (
        "Non-GEMM kernels become more significant as sequence length increases."
    )
    conclusion2 = (
        "Training has a lower GEMM fraction than "
        "inference due to additional non-GEMM operations in the backward "
        "pass (e.g., elementwise ops for gradients)."
    )
    conclusion3 = (
        "As model size increases, GEMM fraction tends to "
        "increase because larger models have more compute-intensive matrix "
        "multiplications relative to non-GEMM ops."
    )
    parts.append(f"{conclusion1}\n\n{conclusion2}\n\n{conclusion3}\n")
    return "\n".join(parts)


def _get_attn_metric(
    run_tag: str,
    size: str,
    seq_len: int,
    oom_status: OomStatus,
    metric_type: str,
) -> str:
    """Get attention metric (softmax time, matmul time, or ratio) or status."""
    if oom_status.get((size, seq_len, "FWD"), False):
        return "OOM"

    stats = _get_nvtx_stats(run_tag, size, "FWD", seq_len, "forward")

    if metric_type == "softmax" and "softmax" in stats:
        return _format_ms(stats["softmax"]["total_ns"])

    if metric_type == "matmul" and "attn_scores" in stats and "final_matmul" in stats:
        matmul_ns = stats["attn_scores"]["total_ns"] + stats["final_matmul"]["total_ns"]
        return _format_ms(matmul_ns)

    if metric_type == "ratio":
        ratio = _calc_softmax_ratio(stats)
        if ratio is not None:
            return f"{ratio:.1%}"

    return "N/A"


def _analyze_attention_ops(run_tag: str, oom_status: OomStatus) -> str:
    """Analyze softmax vs matmul in attention for a single forward pass.

    Uses the iteration specified by NVTX_INSTANCE constant (default: 6th).
    """
    parts = [f"#### (e) Softmax vs Matmul in Attention (iteration {NVTX_INSTANCE})\n"]

    for metric_type, title in [
        ("softmax", "Softmax Time (ms)"),
        ("matmul", "Matmul Time (ms)"),
        ("ratio", "Softmax/Matmul Ratio"),
    ]:
        rows = [
            [size]
            + [
                _get_attn_metric(run_tag, size, seq_len, oom_status, metric_type)
                for seq_len in SEQ_LENS
            ]
            for size in SIZES
        ]
        parts.append(_format_table_aligned(title, SEQ_HEADERS, rows))

    conclusion1 = (
        "Bigger seq length, softmax time increases "
        "faster than matmul time, leading to a higher softmax/matmul ratio. This is "
        "because softmax involves more elementwise operations and reductions "
        "that scale quadratically with sequence length, while matmul benefits "
        "from optimized GPU kernels."
    )
    conclusion2 = (
        "For one head and one "
        "batch, softmax FLOPs per row is 5mn; across attention this is 5 x seq x seq. "
        "Computing attention scores is 2 x seq x head_dim x seq, and the final matmul "
        "is the same, so total matmul FLOPs is 4 x seq x head_dim x seq. The FLOPs "
        "ratio is (5 x seq x seq) : (4 x seq x head_dim x seq) = 5 : (4 x head_dim). "
        "In our medium config, head_dim = 64, so the ratio is 5 : 256, about 1.95%."
    )
    conclusion3 = """
```shell
m x (n-1)  get row max
m x n      minus max
m x n      exp
m x (n-1)  get sum
m x n      divide
```
"""
    conclusion4 = (
        "The time spent computing "
        "softmax is much higher than its FLOPs ratio, likely because softmax is "
        "elementwise and memory-bound (more memory traffic), while GEMM kernels "
        "are highly optimized and more compute-bound. A possible improvement is "
        "to use a fused kernel to avoid intermediate softmax stores/loads, trading "
        "a bit more compute for less memory access."
    )
    parts.append(f"{conclusion1}\n\n{conclusion2}\n\n{conclusion3}\n\n{conclusion4}\n")
    return "\n".join(parts)


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


def _get_gemm_value_or_status(
    run_tag: str, size: str, seq_len: int, mode: str, oom_status: OomStatus
) -> str:
    """Get GEMM percentage or status (OOM/N/A) for a configuration."""
    if oom_status.get((size, seq_len, mode), False):
        return "OOM"
    kernels = _get_kernels(run_tag, size, mode, seq_len)
    return f"{_calc_gemm_pct(kernels):.1f}%" if kernels else "N/A"


def _get_top_kernel_str(
    run_tag: str, size: str, nvtx_range: str, mode: str, seq_len: int
) -> str:
    """Get formatted string for top kernel in NVTX range."""
    kernels = _get_kernels(run_tag, size, mode, seq_len, nvtx_range)
    if kernels:
        top = kernels[0]
        return f"{top['name'][:25]}... ({top['pct']})"
    return "N/A"


def _analyze_single_pass_kernels(run_tag: str, oom_status: OomStatus) -> str:
    """Analyze kernels within single forward/backward pass in compact table format.

    Uses the iteration specified by NVTX_INSTANCE constant (default: 6th).
    """
    parts = [f"#### (b) Dominant Kernels (iteration {NVTX_INSTANCE})\n"]

    def get_kernel(mode: str, nvtx_range: str, seq_len: int) -> str:
        if oom_status.get((size, seq_len, mode), False):
            return "OOM"
        return _get_top_kernel_str(run_tag, size, nvtx_range, mode, seq_len)

    for size in SIZES:
        headers = [
            "Seq Len",
            "Inference Forward",
            "Training Forward",
            "Training Backward",
        ]
        rows = [
            [
                f"seq{seq_len}",
                get_kernel("FWD", "forward", seq_len),
                get_kernel("FWD_BWD", "forward", seq_len),
                get_kernel("FWD_BWD", "backward", seq_len),
            ]
            for seq_len in SEQ_LENS
        ]
        title = f"{size} Top Kernel per Pass"
        parts.append(_format_table_aligned(title, headers, rows))

    conclusion = (
        "The dominant kernel is "
        "always GEMM for matrix multiplication. Forward's GEMM ratio is larger "
        "than backward's because backward has more non-GEMM kernels (e.g., "
        "elementwise ops for gradients). As seq_len increases, non-GEMM "
        "kernels (e.g., elementwise ops) become more prominent due to more activations "
        "and intermediate results.\n"
    )
    parts.append(conclusion)
    return "\n".join(parts)


def analyze(run_tag: str) -> str:
    """Run analysis and return results as string."""
    parts = [
        "### Nsys Profile Analysis\n",
        f">based on nsys reports with run tag `{run_tag}`\n",
    ]

    oom_status = parse_oom_status(run_tag)
    parts.append(_analyze_forward_timing(run_tag, oom_status))
    parts.append(_analyze_single_pass_kernels(run_tag, oom_status))
    parts.append(_analyze_non_gemm_kernels(run_tag, oom_status))
    parts.append(_analyze_gemm_fraction(run_tag, oom_status))
    parts.append(_analyze_attention_ops(run_tag, oom_status))

    result = "\n".join(parts)
    print(result)
    return result


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

    result = analyze(args.run_tag)

    output_path = Path("docs/sections/nsys-profile.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result)
    print(f"\nMarkdown saved: {output_path}")
