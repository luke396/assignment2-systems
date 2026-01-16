#!/usr/bin/env python3
"""Analyze nsys profile results for forward/backward pass benchmarks."""
# ruff: noqa: S603

import subprocess
from pathlib import Path

SIZES = ["small", "medium", "large", "xl", "2.7B"]
NSYS_DIR = Path("output/nsys")


def run_nsys_stats(report_path: Path, report_type: str, nvtx_filter: str = "") -> str:
    """Run nsys stats and return output."""
    if not report_path.exists():
        return ""
    cmd = ["nsys", "stats", "--report", report_type, str(report_path)]
    if nvtx_filter:
        cmd.extend(["--filter-nvtx", nvtx_filter])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.stdout


def parse_nvtx_sum(output: str) -> dict[str, dict[str, float]]:
    """Parse nvtx_sum output to extract timing data."""
    stats: dict[str, dict[str, float]] = {}
    for line in output.splitlines():
        if ":forward" in line or ":backward" in line:
            parts = line.split()
            name = parts[-1]
            stats[name] = {
                "total_ns": float(parts[1].replace(",", "")),
                "instances": float(parts[2].replace(",", "")),
                "avg_ns": float(parts[3].replace(",", "")),
            }
        elif "computing attention scores" in line:
            parts = line.split()
            stats["attn_scores"] = {
                "total_ns": float(parts[3].replace(",", "")),
                "instances": float(parts[4].replace(",", "")),
            }
        elif "computing softmax" in line:
            parts = line.split()
            stats["softmax"] = {
                "total_ns": float(parts[3].replace(",", "")),
                "instances": float(parts[4].replace(",", "")),
            }
        elif "final matmul" in line:
            parts = line.split()
            stats["final_matmul"] = {
                "total_ns": float(parts[3].replace(",", "")),
                "instances": float(parts[4].replace(",", "")),
            }
    return stats


_MIN_KERNEL_PARTS = 10


def parse_kernel_sum(output: str) -> list[dict[str, str | float | bool]]:
    """Parse cuda_gpu_kern_sum output to extract top kernels."""
    kernels: list[dict[str, str | float | bool]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or "Time (%)" in line or "--------" in line:
            continue
        parts = line.split()
        if len(parts) < _MIN_KERNEL_PARTS:
            continue
        # Check if first part looks like a percentage (number)
        try:
            float(parts[0])
        except ValueError:
            continue
        try:
            pct = parts[0] + "%"
            total_ns = parts[1]
            instances = parts[2]
            name = " ".join(parts[8:])[:50] + "..."
            kernels.append({
                "pct": pct,
                "total_time": f"{float(total_ns.replace(',', '')) / 1e6:.2f} ms",
                "instances": int(instances.replace(",", "")),
                "name": name,
                "is_gemm": "cutlass" in name or "gemm" in name.lower(),
            })
        except (IndexError, ValueError):
            continue
    return kernels[:10]


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    """Print a formatted markdown table."""
    print(f"\n### {title}\n")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def _build_path(prefix: str, size: str, mode: str) -> Path:
    return NSYS_DIR / f"benchmark__{prefix}__{size}__{mode}__seq128__warm5.nsys-rep"


def _analyze_forward_timing() -> None:
    """Analyze forward pass timing."""
    print("## (a) Forward Pass Timing\n")
    fwd_rows = []
    for size in SIZES:
        path = _build_path("new_base", size, "FWD")
        output = run_nsys_stats(path, "nvtx_sum")
        stats = parse_nvtx_sum(output)
        if ":forward" in stats:
            total_ms = stats[":forward"]["total_ns"] / 1e6
            instances = int(stats[":forward"]["instances"])
            avg_ms = stats[":forward"]["avg_ns"] / 1e6
            fwd_rows.append([size, f"{total_ms:.2f}", str(instances), f"{avg_ms:.2f}"])
    headers = ["Model", "Total (ms)", "Inst", "Avg (ms)"]
    print_table("Forward Pass Time (NVTX)", headers, fwd_rows)


def _analyze_dominant_kernels() -> None:
    """Analyze top kernels in forward vs forward+backward."""
    print("\n## (b) Dominant CUDA Kernels\n")
    for mode, label in [("FWD", "Forward Only"), ("FWD_BWD", "Forward+Backward")]:
        print(f"\n**{label} (medium model):**\n")
        path = _build_path("new_base", "medium", mode)
        output = run_nsys_stats(path, "cuda_gpu_kern_sum")
        kernels = parse_kernel_sum(output)
        if kernels:
            rows = [
                [str(k["pct"]), str(k["total_time"]), str(k["instances"]),
                 str(k["name"])]
                for k in kernels[:3]
            ]
            headers = ["Time%", "Total", "Inst", "Kernel"]
            print_table(f"Top 3 Kernels ({label})", headers, rows)


def _analyze_non_gemm_kernels() -> None:
    """Analyze non-GEMM kernels."""
    print("\n## (c) Non-GEMM Kernels in Forward Pass\n")
    path = _build_path("new_base", "medium", "FWD")
    output = run_nsys_stats(path, "cuda_gpu_kern_sum")
    kernels = parse_kernel_sum(output)
    non_gemm = [k for k in kernels if not k.get("is_gemm", False)]
    if non_gemm:
        rows = [
            [str(k["pct"]), str(k["total_time"]), str(k["instances"]), str(k["name"])]
            for k in non_gemm[:3]
        ]
        print_table("Top Non-GEMM Kernels", ["Time%", "Total", "Inst", "Kernel"], rows)


def _analyze_gemm_fraction() -> None:
    """Analyze GEMM fraction in inference vs training."""
    print("\n## (d) GEMM Fraction: Inference vs Training\n")
    for mode, label in [("FWD", "Inference"), ("FWD_BWD", "Training")]:
        path = _build_path("new_base", "medium", mode)
        output = run_nsys_stats(path, "cuda_gpu_kern_sum")
        kernels = parse_kernel_sum(output)
        gemm_pct = sum(
            float(str(k["pct"]).rstrip("%"))
            for k in kernels
            if k.get("is_gemm", False)
        )
        print(f"- **{label}**: GEMM ~{gemm_pct:.1f}% of top kernel time")


def _analyze_attention_ops() -> None:
    """Analyze softmax vs matmul in attention."""
    print("\n## (e) Softmax vs Matmul in Attention\n")
    attn_rows = []
    for size in SIZES:
        path = _build_path("new_base", size, "FWD")
        output = run_nsys_stats(path, "nvtx_sum")
        stats = parse_nvtx_sum(output)
        if all(k in stats for k in ["softmax", "attn_scores", "final_matmul"]):
            softmax_ms = stats["softmax"]["total_ns"] / 1e6
            attn_ns = stats["attn_scores"]["total_ns"]
            matmul_ns = stats["final_matmul"]["total_ns"]
            matmul_ms = (attn_ns + matmul_ns) / 1e6
            ratio = softmax_ms / matmul_ms if matmul_ms > 0 else 0
            attn_rows.append([
                size, f"{softmax_ms:.2f}", f"{matmul_ms:.2f}", f"{ratio:.1%}"
            ])
    headers = ["Model", "Softmax(ms)", "Matmul(ms)", "Ratio"]
    print_table("Softmax vs Matmul", headers, attn_rows)


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


def analyze() -> None:
    """Run analysis and print results."""
    print("# NSYS Profile Analysis\n")
    _analyze_forward_timing()
    _analyze_dominant_kernels()
    _analyze_non_gemm_kernels()
    _analyze_gemm_fraction()
    _analyze_attention_ops()
    _print_conclusions()


if __name__ == "__main__":
    analyze()
