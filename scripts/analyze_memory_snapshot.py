"""Analyze PyTorch CUDA memory snapshots from .snap files."""

import argparse
import pickle  # Required: PyTorch memory snapshots use pickle format
import re
from pathlib import Path

from torch.cuda._memory_viz import trace_plot

from scripts.constants import SIZES
from scripts.table_utils import format_table

MEMORY_DIR = Path("output/memory")
HTML_DIR = MEMORY_DIR / "html"

# Type alias for snapshot key: (size, mode, seq_len)
SnapshotKey = tuple[str, str, int]


def _size_sort_key(size: str) -> int:
    """Sort key for model sizes based on SIZES order."""
    return SIZES.index(size) if size in SIZES else 999


def parse_snap_filename(filename: str) -> dict[str, str | int] | None:
    """Parse snap filename to extract metadata.

    Expected format: {timestamp}__{size}__{mode}__seq{seq_len}__warm{n}.snap
    Example: 20260125_153755__2.7B__FWD__seq128__warm5.snap
             20260125_153755__2.7B__FWD_BWD__seq128__warm5.snap
    """
    pattern = r"(\d+_\d+)__([^_]+)__(FWD(?:_BWD)?)__seq(\d+)__warm(\d+)\.snap"
    match = re.match(pattern, filename)
    if not match:
        return None
    return {
        "timestamp": match.group(1),
        "size": match.group(2),
        "mode": match.group(3),
        "seq_len": int(match.group(4)),
        "warmup": int(match.group(5)),
    }


def parse_autocast_filename(filename: str) -> dict[str, str | int] | None:
    """Parse autocast snap filename to extract metadata.

    Expected format: {prefix}__{size}__{mode}__seq{seq_len}__warm{n}.snap
    Example: autocast-a800__2.7B__FWD__seq128__warm5.snap
             autocast-a800__2.7B__FWD_BWD__seq128__warm5.snap
    """
    pattern = r"(autocast-[^_]+)__([^_]+)__(FWD(?:_BWD)?)__seq(\d+)__warm(\d+)\.snap"
    match = re.match(pattern, filename)
    if not match:
        return None
    return {
        "prefix": match.group(1),
        "size": match.group(2),
        "mode": match.group(3),
        "seq_len": int(match.group(4)),
        "warmup": int(match.group(5)),
    }


def load_snapshot(path: Path) -> dict:
    """Load a pickle snapshot file."""
    try:
        with path.open("rb") as f:
            return pickle.load(f)  # noqa: S301 - PyTorch snapshots require pickle
    except (FileNotFoundError, pickle.PickleError) as e:
        msg = f"Failed to load snapshot {path}: {e}"
        raise RuntimeError(msg) from e


def calc_peak_memory(data: dict) -> int:
    """Calculate peak memory from snapshot.

    Uses the sum of all segment total_size, which represents
    the actual GPU memory allocated by CUDA at snapshot time.
    """
    segments = data.get("segments", [])
    return sum(seg.get("total_size", 0) for seg in segments)


def format_bytes(n: int) -> str:
    """Format bytes as human-readable string (GB)."""
    return f"{n / 1024**3:.2f} GB"


def generate_html(data: dict, output_path: Path) -> None:
    """Generate HTML visualization from snapshot data."""
    html_content = trace_plot(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content)


def generate_html_for_snapshots(
    snapshots: dict[SnapshotKey, Path],
    html_dir: Path,
    prefix: str = "",
) -> list[Path]:
    """Generate HTML files for all snapshots.

    Args:
        snapshots: Dict mapping (size, mode, seq_len) to snapshot file path
        html_dir: Directory to save HTML files
        prefix: Optional prefix for HTML filenames

    Returns:
        List of generated HTML file paths

    """
    generated: list[Path] = []
    html_dir.mkdir(parents=True, exist_ok=True)

    for (size, mode, seq_len), snap_path in snapshots.items():
        if prefix:
            name_parts = [prefix, size, mode, f"seq{seq_len}"]
        else:
            name_parts = [size, mode, f"seq{seq_len}"]
        html_name = "__".join(name_parts) + ".html"
        html_path = html_dir / html_name

        print(f"Generating HTML for {size} {mode} seq={seq_len}...")
        data = load_snapshot(snap_path)
        generate_html(data, html_path)
        generated.append(html_path)

    return generated


def get_latest_snapshots(
    memory_dir: Path, run_tag: str | None = None
) -> dict[SnapshotKey, Path]:
    """Get the latest snapshot for each (size, mode, seq_len) combination.

    Args:
        memory_dir: Directory containing .snap files
        run_tag: If provided, only include files with this timestamp prefix

    Returns:
        Dict mapping (size, mode, seq_len) to file path.

    """
    snapshots: dict[SnapshotKey, tuple[str, Path]] = {}

    for snap_file in memory_dir.glob("*.snap"):
        meta = parse_snap_filename(snap_file.name)
        if not meta:
            continue

        timestamp = str(meta["timestamp"])
        if run_tag and not timestamp.startswith(run_tag):
            continue

        key = (str(meta["size"]), str(meta["mode"]), int(meta["seq_len"]))

        if key not in snapshots or timestamp > snapshots[key][0]:
            snapshots[key] = (timestamp, snap_file)

    return {k: v[1] for k, v in snapshots.items()}


def get_autocast_snapshots(
    memory_dir: Path, prefix: str = "autocast-a800"
) -> dict[SnapshotKey, Path]:
    """Get autocast snapshot for each (size, mode, seq_len) combination.

    Args:
        memory_dir: Directory containing .snap files
        prefix: Autocast prefix to match (e.g., "autocast-a800")

    Returns:
        Dict mapping (size, mode, seq_len) to file path.

    """
    snapshots: dict[SnapshotKey, Path] = {}

    for snap_file in memory_dir.glob("*.snap"):
        meta = parse_autocast_filename(snap_file.name)
        if not meta:
            continue

        if str(meta["prefix"]) != prefix:
            continue

        key = (str(meta["size"]), str(meta["mode"]), int(meta["seq_len"]))
        snapshots[key] = snap_file

    return snapshots


def _calc_peak_memory_for_snapshots(
    snapshots: dict[SnapshotKey, Path],
) -> dict[SnapshotKey, int]:
    """Calculate peak memory for each snapshot in the dict."""
    return {
        key: calc_peak_memory(load_snapshot(path)) for key, path in snapshots.items()
    }


def analyze_all_snapshots(
    memory_dir: Path, run_tag: str | None = None
) -> dict[SnapshotKey, int]:
    """Analyze all snapshots and return peak memory for each config."""
    return _calc_peak_memory_for_snapshots(get_latest_snapshots(memory_dir, run_tag))


def analyze_autocast_snapshots(
    memory_dir: Path, prefix: str = "autocast-a800"
) -> dict[SnapshotKey, int]:
    """Analyze autocast snapshots and return peak memory for each config."""
    return _calc_peak_memory_for_snapshots(get_autocast_snapshots(memory_dir, prefix))


def build_combined_table(
    results: dict[SnapshotKey, int],
    sizes: list[str],
    seq_lens: list[int],
) -> list[list[str]]:
    """Build combined table with FWD and FWD_BWD in same row."""
    rows = []
    for size in sizes:
        for seq_len in seq_lens:
            fwd_key = (size, "FWD", seq_len)
            bwd_key = (size, "FWD_BWD", seq_len)

            fwd_val = format_bytes(results[fwd_key]) if fwd_key in results else "N/A"
            bwd_val = format_bytes(results[bwd_key]) if bwd_key in results else "N/A"

            rows.append([size, str(seq_len), fwd_val, bwd_val])
    return rows


def _format_delta(base: int | None, other: int | None) -> str:
    """Format memory delta with percentage change."""
    if not base or not other:
        return "N/A"
    if base == 0:
        return "N/A"
    delta = other - base
    pct = (delta / base) * 100
    return f"{delta / 1024**3:+.2f} GB ({pct:+.1f}%)"


def build_comparison_table(
    baseline: dict[SnapshotKey, int],
    autocast: dict[SnapshotKey, int],
    sizes: list[str],
    seq_lens: list[int],
) -> list[list[str]]:
    """Build comparison table between baseline and autocast results."""
    rows = []
    for size in sizes:
        for seq_len in seq_lens:
            fwd_key = (size, "FWD", seq_len)
            bwd_key = (size, "FWD_BWD", seq_len)

            base_fwd = baseline.get(fwd_key)
            auto_fwd = autocast.get(fwd_key)
            base_bwd = baseline.get(bwd_key)
            auto_bwd = autocast.get(bwd_key)

            rows.append(
                [
                    size,
                    str(seq_len),
                    format_bytes(base_fwd) if base_fwd else "N/A",
                    format_bytes(auto_fwd) if auto_fwd else "N/A",
                    _format_delta(base_fwd, auto_fwd),
                    format_bytes(base_bwd) if base_bwd else "N/A",
                    format_bytes(auto_bwd) if auto_bwd else "N/A",
                    _format_delta(base_bwd, auto_bwd),
                ]
            )
    return rows


AUTOCAST_ANALYSIS_TEXT = """

Autocast increases memory usage in most cases, contrary to expectations.

Root Cause Analysis: The custom implementations in `cs336_basics/blocks.py`
prevent autocast from providing benefits:

1. **Custom `Linear` using `einsum`** (blocks.py:48-50)
   - `einsum` cannot trigger cuBLAS Tensor Core optimizations
   - PyTorch's `torch.matmul` or `F.linear` would automatically use
     Tensor Cores under autocast

2. **Custom `scaled_dot_product_attention`** (blocks.py:289-321)
   - Uses manual `einsum` for Q·K^T and attention·V computations
   - Does not use `F.scaled_dot_product_attention` which provides
     FlashAttention/Memory-Efficient Attention

3. **Custom `softmax`** (blocks.py:281-286)
   - Manual implementation without optimized CUDA kernels
   - `F.softmax` has fused CUDA implementations

4. **Type Conversion Overhead**
   - Since custom ops are not recognized as "autocast-eligible",
     frequent fp32 ↔ bf16 conversions occur
   - Each `einsum` call may trigger unnecessary dtype casts
   - Extra memory is allocated for intermediate tensors in different
     precisions

Expected vs Actual Flow:

```
Expected (with native PyTorch ops):
  Input(fp32) → auto-cast to bf16 → Tensor Core compute → Output(bf16)
  Memory: reduced by ~50% for activations

Actual (with custom einsum ops):
  Input(fp32) → cast bf16 → einsum(no Tensor Core) → cast fp32 → ...
  Memory: increased due to duplicate tensors in both precisions
```

Conclusion:

To benefit from autocast, the model should use PyTorch native operations:
- Replace `einsum` in `Linear` with `F.linear` or `x @ weight.T`
- Replace custom attention with `F.scaled_dot_product_attention`
- Replace custom `softmax` with `F.softmax`

However, since this is a course assignment (cs336), the custom
implementations are intentional for educational purposes.
"""


def analyze(run_tag: str | None = None, *, gen_html: bool = False) -> str:
    """Generate memory analysis report from snapshot files."""
    if not MEMORY_DIR.exists():
        return "Memory snapshot directory not found.\n"

    snapshots = get_latest_snapshots(MEMORY_DIR, run_tag)
    if not snapshots:
        return "No snapshot files found.\n"

    results = _calc_peak_memory_for_snapshots(snapshots)

    if gen_html:
        prefix = run_tag or "baseline"
        html_files = generate_html_for_snapshots(snapshots, HTML_DIR, prefix)
        print(f"\nGenerated {len(html_files)} HTML files in {HTML_DIR}")

    available_sizes = sorted({k[0] for k in results}, key=_size_sort_key)
    available_seq_lens = sorted({k[2] for k in results})

    headers = [
        "Model Size",
        "Sequence Length",
        "Forward Only Peak Memory (GB)",
        "Forward + Backward Peak Memory (GB)",
    ]

    rows = build_combined_table(results, available_sizes, available_seq_lens)
    result = format_table("#### (b) Table of peak memory usage", headers, rows)
    print(result)
    return result


def analyze_autocast(
    prefix: str = "autocast-a800", *, gen_html: bool = False
) -> str:
    """Generate autocast memory analysis report from snapshot files."""
    if not MEMORY_DIR.exists():
        return "Memory snapshot directory not found.\n"

    snapshots = get_autocast_snapshots(MEMORY_DIR, prefix)
    if not snapshots:
        return "No autocast snapshot files found.\n"

    results = _calc_peak_memory_for_snapshots(snapshots)
    baseline_results = analyze_all_snapshots(MEMORY_DIR, run_tag=None)

    if gen_html:
        html_files = generate_html_for_snapshots(snapshots, HTML_DIR, prefix)
        print(f"\nGenerated {len(html_files)} HTML files in {HTML_DIR}")

    available_sizes = sorted({k[0] for k in results}, key=_size_sort_key)
    available_seq_lens = sorted({k[2] for k in results})

    # Build main autocast table
    headers = [
        "Model Size",
        "Sequence Length",
        "Forward Only Peak Memory (GB)",
        "Forward + Backward Peak Memory (GB)",
    ]
    rows = build_combined_table(results, available_sizes, available_seq_lens)
    main_table = format_table(
        "#### (c) Table of autocast peak memory usage", headers, rows
    )

    # Build comparison table
    comparison_headers = [
        "Model Size",
        "Seq Len",
        "Baseline FWD",
        "Autocast FWD",
        "Δ FWD",
        "Baseline FWD+BWD",
        "Autocast FWD+BWD",
        "Δ FWD+BWD",
    ]
    comparison_rows = build_comparison_table(
        baseline_results, results, available_sizes, available_seq_lens
    )
    comparison_table = format_table(
        "##### Comparison: Baseline vs Autocast", comparison_headers, comparison_rows
    )

    result = main_table + "\n" + comparison_table + AUTOCAST_ANALYSIS_TEXT
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze PyTorch CUDA memory snapshots"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML visualizations for each snapshot",
    )
    args = parser.parse_args()

    # Output paths
    baseline_path = Path("docs/sections/memory-profile-(b).md")
    autocast_path = Path("docs/sections/memory-profile-(c).md")

    # Run baseline analysis (b)
    print("=== Analyzing baseline snapshots ===")
    baseline_result = analyze(run_tag=None, gen_html=args.html)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(baseline_result)
    print(f"Saved: {baseline_path}\n")

    # Run autocast analysis (c)
    print("=== Analyzing autocast snapshots ===")
    autocast_result = analyze_autocast(prefix="autocast-a800", gen_html=args.html)
    autocast_path.write_text(autocast_result)
    print(f"Saved: {autocast_path}")
