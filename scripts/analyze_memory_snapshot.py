"""Analyze PyTorch CUDA memory snapshots from .snap files."""

import argparse
import pickle  # Required: PyTorch memory snapshots use pickle format
import re
from pathlib import Path

from scripts.constants import SIZES
from scripts.table_utils import format_table

MEMORY_DIR = Path("output/memory")


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


def load_snapshot(path: Path) -> dict:
    """Load a pickle snapshot file."""
    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301


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


def get_latest_snapshots(
    memory_dir: Path, run_tag: str | None = None
) -> dict[tuple[str, str, int], Path]:
    """Get the latest snapshot for each (size, mode, seq_len) combination.

    Args:
        memory_dir: Directory containing .snap files
        run_tag: If provided, only include files with this timestamp prefix

    Returns:
        Dict mapping (size, mode, seq_len) to file path.

    """
    snapshots: dict[tuple[str, str, int], tuple[str, Path]] = {}

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


def analyze_all_snapshots(
    memory_dir: Path, run_tag: str | None = None
) -> dict[tuple[str, str, int], int]:
    """Analyze all snapshots and return peak memory for each config."""
    snapshots = get_latest_snapshots(memory_dir, run_tag)
    results: dict[tuple[str, str, int], int] = {}

    for key, path in snapshots.items():
        data = load_snapshot(path)
        peak = calc_peak_memory(data)
        results[key] = peak

    return results


def build_combined_table(
    results: dict[tuple[str, str, int], int],
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


def analyze(run_tag: str | None = None) -> str:
    """Generate memory analysis report from snapshot files."""
    if not MEMORY_DIR.exists():
        return "Memory snapshot directory not found.\n"

    results = analyze_all_snapshots(MEMORY_DIR, run_tag)
    if not results:
        return "No snapshot files found.\n"

    # Detect available sizes and seq_lens from results
    def sort_key(s: str) -> int:
        return SIZES.index(s) if s in SIZES else 999

    available_sizes = sorted({k[0] for k in results}, key=sort_key)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze PyTorch CUDA memory snapshots"
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Run tag to analyze (currently unused, reserved for future)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/sections/memory-profile-(b).md",
        help="Output markdown file path",
    )
    args = parser.parse_args()

    result = analyze(args.run_tag)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result)
    print(f"\nMarkdown saved: {output_path}")
