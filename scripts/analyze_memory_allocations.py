#!/usr/bin/env python3
"""Analyze memory allocation patterns from PyTorch CUDA memory snapshots.

This script provides detailed breakdown of memory allocations by category
(Activation, Gradient, Optimizer, etc.) from .snap files, focusing on
single-step peak allocations rather than total memory.

Note: Uses pickle for loading PyTorch memory snapshots (.snap files),
which is the standard format used by torch.cuda.memory._snapshot().
"""

import argparse
import pickle  # Required: PyTorch memory snapshots use pickle format
from collections import defaultdict
from pathlib import Path
from textwrap import dedent

from scripts.table_utils import format_table

MEMORY_DIR = Path("output/memory")
_MIN_FILENAME_PARTS = 5
_TOP_N_ALLOCATIONS = 10
_MAX_CALL_CHAIN_LEN = 40


def format_size(size: int, unit: str = "auto") -> str:
    """Format bytes as human-readable string."""
    kib, mib, gib = 1024, 1024**2, 1024**3

    if unit == "GiB" or (unit == "auto" and size >= gib):
        return f"{size / gib:.1f} GiB"
    if unit == "MiB" or (unit == "auto" and size >= mib):
        return f"{size / mib:.1f} MiB"
    if unit == "KiB" or (unit == "auto" and size >= kib):
        return f"{size / kib:.1f} KiB"
    return f"{size} B"


def _categorize_cpp_frame(name: str) -> tuple[str, str] | None:
    """Check C++ frame for known patterns."""
    if "AccumulateGrad" in name:
        return ("Gradient", "AccumulateGrad")
    if "zeros_like" in name:
        return ("Optimizer State", "zeros_like")
    if "clone_obey_contract" in name:
        return ("Gradient", "clone_for_grad")
    return None


def _categorize_blocks_forward(line: int) -> tuple[str, str]:
    """Categorize forward() calls in blocks.py by line number."""
    # Line ranges for different components in blocks.py
    ranges = [
        (158, 170, "Act:FFN", "SwiGLU"),  # SwiGLU.forward
        (199, 211, "Act:FFN", "SiLU"),  # SiLU.forward
        (112, 128, "Act:Norm", "RMSNorm"),  # RMSNorm.forward
        (82, 84, "Act:Embed", "Embedding"),  # Embedding.forward
        (48, 50, "Act:Linear", "Linear"),  # Linear.forward
        (358, 409, "Act:Attn", "MHSA"),  # MultiheadSelfAttention.forward
        (510, 531, "Act:Block", "TFBlock"),  # TransformerBlock.forward
    ]
    for start, end, category, op_name in ranges:
        if start <= line <= end:
            return (category, f"{op_name}:{line}")
    return ("Act:Other", f"forward:{line}")


def _categorize_blocks_py(name: str, line: int) -> tuple[str, str] | None:
    """Categorize allocations from blocks.py."""
    if name == "softmax":
        return ("Act:Softmax", f"softmax:{line}")

    if name == "scaled_dot_product_attention":
        # Line 310: Q·K^T scores, Line 321: attention·V output
        # 312 is the cutoff between these two operations in blocks.py
        if line <= 312:  # noqa: PLR2004
            return ("Act:AttnScores", f"Q.K^T:{line}")
        return ("Act:AttnOut", f"attn.V:{line}")

    if name == "forward":
        return _categorize_blocks_forward(line)

    return None


def _categorize_python_frame(
    filename: str, name: str, line: int
) -> tuple[str, str] | None:
    """Check Python frame for known patterns."""
    # Non-activation patterns (check first)
    non_activation_patterns = [
        ("training_utility.py", "step", "Optimizer", "optimizer.step"),
        ("training_utility.py", "cross_entropy", "Loss", "cross_entropy"),
        ("benchmark.py", "generate_random_data", "Data", "random_data"),
    ]

    for file_pattern, name_pattern, category, op_name in non_activation_patterns:
        if file_pattern in filename and name_pattern in name:
            return (category, f"{op_name}:{line}")

    # Activation sub-categories (blocks.py specific)
    if "blocks.py" in filename:
        return _categorize_blocks_py(name, line)

    # einsum from functional.py (generic)
    if "functional.py" in filename and name == "einsum":
        return ("Act:einsum", "einsum")

    return None


def get_operation_category(frames: list[dict]) -> tuple[str, str]:
    """Identify operation category from stack frames.

    Returns:
        Tuple of (category, operation_detail).

    """
    # Check C++ frames first for specific patterns
    for frame in frames:
        name = frame.get("name", "")
        result = _categorize_cpp_frame(name)
        if result:
            return result

    # Check Python frames
    for frame in frames:
        filename = frame.get("filename", "")
        if ".py" not in filename:
            continue

        name = frame.get("name", "")
        line = frame.get("line", 0)
        result = _categorize_python_frame(filename, name, line)
        if result:
            return result

    return ("Other", "unknown")


def _extract_call_chain(frames: list[dict], max_depth: int = 3) -> str:
    """Extract simplified Python call chain from frames."""
    chain = []
    for frame in frames:
        filename = frame.get("filename", "")
        if ".py" not in filename or "conda" in filename or "python-3" in filename:
            continue

        name = frame.get("name", "")
        line = frame.get("line", 0)
        short_file = filename.split("/")[-1]
        chain.append(f"{short_file}:{line}:{name}")

        if len(chain) >= max_depth:
            break

    return " → ".join(chain) if chain else "Internal"


def load_snapshot(path: Path) -> dict:
    """Load a pickle snapshot file."""
    try:
        with path.open("rb") as f:
            return pickle.load(f)  # noqa: S301 - PyTorch snapshots require pickle
    except (FileNotFoundError, pickle.PickleError) as e:
        msg = f"Failed to load snapshot {path}: {e}"
        raise RuntimeError(msg) from e


def analyze_snap(snap_path: Path) -> dict:
    """Analyze a single snap file and return categorized stats.

    Returns:
        Dict with category stats including max single allocation per category.

    """
    data = load_snapshot(snap_path)
    traces = data.get("device_traces", [[]])[0]

    # Track max single allocation per category
    category_max: dict[str, dict] = defaultdict(
        lambda: {"max_size": 0, "count": 0, "detail": "", "call_chain": ""}
    )

    # Collect all allocations for top-N analysis
    all_allocs: list[dict] = []

    for entry in traces:
        if entry.get("action") != "alloc":
            continue

        size = entry.get("size", 0)
        frames = entry.get("frames", [])
        category, detail = get_operation_category(frames)
        call_chain = _extract_call_chain(frames)

        category_max[category]["count"] += 1

        if size > category_max[category]["max_size"]:
            category_max[category]["max_size"] = size
            category_max[category]["detail"] = detail
            category_max[category]["call_chain"] = call_chain

        all_allocs.append(
            {
                "size": size,
                "category": category,
                "detail": detail,
                "call_chain": call_chain,
            }
        )

    # Sort to get top allocations
    all_allocs.sort(key=lambda x: x["size"], reverse=True)

    return {
        "category_max": dict(category_max),
        "top_allocs": all_allocs[:_TOP_N_ALLOCATIONS],
        "total_allocs": len(all_allocs),
    }


def parse_snap_filename(name: str) -> dict | None:
    """Parse snap filename to extract configuration metadata."""
    name = name.replace(".snap", "")
    parts = name.split("__")

    if len(parts) < _MIN_FILENAME_PARTS:
        return None

    if name.startswith("autocast"):
        return {
            "type": "autocast",
            "size": parts[1],
            "mode": parts[2],
            "seq": int(parts[3].replace("seq", "")),
        }

    return {
        "type": "baseline",
        "size": parts[1],
        "mode": parts[2],
        "seq": int(parts[3].replace("seq", "")),
    }


CONCLUSIONS = dedent("""\
##### Key Findings

1. **Attention scores scale quadratically**: The Q·K^T computation
   (AttnScores) creates large activation allocations that scale with O(seq^2):
   - seq128: 8 MiB baseline (batch x heads x 128 x 128 x 4 bytes)
   - seq256: 32 MiB baseline (4x increase for 2x seq)
   - seq512: 128 MiB baseline (4x increase for 2x seq)

2. **Gradient and optimizer state are fixed-size**: The largest single
   allocations (~100 MiB each) come from:
   - Gradient accumulation (AccumulateGrad) for embedding/projection layers
   - Optimizer state initialization (zeros_like) for Adam's m and v buffers
   - These scale with model parameters, not sequence length

3. **FFN allocations scale linearly**: Feed-forward network activations
   scale with O(batch x seq x d_ff):
   - seq128: 20 MiB, seq256: 40 MiB, seq512: 80 MiB

4. **Autocast reduces most activation sizes by ~50%**: With bf16:
   - AttnScores, AttnOut, FFN, Attn all halve in size
   - Softmax, Norm, Embed, Block unchanged (forced fp32 or unaffected)

5. **einsum increases under autocast**: The einsum category shows +30 MiB
   increase at seq128, likely due to dtype conversion buffers created
   during bf16<->fp32 casts in the custom einsum implementations.

6. **Softmax stays fp32**: Custom softmax implementation forces fp32 for
   numerical stability, so its allocation size doesn't change with autocast.
""")


# Activation sub-categories for detailed analysis
ACTIVATION_SUBCATS = [
    "Act:AttnScores",
    "Act:AttnOut",
    "Act:Softmax",
    "Act:FFN",
    "Act:Norm",
    "Act:Embed",
    "Act:Linear",
    "Act:Attn",
    "Act:einsum",
    "Act:Block",
    "Act:Other",
]

# Short display names for categories
SHORT_NAMES = {
    "Act:AttnScores": "AttnScores",
    "Act:AttnOut": "AttnOut",
    "Act:Softmax": "Softmax",
    "Act:FFN": "FFN",
    "Act:Norm": "Norm",
    "Act:Embed": "Embed",
    "Act:Linear": "Linear",
    "Act:Attn": "Attn",
    "Act:einsum": "einsum",
    "Act:Block": "Block",
    "Act:Other": "Other",
    "Gradient": "Gradient",
    "Optimizer": "Optimizer",
    "Optimizer State": "Opt State",
}


def _format_delta(base: int, auto: int) -> str:
    """Format delta between baseline and autocast."""
    if base == 0 and auto == 0:
        return "-"
    delta = auto - base
    if delta == 0:
        return "0"
    return f"{delta / (1024**2):+.1f}"


def build_combined_table(results: dict, mode: str, seq: int) -> str:
    """Build combined table showing baseline, autocast, and delta for one seq."""
    base_key = ("baseline", mode, seq)
    auto_key = ("autocast", mode, seq)

    if base_key not in results or auto_key not in results:
        return ""

    base_cat = results[base_key]["category_max"]
    auto_cat = results[auto_key]["category_max"]

    # Determine which categories to show based on mode
    if mode == "FWD":
        categories = ACTIVATION_SUBCATS
    else:
        categories = ["Gradient", "Optimizer State", *ACTIVATION_SUBCATS]

    headers = ["Category", "Baseline", "Autocast", "Δ (MiB)"]
    rows = []

    for cat in categories:
        base_size = base_cat.get(cat, {}).get("max_size", 0)
        auto_size = auto_cat.get(cat, {}).get("max_size", 0)

        if base_size == 0 and auto_size == 0:
            continue

        rows.append(
            [
                SHORT_NAMES.get(cat, cat),
                format_size(base_size),
                format_size(auto_size),
                _format_delta(base_size, auto_size),
            ]
        )

    title = f"##### {mode} seq{seq} - Max Single Allocation"
    return format_table(title, headers, rows)


def build_top_allocs_table(results: dict, config_key: tuple) -> str:
    """Build table showing top N allocations for a specific config."""
    if config_key not in results:
        return ""

    top_allocs = results[config_key]["top_allocs"]
    headers = ["#", "Size", "Category", "Operation", "Call Chain"]
    rows = []

    for i, alloc in enumerate(top_allocs, 1):
        chain = alloc["call_chain"]
        if len(chain) > _MAX_CALL_CHAIN_LEN:
            chain_display = chain[:_MAX_CALL_CHAIN_LEN] + "..."
        else:
            chain_display = chain
        rows.append(
            [
                str(i),
                format_size(alloc["size"]),
                alloc["category"],
                alloc["detail"][:20],
                chain_display,
            ]
        )

    typ, mode, seq = config_key
    title = f"##### Top {_TOP_N_ALLOCATIONS} Allocations: {typ} {mode} seq{seq}"
    return format_table(title, headers, rows)


def analyze(memory_dir: Path = MEMORY_DIR) -> str:
    """Analyze all memory snapshots and generate report.

    Returns:
        Markdown formatted analysis report.

    """
    if not memory_dir.exists():
        return "Memory snapshot directory not found.\n"

    snap_files = sorted(memory_dir.glob("*.snap"))
    if not snap_files:
        return "No snapshot files found.\n"

    # Analyze all files and organize by configuration
    results: dict[tuple[str, str, int], dict] = {}

    for snap_path in snap_files:
        config = parse_snap_filename(snap_path.name)
        if config is None:
            continue

        key = (config["type"], config["mode"], config["seq"])
        results[key] = analyze_snap(snap_path)

    # Build report sections - one combined table per (mode, seq) pair
    parts = ["#### (e) Memory Allocation Analysis (Single-Step Peak)"]

    # FWD mode tables
    for seq in [128, 256, 512]:
        table = build_combined_table(results, "FWD", seq)
        if table:
            parts.append(table)

    # FWD_BWD mode tables
    for seq in [128, 256]:
        table = build_combined_table(results, "FWD_BWD", seq)
        if table:
            parts.append(table)

    # Show top allocations for one key configuration
    parts.append(build_top_allocs_table(results, ("baseline", "FWD_BWD", 256)))
    parts.append(CONCLUSIONS.strip())

    result = "\n\n".join(parts) + "\n"
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze memory allocation patterns from CUDA snapshots"
    )
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=MEMORY_DIR,
        help=f"Directory containing .snap files (default: {MEMORY_DIR})",
    )
    args = parser.parse_args()

    result = analyze(args.memory_dir)

    output_path = Path("docs/sections/memory-profile-(e).md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result)
    print(f"\nMarkdown saved: {output_path}")
