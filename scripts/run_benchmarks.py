"""Run benchmarks with multiple parameter combinations and output markdown table."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import strftime

import pandas as pd

from cs336_systems.configs import CONFIGS, ModelConfig
from cs336_systems.output_utils import get_output_dir

SEQ_LENGTHS = (128, 256, 512, 1024)
CONFIG_ORDER = [config.name for config in CONFIGS]
MODE_LABELS = {True: "forward-only", False: "forward+backward"}
MODE_ORDER = ("forward-only", "forward+backward")
MODE_SHORT_LABELS = {"forward-only": "FWD", "forward+backward": "FWD+BWD"}
MODE_FILE_LABELS = {"forward-only": "FWD", "forward+backward": "FWD_BWD"}
INDEX_COLS = [
    "Config",
    "Seq Len",
    "Warmup Steps",
]
VALUE_COLS = [
    "Result",
    "Total Time (s)",
    "Warmup Time (s)",
    "Avg Time/Step (s)",
    "Std Time/Step (s)",
]
METRIC_PATTERNS = {
    "Total Time (s)": r"Total time for \d+ steps: ([\d.]+)",
    "Warmup Time (s)": r"Warm-up time \(\d+ steps\): ([\d.]+)",
    "Avg Time/Step (s)": r"Avg time per step after warm-up: ([\d.]+)",
    "Std Time/Step (s)": r"Std time per step after warm-up: ([\d.]+)",
}
METRIC_LABELS = {
    "Result": "Status",
    "Total Time (s)": "Total(s)",
    "Warmup Time (s)": "Warmup(s)",
    "Avg Time/Step (s)": "Avg/Step(s)",
    "Std Time/Step (s)": "Std/Step(s)",
}


@dataclass(frozen=True)
class RunOptions:
    """Runtime options for batch benchmark execution."""

    nsys_output_dir: Path
    run_tag: str | None
    output_dir: Path
    autocast: bool
    warmup_steps: int
    python_time: bool
    memory_profile: bool
    memory_output_dir: Path
    configs: tuple[ModelConfig, ...]
    seq_lengths: tuple[int, ...]


def _extract_float(pattern: str, text: str) -> float | None:
    """Extract a float value from text using regex pattern."""
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def _slugify(value: str) -> str:
    """Make a filesystem-safe token from user-provided text."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-") or "run"


def _nsys_available() -> bool:
    """Check if nsys is available on PATH."""
    return shutil.which("nsys") is not None


def _nsys_cmd(report_prefix: Path) -> list[str]:
    """Build nsys profiling command."""
    return [
        "nsys",
        "profile",
        "-o",
        str(report_prefix),
        "--force-overwrite=true",
        "--trace=cuda,osrt,nvtx",
        "--sample=cpu",
    ]


def _benchmark_cmd(
    config: ModelConfig,
    *,
    forward_only: bool,
    context_length: int,
    warmup_steps: int,
    options: RunOptions,
    memory_output: Path | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "cs336_systems.benchmark",
        "--d_model",
        str(config.d_model),
        "--d_ff",
        str(config.d_ff),
        "--n_layers",
        str(config.n_layers),
        "--num_heads",
        str(config.num_heads),
        "--context_length",
        str(context_length),
        "--warm_up_steps",
        str(warmup_steps),
    ]
    if forward_only:
        cmd.append("--forward_only")
    if options.autocast:
        cmd.append("--autocast")
    if options.python_time:
        cmd.append("--python-time")
    if options.memory_profile and memory_output:
        cmd.extend(["--memory-profile", "--memory-output", str(memory_output)])
    return cmd


def _is_oom_error(stdout: str | None, stderr: str | None) -> bool:
    """Best-effort OOM detection from subprocess output."""
    text = f"{stdout or ''} {stderr or ''}".lower()
    oom_phrases = (
        "out of memory",
        "cuda out of memory",
        "oom",
        "cudnn error: cudnn_status_alloc_failed",
    )
    return any(phrase in text for phrase in oom_phrases)


def _result_row(
    config: ModelConfig,
    *,
    forward_only: bool,
    context_length: int,
    warmup_steps: int,
    result: str = "OK",
) -> dict:
    """Build result row with common fields."""
    return {
        "Config": config.name,
        "Mode": MODE_LABELS[forward_only],
        "Seq Len": context_length,
        "Warmup Steps": warmup_steps,
        "Result": result,
        "Total Time (s)": None,
        "Warmup Time (s)": None,
        "Avg Time/Step (s)": None,
        "Std Time/Step (s)": None,
    }


def _build_output_filename(
    options: RunOptions,
    config: ModelConfig,
    *,
    forward_only: bool,
    context_length: int,
    warmup_steps: int,
) -> str:
    """Build base filename for benchmark outputs (without extension)."""
    mode_label = MODE_FILE_LABELS[MODE_LABELS[forward_only]]
    parts = [
        options.run_tag,
        config.name,
        mode_label,
        f"seq{context_length}",
        f"warm{warmup_steps}",
    ]
    return "__".join(_slugify(part) for part in parts if part)


def run_benchmark(
    config: ModelConfig,
    *,
    forward_only: bool,
    context_length: int,
    warmup_steps: int,
    options: RunOptions,
) -> dict:
    """Run benchmark for a given configuration and parse results."""
    base_name = _build_output_filename(
        options,
        config,
        forward_only=forward_only,
        context_length=context_length,
        warmup_steps=warmup_steps,
    )

    memory_output = None
    if options.memory_profile:
        memory_output = options.memory_output_dir / f"{base_name}.snap"

    cmd = _benchmark_cmd(
        config,
        forward_only=forward_only,
        context_length=context_length,
        warmup_steps=warmup_steps,
        options=options,
        memory_output=memory_output,
    )

    # NVTX mode: wrap with nsys (not in python_time or memory_profile mode)
    if not options.python_time and not options.memory_profile:
        report_prefix = options.nsys_output_dir / base_name
        cmd = [*_nsys_cmd(report_prefix), *cmd]

    mode = MODE_LABELS[forward_only]
    label = f"{config.name} ({mode}, seq={context_length}, warmup={warmup_steps})"
    print(f"Running benchmark for {label}...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark for {label}: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        error_type = "OOM" if _is_oom_error(e.stdout, e.stderr) else "ERROR"
        return _result_row(
            config,
            forward_only=forward_only,
            context_length=context_length,
            warmup_steps=warmup_steps,
            result=error_type,
        )

    row = _result_row(
        config,
        forward_only=forward_only,
        context_length=context_length,
        warmup_steps=warmup_steps,
    )

    # In Python timing mode, parse metrics from stdout
    if options.python_time:
        for metric, pattern in METRIC_PATTERNS.items():
            row[metric] = _extract_float(pattern, result.stdout)

    return row


def _format_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df["Config"] = pd.Categorical(df["Config"], CONFIG_ORDER, ordered=True)

    df = (
        df.pivot_table(
            index=INDEX_COLS,
            columns="Mode",
            values=VALUE_COLS,
            aggfunc="first",
            sort=False,
        )
        .reindex(columns=VALUE_COLS, level=0)
        .reindex(columns=MODE_ORDER, level=1)
    )
    df.columns = [
        f"{METRIC_LABELS[metric]} {MODE_SHORT_LABELS[mode]}"
        for metric, mode in df.columns
    ]
    df = df.reset_index()
    return df.sort_values(INDEX_COLS, kind="stable", ignore_index=True)


def _parse_args() -> RunOptions:
    config_names = [c.name for c in CONFIGS]
    parser = argparse.ArgumentParser(description="Run benchmark grid.")
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional tag added to output filenames.",
    )
    parser.add_argument(
        "--nsys-output-dir",
        type=Path,
        default=None,
        help="Directory to store nsys reports (default: output/nsys).",
    )
    parser.add_argument(
        "--memory-output-dir",
        type=Path,
        default=None,
        help="Directory to store memory snapshots (default: output/memory).",
    )
    parser.add_argument(
        "--autocast",
        action="store_true",
        help="Enable autocasting for CUDA.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of warm-up steps (default: 5).",
    )
    parser.add_argument(
        "--python-time",
        action="store_true",
        help="Enable Python time mode with synchronize (generates md file).",
    )
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="Enable memory profiling (generates .snap files).",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=config_names,
        default=None,
        metavar="CONFIG",
        help=f"Model configs to run (default: all). Choices: {', '.join(config_names)}",
    )
    parser.add_argument(
        "--seq-lengths",
        nargs="+",
        type=int,
        default=None,
        metavar="LEN",
        help=f"Sequence lengths to run (default: {', '.join(map(str, SEQ_LENGTHS))}).",
    )
    args = parser.parse_args()

    if args.python_time and args.memory_profile:
        parser.error("--python-time and --memory-profile are mutually exclusive")

    output_dir = get_output_dir()
    nsys_dir = args.nsys_output_dir or (output_dir / "nsys")
    nsys_dir.mkdir(parents=True, exist_ok=True)
    memory_dir = args.memory_output_dir or (output_dir / "memory")
    memory_dir.mkdir(parents=True, exist_ok=True)

    # Auto-generate run_tag for modes that produce output files (NVTX, memory)
    run_tag = args.run_tag or (None if args.python_time else strftime("%Y%m%d_%H%M%S"))

    # Filter configs and seq_lengths
    config_map = {c.name: c for c in CONFIGS}
    configs = (
        tuple(config_map[name] for name in args.configs)
        if args.configs
        else tuple(CONFIGS)
    )
    seq_lengths = tuple(args.seq_lengths) if args.seq_lengths else SEQ_LENGTHS

    return RunOptions(
        nsys_output_dir=nsys_dir,
        run_tag=run_tag,
        output_dir=output_dir,
        autocast=args.autocast,
        warmup_steps=args.warmup_steps,
        python_time=args.python_time,
        memory_profile=args.memory_profile,
        memory_output_dir=memory_dir,
        configs=configs,
        seq_lengths=seq_lengths,
    )


def _run_all_benchmarks(options: RunOptions) -> list[dict]:
    """Run benchmarks for all configurations."""
    return [
        run_benchmark(
            config,
            forward_only=forward_only,
            context_length=context_length,
            warmup_steps=options.warmup_steps,
            options=options,
        )
        for config in options.configs
        for context_length in options.seq_lengths
        for forward_only in [True, False]
    ]


def _print_summary(results: list[dict], options: RunOptions) -> None:
    """Print benchmark summary for NVTX/memory modes."""
    successful = sum(1 for r in results if r["Result"] == "OK")
    failed = [r for r in results if r["Result"] != "OK"]

    print(f"\n{'=' * 60}")
    print("Benchmark Summary")
    print(f"{'=' * 60}")
    print(f"Total benchmarks: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed benchmarks:")
        for r in failed:
            print(f"  - {r['Config']} ({r['Mode']}, seq={r['Seq Len']}): {r['Result']}")

    if options.memory_profile:
        print(f"\nMemory snapshots saved to: {options.memory_output_dir}")
        print("Use PyTorch Memory Visualizer to analyze the snapshots.")
    else:
        print(f"\nNsys reports saved to: {options.nsys_output_dir}")
        print("Use 'nsys-ui' or 'nsys stats' to analyze the reports.")


def main() -> None:
    """Run benchmarks for all configurations and output markdown table."""
    options = _parse_args()

    # Check nsys availability only in NVTX mode (not python_time or memory_profile)
    if not options.python_time and not options.memory_profile and not _nsys_available():
        msg = (
            "nsys not found on PATH. Install Nsight Systems or use --python-time mode."
        )
        raise RuntimeError(msg)

    results = _run_all_benchmarks(options)

    if options.python_time:
        df = _format_results(pd.DataFrame(results))
        tag_suffix = f"__{_slugify(options.run_tag)}" if options.run_tag else ""
        output_file = options.output_dir / f"benchmark_results{tag_suffix}.md"
        md_content = f"### Benchmark Results\n\n{df.to_markdown(index=False)}\n"
        output_file.write_text(md_content)
        print(f"Markdown table saved to {output_file}")
        return

    _print_summary(results, options)


if __name__ == "__main__":
    main()
