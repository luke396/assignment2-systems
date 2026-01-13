"""Run benchmarks with multiple parameter combinations and output markdown table."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from time import strftime

import pandas as pd

from cs336_systems.configs import CONFIGS, ModelConfig
from cs336_systems.output_utils import get_output_dir

SEQ_LENGTHS = (128, 256, 512, 1024)
WARMUP_STEPS = (0, 1, 5)
CONFIG_ORDER = [config.name for config in CONFIGS]
MODE_LABELS = {True: "forward-only", False: "forward+backward"}
MODE_ORDER = ("forward-only", "forward+backward")
MODE_SHORT_LABELS = {"forward-only": "FWD", "forward+backward": "FWD+BWD"}
MODE_FILE_LABELS = {"forward-only": "FWD", "forward+backward": "FWD_BWD"}
INDEX_COLS = [
    "Config",
    "Seq Len",
    "Warmup Steps",
    "d_model",
    "d_ff",
    "n_layers",
    "num_heads",
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

    nsys: bool
    nsys_output_dir: Path
    nsys_prefix: str
    nsys_delay: float
    run_tag: str | None
    output_dir: Path


def _extract_float(pattern: str, text: str) -> float | None:
    """Extract a float value from text using regex pattern."""
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def _slugify(value: str) -> str:
    """Make a filesystem-safe token from user-provided text."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-") or "run"


def _nsys_available() -> bool:
    return shutil.which("nsys") is not None


def _benchmark_cmd(
    config: ModelConfig, *, forward_only: bool, context_length: int, warmup_steps: int
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
    return cmd


def _nsys_cmd(report_prefix: Path, delay: float) -> list[str]:
    cmd = [
        "nsys",
        "profile",
        "-o",
        str(report_prefix),
        "--force-overwrite=true",
        "--trace=cuda,osrt,nvtx",
        "--sample=cpu",
    ]
    if delay > 0:
        cmd.insert(6, f"--delay={delay}")
    return cmd


def _is_oom_error(stdout: str | None, stderr: str | None) -> bool:
    """Best-effort OOM detection from subprocess output."""
    text = (stdout or "") + " " + (stderr or "")
    text_lower = text.lower()
    return any(
        phrase in text_lower
        for phrase in (
            "out of memory",
            "cuda out of memory",
            "oom",
            "cudnn error: cudnn_status_alloc_failed",
        )
    )


def _base_row(
    config: ModelConfig,
    *,
    context_length: int,
    warmup_steps: int,
    forward_only: bool,
) -> dict:
    return {
        "Config": config.name,
        "Mode": MODE_LABELS[forward_only],
        "Seq Len": context_length,
        "Warmup Steps": warmup_steps,
        "d_model": config.d_model,
        "d_ff": config.d_ff,
        "n_layers": config.n_layers,
        "num_heads": config.num_heads,
    }


def _parse_metrics(output: str) -> dict[str, float | None]:
    return {
        label: _extract_float(pattern, output)
        for label, pattern in METRIC_PATTERNS.items()
    }


def _error_row(
    config: ModelConfig,
    *,
    forward_only: bool,
    context_length: int,
    warmup_steps: int,
    error: subprocess.CalledProcessError,
) -> dict:
    row = _base_row(
        config,
        forward_only=forward_only,
        context_length=context_length,
        warmup_steps=warmup_steps,
    )
    row.update(
        {
            "Result": "OOM" if _is_oom_error(error.stdout, error.stderr) else "ERROR",
            "Total Time (s)": None,
            "Warmup Time (s)": None,
            "Avg Time/Step (s)": None,
            "Std Time/Step (s)": None,
        }
    )
    return row


def _build_nsys_report_prefix(
    options: RunOptions,
    config: ModelConfig,
    *,
    forward_only: bool,
    context_length: int,
    warmup_steps: int,
) -> Path:
    mode_label = MODE_FILE_LABELS[MODE_LABELS[forward_only]]
    parts = [
        options.nsys_prefix,
        options.run_tag,
        config.name,
        mode_label,
        f"seq{context_length}",
        f"warm{warmup_steps}",
    ]
    name = "__".join(_slugify(part) for part in parts if part)
    return options.nsys_output_dir / name


def run_benchmark(
    config: ModelConfig,
    *,
    forward_only: bool,
    context_length: int,
    warmup_steps: int,
    options: RunOptions,
) -> dict:
    """Run benchmark for a given configuration and parse results."""
    cmd = _benchmark_cmd(
        config,
        forward_only=forward_only,
        context_length=context_length,
        warmup_steps=warmup_steps,
    )

    if options.nsys:
        report_prefix = _build_nsys_report_prefix(
            options,
            config,
            forward_only=forward_only,
            context_length=context_length,
            warmup_steps=warmup_steps,
        )
        cmd = [*_nsys_cmd(report_prefix, options.nsys_delay), *cmd]

    print(
        "Running benchmark for "
        f"{config.name} ({MODE_LABELS[forward_only]}, "
        f"seq={context_length}, warmup={warmup_steps})..."
    )
    # S603: cmd is constructed internally from hardcoded configs, not from user input
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)  # noqa: S603
    output = result.stdout

    row = _base_row(
        config,
        forward_only=forward_only,
        context_length=context_length,
        warmup_steps=warmup_steps,
    )
    row.update({"Result": "OK", **_parse_metrics(output)})
    return row


def _run_case(
    config: ModelConfig,
    *,
    forward_only: bool,
    context_length: int,
    warmup_steps: int,
    options: RunOptions,
) -> dict:
    """Run a single benchmark case with error handling."""
    try:
        return run_benchmark(
            config,
            forward_only=forward_only,
            context_length=context_length,
            warmup_steps=warmup_steps,
            options=options,
        )
    except subprocess.CalledProcessError as e:
        print(
            "Error running benchmark for "
            f"{config.name} ({MODE_LABELS[forward_only]}, "
            f"seq={context_length}, warmup={warmup_steps}): {e}"
        )
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return _error_row(
            config,
            forward_only=forward_only,
            context_length=context_length,
            warmup_steps=warmup_steps,
            error=e,
        )


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

    sort_cols = [
        "Config",
        "Seq Len",
        "Warmup Steps",
        "d_model",
        "d_ff",
        "n_layers",
        "num_heads",
    ]
    return df.sort_values(sort_cols, kind="stable", ignore_index=True)


def _parse_args() -> RunOptions:
    parser = argparse.ArgumentParser(
        description="Run benchmark grid and optionally profile with nsys."
    )
    parser.add_argument(
        "--nsys",
        action="store_true",
        help="Profile each benchmark run with Nsight Systems.",
    )
    parser.add_argument(
        "--nsys-output-dir",
        type=Path,
        default=None,
        help="Directory to store nsys reports (default: output/nsys).",
    )
    parser.add_argument(
        "--nsys-prefix",
        type=str,
        default="benchmark",
        help="Prefix for nsys report filenames.",
    )
    parser.add_argument(
        "--nsys-delay",
        type=float,
        default=0.0,
        help="Delay (seconds) before profiling starts.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional tag added to output filenames (auto when --nsys).",
    )
    args = parser.parse_args()

    run_tag = args.run_tag
    if run_tag is None and args.nsys:
        run_tag = strftime("%Y%m%d_%H%M%S")

    output_dir = get_output_dir()
    nsys_dir = args.nsys_output_dir or (output_dir / "nsys")
    nsys_dir.mkdir(parents=True, exist_ok=True)

    return RunOptions(
        nsys=args.nsys,
        nsys_output_dir=nsys_dir,
        nsys_prefix=args.nsys_prefix,
        nsys_delay=args.nsys_delay,
        run_tag=run_tag,
        output_dir=output_dir,
    )


def main() -> None:
    """Run benchmarks for all configurations and output markdown table."""
    options = _parse_args()
    if options.nsys and not _nsys_available():
        msg = "nsys not found on PATH. Install Nsight Systems or disable --nsys."
        raise RuntimeError(msg)

    results = []

    for config in CONFIGS:
        for forward_only, context_length, warmup_steps in product(
            (True, False), SEQ_LENGTHS, WARMUP_STEPS
        ):
            results.append(
                _run_case(
                    config,
                    forward_only=forward_only,
                    context_length=context_length,
                    warmup_steps=warmup_steps,
                    options=options,
                )
            )

    df = _format_results(pd.DataFrame(results))

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80 + "\n")
    print(df.to_markdown(index=False))

    tag_suffix = f"__{_slugify(options.run_tag)}" if options.run_tag else ""
    output_file = options.output_dir / f"benchmark_results{tag_suffix}.md"
    output_file.write_text(f"# Benchmark Results\n\n{df.to_markdown(index=False)}\n")
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
