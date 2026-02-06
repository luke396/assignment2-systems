"""Compare JIT vs Eager mode for 2.7B model."""

import re
import subprocess
import sys
from pathlib import Path

DOC_DIR = Path("docs/sections")

# Model configs
CONFIGS = {
    "xl": {
        "d_model": 1600,
        "d_ff": 6400,
        "n_layers": 48,
        "num_heads": 25,
    },
    "2.7B": {
        "d_model": 2560,
        "d_ff": 10240,
        "n_layers": 32,
        "num_heads": 32,
    },
}

# Sequence lengths per model
SEQ_LENS = {
    "xl": [128, 256, 512, 1024],
    "2.7B": [128, 256],
}
MODES = [False, True]  # Eager, JIT


def parse_output(output: str) -> dict[str, float | None]:
    """Parse benchmark output for time and memory stats."""
    result: dict[str, float | None] = {
        "avg_time": None,
        "std_time": None,
        "avg_mem_gb": None,
    }

    # Parse avg time per step
    match = re.search(r"Avg time per step after warm-up: ([\d.]+) seconds", output)
    if match:
        result["avg_time"] = float(match.group(1))

    # Parse std time per step
    match = re.search(r"Std time per step after warm-up: ([\d.]+) seconds", output)
    if match:
        result["std_time"] = float(match.group(1))

    # Parse avg memory before backward
    match = re.search(r"Avg memory before backward: ([\d.]+) GB", output)
    if match:
        result["avg_mem_gb"] = float(match.group(1))

    return result


def run_benchmark(
    model_name: str, seq_len: int, *, jit: bool
) -> dict[str, float | None]:
    """Run benchmark with given configuration."""
    config = CONFIGS[model_name]
    cmd = [
        sys.executable,
        "-m",
        "cs336_systems.benchmark",
        "--d_model",
        str(config["d_model"]),
        "--d_ff",
        str(config["d_ff"]),
        "--n_layers",
        str(config["n_layers"]),
        "--num_heads",
        str(config["num_heads"]),
        "--context_length",
        str(seq_len),
        "--python-time",
    ]
    if jit:
        cmd.append("--jit")

    mode_str = "JIT" if jit else "Eager"
    print(f"Running: model={model_name}, seq_len={seq_len}, mode={mode_str}...")

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        return {"avg_time": None, "std_time": None, "avg_mem_gb": None}

    return parse_output(result.stdout)


def format_table(model_name: str, results: list[dict]) -> str:
    """Format comparison table for a model."""
    lines = []
    lines.append(f"#### JIT vs Eager: {model_name}\n")
    lines.append("| Seq Len | Mode  | Avg Time (s) | Mem (GB) |")
    lines.append("|---------|-------|--------------|----------|")

    for r in results:
        mode_str = "JIT" if r["jit"] else "Eager"
        avg_time = f"{r['avg_time']:.4f}" if r["avg_time"] else "N/A"
        avg_mem = f"{r['avg_mem_gb']:.2f}" if r["avg_mem_gb"] else "N/A"
        seq = r["seq_len"]
        lines.append(f"| {seq:<7} | {mode_str:<5} | {avg_time:<12} | {avg_mem:<8} |")

    return "\n".join(lines)


def main() -> None:
    """Run JIT comparison benchmark."""
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    all_tables = []
    all_tables.append("### End-to-End JIT vs Eager Comparison Results\n")

    for model_name in CONFIGS:
        results = []
        seq_lens = SEQ_LENS[model_name]
        for seq_len in seq_lens:
            for jit in MODES:
                result = run_benchmark(model_name, seq_len, jit=jit)
                results.append({"seq_len": seq_len, "jit": jit, **result})

        table = format_table(model_name, results)
        all_tables.append(table)
        print(table)

    output_file = DOC_DIR / "e2e_jit_comparison.md"
    output_file.write_text("\n\n".join(all_tables) + "\n")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
