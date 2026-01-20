"""Generate benchmarking documentation from benchmark results."""

import argparse
from pathlib import Path

from cs336_systems.output_utils import get_output_dir


def analyze(run_tag: str | None = None) -> str:
    """Generate benchmarking section from results file."""
    output_dir = get_output_dir()
    tag_suffix = f"__{run_tag}" if run_tag else ""
    results_file = output_dir / f"benchmark_results{tag_suffix}.md"

    if not results_file.exists():
        msg = f"Results file not found: {results_file}"
        raise FileNotFoundError(msg)

    table = results_file.read_text()

    result = (
        "### Basic benchmarking Results\n"
        f"\n>based on `benchmark_results{tag_suffix}.md`\n"
        f"\n{table}"
        "\n#### Analysis\n"
        "\nThe table above shows benchmark results for different model configurations "
        "across various sequence lengths. Results include both forward-only "
        "(inference) and forward+backward (training) passes. Time is recorded "
        "and measured with python's time module.\n"
        "\n#### Warm up \n"
        "\nThe first step is cost longer time than the rest. With warm up,"
        " the time of train or inference are stable.\n"
    )
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate benchmarking documentation from results"
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="5090",
        help="Run tag to analyze (default: latest run without tag)",
    )
    args = parser.parse_args()

    result = analyze(args.run_tag)

    output_path = Path("docs/sections/benchmarking.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result)
    print(f"Markdown saved: {output_path}")
