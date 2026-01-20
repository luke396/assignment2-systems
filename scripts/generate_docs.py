"""Generate complete documentation from all analysis results."""

import argparse
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from scripts.analyze_basic_benchmarks import analyze as analyze_benchmarks
from scripts.analyze_mixed_precision import analyze as analyze_mixed_precision
from scripts.analyze_nsys_profile import analyze as analyze_nsys_profile


def regenerate_all_sections(
    benchmark_run_tag: str = "5090",
    nsys_run_tag: str = "new_base",
    base_prefix: str = "new_base",
    mixed_prefix: str = "autocastBF",
) -> None:
    """Regenerate all documentation sections by running analysis scripts."""
    sections_dir = Path("docs/sections")
    sections_dir.mkdir(parents=True, exist_ok=True)

    # Generate basic benchmarking section
    result = analyze_benchmarks(benchmark_run_tag)
    (sections_dir / "benchmarking.md").write_text(result)
    print(f"Generated {sections_dir / 'benchmarking.md'}")

    # Generate nsys profile section
    result = analyze_nsys_profile(nsys_run_tag)
    (sections_dir / "nsys-profile.md").write_text(result)
    print(f"Generated {sections_dir / 'nsys-profile.md'}")

    # Generate mixed precision section
    result = analyze_mixed_precision(base_prefix, mixed_prefix)
    (sections_dir / "mixed-precision.md").write_text(result)
    print(f"Generated {sections_dir / 'mixed-precision.md'}")


def generate_main_readme() -> None:
    """Generate main docs/README.md from template."""
    # Render main README using template (S701: autoescape not needed for markdown)
    env = Environment(loader=FileSystemLoader("docs"))  # noqa: S701
    template = env.get_template("templates/main.md.j2")
    output = template.render()

    # Save to docs/README.md
    output_path = Path("docs/README.md")
    output_path.write_text(output)
    print(f"Generated {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate complete documentation from analysis results"
    )
    parser.add_argument(
        "--regenerate-all",
        action="store_true",
        help="Regenerate all documentation sections before generating main README",
    )
    parser.add_argument(
        "--benchmark-run-tag",
        type=str,
        default="5090",
        help="Run tag for benchmark analysis (default: 5090)",
    )
    parser.add_argument(
        "--nsys-run-tag",
        type=str,
        default="new_base",
        help="Run tag for nsys analysis (default: new_base)",
    )
    parser.add_argument(
        "--base-prefix",
        type=str,
        default="new_base",
        help="Base run prefix for mixed precision analysis (default: new_base)",
    )
    parser.add_argument(
        "--mixed-prefix",
        type=str,
        default="autocastBF",
        help="Mixed precision run prefix (default: autocastBF)",
    )
    args = parser.parse_args()

    if args.regenerate_all:
        regenerate_all_sections(
            args.benchmark_run_tag,
            args.nsys_run_tag,
            args.base_prefix,
            args.mixed_prefix,
        )

    generate_main_readme()
