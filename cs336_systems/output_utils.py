"""Utilities for managing output files and directories."""

from pathlib import Path


def get_output_dir() -> Path:
    """Get the output directory path, creating it if necessary."""
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir
