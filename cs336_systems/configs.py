"""Shared model configurations for benchmarking."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a benchmark model."""

    name: str
    d_model: int
    d_ff: int
    n_layers: int
    num_heads: int


CONFIGS = [
    ModelConfig("small", d_model=768, d_ff=3072, n_layers=12, num_heads=12),
    ModelConfig("medium", d_model=1024, d_ff=4096, n_layers=24, num_heads=16),
    ModelConfig("large", d_model=1280, d_ff=5120, n_layers=36, num_heads=20),
    ModelConfig("xl", d_model=1600, d_ff=6400, n_layers=48, num_heads=25),
    ModelConfig("2.7B", d_model=2560, d_ff=10240, n_layers=32, num_heads=32),
]
