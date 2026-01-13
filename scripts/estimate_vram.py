"""Estimate VRAM usage for benchmark model configs.

Formulas
========

1. Parameter Count (per Transformer layer)
------------------------------------------

Attention:
    W_Q, W_K, W_V: 3 * d_model * d_model
    W_O:           1 * d_model * d_model
    Total:         4 * d_model^2

FFN (SwiGLU/GeGLU style with 3 projections):
    W_gate: d_model * d_ff
    W_up:   d_model * d_ff
    W_down: d_ff * d_model
    Total:  3 * d_model * d_ff

LayerNorm (2 per layer):
    gamma, beta for each: 2 * d_model
    Total:                2 * d_model

Per-layer total:
    P_layer = 4 * d_model^2 + 3 * d_model * d_ff + 2 * d_model

Non-layer parameters:
    Token embedding:  vocab_size * d_model
    LM head:          d_model * vocab_size
    Final LayerNorm:  d_model

Total parameters:
    P_total = n_layers * P_layer + 2 * vocab_size * d_model + d_model

2. VRAM Estimation
------------------

Model weights (fp32):
    M_params = P_total * 4 bytes

Attention activations (per layer):
    Shape: [batch_size, num_heads, seq_len, seq_len]
    Size:  batch_size * num_heads * seq_len^2 * 4 bytes
    Total: n_layers * above

Forward pass VRAM:
    V_forward = M_params + 2 * M_attn_activations

Training VRAM (simplified estimate):
    V_train = 4 * M_params + 3 * M_attn_activations

    Breakdown of 4x multiplier (approximation):
    - Model weights (fp32):     1x
    - Gradients (fp32):         1x
    - Optimizer states (AdamW): 2x (momentum + variance)
    Total:                      4x

    Note: Mixed precision training actually uses ~4.5x (18 bytes/param)
    due to maintaining both fp16 and fp32 copies. This script uses 4x
    as a simplified lower bound.

References:
----------
- HuggingFace: https://huggingface.co/docs/transformers/perf_train_gpu_one
- Standard Transformer: "Attention Is All You Need" (Vaswani et al., 2017)
- SwiGLU FFN: "GLU Variants Improve Transformer" (Shazeer, 2020)

"""

from __future__ import annotations

from cs336_systems.configs import CONFIGS, ModelConfig
from cs336_systems.output_utils import get_output_dir

GIB = 1024**3


def param_count(cfg: ModelConfig, vocab_size: int) -> int:
    """Return parameter count for the model configuration."""
    per_layer = 4 * cfg.d_model**2 + 3 * cfg.d_model * cfg.d_ff + 2 * cfg.d_model
    embeddings = 2 * vocab_size * cfg.d_model + cfg.d_model
    return per_layer * cfg.n_layers + embeddings


def estimate_vram_gib(
    cfg: ModelConfig,
    *,
    seq_len: int,
    batch_size: int,
    vocab_size: int,
) -> tuple[float, float]:
    """Return (forward_only_gib, train_gib) estimates."""
    param_gib = param_count(cfg, vocab_size) * 4 / GIB
    attn_gib = batch_size * cfg.num_heads * seq_len**2 * 4 * cfg.n_layers / GIB

    forward_total = param_gib + attn_gib * 2
    train_total = param_gib * 4 + attn_gib * 3
    return forward_total, train_total


def main() -> None:
    """Print forward and training VRAM estimates for benchmark configs."""
    batch_size = 4
    vocab_size = 10000
    seq_lens = [0, 128, 256, 512, 1024]

    lines: list[str] = [f"batch_size={batch_size}, vocab_size={vocab_size}"]

    for seq_len in seq_lens:
        if seq_len == 0:
            label = "base_vram (seq/batch independent)"
        else:
            label = f"seq_len={seq_len}"
        lines.append(f"\n{label}")
        lines.append(f"{'config':<8}{'forward_gib':>14}{'train_gib':>12}")
        for cfg in CONFIGS:
            fwd, train = estimate_vram_gib(
                cfg, seq_len=seq_len, batch_size=batch_size, vocab_size=vocab_size
            )
            lines.append(f"{cfg.name:<8}{fwd:>14.2f}{train:>12.2f}")

    output = "\n".join(lines)
    print(output)

    output_file = get_output_dir() / "vram_estimates.txt"
    output_file.write_text(output + "\n")
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
