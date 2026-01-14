"""NVTX functions for nsys."""

import torch
from cs336_basics.blocks import softmax
from einops import einsum
from torch.cuda import nvtx


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention.

    Args:
        q: Query tensor of shape (batch_size, ..., seq_len_q, d_k).
        k: Key tensor of shape (batch_size, ..., seq_len_k, d_k).
        v: Value tensor of shape (batch_size, ..., seq_len_k, d_v).
        mask: Optional boolean mask tensor of shape (..., seq_len_q, seq_len_k).
              True indicates positions to keep, False indicates positions to mask out.

    Returns:
        Attention output tensor of shape (batch_size, ..., seq_len_q, d_v).

    """
    # Scaled dot-product for numerical stability
    d_k = k.shape[-1]
    with nvtx.range("computing attention scores"):
        scores = einsum(q, k, "... s_q d_k, ... s_k d_k -> ... s_q s_k") / torch.sqrt(
            torch.tensor(d_k, dtype=q.dtype, device=q.device)
        )
    if mask is not None:
        neg_inf = torch.tensor(
            float("-inf"), dtype=scores.dtype, device=scores.device
        )  # Convert boolean mask to additive mask: True -> 0.0, False -> -inf
        scores = torch.where(mask, scores, neg_inf)
    with nvtx.range("computing softmax"):
        attention_weights = softmax(scores, dim=-1)
    with nvtx.range("final matmul"):
        return einsum(attention_weights, v, "... s_q s_k, ... s_k d_k -> ... s_q d_k")
