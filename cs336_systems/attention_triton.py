import math

import torch
import triton
import triton.language as tl


class AttentionTorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal=False,  # noqa: FBT002
        q_tile_size=16,  # bq
        kv_tile_size=16,  # bk
    ):
        """
        Q : (... , seq_len, d)
        K : (... , seq_len, d)
        V : (... , seq_len, d)
        """
        _ = is_causal  # causal masking not implemented for PyTorch version

        seq, d = Q.size(-2), Q.size(-1)
        scale = 1 / math.sqrt(d)
        num_q_tiles = math.ceil(seq / q_tile_size)
        num_kv_tiles = math.ceil(seq / kv_tile_size)

        o = Q.new_zeros(Q.size())  # (seq_len, d)
        l = Q.new_zeros(Q.shape[:-1])  # (seq_len)  # noqa: E741

        for i in range(num_q_tiles):
            q_start, q_end = i * q_tile_size, (i + 1) * q_tile_size
            tile_q = Q[..., q_start:q_end, :]  # (b_q, d)
            # unnormalized output
            o_i = tile_q.new_zeros(tile_q.size())  # (b_q, d)
            # normalization factor
            l_i = tile_q.new_zeros(tile_q.shape[:-1])  # (b_q,)
            # online max
            m_i = tile_q.new_full(tile_q.shape[:-1], float("-inf"))  # (b_q,)

            for j in range(num_kv_tiles):
                kv_start, kv_end = j * kv_tile_size, (j + 1) * kv_tile_size
                tile_k = K[..., kv_start:kv_end, :]  # (b_k, d)
                tile_v = V[..., kv_start:kv_end, :]  # (b_k, d)

                s_ij = tile_q @ tile_k.transpose(-2, -1) * scale  # (b_q, b_k)
                m_i_prev = m_i
                m_i = torch.maximum(m_i_prev, s_ij.max(dim=-1).values)  # (b_q,)
                # minus max to improve numerical stability
                p_ij = torch.exp(s_ij - m_i[..., None])  # (b_q, b_k)
                # update normalization factor and output
                alpha = torch.exp(m_i_prev - m_i)
                l_i = alpha * l_i + p_ij.sum(dim=-1)  # (b_q,)
                # update unnormalized output
                o_i = alpha[..., None] * o_i + p_ij @ tile_v  # (b_q, d)
            # finalize normalize output
            o_i = o_i / l_i[..., None]  # (b_q, d)
            # log sum exp - lse for backward
            l_i = m_i + torch.log(l_i)  # (b_q,)
            o[..., q_start:q_end, :] = o_i
            l[..., q_start:q_end] = l_i

        ctx.save_for_backward(l, Q, K, V, o)

        return o

    @staticmethod
    def backward(ctx, *grad_outputs):
        l, Q, K, V, o = ctx.saved_tensors  # noqa: E741
        return _backward_impl(l, Q, K, V, o, grad_outputs)


@torch.compile
def _backward_impl(l, Q, K, V, o, grad_outputs):  # noqa: E741
    """Flash Attention backward pass."""
    do = grad_outputs[0]  # (..., seq_len, d)
    scale = 1 / math.sqrt(Q.size(-1))
    d = torch.sum(o * do, dim=-1)  # (..., seq_len)

    s = Q @ K.transpose(-2, -1) * scale  # (..., seq_len, seq_len)
    p = torch.exp(s - l[..., :, None])  # (..., seq_len, seq_len)
    dv = p.transpose(-2, -1) @ do  # (..., seq_len, d)
    dp = do @ V.transpose(-2, -1)  # (..., seq_len, seq_len)
    ds = p * (dp - d[..., :, None])  # (..., seq_len, seq_len)
    dq = ds @ K * scale  # (..., seq_len, d)
    dk = ds.transpose(-2, -1) @ Q * scale  # (..., seq_len, d)
    return dq, dk, dv, None  # we need return grad for every forward's input


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    n_queries,
    n_keys,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr = False,  # noqa: FBT002  # type: ignore[invalid-parameter-default]
):
    # program indices
    q_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_idx * stride_qb,
        shape=(n_queries, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_idx * stride_kb,
        shape=(n_keys, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_idx * stride_vb,
        shape=(n_keys, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_idx * stride_ob,
        shape=(n_queries, D),
        strides=(stride_oq, stride_od),
        offsets=(q_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_idx * stride_lb,
        shape=(n_queries,),
        strides=(stride_lq,),
        offsets=(q_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # tile assumed always not out of bound
    tile_q = tl.load(Q_block_ptr)  # (b_q, d)

    # keep inner var high precision
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)  # (b_q, d)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)  # (b_q,)
    m_i = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

    if IS_CAUSAL:
        q_end = (q_tile_idx + 1) * Q_TILE_SIZE
        num_kv_tiles = tl.cdiv(q_end, K_TILE_SIZE)
    else:
        num_kv_tiles = tl.cdiv(n_keys, K_TILE_SIZE)

    for kv_tile_idx in range(num_kv_tiles):
        tile_k = tl.load(K_block_ptr)  # (b_k, d)
        tile_v = tl.load(V_block_ptr)  # (b_k, d)

        s_ij = tl.dot(tile_q, tl.trans(tile_k)) * scale  # (b_q, b_k)

        if IS_CAUSAL:
            q_offset = q_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_offset = kv_tile_idx * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            s_ij = tl.where(q_offset[:, None] >= k_offset[None, :], s_ij, -1e6)

        m_i_prev = m_i
        m_i = tl.maximum(m_i_prev, tl.max(s_ij, axis=1))  # (b_q,)
        p_ij = tl.exp(s_ij - m_i[:, None])  # (b_q, b_k)
        alpha = tl.exp(m_i_prev - m_i)
        l_i = alpha * l_i + tl.sum(p_ij, axis=1)  # (b_q,)
        # temp cast p_ij to tile_v's same dtype for dot
        # use acc avoid creating temp tensor to save memory
        o_i = tl.dot(
            p_ij.to(tile_v.dtype), tile_v, acc=alpha[:, None] * o_i
        )  # (b_q, d)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    o_i = o_i / l_i[:, None]  # (b_q, d)
    l_i = m_i + tl.log(l_i)  # (b_q,)
    tl.store(O_block_ptr, o_i.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, l_i)


class AttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal=False,  # noqa: FBT002
        q_tile_size=16,  # bq
        kv_tile_size=16,  # bk
    ):
        """
        Q : (batch, seq_len, d)
        K : (batch, seq_len, d)
        V : (batch, seq_len, d)
        """
        batch, seq, d = Q.size(0), Q.size(-2), Q.size(-1)
        scale = 1 / math.sqrt(d)
        n_q_tiles = math.ceil(seq / q_tile_size)

        o = Q.new_zeros(Q.size())  # (seq_len, d)
        l = Q.new_zeros(Q.shape[:-1])  # (seq_len)  # noqa: E741

        ctx.Q_TILE_SIZE = q_tile_size
        ctx.K_TILE_SIZE = kv_tile_size

        flash_fwd_kernel[(n_q_tiles, batch)](
            Q,
            K,
            V,
            o,
            l,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            l.stride(0),
            l.stride(1),
            n_queries=seq,
            n_keys=seq,
            scale=scale,
            D=d,  # type: ignore[invalid-argument-type]
            Q_TILE_SIZE=q_tile_size,  # type: ignore[invalid-argument-type]
            K_TILE_SIZE=kv_tile_size,  # type: ignore[invalid-argument-type]
            IS_CAUSAL=is_causal,  # type: ignore[invalid-argument-type]
        )

        ctx.save_for_backward(l, Q, K, V, o)

        return o
