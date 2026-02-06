import math

import torch
import triton
import triton.language as tl


def attention_regular(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = False,  # noqa: FBT001, FBT002
) -> torch.Tensor:
    """Standard (naive) attention: O(n^2) memory, no tiling."""
    scale = 1 / math.sqrt(Q.size(-1))
    s = Q @ K.transpose(-2, -1) * scale  # (..., seq, seq)
    if is_causal:
        seq_len = Q.size(-2)
        q_idx = torch.arange(seq_len, device=Q.device)
        k_idx = torch.arange(seq_len, device=Q.device)
        s = s.masked_fill(q_idx[:, None] < k_idx[None, :], float("-inf"))
    p = torch.softmax(s, dim=-1)
    return p @ V


@torch.compile
def _forward_impl(Q, K, V, is_causal, q_tile_size, kv_tile_size):
    seq, d = Q.size(-2), Q.size(-1)
    scale = 1 / math.sqrt(d)
    num_q_tiles = math.ceil(seq / q_tile_size)
    num_kv_tiles = math.ceil(seq / kv_tile_size)

    o = Q.new_zeros(Q.size())  # (seq_len, d)
    l = Q.new_zeros(Q.shape[:-1])  # (seq_len)  # noqa: E741

    for i in range(num_q_tiles):
        q_start, q_end = i * q_tile_size, (i + 1) * q_tile_size
        tile_q = Q[..., q_start:q_end, :]  # (b_q, d)
        o_i = tile_q.new_zeros(tile_q.size())  # (b_q, d)
        l_i = tile_q.new_zeros(tile_q.shape[:-1])  # (b_q,)
        m_i = tile_q.new_full(tile_q.shape[:-1], float("-inf"))  # (b_q,)

        max_kv_tiles = (
            math.ceil((i + 1) * q_tile_size / kv_tile_size)
            if is_causal
            else num_kv_tiles
        )
        for j in range(max_kv_tiles):
            kv_start = j * kv_tile_size
            kv_end = (j + 1) * kv_tile_size
            tile_k = K[..., kv_start:kv_end, :]  # (b_k, d)
            tile_v = V[..., kv_start:kv_end, :]  # (b_k, d)

            s_ij = tile_q @ tile_k.transpose(-2, -1) * scale

            if is_causal:
                q_indices = torch.arange(q_start, q_end, device=Q.device)
                k_indices = torch.arange(kv_start, kv_end, device=Q.device)
                causal_mask = q_indices[:, None] < k_indices[None, :]
                s_ij = s_ij.masked_fill(causal_mask, float("-inf"))
            m_i_prev = m_i
            m_i = torch.maximum(m_i_prev, s_ij.max(dim=-1).values)
            p_ij = torch.exp(s_ij - m_i[..., None])
            alpha = torch.exp(m_i_prev - m_i)
            l_i = alpha * l_i + p_ij.sum(dim=-1)
            o_i = alpha[..., None] * o_i + p_ij @ tile_v
        o_i = o_i / l_i[..., None]
        l_i = m_i + torch.log(l_i)
        o[..., q_start:q_end, :] = o_i
        l[..., q_start:q_end] = l_i

    return o, l


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
        o, l = _forward_impl(  # noqa: E741
            Q,
            K,
            V,
            is_causal,
            q_tile_size,
            kv_tile_size,
        )

        ctx.save_for_backward(l, Q, K, V, o)
        ctx.is_causal = is_causal

        return o

    @staticmethod
    def backward(ctx, *grad_outputs):
        l, Q, K, V, o = ctx.saved_tensors  # noqa: E741
        is_causal = ctx.is_causal
        return _backward_impl(l, Q, K, V, o, grad_outputs, is_causal)


@torch.compile
def _backward_impl(l, Q, K, V, o, grad_outputs, is_causal):  # noqa: E741
    """Flash Attention backward pass."""
    do = grad_outputs[0]  # (..., seq_len, d)
    scale = 1 / math.sqrt(Q.size(-1))
    d = torch.sum(o * do, dim=-1)  # (..., seq_len)

    s = Q @ K.transpose(-2, -1) * scale  # (..., seq_len, seq_len)

    # Apply causal mask
    if is_causal:
        seq_len = Q.size(-2)
        q_indices = torch.arange(seq_len, device=Q.device)
        k_indices = torch.arange(seq_len, device=Q.device)
        causal_mask = q_indices[:, None] < k_indices[None, :]
        s = s.masked_fill(causal_mask, float("-inf"))

    p = torch.exp(s - l[..., :, None])  # (..., seq_len, seq_len)
    dv = p.transpose(-2, -1) @ do  # (..., seq_len, d)
    dp = do @ V.transpose(-2, -1)  # (..., seq_len, seq_len)
    ds = p * (dp - d[..., :, None])  # (..., seq_len, seq_len)
    dq = ds @ K * scale  # (..., seq_len, d)
    dk = ds.transpose(-2, -1) @ Q * scale  # (..., seq_len, d)
    return dq, dk, dv, None, None, None  # grad for every forward input


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
    tl.store(O_block_ptr, o_i.to(tile_q.dtype))
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
        l = torch.zeros(Q.shape[:-1], dtype=torch.float32, device=Q.device)  # noqa: E741

        ctx.Q_TILE_SIZE = q_tile_size
        ctx.K_TILE_SIZE = kv_tile_size
        ctx.is_causal = is_causal

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

    @staticmethod
    def backward(
        # the params is fixed, other should passed by ctx
        ctx,
        *grad_outputs,
    ):
        l, Q, K, V, o = ctx.saved_tensors  # noqa: E741
        q_tile_size = ctx.Q_TILE_SIZE
        k_tile_size = ctx.K_TILE_SIZE
        is_causal = ctx.is_causal

        do = grad_outputs[0]  # (..., seq_len, d)
        D = torch.empty_like(l)  # computed by flash_bwd_preprocess

        batch, seq, d = Q.size(0), Q.size(-2), Q.size(-1)
        scale = 1 / math.sqrt(d)
        n_q_tiles = math.ceil(seq / q_tile_size)
        n_k_tiles = math.ceil(seq / k_tile_size)

        dq = torch.zeros_like(Q)  # (..., seq_len, d)
        dk = torch.zeros_like(K)
        dv = torch.zeros_like(V)

        flash_bwd_preprocess[(n_q_tiles, batch)](
            o,
            do,
            D,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            D.stride(0),
            D.stride(1),
            n_queries=seq,
            DIM=d,
            Q_TILE_SIZE=q_tile_size,  # type: ignore[invalid-argument-type]
        )

        # outer kv innder q
        flash_dkdv_kerel[(n_k_tiles, batch)](
            Q,
            K,
            V,
            D,
            l,
            dk,
            dv,
            do,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            D.stride(0),
            D.stride(1),
            l.stride(0),
            l.stride(1),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            n_queries=seq,
            n_keys=seq,
            scale=scale,
            DIM=d,
            Q_TILE_SIZE=q_tile_size,  # type: ignore[invalid-argument-type]
            K_TILE_SIZE=k_tile_size,  # type: ignore[invalid-argument-type]
            IS_CAUSAL=is_causal,  # type: ignore[invalid-argument-type]
        )

        # outer q inner kv
        flash_dq_kerel[(n_q_tiles, batch)](
            Q,
            K,
            V,
            D,
            l,
            do,
            dq,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            D.stride(0),
            D.stride(1),
            l.stride(0),
            l.stride(1),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            n_queries=seq,
            n_keys=seq,
            scale=scale,
            DIM=d,
            Q_TILE_SIZE=q_tile_size,  # type: ignore[invalid-argument-type]
            K_TILE_SIZE=k_tile_size,  # type: ignore[invalid-argument-type]
            IS_CAUSAL=is_causal,  # type: ignore[invalid-argument-type]
        )

        return dq, dk, dv, None


@triton.jit
def flash_bwd_preprocess(
    O_ptr,
    dO_ptr,
    D_ptr,
    stride_ob,
    stride_oq,
    stride_od,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_db,
    stride_dq,
    n_queries,
    DIM: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
):
    """Precompute D = sum_i O_i * dO_i for backward pass."""
    q_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_idx * stride_ob,
        shape=(n_queries, DIM),
        strides=(stride_oq, stride_od),
        offsets=(q_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_idx * stride_dob,
        shape=(n_queries, DIM),
        strides=(stride_doq, stride_dod),
        offsets=(q_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_idx * stride_db,
        shape=(n_queries,),
        strides=(stride_dq,),
        offsets=(q_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    tile_o = tl.load(O_block_ptr)  # (b_q, d)
    tile_do = tl.load(dO_block_ptr)  # (b_q, d)

    d_i = tl.sum(tile_o.to(tl.float32) * tile_do.to(tl.float32), axis=1)

    tl.store(D_block_ptr, d_i)


@triton.jit
def flash_dkdv_kerel(
    Q_ptr,
    K_ptr,
    V_ptr,
    D_ptr,
    L_ptr,
    dK_ptr,
    dV_ptr,
    dO_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_db,
    stride_dq,
    stride_lb,
    stride_lq,
    stride_dkb,
    stride_dkk,
    stride_dkd,
    stride_dvb,
    stride_dvk,
    stride_dvd,
    stride_dob,
    stride_doq,
    stride_dod,
    n_queries,
    n_keys,
    scale,
    DIM: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr = False,  # noqa: FBT002 # type: ignore[invalid-parameter-default]
):
    k_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_idx * stride_kb,
        shape=(n_keys, DIM),
        strides=(stride_kk, stride_kd),
        offsets=(k_tile_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_idx * stride_vb,
        shape=(n_keys, DIM),
        strides=(stride_vk, stride_vd),
        offsets=(k_tile_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_idx * stride_dkb,
        shape=(n_keys, DIM),
        strides=(stride_dkk, stride_dkd),
        offsets=(k_tile_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_idx * stride_dvb,
        shape=(n_keys, DIM),
        strides=(stride_dvk, stride_dvd),
        offsets=(k_tile_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0),
    )
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_idx * stride_qb,
        shape=(n_queries, DIM),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_idx * stride_dob,
        shape=(n_queries, DIM),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_idx * stride_lb,
        shape=(n_queries,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_idx * stride_db,
        shape=(n_queries,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    tile_k = tl.load(K_block_ptr)  # (b_k, d)
    tile_v = tl.load(V_block_ptr)  # (b_k, d)

    tile_dk = tl.zeros((K_TILE_SIZE, DIM), dtype=tl.float32)  # (b_k, d)
    tile_dv = tl.zeros((K_TILE_SIZE, DIM), dtype=tl.float32)  # (b_k, d)

    for q_tile_idx in range(tl.cdiv(n_queries, Q_TILE_SIZE)):
        tile_q = tl.load(Q_block_ptr)  # (b_q, d)
        tile_l = tl.load(L_block_ptr)  # (b_q,)
        tile_do = tl.load(dO_block_ptr)  # (b_q, d)
        tile_d = tl.load(D_block_ptr)  # (b_q,)

        s_ij = tl.dot(tile_q, tl.trans(tile_k)) * scale  # (b_q, b_k)

        if IS_CAUSAL:
            q_offset = q_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_offset = k_tile_idx * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            s_ij = tl.where(q_offset[:, None] >= k_offset[None, :], s_ij, -1e6)

        p_ij = tl.exp(s_ij - tile_l[:, None])  # (b_q, b_k)
        tile_dv += tl.dot(tl.trans(p_ij.to(tile_do.dtype)), tile_do)  # (b_k, d)
        dp_ij = tl.dot(tile_do, tl.trans(tile_v))  # (b_q, b_k)
        ds_ij = p_ij * (dp_ij - tile_d[:, None]) * scale  # (b_q, b_k)
        tile_dk += tl.dot(tl.trans(ds_ij.to(tile_q.dtype)), tile_q)  # (b_k, d)

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))

    tl.store(dK_block_ptr, tile_dk.to(tile_k.dtype))
    tl.store(dV_block_ptr, tile_dv.to(tile_k.dtype))


@triton.jit
def flash_dq_kerel(
    Q_ptr,
    K_ptr,
    V_ptr,
    D_ptr,
    L_ptr,
    dO_ptr,
    dQ_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_db,
    stride_dq,
    stride_lb,
    stride_lq,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_dqb,
    stride_dqq,
    stride_dqd,
    n_queries,
    n_keys,
    scale,
    DIM: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr = False,  # noqa: FBT002 # type: ignore[invalid-parameter-default]
):
    q_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_idx * stride_qb,
        shape=(n_queries, DIM),
        strides=(stride_qq, stride_qd),
        offsets=(q_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_idx * stride_kb,
        shape=(n_keys, DIM),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_idx * stride_vb,
        shape=(n_keys, DIM),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_idx * stride_dob,
        shape=(n_queries, DIM),
        strides=(stride_doq, stride_dod),
        offsets=(q_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_idx * stride_dqb,
        shape=(n_queries, DIM),
        strides=(stride_dqq, stride_dqd),
        offsets=(q_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_idx * stride_db,
        shape=(n_queries,),
        strides=(stride_dq,),
        offsets=(q_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_idx * stride_lb,
        shape=(n_queries,),
        strides=(stride_lq,),
        offsets=(q_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    if IS_CAUSAL:
        q_end = (q_tile_idx + 1) * Q_TILE_SIZE
        num_k_tiles = tl.cdiv(q_end, K_TILE_SIZE)
    else:
        num_k_tiles = tl.cdiv(n_keys, K_TILE_SIZE)

    tile_q = tl.load(Q_block_ptr)  # (b_q, d)
    tile_d = tl.load(D_block_ptr)  # (b_q,)
    tile_l = tl.load(L_block_ptr)  # (b_q,)
    tile_do = tl.load(dO_block_ptr)  # (b_q, d)

    tile_dq = tl.zeros((Q_TILE_SIZE, DIM), dtype=tl.float32)  # (b_q, d)

    for k_tile_idx in range(num_k_tiles):
        tile_k = tl.load(K_block_ptr)  # (b_k, d)
        tile_v = tl.load(V_block_ptr)  # (b_k, d)

        s_ij = tl.dot(tile_q, tl.trans(tile_k)) * scale  # (b_q, b_k)
        if IS_CAUSAL:
            q_offset = q_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_offset = k_tile_idx * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            s_ij = tl.where(q_offset[:, None] >= k_offset[None, :], s_ij, -1e6)

        p_ij = tl.exp(s_ij - tile_l[:, None])  # (b_q, b_k)
        dp_ij = tl.dot(tile_do, tl.trans(tile_v))  # (b_q, b_k)
        ds_ij = p_ij * (dp_ij - tile_d[:, None]) * scale  # (b_q, b_k)

        tile_dq += tl.dot(ds_ij.to(tile_k.dtype), tile_k)  # (b_q, d)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(dQ_block_ptr, tile_dq.to(tile_q.dtype))
