"""
Flash Attention v2 — Triton Kernel (Dao 2023)

Key differences from v1 (Dao et al. 2022):

┌──────────────────────────────────────────────────────────────────────────┐
│                        v1 vs v2 Changes                                  │
├──────────────────────┬───────────────────────┬───────────────────────────┤
│ Aspect               │ v1 (Dao et al. 2022)  │ v2 (Dao 2023)            │
├──────────────────────┼───────────────────────┼───────────────────────────┤
│ Forward inner loop   │ Rescale O by l/l_new  │ Defer all normalization   │
│                      │ each iteration        │ to end (fewer non-matmul  │
│                      │                       │ FLOPs in inner loop)      │
├──────────────────────┼───────────────────────┼───────────────────────────┤
│ Backward structure   │ Single kernel:        │ Two separate kernels:     │
│                      │ outer loop over KV,   │ 1. dK/dV: outer=KV,      │
│                      │ atomic_add for dQ     │    inner=Q (no atomics)   │
│                      │                       │ 2. dQ: outer=Q,           │
│                      │                       │    inner=KV (no atomics)  │
├──────────────────────┼───────────────────────┼───────────────────────────┤
│ Causal optimization  │ Skip KV blocks past   │ Skip KV blocks past      │
│                      │ diagonal              │ diagonal AND skip Q       │
│                      │                       │ blocks before diagonal    │
│                      │                       │ in bwd (tighter bounds)   │
├──────────────────────┼───────────────────────┼───────────────────────────┤
│ Autotuning           │ Fixed block sizes     │ @triton.autotune over     │
│                      │                       │ BLOCK_Q, BLOCK_KV,        │
│                      │                       │ num_warps, num_stages     │
└──────────────────────┴───────────────────────┴───────────────────────────┘

Why v2 is faster:
  1. Fewer non-matmul FLOPs in forward inner loop (no l/l_new division per step)
  2. No atomic_add in backward → eliminates memory contention
  3. Tighter causal bounds → skip more useless work
  4. Autotuning → adapts block sizes to hardware
"""

import torch
import triton
import triton.language as tl


# ============================================================
# Forward Kernel — with autotuning
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 64}, num_warps=8, num_stages=2),
    ],
    key=['S', 'D'],
)
@triton.jit
def _flash_attn_v2_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, D: tl.constexpr,
    scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """Flash Attention v2 forward.

    Grid: (cdiv(S, BLOCK_Q), B * H)

    Key difference from v1:
      v1 inner loop: O = (alpha * l / l_new) * O + P @ V / l_new    [division per step]
      v2 inner loop: O = alpha * O + P @ V                           [NO division]
      v2 after loop: O = O / l                                       [single division]

    This eliminates one element-wise division per inner loop iteration.
    Sounds small, but non-matmul FLOPs compete with matmul FLOPs for
    execution units — reducing them improves occupancy and throughput.
    """
    q_block_id = tl.program_id(0)
    bh_id = tl.program_id(1)

    # Base pointers for this (batch, head)
    Q_bh = Q_ptr + bh_id * S * D
    K_bh = K_ptr + bh_id * S * D
    V_bh = V_ptr + bh_id * S * D
    O_bh = O_ptr + bh_id * S * D
    LSE_bh = LSE_ptr + bh_id * S

    q_start = q_block_id * BLOCK_Q
    q_offsets = q_start + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < S
    d_range = tl.arange(0, D)

    # Load Q: (BLOCK_Q, D)
    q = tl.load(Q_bh + q_offsets[:, None] * D + d_range[None, :],
                mask=q_mask[:, None], other=0.0)

    # Accumulators — v2 keeps O unnormalized until the end
    m_i = tl.full([BLOCK_Q], value=float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_Q, D], dtype=tl.float32)

    # KV iteration bounds
    if IS_CAUSAL:
        kv_end = tl.minimum(q_start + BLOCK_Q, S)
        n_kv_blocks = tl.cdiv(kv_end, BLOCK_KV)
    else:
        n_kv_blocks = tl.cdiv(S, BLOCK_KV)

    for kv_block_id in range(0, n_kv_blocks):
        kv_start = kv_block_id * BLOCK_KV
        kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offsets < S

        # Load K: (BLOCK_KV, D)
        k = tl.load(K_bh + kv_offsets[:, None] * D + d_range[None, :],
                     mask=kv_mask[:, None], other=0.0)

        # Load V: (BLOCK_KV, D)  [v2: load V early, separate from K]
        v = tl.load(V_bh + kv_offsets[:, None] * D + d_range[None, :],
                     mask=kv_mask[:, None], other=0.0)

        # S = Q @ K^T * scale
        s_block = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * scale

        # Masking
        if IS_CAUSAL:
            causal_mask = q_offsets[:, None] < kv_offsets[None, :]
            s_block = tl.where(causal_mask, float('-inf'), s_block)
        s_block = tl.where(kv_mask[None, :], s_block, float('-inf'))

        # ===== Online softmax (v2 simplified) =====
        # Step 1: row max of this block
        m_block = tl.max(s_block, axis=1)
        m_new = tl.maximum(m_i, m_block)

        # Step 2: rescale factor for old accumulators
        alpha = tl.exp(m_i - m_new)

        # Step 3: new block's softmax numerators
        p_block = tl.exp(s_block - m_new[:, None])

        # Step 4: update l (running sum of exp)
        l_i = alpha * l_i + tl.sum(p_block, axis=1)

        # Step 5: update O — NO DIVISION HERE (v2 key difference)
        # v1 would do: O = (alpha * l_old / l_new) * O + P @ V / l_new
        # v2 just does: O = alpha * O + P @ V
        o_i = alpha[:, None] * o_i + tl.dot(p_block.to(tl.float32), v.to(tl.float32))

        # Step 6: update max
        m_i = m_new

    # ===== Final normalization (v2: single division at the end) =====
    o_i = o_i / l_i[:, None]

    # Store output
    tl.store(O_bh + q_offsets[:, None] * D + d_range[None, :],
             o_i, mask=q_mask[:, None])

    # Store LSE for backward
    lse = m_i + tl.log(l_i)
    tl.store(LSE_bh + q_offsets, lse, mask=q_mask)


# ============================================================
# Backward Kernel 1: dK, dV (outer loop = KV blocks, no atomics)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 128}, num_warps=8, num_stages=2),
    ],
    key=['S', 'D'],
)
@triton.jit
def _flash_attn_v2_bwd_dkdv(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, dO_ptr,
    dK_ptr, dV_ptr,
    DELTA_ptr,
    S, D: tl.constexpr,
    scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """Backward kernel for dK and dV.

    Grid: (cdiv(S, BLOCK_KV), B * H)
    Each program owns one KV block and accumulates dK, dV locally.
    Inner loop iterates over Q blocks.

    v2 difference from v1: identical grid structure, but now this is a
    DEDICATED kernel for dK/dV — no dQ computation here, no atomic_add.
    """
    kv_block_id = tl.program_id(0)
    bh_id = tl.program_id(1)

    base = bh_id * S * D
    lse_base = bh_id * S

    kv_start = kv_block_id * BLOCK_KV
    kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
    kv_mask = kv_offsets < S
    d_range = tl.arange(0, D)

    # Load K, V: owned by this program, stays in registers
    k = tl.load(K_ptr + base + kv_offsets[:, None] * D + d_range[None, :],
                mask=kv_mask[:, None], other=0.0)
    v = tl.load(V_ptr + base + kv_offsets[:, None] * D + d_range[None, :],
                mask=kv_mask[:, None], other=0.0)

    dk = tl.zeros([BLOCK_KV, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_KV, D], dtype=tl.float32)

    # Q block iteration bounds for causal
    if IS_CAUSAL:
        # Only Q blocks where q_start >= kv_start need to attend to this KV block
        q_block_start = kv_start // BLOCK_Q
    else:
        q_block_start = 0

    n_q_blocks = tl.cdiv(S, BLOCK_Q)

    for q_block_id in range(q_block_start, n_q_blocks):
        q_start = q_block_id * BLOCK_Q
        q_offsets = q_start + tl.arange(0, BLOCK_Q)
        q_mask = q_offsets < S

        # Load Q, dO for this Q block
        q = tl.load(Q_ptr + base + q_offsets[:, None] * D + d_range[None, :],
                     mask=q_mask[:, None], other=0.0)
        do = tl.load(dO_ptr + base + q_offsets[:, None] * D + d_range[None, :],
                      mask=q_mask[:, None], other=0.0)
        lse = tl.load(LSE_ptr + lse_base + q_offsets, mask=q_mask, other=0.0)
        delta = tl.load(DELTA_ptr + lse_base + q_offsets, mask=q_mask, other=0.0)

        # Recompute: S = Q @ K^T * scale
        s_block = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * scale

        if IS_CAUSAL:
            causal_mask = q_offsets[:, None] < kv_offsets[None, :]
            s_block = tl.where(causal_mask, float('-inf'), s_block)

        # Recompute P = exp(S - LSE)
        p_block = tl.exp(s_block - lse[:, None])

        # dV += P^T @ dO
        dv += tl.dot(tl.trans(p_block.to(tl.float32)), do.to(tl.float32))

        # dP = dO @ V^T
        dp = tl.dot(do.to(tl.float32), tl.trans(v.to(tl.float32)))

        # dS = P * (dP - delta)
        ds = p_block * (dp - delta[:, None])

        # dK += dS^T @ Q * scale
        dk += tl.dot(tl.trans(ds.to(tl.float32)), q.to(tl.float32)) * scale

    # Store dK, dV — no atomics needed!
    tl.store(dK_ptr + base + kv_offsets[:, None] * D + d_range[None, :],
             dk, mask=kv_mask[:, None])
    tl.store(dV_ptr + base + kv_offsets[:, None] * D + d_range[None, :],
             dv, mask=kv_mask[:, None])


# ============================================================
# Backward Kernel 2: dQ (outer loop = Q blocks, no atomics)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 128}, num_warps=8, num_stages=2),
    ],
    key=['S', 'D'],
)
@triton.jit
def _flash_attn_v2_bwd_dq(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, dO_ptr,
    dQ_ptr,
    DELTA_ptr,
    S, D: tl.constexpr,
    scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """Backward kernel for dQ.

    Grid: (cdiv(S, BLOCK_Q), B * H)
    Each program owns one Q block and accumulates dQ locally.
    Inner loop iterates over KV blocks.

    This is the v2 key insight: by splitting backward into two kernels
    (one for dK/dV, one for dQ), NEITHER kernel needs atomic_add.
    v1 had to atomic_add dQ because multiple KV-block programs contributed
    to the same dQ rows. Now each Q-block program computes its own dQ.
    """
    q_block_id = tl.program_id(0)
    bh_id = tl.program_id(1)

    base = bh_id * S * D
    lse_base = bh_id * S

    q_start = q_block_id * BLOCK_Q
    q_offsets = q_start + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < S
    d_range = tl.arange(0, D)

    # Load Q, dO, LSE, delta: owned by this program
    q = tl.load(Q_ptr + base + q_offsets[:, None] * D + d_range[None, :],
                mask=q_mask[:, None], other=0.0)
    do = tl.load(dO_ptr + base + q_offsets[:, None] * D + d_range[None, :],
                  mask=q_mask[:, None], other=0.0)
    lse = tl.load(LSE_ptr + lse_base + q_offsets, mask=q_mask, other=0.0)
    delta = tl.load(DELTA_ptr + lse_base + q_offsets, mask=q_mask, other=0.0)

    dq = tl.zeros([BLOCK_Q, D], dtype=tl.float32)

    # KV iteration bounds
    if IS_CAUSAL:
        n_kv_blocks = tl.cdiv(q_start + BLOCK_Q, BLOCK_KV)
    else:
        n_kv_blocks = tl.cdiv(S, BLOCK_KV)

    for kv_block_id in range(0, n_kv_blocks):
        kv_start = kv_block_id * BLOCK_KV
        kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offsets < S

        k = tl.load(K_ptr + base + kv_offsets[:, None] * D + d_range[None, :],
                     mask=kv_mask[:, None], other=0.0)
        v = tl.load(V_ptr + base + kv_offsets[:, None] * D + d_range[None, :],
                     mask=kv_mask[:, None], other=0.0)

        # Recompute S and P
        s_block = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * scale

        if IS_CAUSAL:
            causal_mask = q_offsets[:, None] < kv_offsets[None, :]
            s_block = tl.where(causal_mask, float('-inf'), s_block)

        p_block = tl.exp(s_block - lse[:, None])

        # dP = dO @ V^T
        dp = tl.dot(do.to(tl.float32), tl.trans(v.to(tl.float32)))

        # dS = P * (dP - delta)
        ds = p_block * (dp - delta[:, None])

        # dQ += dS @ K * scale  — local accumulation, no atomics!
        dq += tl.dot(ds.to(tl.float32), k.to(tl.float32)) * scale

    # Store dQ — direct store, no atomic needed
    tl.store(dQ_ptr + base + q_offsets[:, None] * D + d_range[None, :],
             dq, mask=q_mask[:, None])


# ============================================================
# Autograd Function
# ============================================================

class FlashAttentionV2Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal=False):
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
        B, H, S, D = Q.shape
        assert D in (16, 32, 64, 128), f"Head dim {D} not supported"

        O = torch.empty_like(Q)
        lse = torch.empty(B, H, S, device=Q.device, dtype=torch.float32)

        scale = 1.0 / (D ** 0.5)

        grid = lambda META: (triton.cdiv(S, META['BLOCK_Q']), B * H)
        _flash_attn_v2_fwd[grid](
            Q, K, V, O, lse,
            S, D, scale,
            IS_CAUSAL=causal,
        )

        ctx.save_for_backward(Q, K, V, O, lse)
        ctx.causal = causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, lse = ctx.saved_tensors
        causal = ctx.causal
        B, H, S, D = Q.shape

        dO = dO.contiguous()
        delta = (O * dO).sum(dim=-1)  # (B, H, S)

        dQ = torch.empty_like(Q)  # empty, not zeros — no atomics needed!
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        scale = 1.0 / (D ** 0.5)

        # Kernel 1: dK, dV
        grid_kv = lambda META: (triton.cdiv(S, META['BLOCK_KV']), B * H)
        _flash_attn_v2_bwd_dkdv[grid_kv](
            Q, K, V, O, lse, dO,
            dK, dV, delta,
            S, D, scale,
            IS_CAUSAL=causal,
        )

        # Kernel 2: dQ (separate kernel, no atomics)
        grid_q = lambda META: (triton.cdiv(S, META['BLOCK_Q']), B * H)
        _flash_attn_v2_bwd_dq[grid_q](
            Q, K, V, O, lse, dO,
            dQ, delta,
            S, D, scale,
            IS_CAUSAL=causal,
        )

        return dQ, dK, dV, None


# ============================================================
# Public API
# ============================================================

def flash_attention_v2_forward(Q, K, V, causal=False):
    """Flash Attention v2 forward pass.

    Args:
        Q, K, V: (B, H, S, D) tensors
        causal: whether to apply causal mask

    Returns:
        O: (B, H, S, D) attention output
    """
    return FlashAttentionV2Function.apply(Q, K, V, causal)


def flash_attention_v2_backward(Q, K, V, dO, causal=False):
    """Flash Attention v2 backward pass (for testing)."""
    Q = Q.detach().requires_grad_(True)
    K = K.detach().requires_grad_(True)
    V = V.detach().requires_grad_(True)
    O = flash_attention_v2_forward(Q, K, V, causal)
    O.backward(dO)
    return Q.grad, K.grad, V.grad
