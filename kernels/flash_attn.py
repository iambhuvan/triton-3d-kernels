"""
K2: Flash Attention — Triton Kernel (Forward + Backward)

Forward Architecture:
  Grid: one program per (batch, head, q_block)
  Each program computes attention for BLOCK_Q query rows.

  Algorithm (online softmax from Dao et al.):
    1. Load Q_block: (BLOCK_Q, D)
    2. Initialize accumulators:
       - O = zeros(BLOCK_Q, D)          output accumulator
       - m = -inf * ones(BLOCK_Q)       running row-wise max
       - l = zeros(BLOCK_Q)             running row-wise sum of exp
    3. For each KV block j:
       a. Load K_block, V_block: (BLOCK_KV, D)
       b. S_block = Q_block @ K_block^T * scale     (BLOCK_Q, BLOCK_KV)
       c. If causal: mask upper triangle with -inf
       d. m_new = max(m, rowmax(S_block))            per-row max update
       e. alpha = exp(m - m_new)                     rescaling factor
       f. P_block = exp(S_block - m_new[:, None])    stable softmax numerator
       g. l_new = alpha * l + rowsum(P_block)        running denominator
       h. O = diag(alpha * l / l_new) @ O + (1/l_new)[:, None] * P_block @ V_block
            simplified: O = (alpha * l / l_new)[:, None] * O + P_block @ V_block / l_new[:, None]
          Actually cleaner: O = alpha[:, None] * O + P_block @ V_block
          Then divide by l at the end.
       i. m, l = m_new, l_new
    4. O = O / l[:, None]
    5. Store O, LSE = m + log(l)

Backward Architecture:
  Recomputes attention weights from Q, K, V and stored LSE (no S×S cache).
  Computes dQ, dK, dV using the chain rule through softmax.
"""

import torch
import triton
import triton.language as tl


# ============================================================
# Forward Kernel
# ============================================================

@triton.jit
def _flash_attn_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    # Dims (assumes contiguous (B, H, S, D) layout)
    H, S, D: tl.constexpr,
    scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """Flash Attention forward — clean implementation.

    Grid: (cdiv(S, BLOCK_Q), B * H)
    Program (q_block_id, bh_id) computes O[b, h, q_start:q_start+BLOCK_Q, :]
    """
    q_block_id = tl.program_id(0)
    bh_id = tl.program_id(1)

    # Offset into Q, K, V, O for this (batch, head)
    # bh_id * (S * D) for contiguous (B, H, S, D) layout
    Q_block_ptr = Q_ptr + bh_id * S * D
    K_block_ptr = K_ptr + bh_id * S * D
    V_block_ptr = V_ptr + bh_id * S * D
    O_block_ptr = O_ptr + bh_id * S * D
    LSE_block_ptr = LSE_ptr + bh_id * S

    # Q row indices for this block
    q_start = q_block_id * BLOCK_Q
    q_offsets = q_start + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < S

    # Load Q block: (BLOCK_Q, D)
    d_range = tl.arange(0, D)
    q = tl.load(
        Q_block_ptr + q_offsets[:, None] * D + d_range[None, :],
        mask=q_mask[:, None],
        other=0.0,
    )

    # Initialize accumulators
    m_i = tl.full([BLOCK_Q], value=float('-inf'), dtype=tl.float32)  # running max
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)                      # running sum
    o_i = tl.zeros([BLOCK_Q, D], dtype=tl.float32)                   # running output

    # Number of KV blocks to iterate over
    if IS_CAUSAL:
        # Only need KV blocks up to the diagonal
        n_kv_blocks = tl.cdiv(q_start + BLOCK_Q, BLOCK_KV)
    else:
        n_kv_blocks = tl.cdiv(S, BLOCK_KV)

    # Inner loop over KV blocks
    for kv_block_id in range(0, n_kv_blocks):
        kv_start = kv_block_id * BLOCK_KV
        kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offsets < S

        # Load K block: (BLOCK_KV, D)
        k = tl.load(
            K_block_ptr + kv_offsets[:, None] * D + d_range[None, :],
            mask=kv_mask[:, None],
            other=0.0,
        )

        # S_block = Q @ K^T * scale: (BLOCK_Q, BLOCK_KV)
        s_block = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * scale

        # Causal mask: positions where q_pos < kv_pos get -inf
        if IS_CAUSAL:
            causal_mask = q_offsets[:, None] < kv_offsets[None, :]
            s_block = tl.where(causal_mask, float('-inf'), s_block)

        # Mask out-of-bounds KV positions
        s_block = tl.where(kv_mask[None, :], s_block, float('-inf'))

        # Online softmax update
        # Step 1: new row max
        m_block = tl.max(s_block, axis=1)          # (BLOCK_Q,)
        m_new = tl.maximum(m_i, m_block)            # (BLOCK_Q,)

        # Step 2: rescale old accumulators
        alpha = tl.exp(m_i - m_new)                 # (BLOCK_Q,)

        # Step 3: compute exp(scores - new_max)
        p_block = tl.exp(s_block - m_new[:, None])  # (BLOCK_Q, BLOCK_KV)

        # Step 4: update running sum
        l_new = alpha * l_i + tl.sum(p_block, axis=1)  # (BLOCK_Q,)

        # Step 5: update output accumulator
        # O_new = alpha * O_old + P @ V
        v = tl.load(
            V_block_ptr + kv_offsets[:, None] * D + d_range[None, :],
            mask=kv_mask[:, None],
            other=0.0,
        )
        o_i = alpha[:, None] * o_i + tl.dot(p_block.to(tl.float32), v.to(tl.float32))

        # Step 6: update state
        m_i = m_new
        l_i = l_new

    # Final normalization: O = O / l
    o_i = o_i / l_i[:, None]

    # Store output: (BLOCK_Q, D)
    tl.store(
        O_block_ptr + q_offsets[:, None] * D + d_range[None, :],
        o_i,
        mask=q_mask[:, None],
    )

    # Store LSE = m + log(l) for backward pass
    lse = m_i + tl.log(l_i)
    tl.store(
        LSE_block_ptr + q_offsets,
        lse,
        mask=q_mask,
    )


# ============================================================
# Backward Kernel
# ============================================================

@triton.jit
def _flash_attn_bwd_kernel(
    # Forward inputs
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, dO_ptr,
    # Gradient outputs
    dQ_ptr, dK_ptr, dV_ptr,
    # Precomputed D = rowsum(O * dO) — the "delta" term
    DELTA_ptr,
    # Layout
    S, D: tl.constexpr,
    scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """Flash Attention backward.

    Grid: (cdiv(S, BLOCK_KV), B * H)
    Each program computes dK, dV for one KV block, accumulating over all Q blocks.

    Math:
      P_ij = exp(Q_i @ K_j^T * scale - LSE_i)    (recomputed, not cached)
      dV_j += P_ij^T @ dO_i
      dP_ij = dO_i @ V_j^T
      dS_ij = P_ij * (dP_ij - delta_i)            (softmax backward)
        where delta_i = sum_j(P_ij * dP_ij) = sum_j(dO_i * O_i) = rowsum(dO * O)
      dQ_i += dS_ij @ K_j * scale
      dK_j += dS_ij^T @ Q_i * scale
    """
    kv_block_id = tl.program_id(0)
    bh_id = tl.program_id(1)

    # Offsets
    base = bh_id * S * D
    lse_base = bh_id * S
    delta_base = bh_id * S

    kv_start = kv_block_id * BLOCK_KV
    kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
    kv_mask = kv_offsets < S
    d_range = tl.arange(0, D)

    # Load K, V for this block: (BLOCK_KV, D)
    k = tl.load(K_ptr + base + kv_offsets[:, None] * D + d_range[None, :],
                mask=kv_mask[:, None], other=0.0)
    v = tl.load(V_ptr + base + kv_offsets[:, None] * D + d_range[None, :],
                mask=kv_mask[:, None], other=0.0)

    # Accumulators for dK, dV
    dk = tl.zeros([BLOCK_KV, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_KV, D], dtype=tl.float32)

    # Iterate over Q blocks
    n_q_blocks = tl.cdiv(S, BLOCK_Q)

    for q_block_id in range(0, n_q_blocks):
        q_start = q_block_id * BLOCK_Q
        q_offsets = q_start + tl.arange(0, BLOCK_Q)
        q_mask = q_offsets < S

        # Causal: skip Q blocks that are entirely before this KV block
        should_compute = True
        if IS_CAUSAL:
            if q_start + BLOCK_Q <= kv_start:
                should_compute = False

        if should_compute:
            # Load Q, dO, LSE, delta for this Q block
            q = tl.load(Q_ptr + base + q_offsets[:, None] * D + d_range[None, :],
                         mask=q_mask[:, None], other=0.0)
            do = tl.load(dO_ptr + base + q_offsets[:, None] * D + d_range[None, :],
                          mask=q_mask[:, None], other=0.0)
            lse = tl.load(LSE_ptr + lse_base + q_offsets, mask=q_mask, other=0.0)
            delta = tl.load(DELTA_ptr + delta_base + q_offsets, mask=q_mask, other=0.0)

            # Recompute attention: S = Q @ K^T * scale
            s_block = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * scale  # (BLOCK_Q, BLOCK_KV)

            # Causal mask
            if IS_CAUSAL:
                causal_mask = q_offsets[:, None] < kv_offsets[None, :]
                s_block = tl.where(causal_mask, float('-inf'), s_block)

            # Recompute P from LSE: P = exp(S - LSE)
            p_block = tl.exp(s_block - lse[:, None])  # (BLOCK_Q, BLOCK_KV)

            # dV += P^T @ dO
            dv += tl.dot(tl.trans(p_block.to(tl.float32)), do.to(tl.float32))

            # dP = dO @ V^T
            dp = tl.dot(do.to(tl.float32), tl.trans(v.to(tl.float32)))  # (BLOCK_Q, BLOCK_KV)

            # dS = P * (dP - delta)  — softmax backward
            ds = p_block * (dp - delta[:, None])  # (BLOCK_Q, BLOCK_KV)

            # dK += dS^T @ Q * scale
            dk += tl.dot(tl.trans(ds.to(tl.float32)), q.to(tl.float32)) * scale

            # dQ += dS @ K * scale — need atomic add since multiple KV blocks contribute
            dq_contrib = tl.dot(ds.to(tl.float32), k.to(tl.float32)) * scale  # (BLOCK_Q, D)
            # Atomic add to dQ
            tl.atomic_add(
                dQ_ptr + base + q_offsets[:, None] * D + d_range[None, :],
                dq_contrib,
                mask=q_mask[:, None],
            )

    # Store dK, dV
    tl.store(dK_ptr + base + kv_offsets[:, None] * D + d_range[None, :],
             dk, mask=kv_mask[:, None])
    tl.store(dV_ptr + base + kv_offsets[:, None] * D + d_range[None, :],
             dv, mask=kv_mask[:, None])


# ============================================================
# Autograd Function
# ============================================================

class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal=False):
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
        B, H, S, D = Q.shape

        O = torch.empty_like(Q)
        lse = torch.empty(B, H, S, device=Q.device, dtype=torch.float32)

        # Block sizes — tuned for common head dims
        BLOCK_Q = 64
        BLOCK_KV = 64
        # D must be power of 2 for tl.dot
        assert D in (16, 32, 64, 128), f"Head dim {D} not supported, must be power of 2 <= 128"

        scale = 1.0 / (D ** 0.5)

        grid = (triton.cdiv(S, BLOCK_Q), B * H)
        _flash_attn_fwd[grid](
            Q, K, V, O, lse,
            H, S, D,
            scale,
            IS_CAUSAL=causal,
            BLOCK_Q=BLOCK_Q,
            BLOCK_KV=BLOCK_KV,
        )

        ctx.save_for_backward(Q, K, V, O, lse)
        ctx.causal = causal
        ctx.BLOCK_Q = BLOCK_Q
        ctx.BLOCK_KV = BLOCK_KV
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, lse = ctx.saved_tensors
        causal = ctx.causal
        B, H, S, D = Q.shape

        dO = dO.contiguous()

        # Precompute delta = rowsum(O * dO) — used in softmax backward
        delta = (O * dO).sum(dim=-1)  # (B, H, S)

        dQ = torch.zeros_like(Q)  # zeros because we atomic_add
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        scale = 1.0 / (D ** 0.5)
        BLOCK_Q = ctx.BLOCK_Q
        BLOCK_KV = ctx.BLOCK_KV

        grid = (triton.cdiv(S, BLOCK_KV), B * H)
        _flash_attn_bwd_kernel[grid](
            Q, K, V, O, lse, dO,
            dQ, dK, dV,
            delta,
            S, D,
            scale,
            IS_CAUSAL=causal,
            BLOCK_Q=BLOCK_Q,
            BLOCK_KV=BLOCK_KV,
        )

        return dQ, dK, dV, None


# ============================================================
# Public API
# ============================================================

def flash_attention_forward(Q, K, V, causal=False):
    """Flash Attention forward pass.

    Args:
        Q, K, V: (B, H, S, D) query, key, value tensors
        causal: whether to apply causal mask

    Returns:
        O: (B, H, S, D) attention output
    """
    return FlashAttentionFunction.apply(Q, K, V, causal)


def flash_attention_backward(Q, K, V, dO, causal=False):
    """Flash Attention backward pass (for explicit testing).

    Returns:
        dQ, dK, dV: gradients
    """
    Q = Q.detach().requires_grad_(True)
    K = K.detach().requires_grad_(True)
    V = V.detach().requires_grad_(True)
    O = flash_attention_forward(Q, K, V, causal)
    O.backward(dO)
    return Q.grad, K.grad, V.grad
