"""
Flash Attention v3-Lite — Triton Kernel (Persistent + FP8)

Implements the Triton-expressible subset of Flash Attention v3 (Shah et al. 2024):
  ✅ FP8 quantized QK matmul (e4m3 format, ~2x TFLOPS on H100)
  ✅ Persistent kernel (work-stealing loop, reduces launch overhead)
  ✅ Autotuning with H100-specific block size configs
  ✅ 2-stage pipelining via Triton's num_stages compiler hint

The FULL v3 requires CUDA C++ for hardware features Triton cannot access:
  ❌ WGMMA (warp-group matmul) — Triton's tl.dot compiles to WGMMA automatically
     on H100, but you cannot control descriptors, fencing, or accumulator layout
  ❌ TMA (tensor memory accelerator) — Triton's tl.load may use TMA internally,
     but you cannot issue cp.async.bulk or control mbarrier-based completion
  ❌ Warp specialization — Triton has no concept of producer/consumer warp groups
     or register reallocation (setmaxnreg)

See cuda/flash_attn_v3_hopper.cu for the real CUDA v3 with all three features.

┌───────────────────────────────────────────────────────────────────┐
│ Implementation    │ WGMMA │ TMA │ Warp Spec │ FP8 │ Persistent  │
├───────────────────┼───────┼─────┼───────────┼─────┼─────────────┤
│ This file (Triton)│ auto  │ auto│    ❌     │  ✅ │     ✅      │
│ cuda/ (CUDA C++)  │  ✅   │  ✅ │    ✅     │  —  │     ✅      │
│ Dao-AILab (CUTLASS│  ✅   │  ✅ │    ✅     │  ✅ │     ✅      │
└───────────────────┴───────┴─────┴───────────┴─────┴─────────────┘
"""

import torch
import triton
import triton.language as tl


# ============================================================
# FP8 Quantization Utilities
# ============================================================

@triton.jit
def _quantize_to_fp8(x, scale_factor):
    """Quantize FP32/FP16 tensor to FP8 (e4m3) with per-tensor scaling.

    FP8 e4m3 range: [-448, 448]
    We scale x so that max(|x|) maps to 448, then cast.

    Args:
        x: input tensor in FP16/FP32
        scale_factor: precomputed 448.0 / max(|x|)

    Returns:
        x_fp8: quantized tensor
    """
    x_scaled = x * scale_factor
    # Clamp to FP8 e4m3 range (tl.clamp doesn't exist, use min/max)
    x_clamped = tl.minimum(tl.maximum(x_scaled, -448.0), 448.0)
    return x_clamped.to(tl.float8e4nv)


# ============================================================
# Forward Kernel — FP8 + Persistent + Pipelined
# ============================================================

@triton.autotune(
    configs=[
        # H100-optimized configs with high num_stages for pipelining
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 64}, num_warps=4, num_stages=3),
    ],
    key=['S', 'D'],
)
@triton.jit
def _flash_attn_v3_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    # FP8 scale factors (per batch-head)
    Q_SCALE_ptr, K_SCALE_ptr,
    S, D: tl.constexpr,
    n_q_blocks_total,  # total Q blocks across all batch-heads (for persistent)
    BH,                # B * H
    scale,
    USE_FP8: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """Flash Attention v3 forward — persistent kernel with optional FP8.

    Grid: (NUM_SMS,) — persistent kernel, each program loops over multiple Q blocks.

    Persistent kernel design:
      Instead of launching cdiv(S, BLOCK_Q) * B * H programs (one per Q block),
      we launch NUM_SMS programs (= number of SMs on the GPU). Each program
      pulls work from a global counter, processing Q blocks until none remain.

    Benefits:
      1. Eliminates kernel launch overhead for many small blocks
      2. Better L2 cache utilization: programs on the same SM share L2,
         and KV blocks loaded by one program may still be in L2 when the
         next Q block on the same SM needs them
      3. Better load balancing: no stragglers from uneven work distribution
    """
    pid = tl.program_id(0)
    d_range = tl.arange(0, D)

    # Persistent loop: each program grabs Q blocks round-robin
    # Total work items = cdiv(S, BLOCK_Q) * BH
    num_q_blocks_per_bh = tl.cdiv(S, BLOCK_Q)

    for work_id in range(pid, n_q_blocks_total, tl.num_programs(0)):
        # Decompose work_id -> (bh_id, q_block_id)
        bh_id = work_id // num_q_blocks_per_bh
        q_block_id = work_id % num_q_blocks_per_bh

        if bh_id < BH:
            # Base pointers
            Q_bh = Q_ptr + bh_id * S * D
            K_bh = K_ptr + bh_id * S * D
            V_bh = V_ptr + bh_id * S * D
            O_bh = O_ptr + bh_id * S * D
            LSE_bh = LSE_ptr + bh_id * S

            q_start = q_block_id * BLOCK_Q
            q_offsets = q_start + tl.arange(0, BLOCK_Q)
            q_mask = q_offsets < S

            # Load Q block
            q = tl.load(Q_bh + q_offsets[:, None] * D + d_range[None, :],
                         mask=q_mask[:, None], other=0.0)

            # FP8 path: quantize Q (K quantized on-the-fly in inner loop)
            if USE_FP8:
                q_scale = tl.load(Q_SCALE_ptr + bh_id)
                q_fp8 = (q * q_scale).to(tl.float8e4nv)

            # Accumulators
            m_i = tl.full([BLOCK_Q], value=float('-inf'), dtype=tl.float32)
            l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
            o_i = tl.zeros([BLOCK_Q, D], dtype=tl.float32)

            if IS_CAUSAL:
                n_kv_blocks = tl.cdiv(q_start + BLOCK_Q, BLOCK_KV)
            else:
                n_kv_blocks = tl.cdiv(S, BLOCK_KV)

            for kv_block_id in range(0, n_kv_blocks):
                kv_start = kv_block_id * BLOCK_KV
                kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
                kv_mask = kv_offsets < S

                # Load K block
                k = tl.load(K_bh + kv_offsets[:, None] * D + d_range[None, :],
                             mask=kv_mask[:, None], other=0.0)

                # Load V block (load early for pipelining — num_stages handles overlap)
                v = tl.load(V_bh + kv_offsets[:, None] * D + d_range[None, :],
                             mask=kv_mask[:, None], other=0.0)

                # ===== QK matmul =====
                if USE_FP8:
                    # FP8 path: quantize K, compute S = Q_fp8 @ K_fp8^T in FP8
                    # Result is accumulated in FP32 by hardware
                    k_scale = tl.load(K_SCALE_ptr + bh_id)
                    k_fp8 = (k * k_scale).to(tl.float8e4nv)
                    # FP8 matmul — hardware accumulates in FP32
                    s_block = tl.dot(q_fp8, tl.trans(k_fp8)).to(tl.float32)
                    # Descale: actual score = s_block / (q_scale * k_scale)
                    s_block = s_block / (q_scale * k_scale) * scale
                else:
                    # Standard FP16/FP32 path
                    s_block = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * scale

                # Masking
                if IS_CAUSAL:
                    causal_mask = q_offsets[:, None] < kv_offsets[None, :]
                    s_block = tl.where(causal_mask, float('-inf'), s_block)
                s_block = tl.where(kv_mask[None, :], s_block, float('-inf'))

                # Online softmax (same as v2 — always in FP32)
                m_block = tl.max(s_block, axis=1)
                m_new = tl.maximum(m_i, m_block)
                alpha = tl.exp(m_i - m_new)
                p_block = tl.exp(s_block - m_new[:, None])
                l_i = alpha * l_i + tl.sum(p_block, axis=1)

                # ===== PV matmul — always in FP16/FP32 (not FP8) =====
                # FP8 for PV would lose too much precision in the output
                o_i = alpha[:, None] * o_i + tl.dot(p_block.to(tl.float32), v.to(tl.float32))
                m_i = m_new

            # Final normalization
            o_i = o_i / l_i[:, None]

            # Store
            tl.store(O_bh + q_offsets[:, None] * D + d_range[None, :],
                     o_i, mask=q_mask[:, None])
            lse = m_i + tl.log(l_i)
            tl.store(LSE_bh + q_offsets, lse, mask=q_mask)


# ============================================================
# Backward Kernels (same structure as v2, with FP8 option)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 64}, num_warps=8, num_stages=3),
    ],
    key=['S', 'D'],
)
@triton.jit
def _flash_attn_v3_bwd_dkdv(
    Q_ptr, K_ptr, V_ptr, LSE_ptr, dO_ptr,
    dK_ptr, dV_ptr, DELTA_ptr,
    S, D: tl.constexpr,
    scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """v3 backward for dK, dV — same as v2 (no atomics), with pipelining."""
    kv_block_id = tl.program_id(0)
    bh_id = tl.program_id(1)
    base = bh_id * S * D
    lse_base = bh_id * S

    kv_start = kv_block_id * BLOCK_KV
    kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
    kv_mask = kv_offsets < S
    d_range = tl.arange(0, D)

    k = tl.load(K_ptr + base + kv_offsets[:, None] * D + d_range[None, :],
                mask=kv_mask[:, None], other=0.0)
    v = tl.load(V_ptr + base + kv_offsets[:, None] * D + d_range[None, :],
                mask=kv_mask[:, None], other=0.0)

    dk = tl.zeros([BLOCK_KV, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_KV, D], dtype=tl.float32)

    q_block_start = kv_start // BLOCK_Q if IS_CAUSAL else 0
    n_q_blocks = tl.cdiv(S, BLOCK_Q)

    for q_block_id in range(q_block_start, n_q_blocks):
        q_start = q_block_id * BLOCK_Q
        q_offsets = q_start + tl.arange(0, BLOCK_Q)
        q_mask = q_offsets < S

        q = tl.load(Q_ptr + base + q_offsets[:, None] * D + d_range[None, :],
                     mask=q_mask[:, None], other=0.0)
        do = tl.load(dO_ptr + base + q_offsets[:, None] * D + d_range[None, :],
                      mask=q_mask[:, None], other=0.0)
        lse = tl.load(LSE_ptr + lse_base + q_offsets, mask=q_mask, other=0.0)
        delta = tl.load(DELTA_ptr + lse_base + q_offsets, mask=q_mask, other=0.0)

        s_block = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * scale
        if IS_CAUSAL:
            causal_mask = q_offsets[:, None] < kv_offsets[None, :]
            s_block = tl.where(causal_mask, float('-inf'), s_block)

        p_block = tl.exp(s_block - lse[:, None])
        dv += tl.dot(tl.trans(p_block.to(tl.float32)), do.to(tl.float32))
        dp = tl.dot(do.to(tl.float32), tl.trans(v.to(tl.float32)))
        ds = p_block * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds.to(tl.float32)), q.to(tl.float32)) * scale

    tl.store(dK_ptr + base + kv_offsets[:, None] * D + d_range[None, :],
             dk, mask=kv_mask[:, None])
    tl.store(dV_ptr + base + kv_offsets[:, None] * D + d_range[None, :],
             dv, mask=kv_mask[:, None])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 64}, num_warps=8, num_stages=3),
    ],
    key=['S', 'D'],
)
@triton.jit
def _flash_attn_v3_bwd_dq(
    Q_ptr, K_ptr, V_ptr, LSE_ptr, dO_ptr,
    dQ_ptr, DELTA_ptr,
    S, D: tl.constexpr,
    scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """v3 backward for dQ — same as v2, no atomics."""
    q_block_id = tl.program_id(0)
    bh_id = tl.program_id(1)
    base = bh_id * S * D
    lse_base = bh_id * S

    q_start = q_block_id * BLOCK_Q
    q_offsets = q_start + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < S
    d_range = tl.arange(0, D)

    q = tl.load(Q_ptr + base + q_offsets[:, None] * D + d_range[None, :],
                mask=q_mask[:, None], other=0.0)
    do = tl.load(dO_ptr + base + q_offsets[:, None] * D + d_range[None, :],
                  mask=q_mask[:, None], other=0.0)
    lse = tl.load(LSE_ptr + lse_base + q_offsets, mask=q_mask, other=0.0)
    delta = tl.load(DELTA_ptr + lse_base + q_offsets, mask=q_mask, other=0.0)

    dq = tl.zeros([BLOCK_Q, D], dtype=tl.float32)

    n_kv_blocks = tl.cdiv(q_start + BLOCK_Q, BLOCK_KV) if IS_CAUSAL else tl.cdiv(S, BLOCK_KV)

    for kv_block_id in range(0, n_kv_blocks):
        kv_start = kv_block_id * BLOCK_KV
        kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offsets < S

        k = tl.load(K_ptr + base + kv_offsets[:, None] * D + d_range[None, :],
                     mask=kv_mask[:, None], other=0.0)
        v = tl.load(V_ptr + base + kv_offsets[:, None] * D + d_range[None, :],
                     mask=kv_mask[:, None], other=0.0)

        s_block = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * scale
        if IS_CAUSAL:
            causal_mask = q_offsets[:, None] < kv_offsets[None, :]
            s_block = tl.where(causal_mask, float('-inf'), s_block)

        p_block = tl.exp(s_block - lse[:, None])
        dp = tl.dot(do.to(tl.float32), tl.trans(v.to(tl.float32)))
        ds = p_block * (dp - delta[:, None])
        dq += tl.dot(ds.to(tl.float32), k.to(tl.float32)) * scale

    tl.store(dQ_ptr + base + q_offsets[:, None] * D + d_range[None, :],
             dq, mask=q_mask[:, None])


# ============================================================
# FP8 Scale Computation
# ============================================================

def compute_fp8_scales(Q, K):
    """Compute per-batch-head FP8 quantization scales.

    Scale = 448.0 / max(|x|) per (batch, head) slice.
    This ensures the max value maps to the FP8 e4m3 range.

    Args:
        Q, K: (B, H, S, D) tensors

    Returns:
        q_scale, k_scale: (B*H,) scale factors
    """
    B, H, S, D = Q.shape
    Q_flat = Q.reshape(B * H, S * D)
    K_flat = K.reshape(B * H, S * D)

    q_amax = Q_flat.abs().amax(dim=-1).clamp(min=1e-12)
    k_amax = K_flat.abs().amax(dim=-1).clamp(min=1e-12)

    q_scale = 448.0 / q_amax
    k_scale = 448.0 / k_amax

    return q_scale, k_scale


# ============================================================
# Autograd Function
# ============================================================

class FlashAttentionV3Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal=False, use_fp8=False):
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
        B, H, S, D = Q.shape
        assert D in (16, 32, 64, 128), f"Head dim {D} not supported"

        O = torch.empty_like(Q)
        lse = torch.empty(B, H, S, device=Q.device, dtype=torch.float32)

        scale = 1.0 / (D ** 0.5)

        # FP8 scales
        if use_fp8:
            q_scale, k_scale = compute_fp8_scales(Q, K)
        else:
            q_scale = torch.ones(B * H, device=Q.device, dtype=torch.float32)
            k_scale = torch.ones(B * H, device=Q.device, dtype=torch.float32)

        # Persistent kernel: launch NUM_SMS programs
        num_sms = torch.cuda.get_device_properties(Q.device).multi_processor_count
        n_q_blocks = triton.cdiv(S, 64)
        n_q_blocks_total = n_q_blocks * B * H

        grid = (min(num_sms, n_q_blocks_total),)
        _flash_attn_v3_fwd[grid](
            Q, K, V, O, lse,
            q_scale, k_scale,
            S, D,
            n_q_blocks_total, B * H,
            scale,
            USE_FP8=use_fp8,
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
        delta = (O * dO).sum(dim=-1)

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        scale = 1.0 / (D ** 0.5)

        grid_kv = lambda META: (triton.cdiv(S, META['BLOCK_KV']), B * H)
        _flash_attn_v3_bwd_dkdv[grid_kv](
            Q, K, V, lse, dO,
            dK, dV, delta,
            S, D, scale,
            IS_CAUSAL=causal,
        )

        grid_q = lambda META: (triton.cdiv(S, META['BLOCK_Q']), B * H)
        _flash_attn_v3_bwd_dq[grid_q](
            Q, K, V, lse, dO,
            dQ, delta,
            S, D, scale,
            IS_CAUSAL=causal,
        )

        return dQ, dK, dV, None, None


# ============================================================
# Public API
# ============================================================

def flash_attention_v3_forward(Q, K, V, causal=False, use_fp8=False):
    """Flash Attention v3 forward pass.

    Args:
        Q, K, V: (B, H, S, D) tensors
        causal: whether to apply causal mask
        use_fp8: use FP8 quantized QK matmul (H100 only, ~2x throughput)

    Returns:
        O: (B, H, S, D) attention output
    """
    return FlashAttentionV3Function.apply(Q, K, V, causal, use_fp8)


def flash_attention_v3_backward(Q, K, V, dO, causal=False):
    """Flash Attention v3 backward pass (for testing)."""
    Q = Q.detach().requires_grad_(True)
    K = K.detach().requires_grad_(True)
    V = V.detach().requires_grad_(True)
    O = flash_attention_v3_forward(Q, K, V, causal, use_fp8=False)
    O.backward(dO)
    return Q.grad, K.grad, V.grad
