"""
K1: Fused RoPE — Triton Kernel

Architecture:
  Grid: one program per (batch * head * seq_block) — 1D grid
  Each program handles one (batch, head, seq_position) and processes D elements.

  For each pair of dimensions (x_even, x_odd) at position pos:
    cos_theta = cos[pos, d//2]
    sin_theta = sin[pos, d//2]
    x_even_new = x_even * cos_theta - x_odd * sin_theta
    x_odd_new  = x_even * sin_theta + x_odd * cos_theta

  Why fuse: The naive approach materializes (S, D/2) cos/sin tables in HBM,
  then does 4 element-wise ops with broadcasting. Fused kernel loads cos/sin
  once per position and applies rotation in registers — saves 2 HBM round-trips.
"""

import torch
import triton
import triton.language as tl


# ============================================================
# Triton Kernel
# ============================================================

@triton.jit
def _rope_kernel(
    # Pointers
    X_ptr, OUT_ptr, COS_ptr, SIN_ptr,
    # Strides for X: (B, S, H, D) layout
    stride_xb, stride_xs, stride_xh, stride_xd,
    # Strides for cos/sin: (S, D_half) layout
    stride_cs, stride_cd,
    # Dimensions
    S, H, D_half,
    # Block size
    BLOCK_D: tl.constexpr,  # must be >= D_half
):
    """Fused RoPE kernel.

    Each program handles one (batch, seq_pos, head) combination.
    Processes all D dimensions (D_half pairs) in one shot.

    Grid: (B * S * H,)
    """
    # Map program_id -> (b, s, h)
    pid = tl.program_id(0)
    # Decompose: pid = b * (S * H) + s * H + h
    b = pid // (S * H)
    remainder = pid % (S * H)
    s = remainder // H
    h = remainder % H

    # Base pointer for this (b, s, h) in X
    x_base = X_ptr + b * stride_xb + s * stride_xs + h * stride_xh

    # Offsets for the D_half pairs: [0, 1, ..., D_half-1]
    d_range = tl.arange(0, BLOCK_D)
    mask = d_range < D_half

    # Load even dimensions: x[..., 0], x[..., 2], x[..., 4], ...
    # Even dims are at offset d*2, odd at d*2+1 (interleaved layout)
    even_offsets = d_range * 2 * stride_xd
    odd_offsets = (d_range * 2 + 1) * stride_xd

    x_even = tl.load(x_base + even_offsets, mask=mask, other=0.0)
    x_odd = tl.load(x_base + odd_offsets, mask=mask, other=0.0)

    # Load cos/sin for this position s: cos[s, d], sin[s, d]
    cs_base = COS_ptr + s * stride_cs
    sn_base = SIN_ptr + s * stride_cs

    cos_val = tl.load(cs_base + d_range * stride_cd, mask=mask, other=1.0)
    sin_val = tl.load(sn_base + d_range * stride_cd, mask=mask, other=0.0)

    # Apply 2D rotation to each pair:
    #   x_even_new = x_even * cos - x_odd * sin
    #   x_odd_new  = x_even * sin + x_odd * cos
    out_even = x_even * cos_val - x_odd * sin_val
    out_odd = x_even * sin_val + x_odd * cos_val

    # Store back (same interleaved layout)
    out_base = OUT_ptr + b * stride_xb + s * stride_xs + h * stride_xh
    tl.store(out_base + even_offsets, out_even, mask=mask)
    tl.store(out_base + odd_offsets, out_odd, mask=mask)


# ============================================================
# Python wrapper
# ============================================================

def fused_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply fused RoPE.

    Args:
        x:   (B, S, H, D) input tensor, contiguous
        cos: (S, D//2) precomputed cosine table
        sin: (S, D//2) precomputed sine table

    Returns:
        (B, S, H, D) rotated tensor
    """
    assert x.is_contiguous(), "x must be contiguous"
    assert cos.is_contiguous() and sin.is_contiguous(), "cos/sin must be contiguous"
    B, S, H, D = x.shape
    D_half = D // 2
    assert cos.shape == (S, D_half) and sin.shape == (S, D_half)

    out = torch.empty_like(x)

    # Block size must be power-of-2 >= D_half
    BLOCK_D = triton.next_power_of_2(D_half)

    # Launch one program per (b, s, h)
    grid = (B * S * H,)
    _rope_kernel[grid](
        x, out, cos, sin,
        # Strides for x
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        # Strides for cos/sin
        cos.stride(0), cos.stride(1),
        # Dimensions
        S, H, D_half,
        # Block size
        BLOCK_D=BLOCK_D,
    )

    return out
