"""
K1: Fused Rotary Positional Embedding (RoPE) — Reference Implementation

RoPE applies a rotation to pairs of dimensions in the query/key vectors based on
their position in the sequence. This allows the model to encode relative position
information directly into the attention scores.

Math:
  Given input x of shape (B, S, H, D) where D is even:
    x_rot = x reshaped as pairs: (x0, x1), (x2, x3), ...
    For each pair (x_even, x_odd) at position pos and dimension d:
      cos_theta = cos(pos * freq_d)
      sin_theta = sin(pos * freq_d)
      x_even_new = x_even * cos_theta - x_odd * sin_theta
      x_odd_new  = x_even * sin_theta + x_odd * cos_theta

    where freq_d = 1 / (base^(2d / D)), base=10000

Why this kernel matters:
  The naive implementation materializes the full (S, D) sin/cos table in HBM,
  reads it back, and applies element-wise ops. A fused kernel computes sin/cos
  on-the-fly in SRAM, avoiding the HBM round-trip entirely.
"""

import torch
import torch.nn.functional as F


def precompute_freqs(dim: int, seq_len: int, base: float = 10000.0,
                     device: torch.device = None, dtype: torch.dtype = torch.float32):
    """Precompute the frequency table for RoPE.

    Returns:
        cos: (seq_len, dim//2) cosine table
        sin: (seq_len, dim//2) sine table
    """
    # freq_d = 1 / (base^(2d / D)) for d in [0, 1, ..., D/2 - 1]
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    # positions: [0, 1, ..., seq_len-1]
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    # outer product: (seq_len, dim//2)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def rope_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to input tensor.

    Args:
        x:   (B, S, H, D) input query or key tensor
        cos: (S, D//2) precomputed cosine table
        sin: (S, D//2) precomputed sine table

    Returns:
        (B, S, H, D) rotated tensor
    """
    B, S, H, D = x.shape
    assert D % 2 == 0, "Head dimension must be even for RoPE"

    # Split into even and odd dimensions
    x_even = x[..., 0::2]  # (B, S, H, D//2)
    x_odd = x[..., 1::2]   # (B, S, H, D//2)

    # Reshape cos/sin for broadcasting: (1, S, 1, D//2)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    # Apply rotation
    out_even = x_even * cos - x_odd * sin
    out_odd  = x_even * sin + x_odd * cos

    # Interleave back: stack on last dim then flatten
    out = torch.stack([out_even, out_odd], dim=-1)  # (B, S, H, D//2, 2)
    out = out.reshape(B, S, H, D)

    return out
