"""
K5: Fused SwiGLU MLP — Reference Implementation

SwiGLU is the activation function used in modern transformers (LLaMA, etc.).
It replaces the standard FFN (Linear → ReLU → Linear) with a gated variant:

Standard FFN:
  h = W2 @ ReLU(W1 @ x)

SwiGLU FFN:
  gate = W_gate @ x        # gate projection
  up   = W_up @ x          # up projection
  h    = (gate * SiLU(gate)) * up    # actually: SiLU(gate) * up
  out  = W_down @ h         # down projection

Wait — let me be precise. SwiGLU as used in practice:
  gate_out = SiLU(W_gate @ x)    # gate path with SiLU activation
  up_out   = W_up @ x            # up path (no activation)
  hidden   = gate_out * up_out   # element-wise gating
  out      = W_down @ hidden     # project back to model dim

where SiLU(x) = x * sigmoid(x)

Why this kernel matters:
  The naive implementation does 3 separate operations:
    1. Compute gate = W_gate @ x  (HBM write)
    2. Compute up = W_up @ x      (HBM write)
    3. Compute SiLU(gate) * up    (HBM read gate + up, HBM write result)

  That's 3 HBM round-trips for the intermediate activations. A fused kernel
  can combine steps 1-3: compute both projections, apply SiLU, and multiply
  in one pass, reading x once and writing the result once.

  Note: We fuse the post-projection part (SiLU + elementwise multiply).
  The actual matmuls (W_gate, W_up) stay as cuBLAS calls — we fuse the
  activation and gating that happens BETWEEN the matmuls.
"""

import torch
import torch.nn.functional as F


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


def swiglu_reference(
    x: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
) -> torch.Tensor:
    """Full SwiGLU MLP forward pass.

    Args:
        x:      (B, S, D) input
        W_gate: (D, D_ff) gate projection weights
        W_up:   (D, D_ff) up projection weights
        W_down: (D_ff, D) down projection weights

    Returns:
        out: (B, S, D) output
    """
    gate = x @ W_gate          # (B, S, D_ff)
    up = x @ W_up              # (B, S, D_ff)
    hidden = silu(gate) * up   # (B, S, D_ff) — this is what we fuse
    out = hidden @ W_down      # (B, S, D)
    return out


def swiglu_fused_part_reference(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """Just the fuseable part: SiLU(gate) * up.

    This is the kernel boundary — we fuse this element-wise operation.
    The matmuls before and after stay as cuBLAS calls.

    Args:
        gate: (B, S, D_ff) output of gate projection
        up:   (B, S, D_ff) output of up projection

    Returns:
        hidden: (B, S, D_ff) = SiLU(gate) * up
    """
    return silu(gate) * up
