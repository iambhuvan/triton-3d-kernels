"""
K5: Fused SwiGLU — Triton Kernel

Architecture:
  We fuse the element-wise part: SiLU(gate) * up
  The matmuls (W_gate @ x, W_up @ x, W_down @ hidden) stay as cuBLAS calls.

  Grid: one program per BLOCK_SIZE chunk of the flattened tensor
  Each program:
    1. Load gate[offsets] and up[offsets] from HBM -> registers
    2. Compute sigmoid(gate) in registers
    3. Compute silu = gate * sigmoid(gate)
    4. Compute output = silu * up
    5. Store output[offsets] to HBM

  This is a memory-bound kernel (just element-wise ops), so the win comes
  from reducing 3 HBM passes to 1:
    Naive: read gate, write silu(gate), read silu(gate)+up, write result
    Fused: read gate+up, write result

  Backward fuses d_gate and d_up computation in one pass:
    d_up   = dout * SiLU(gate)
    d_gate = dout * up * SiLU'(gate)
    where SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
"""

import torch
import triton
import triton.language as tl


# ============================================================
# Forward Kernel
# ============================================================

@triton.jit
def _swiglu_fwd_kernel(
    GATE_ptr, UP_ptr, OUT_ptr,
    N,  # total number of elements (flattened)
    BLOCK_SIZE: tl.constexpr,
):
    """Fused SiLU(gate) * up forward.

    Math:
      sigmoid(x) = 1 / (1 + exp(-x))
      SiLU(x) = x * sigmoid(x)
      out = SiLU(gate) * up
    """
    # Which block of elements this program handles
    pid = tl.program_id(0)
    # Compute element offsets for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask for out-of-bounds elements (last block may be partial)
    mask = offsets < N

    # Load gate and up from HBM -> registers
    gate = tl.load(GATE_ptr + offsets, mask=mask, other=0.0)
    up = tl.load(UP_ptr + offsets, mask=mask, other=0.0)

    # Cast to FP32 for sigmoid/exp (Triton tl.math.exp requires FP32+)
    gate_f32 = gate.to(tl.float32)
    up_f32 = up.to(tl.float32)

    # Compute SiLU(gate) = gate * sigmoid(gate) entirely in registers
    # sigmoid(x) = 1 / (1 + exp(-x))
    sigmoid_gate = tl.sigmoid(gate_f32)
    silu_gate = gate_f32 * sigmoid_gate

    # Fused output: SiLU(gate) * up
    out = silu_gate * up_f32
    # Cast back to input dtype
    out = out.to(gate.dtype)

    # Store result back to HBM
    tl.store(OUT_ptr + offsets, out, mask=mask)


# ============================================================
# Backward Kernel
# ============================================================

@triton.jit
def _swiglu_bwd_kernel(
    GATE_ptr, UP_ptr, DOUT_ptr,
    DGATE_ptr, DUP_ptr,
    N,  # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """Fused SwiGLU backward — computes d_gate and d_up in one pass.

    Math:
      Forward: out = SiLU(gate) * up

      d_up = dout * SiLU(gate)

      d_gate = dout * up * SiLU'(gate)
      where SiLU'(x) = d/dx [x * sigmoid(x)]
                      = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                      = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

    Why fuse: without fusion, we'd need 3 separate reads of gate from HBM
    (once for sigmoid, once for silu, once for d_silu). Fused = 1 read.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load all inputs: gate, up, dout (3 HBM reads)
    gate = tl.load(GATE_ptr + offsets, mask=mask, other=0.0)
    up = tl.load(UP_ptr + offsets, mask=mask, other=0.0)
    dout = tl.load(DOUT_ptr + offsets, mask=mask, other=0.0)

    # Cast to FP32 for sigmoid/exp (Triton tl.math.exp requires FP32+)
    gate_f32 = gate.to(tl.float32)
    up_f32 = up.to(tl.float32)
    dout_f32 = dout.to(tl.float32)

    # Compute sigmoid(gate) once — reused for both d_gate and d_up
    sigmoid_gate = tl.sigmoid(gate_f32)

    # SiLU(gate) = gate * sigmoid(gate)
    silu_gate = gate_f32 * sigmoid_gate

    # d_up = dout * SiLU(gate)
    d_up = dout_f32 * silu_gate

    # SiLU'(gate) = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
    dsilu_gate = sigmoid_gate * (1.0 + gate_f32 * (1.0 - sigmoid_gate))

    # d_gate = dout * up * SiLU'(gate)
    d_gate = dout_f32 * up_f32 * dsilu_gate

    # Store both gradients, cast back to input dtype (2 HBM writes)
    tl.store(DGATE_ptr + offsets, d_gate.to(gate.dtype), mask=mask)
    tl.store(DUP_ptr + offsets, d_up.to(up.dtype), mask=mask)


# ============================================================
# Autograd wrapper
# ============================================================

class FusedSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        assert gate.is_contiguous() and up.is_contiguous()
        assert gate.shape == up.shape

        ctx.save_for_backward(gate, up)
        out = torch.empty_like(gate)
        N = gate.numel()
        BLOCK_SIZE = 1024

        # Launch grid: one program per BLOCK_SIZE chunk
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        _swiglu_fwd_kernel[grid](
            gate, up, out,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out

    @staticmethod
    def backward(ctx, dout):
        gate, up = ctx.saved_tensors
        dout = dout.contiguous()

        d_gate = torch.empty_like(gate)
        d_up = torch.empty_like(up)
        N = gate.numel()
        BLOCK_SIZE = 1024

        grid = (triton.cdiv(N, BLOCK_SIZE),)
        _swiglu_bwd_kernel[grid](
            gate, up, dout,
            d_gate, d_up,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return d_gate, d_up


# ============================================================
# Public API
# ============================================================

def fused_swiglu(
    x: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
) -> torch.Tensor:
    """Full SwiGLU MLP with fused activation.

    Args:
        x:      (B, S, D) input
        W_gate: (D, D_ff) gate projection
        W_up:   (D, D_ff) up projection
        W_down: (D_ff, D) down projection

    Returns:
        out: (B, S, D)
    """
    gate = x @ W_gate          # cuBLAS
    up = x @ W_up              # cuBLAS
    hidden = FusedSwiGLUFunction.apply(gate, up)  # Triton fused kernel
    out = hidden @ W_down      # cuBLAS
    return out
