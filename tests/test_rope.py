"""Tests for K1: Fused RoPE kernel."""

import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reference.rope_ref import rope_reference, precompute_freqs
from kernels.rope import fused_rope

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_rope_inputs(B=2, S=128, H=8, D=64, device='cuda', dtype=torch.float32):
    """Generate test inputs for RoPE."""
    x = torch.randn(B, S, H, D, device=device, dtype=dtype)
    cos, sin = precompute_freqs(D, S, device=device, dtype=dtype)
    return x, cos, sin


class TestRoPECorrectness:
    """Verify Triton kernel matches reference implementation."""

    def test_basic_shapes(self):
        x, cos, sin = make_rope_inputs()
        out = fused_rope(x, cos, sin)
        assert out.shape == x.shape

    def test_matches_reference(self):
        x, cos, sin = make_rope_inputs()
        ref = rope_reference(x, cos, sin)
        tri = fused_rope(x, cos, sin)
        torch.testing.assert_close(tri, ref, atol=5e-3, rtol=5e-3)

    @pytest.mark.parametrize("B,S,H,D", [
        (1, 32, 1, 64),
        (4, 256, 16, 128),
        (2, 512, 8, 64),
        (1, 1024, 32, 64),
    ])
    def test_various_shapes(self, B, S, H, D):
        x, cos, sin = make_rope_inputs(B=B, S=S, H=H, D=D)
        ref = rope_reference(x, cos, sin)
        tri = fused_rope(x, cos, sin)
        torch.testing.assert_close(tri, ref, atol=5e-3, rtol=5e-3)

    def test_dtype_float16(self):
        x, cos, sin = make_rope_inputs(dtype=torch.float16)
        ref = rope_reference(x, cos, sin)
        tri = fused_rope(x, cos, sin)
        torch.testing.assert_close(tri, ref, atol=1e-2, rtol=1e-2)

    def test_deterministic(self):
        x, cos, sin = make_rope_inputs()
        out1 = fused_rope(x, cos, sin)
        out2 = fused_rope(x, cos, sin)
        torch.testing.assert_close(out1, out2)

    def test_zero_input(self):
        x, cos, sin = make_rope_inputs()
        x = torch.zeros_like(x)
        out = fused_rope(x, cos, sin)
        assert torch.all(out == 0)

    def test_rotation_preserves_norm(self):
        """RoPE is a rotation — should preserve vector norms per (position, head)."""
        x, cos, sin = make_rope_inputs(B=1, S=64, H=4, D=64)
        out = fused_rope(x, cos, sin)
        # Norms per (batch, seq, head) should be preserved
        x_norms = torch.norm(x, dim=-1)
        out_norms = torch.norm(out, dim=-1)
        torch.testing.assert_close(x_norms, out_norms, atol=5e-3, rtol=5e-3)


class TestRoPEGradient:
    """Verify backward pass correctness.

    Note: fused_rope is a forward-only Triton kernel (no custom backward).
    PyTorch cannot autograd through Triton JIT code, so we test that the
    reference (which uses standard PyTorch ops) supports gradients correctly.
    """

    def test_reference_gradcheck(self):
        """Verify the reference implementation supports autograd."""
        from reference.rope_ref import rope_reference
        x, cos, sin = make_rope_inputs(B=1, S=16, H=2, D=32)
        x = x.double().requires_grad_(True)
        cos = cos.double()
        sin = sin.double()
        torch.autograd.gradcheck(
            lambda x: rope_reference(x, cos, sin),
            x, eps=1e-6, atol=5e-3, rtol=5e-3,
        )
