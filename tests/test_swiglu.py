"""Tests for K5: Fused SwiGLU MLP kernel."""

import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reference.swiglu_ref import swiglu_reference, swiglu_fused_part_reference
from kernels.swiglu import fused_swiglu, FusedSwiGLUFunction

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_swiglu_inputs(B=2, S=64, D=128, D_ff=512, device='cuda', dtype=torch.float32):
    x = torch.randn(B, S, D, device=device, dtype=dtype)
    W_gate = torch.randn(D, D_ff, device=device, dtype=dtype) * 0.02
    W_up = torch.randn(D, D_ff, device=device, dtype=dtype) * 0.02
    W_down = torch.randn(D_ff, D, device=device, dtype=dtype) * 0.02
    return x, W_gate, W_up, W_down


class TestSwiGLUCorrectness:
    def test_basic_shapes(self):
        x, W_gate, W_up, W_down = make_swiglu_inputs()
        out = fused_swiglu(x, W_gate, W_up, W_down)
        assert out.shape == x.shape

    def test_matches_reference(self):
        x, W_gate, W_up, W_down = make_swiglu_inputs()
        ref = swiglu_reference(x, W_gate, W_up, W_down)
        tri = fused_swiglu(x, W_gate, W_up, W_down)
        torch.testing.assert_close(tri, ref, atol=5e-3, rtol=5e-3)

    def test_fused_part_matches(self):
        """Test just the fused element-wise part (SiLU(gate) * up)."""
        gate = torch.randn(2, 64, 256, device='cuda')
        up = torch.randn(2, 64, 256, device='cuda')
        ref = swiglu_fused_part_reference(gate, up)
        tri = FusedSwiGLUFunction.apply(gate, up)
        torch.testing.assert_close(tri, ref, atol=5e-3, rtol=5e-3)

    @pytest.mark.parametrize("B,S,D,D_ff", [
        (1, 32, 64, 256),
        (4, 128, 256, 1024),
        (2, 256, 512, 2048),
    ])
    def test_various_shapes(self, B, S, D, D_ff):
        x, W_gate, W_up, W_down = make_swiglu_inputs(B=B, S=S, D=D, D_ff=D_ff)
        ref = swiglu_reference(x, W_gate, W_up, W_down)
        tri = fused_swiglu(x, W_gate, W_up, W_down)
        torch.testing.assert_close(tri, ref, atol=5e-3, rtol=5e-3)

    def test_zero_gate(self):
        """SiLU(0) = 0, so output should be 0 regardless of up."""
        gate = torch.zeros(1, 16, 64, device='cuda')
        up = torch.randn(1, 16, 64, device='cuda')
        out = FusedSwiGLUFunction.apply(gate, up)
        torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-6, rtol=1e-6)


class TestSwiGLUGradient:
    def test_backward_shapes(self):
        x, W_gate, W_up, W_down = make_swiglu_inputs(B=1, S=16, D=32, D_ff=64)
        x.requires_grad_(True)
        out = fused_swiglu(x, W_gate, W_up, W_down)
        out.sum().backward()
        assert x.grad.shape == x.shape

    def test_fused_part_gradcheck(self):
        gate = torch.randn(1, 8, 32, dtype=torch.float64, device='cuda', requires_grad=True)
        up = torch.randn(1, 8, 32, dtype=torch.float64, device='cuda', requires_grad=True)
        torch.autograd.gradcheck(
            FusedSwiGLUFunction.apply, (gate, up),
            eps=1e-6, atol=1e-4, rtol=1e-3,
        )
