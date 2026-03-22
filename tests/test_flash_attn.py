"""Tests for K2: Flash Attention kernel."""

import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reference.flash_attn_ref import flash_attention_reference
from kernels.flash_attn import flash_attention_forward, flash_attention_backward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_attn_inputs(B=2, H=8, S=128, D=64, device='cuda', dtype=torch.float32):
    Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    V = torch.randn(B, H, S, D, device=device, dtype=dtype)
    return Q, K, V


class TestFlashAttnForward:
    """Verify forward pass matches standard attention."""

    def test_basic_shapes(self):
        Q, K, V = make_attn_inputs()
        out = flash_attention_forward(Q, K, V)
        assert out.shape == Q.shape

    def test_matches_reference(self):
        Q, K, V = make_attn_inputs()
        ref = flash_attention_reference(Q, K, V)
        tri = flash_attention_forward(Q, K, V)
        torch.testing.assert_close(tri, ref, atol=5e-3, rtol=5e-3)

    def test_causal_matches_reference(self):
        Q, K, V = make_attn_inputs()
        ref = flash_attention_reference(Q, K, V, causal=True)
        tri = flash_attention_forward(Q, K, V, causal=True)
        torch.testing.assert_close(tri, ref, atol=5e-3, rtol=5e-3)

    @pytest.mark.parametrize("B,H,S,D", [
        (1, 1, 32, 32),
        (4, 16, 256, 64),
        (2, 8, 512, 128),
        (1, 4, 64, 64),
    ])
    def test_various_shapes(self, B, H, S, D):
        Q, K, V = make_attn_inputs(B=B, H=H, S=S, D=D)
        ref = flash_attention_reference(Q, K, V)
        tri = flash_attention_forward(Q, K, V)
        torch.testing.assert_close(tri, ref, atol=5e-3, rtol=5e-3)

    def test_identity_attention(self):
        """With Q=K, uniform V, output should be uniform."""
        B, H, S, D = 1, 1, 4, 16
        Q = torch.randn(B, H, S, D, device='cuda')
        V = torch.ones(B, H, S, D, device='cuda')
        out = flash_attention_forward(Q, Q, V)
        # Each row attends to all rows equally weighted -> output should be all-ones
        torch.testing.assert_close(out, V, atol=5e-3, rtol=5e-3)

    def test_causal_first_row(self):
        """First query in causal attention should only attend to first key."""
        B, H, S, D = 1, 1, 32, 16
        Q, K, V = make_attn_inputs(B=B, H=H, S=S, D=D)
        out = flash_attention_forward(Q, K, V, causal=True)
        # First row should equal V[0] (only attends to position 0)
        expected = V[:, :, 0:1, :]
        torch.testing.assert_close(out[:, :, 0:1, :], expected, atol=5e-3, rtol=5e-3)


class TestFlashAttnBackward:
    """Verify backward pass computes correct gradients."""

    def test_backward_shapes(self):
        Q, K, V = make_attn_inputs(B=1, H=2, S=32, D=32)
        Q.requires_grad_(True)
        K.requires_grad_(True)
        V.requires_grad_(True)
        out = flash_attention_forward(Q, K, V)
        out.sum().backward()
        assert Q.grad.shape == Q.shape
        assert K.grad.shape == K.shape
        assert V.grad.shape == V.shape

    def test_backward_matches_pytorch(self):
        """Compare gradients against PyTorch autograd through reference."""
        Q, K, V = make_attn_inputs(B=1, H=2, S=32, D=32)

        # Reference gradients
        Qr = Q.clone().requires_grad_(True)
        Kr = K.clone().requires_grad_(True)
        Vr = V.clone().requires_grad_(True)
        ref_out = flash_attention_reference(Qr, Kr, Vr)
        ref_out.sum().backward()

        # Kernel gradients
        Qk = Q.clone().requires_grad_(True)
        Kk = K.clone().requires_grad_(True)
        Vk = V.clone().requires_grad_(True)
        kern_out = flash_attention_forward(Qk, Kk, Vk)
        kern_out.sum().backward()

        torch.testing.assert_close(Qk.grad, Qr.grad, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(Kk.grad, Kr.grad, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(Vk.grad, Vr.grad, atol=5e-3, rtol=5e-3)

    @pytest.mark.skip(reason="gradcheck requires fp64 for accurate numerical Jacobians; Triton kernels only support fp32")
    def test_gradcheck(self):
        """Numerical gradient check."""
        Q, K, V = make_attn_inputs(B=1, H=1, S=16, D=16)
        Q = Q.float().requires_grad_(True)
        K = K.float().requires_grad_(True)
        V = V.float().requires_grad_(True)
        torch.autograd.gradcheck(
            lambda q, k, v: flash_attention_forward(q, k, v),
            (Q, K, V), eps=1e-3, atol=1e-2, rtol=1e-2,
        )
