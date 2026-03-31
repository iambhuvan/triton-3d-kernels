"""
Flash Attention v2 Tests

Tests verify:
  1. v2 matches the PyTorch reference implementation
  2. Backward pass produces correct gradients
  3. Causal masking works correctly
  4. Various shapes and configurations
"""

import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reference.flash_attn_ref import flash_attention_reference
from kernels.flash_attn_v2 import flash_attention_v2_forward as fa_v2


# ============================================================
# Helpers
# ============================================================

def make_qkv(B=2, H=4, S=128, D=64, device='cuda', dtype=torch.float32):
    Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    V = torch.randn(B, H, S, D, device=device, dtype=dtype)
    return Q, K, V


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SKIP_IF_CPU = pytest.mark.skipif(DEVICE == 'cpu', reason="Triton requires CUDA")


# ============================================================
# Test: v2 matches reference
# ============================================================

class TestV2MatchesReference:
    """v2 should produce output identical to standard PyTorch attention."""

    @SKIP_IF_CPU
    def test_v2_matches_reference(self):
        Q, K, V = make_qkv(device=DEVICE)
        ref = flash_attention_reference(Q, K, V)
        v2 = fa_v2(Q, K, V)
        torch.testing.assert_close(v2, ref, atol=5e-3, rtol=5e-3)

    @SKIP_IF_CPU
    @pytest.mark.parametrize("B,H,S,D", [
        (1, 1, 64, 32),
        (2, 8, 256, 64),
        (1, 16, 512, 128),
        (4, 4, 128, 64),
    ])
    def test_v2_various_shapes(self, B, H, S, D):
        Q, K, V = make_qkv(B=B, H=H, S=S, D=D, device=DEVICE)
        ref = flash_attention_reference(Q, K, V)
        out = fa_v2(Q, K, V)
        torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)


# ============================================================
# Test: Causal masking
# ============================================================

class TestCausalMasking:
    """Causal masking should work correctly."""

    @SKIP_IF_CPU
    def test_causal_matches_reference(self):
        Q, K, V = make_qkv(device=DEVICE)
        ref = flash_attention_reference(Q, K, V, causal=True)
        out = fa_v2(Q, K, V, causal=True)
        torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)

    @SKIP_IF_CPU
    def test_causal_first_row(self):
        """First row should only attend to first position."""
        B, H, S, D = 1, 2, 64, 32
        Q, K, V = make_qkv(B=B, H=H, S=S, D=D, device=DEVICE)
        out = fa_v2(Q, K, V, causal=True)
        expected = V[:, :, 0:1, :]
        torch.testing.assert_close(out[:, :, 0:1, :], expected, atol=5e-3, rtol=5e-3)


# ============================================================
# Test: Backward pass
# ============================================================

class TestBackward:
    """Gradients should match PyTorch autograd reference."""

    @SKIP_IF_CPU
    def test_backward_matches_reference(self):
        Q, K, V = make_qkv(B=1, H=2, S=64, D=32, device=DEVICE)
        dO = torch.randn_like(Q)

        # Reference
        Qr = Q.clone().requires_grad_(True)
        Kr = K.clone().requires_grad_(True)
        Vr = V.clone().requires_grad_(True)
        ref = flash_attention_reference(Qr, Kr, Vr)
        ref.backward(dO)

        Qi = Q.clone().requires_grad_(True)
        Ki = K.clone().requires_grad_(True)
        Vi = V.clone().requires_grad_(True)
        out = fa_v2(Qi, Ki, Vi)
        out.backward(dO)

        torch.testing.assert_close(Qi.grad, Qr.grad, atol=5e-2, rtol=5e-2, msg="dQ mismatch")
        torch.testing.assert_close(Ki.grad, Kr.grad, atol=5e-2, rtol=5e-2, msg="dK mismatch")
        torch.testing.assert_close(Vi.grad, Vr.grad, atol=5e-2, rtol=5e-2, msg="dV mismatch")

    @SKIP_IF_CPU
    def test_causal_backward_matches_reference(self):
        Q, K, V = make_qkv(B=1, H=2, S=64, D=32, device=DEVICE)
        dO = torch.randn_like(Q)

        Qr = Q.clone().requires_grad_(True)
        Kr = K.clone().requires_grad_(True)
        Vr = V.clone().requires_grad_(True)
        flash_attention_reference(Qr, Kr, Vr, causal=True).backward(dO)

        Qi = Q.clone().requires_grad_(True)
        Ki = K.clone().requires_grad_(True)
        Vi = V.clone().requires_grad_(True)
        fa_v2(Qi, Ki, Vi, causal=True).backward(dO)

        torch.testing.assert_close(Qi.grad, Qr.grad, atol=5e-2, rtol=5e-2, msg="causal dQ")
        torch.testing.assert_close(Ki.grad, Kr.grad, atol=5e-2, rtol=5e-2, msg="causal dK")
        torch.testing.assert_close(Vi.grad, Vr.grad, atol=5e-2, rtol=5e-2, msg="causal dV")
