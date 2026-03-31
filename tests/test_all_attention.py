"""
Attention Test Suite: Flash Attention v2

Cross-validates my Flash Attention v2 implementation against the PyTorch
reference. Tests forward, backward, causal masking, various shapes, and
edge cases.

Run: pytest tests/test_all_attention.py -v
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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SKIP_IF_CPU = pytest.mark.skipif(DEVICE == 'cpu', reason="Triton requires CUDA")


def make_qkv(B=2, H=4, S=128, D=64, device=DEVICE, dtype=torch.float32):
    Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    V = torch.randn(B, H, S, D, device=device, dtype=dtype)
    return Q, K, V


# ============================================================
# 1. v2 matches PyTorch reference
# ============================================================

class TestV2MatchesReference:
    """Flash Attention v2 must match standard PyTorch attention."""

    @SKIP_IF_CPU
    def test_forward_matches_reference(self):
        Q, K, V = make_qkv()
        ref = flash_attention_reference(Q, K, V)
        out = fa_v2(Q, K, V)
        torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3,
                                   msg="v2 forward doesn't match reference")

    @SKIP_IF_CPU
    def test_causal_matches_reference(self):
        Q, K, V = make_qkv()
        ref = flash_attention_reference(Q, K, V, causal=True)
        out = fa_v2(Q, K, V, causal=True)
        torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3,
                                   msg="v2 causal doesn't match reference")

    @SKIP_IF_CPU
    @pytest.mark.parametrize("B,H,S,D", [
        (1, 1, 32, 16),
        (2, 8, 256, 64),
        (1, 16, 512, 128),
        (4, 4, 64, 32),
    ])
    def test_various_shapes(self, B, H, S, D):
        Q, K, V = make_qkv(B=B, H=H, S=S, D=D)
        ref = flash_attention_reference(Q, K, V)
        out = fa_v2(Q, K, V)
        torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3,
                                   msg=f"v2 shape ({B},{H},{S},{D}) mismatch")


# ============================================================
# 2. Backward pass
# ============================================================

class TestBackward:
    """Gradients must match the reference implementation."""

    @SKIP_IF_CPU
    def test_gradients_match_reference(self):
        Q, K, V = make_qkv(B=1, H=2, S=64, D=32)
        dO = torch.randn_like(Q)

        # Reference gradients
        Qr, Kr, Vr = [t.clone().requires_grad_(True) for t in (Q, K, V)]
        flash_attention_reference(Qr, Kr, Vr).backward(dO)

        Qi, Ki, Vi = [t.clone().requires_grad_(True) for t in (Q, K, V)]
        fa_v2(Qi, Ki, Vi).backward(dO)
        torch.testing.assert_close(Qi.grad, Qr.grad, atol=5e-2, rtol=5e-2, msg="v2 dQ")
        torch.testing.assert_close(Ki.grad, Kr.grad, atol=5e-2, rtol=5e-2, msg="v2 dK")
        torch.testing.assert_close(Vi.grad, Vr.grad, atol=5e-2, rtol=5e-2, msg="v2 dV")

    @SKIP_IF_CPU
    def test_causal_gradients_match(self):
        Q, K, V = make_qkv(B=1, H=2, S=64, D=32)
        dO = torch.randn_like(Q)

        Qr, Kr, Vr = [t.clone().requires_grad_(True) for t in (Q, K, V)]
        flash_attention_reference(Qr, Kr, Vr, causal=True).backward(dO)

        Qi, Ki, Vi = [t.clone().requires_grad_(True) for t in (Q, K, V)]
        fa_v2(Qi, Ki, Vi, causal=True).backward(dO)
        torch.testing.assert_close(Qi.grad, Qr.grad, atol=5e-2, rtol=5e-2, msg="v2 causal dQ")
        torch.testing.assert_close(Ki.grad, Kr.grad, atol=5e-2, rtol=5e-2, msg="v2 causal dK")
        torch.testing.assert_close(Vi.grad, Vr.grad, atol=5e-2, rtol=5e-2, msg="v2 causal dV")


# ============================================================
# 3. Edge cases
# ============================================================

class TestEdgeCases:
    """Edge cases that should work correctly."""

    @SKIP_IF_CPU
    def test_seq_len_1(self):
        """S=1: output should equal V (only one position to attend to)."""
        B, H, S, D = 1, 2, 1, 32
        Q, K, V = make_qkv(B=B, H=H, S=S, D=D)
        out = fa_v2(Q, K, V)
        torch.testing.assert_close(out, V, atol=5e-3, rtol=5e-3, msg="v2 S=1")

    @SKIP_IF_CPU
    def test_single_head(self):
        Q, K, V = make_qkv(B=2, H=1, S=64, D=32)
        ref = flash_attention_reference(Q, K, V)
        out = fa_v2(Q, K, V)
        torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3, msg="v2 H=1")

    @SKIP_IF_CPU
    def test_causal_first_row_is_v0(self):
        """In causal mode, first query only attends to first key -> output = V[0]."""
        B, H, S, D = 1, 2, 64, 32
        Q, K, V = make_qkv(B=B, H=H, S=S, D=D)
        expected = V[:, :, 0:1, :]
        out = fa_v2(Q, K, V, causal=True)
        torch.testing.assert_close(out[:, :, 0:1, :], expected, atol=5e-3, rtol=5e-3,
                                   msg="v2 causal first row")

    @SKIP_IF_CPU
    def test_uniform_v_gives_uniform_output(self):
        """When V is all-ones, output should be all-ones (regardless of attention weights)."""
        B, H, S, D = 1, 2, 32, 16
        Q = torch.randn(B, H, S, D, device=DEVICE)
        K = torch.randn(B, H, S, D, device=DEVICE)
        V = torch.ones(B, H, S, D, device=DEVICE)
        out = fa_v2(Q, K, V)
        torch.testing.assert_close(out, V, atol=5e-3, rtol=5e-3, msg="v2 uniform V")
