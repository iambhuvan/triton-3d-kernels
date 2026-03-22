"""
Comparative tests: Flash Attention v1 vs v2 vs v3

Tests verify:
  1. All three versions produce identical output (within numerical tolerance)
  2. All three versions match the PyTorch reference implementation
  3. Backward passes produce matching gradients
  4. FP8 mode (v3) stays within acceptable tolerance
  5. Causal masking works identically across versions
"""

import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reference.flash_attn_ref import flash_attention_reference
from kernels.flash_attn import flash_attention_forward as fa_v1
from kernels.flash_attn_v2 import flash_attention_v2_forward as fa_v2
from kernels.flash_attn_v3 import flash_attention_v3_forward as fa_v3


# ============================================================
# Helpers
# ============================================================

def make_qkv(B=2, H=4, S=128, D=64, device='cuda', dtype=torch.float32):
    Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    V = torch.randn(B, H, S, D, device=device, dtype=dtype)
    return Q, K, V


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# v2/v3 use @triton.autotune which requires CUDA
SKIP_IF_CPU = pytest.mark.skipif(DEVICE == 'cpu', reason="Triton requires CUDA")


# ============================================================
# Test: All versions match reference
# ============================================================

class TestAllVersionsMatchReference:
    """Each version should produce output identical to standard PyTorch attention."""

    @SKIP_IF_CPU
    def test_v1_matches_reference(self):
        Q, K, V = make_qkv(device=DEVICE)
        ref = flash_attention_reference(Q, K, V)
        v1 = fa_v1(Q, K, V)
        torch.testing.assert_close(v1, ref, atol=5e-3, rtol=5e-3)

    @SKIP_IF_CPU
    def test_v2_matches_reference(self):
        Q, K, V = make_qkv(device=DEVICE)
        ref = flash_attention_reference(Q, K, V)
        v2 = fa_v2(Q, K, V)
        torch.testing.assert_close(v2, ref, atol=5e-3, rtol=5e-3)

    @SKIP_IF_CPU
    def test_v3_matches_reference(self):
        Q, K, V = make_qkv(device=DEVICE)
        ref = flash_attention_reference(Q, K, V)
        v3 = fa_v3(Q, K, V)
        torch.testing.assert_close(v3, ref, atol=5e-3, rtol=5e-3)

    @SKIP_IF_CPU
    def test_v3_fp8_within_tolerance(self):
        """FP8 mode should be close but not exact — quantization introduces error."""
        Q, K, V = make_qkv(device=DEVICE)
        ref = flash_attention_reference(Q, K, V)
        v3_fp8 = fa_v3(Q, K, V, use_fp8=True)
        # FP8 has wider tolerance due to quantization (e4m3 has only 3 mantissa bits)
        torch.testing.assert_close(v3_fp8, ref, atol=0.1, rtol=0.1)


# ============================================================
# Test: Versions match each other
# ============================================================

class TestVersionsMatchEachOther:
    """v1, v2, v3 should produce identical output (they're the same algorithm)."""

    @SKIP_IF_CPU
    def test_v1_v2_match(self):
        Q, K, V = make_qkv(device=DEVICE)
        o1 = fa_v1(Q, K, V)
        o2 = fa_v2(Q, K, V)
        torch.testing.assert_close(o1, o2, atol=5e-3, rtol=5e-3)

    @SKIP_IF_CPU
    def test_v1_v3_match(self):
        Q, K, V = make_qkv(device=DEVICE)
        o1 = fa_v1(Q, K, V)
        o3 = fa_v3(Q, K, V)
        torch.testing.assert_close(o1, o3, atol=5e-3, rtol=5e-3)

    @SKIP_IF_CPU
    def test_v2_v3_match(self):
        Q, K, V = make_qkv(device=DEVICE)
        o2 = fa_v2(Q, K, V)
        o3 = fa_v3(Q, K, V)
        torch.testing.assert_close(o2, o3, atol=5e-3, rtol=5e-3)

    @SKIP_IF_CPU
    @pytest.mark.parametrize("B,H,S,D", [
        (1, 1, 64, 32),
        (2, 8, 256, 64),
        (1, 16, 512, 128),
        (4, 4, 128, 64),
    ])
    def test_all_match_various_shapes(self, B, H, S, D):
        Q, K, V = make_qkv(B=B, H=H, S=S, D=D, device=DEVICE)
        o1 = fa_v1(Q, K, V)
        o2 = fa_v2(Q, K, V)
        o3 = fa_v3(Q, K, V)
        torch.testing.assert_close(o1, o2, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(o2, o3, atol=5e-3, rtol=5e-3)


# ============================================================
# Test: Causal masking
# ============================================================

class TestCausalMasking:
    """Causal masking should work identically across all versions."""

    @SKIP_IF_CPU
    def test_causal_all_match_reference(self):
        Q, K, V = make_qkv(device=DEVICE)
        ref = flash_attention_reference(Q, K, V, causal=True)
        o1 = fa_v1(Q, K, V, causal=True)
        o2 = fa_v2(Q, K, V, causal=True)
        o3 = fa_v3(Q, K, V, causal=True)
        torch.testing.assert_close(o1, ref, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(o2, ref, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(o3, ref, atol=5e-3, rtol=5e-3)

    @SKIP_IF_CPU
    def test_causal_first_row_all_versions(self):
        """First row should only attend to first position."""
        B, H, S, D = 1, 2, 64, 32
        Q, K, V = make_qkv(B=B, H=H, S=S, D=D, device=DEVICE)
        o1 = fa_v1(Q, K, V, causal=True)
        o2 = fa_v2(Q, K, V, causal=True)
        o3 = fa_v3(Q, K, V, causal=True)
        # First row = V[0] (only attends to position 0)
        expected = V[:, :, 0:1, :]
        torch.testing.assert_close(o1[:, :, 0:1, :], expected, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(o2[:, :, 0:1, :], expected, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(o3[:, :, 0:1, :], expected, atol=5e-3, rtol=5e-3)


# ============================================================
# Test: Backward passes match
# ============================================================

class TestBackwardAllVersions:
    """Gradients should match across all versions."""

    @SKIP_IF_CPU
    def test_backward_v1_v2_match(self):
        Q, K, V = make_qkv(B=1, H=2, S=64, D=32, device=DEVICE)
        dO = torch.randn_like(Q)

        # v1 gradients
        Q1 = Q.clone().requires_grad_(True)
        K1 = K.clone().requires_grad_(True)
        V1 = V.clone().requires_grad_(True)
        o1 = fa_v1(Q1, K1, V1)
        o1.backward(dO)

        # v2 gradients
        Q2 = Q.clone().requires_grad_(True)
        K2 = K.clone().requires_grad_(True)
        V2 = V.clone().requires_grad_(True)
        o2 = fa_v2(Q2, K2, V2)
        o2.backward(dO)

        torch.testing.assert_close(Q1.grad, Q2.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(K1.grad, K2.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(V1.grad, V2.grad, atol=5e-2, rtol=5e-2)

    @SKIP_IF_CPU
    def test_backward_v2_v3_match(self):
        Q, K, V = make_qkv(B=1, H=2, S=64, D=32, device=DEVICE)
        dO = torch.randn_like(Q)

        Q2 = Q.clone().requires_grad_(True)
        K2 = K.clone().requires_grad_(True)
        V2 = V.clone().requires_grad_(True)
        o2 = fa_v2(Q2, K2, V2)
        o2.backward(dO)

        Q3 = Q.clone().requires_grad_(True)
        K3 = K.clone().requires_grad_(True)
        V3 = V.clone().requires_grad_(True)
        o3 = fa_v3(Q3, K3, V3)
        o3.backward(dO)

        torch.testing.assert_close(Q2.grad, Q3.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(K2.grad, K3.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(V2.grad, V3.grad, atol=5e-2, rtol=5e-2)

    @SKIP_IF_CPU
    def test_backward_all_match_reference(self):
        """All versions' gradients should match PyTorch autograd reference."""
        Q, K, V = make_qkv(B=1, H=2, S=64, D=32, device=DEVICE)
        dO = torch.randn_like(Q)

        # Reference
        Qr = Q.clone().requires_grad_(True)
        Kr = K.clone().requires_grad_(True)
        Vr = V.clone().requires_grad_(True)
        ref = flash_attention_reference(Qr, Kr, Vr)
        ref.backward(dO)

        for fa_fn, name in [(fa_v1, "v1"), (fa_v2, "v2"), (fa_v3, "v3")]:
            Qi = Q.clone().requires_grad_(True)
            Ki = K.clone().requires_grad_(True)
            Vi = V.clone().requires_grad_(True)
            oi = fa_fn(Qi, Ki, Vi)
            oi.backward(dO)
            torch.testing.assert_close(Qi.grad, Qr.grad, atol=5e-2, rtol=5e-2,
                                       msg=f"{name} dQ mismatch")
            torch.testing.assert_close(Ki.grad, Kr.grad, atol=5e-2, rtol=5e-2,
                                       msg=f"{name} dK mismatch")
            torch.testing.assert_close(Vi.grad, Vr.grad, atol=5e-2, rtol=5e-2,
                                       msg=f"{name} dV mismatch")


# ============================================================
# Test: FP8 specific tests
# ============================================================

class TestFP8:
    """Tests specific to v3's FP8 quantization path."""

    @SKIP_IF_CPU
    def test_fp8_output_not_nan(self):
        Q, K, V = make_qkv(device=DEVICE)
        o = fa_v3(Q, K, V, use_fp8=True)
        assert not torch.isnan(o).any(), "FP8 output contains NaN"

    @SKIP_IF_CPU
    def test_fp8_output_not_inf(self):
        Q, K, V = make_qkv(device=DEVICE)
        o = fa_v3(Q, K, V, use_fp8=True)
        assert not torch.isinf(o).any(), "FP8 output contains Inf"

    @SKIP_IF_CPU
    def test_fp8_vs_fp16_relative_error(self):
        """FP8 should have bounded relative error vs FP16 for typical inputs.

        FP8 e4m3 has only 3 mantissa bits, giving ~1/8 relative precision per
        element.  After matrix multiplications the errors compound, so mean
        relative error of 30-50% is expected for attention output.
        """
        Q, K, V = make_qkv(B=2, H=8, S=256, D=64, device=DEVICE)
        o_fp16 = fa_v3(Q, K, V, use_fp8=False)
        o_fp8 = fa_v3(Q, K, V, use_fp8=True)

        rel_error = (o_fp8 - o_fp16).abs() / (o_fp16.abs() + 1e-8)
        mean_rel_error = rel_error.mean().item()
        assert mean_rel_error < 0.50, f"FP8 mean relative error {mean_rel_error:.4f} > 50%"

    @SKIP_IF_CPU
    def test_fp8_large_values_handled(self):
        """FP8 should handle inputs with large magnitudes via scaling."""
        Q, K, V = make_qkv(device=DEVICE)
        Q = Q * 100  # large values
        K = K * 100
        o = fa_v3(Q, K, V, use_fp8=True)
        assert not torch.isnan(o).any()

    @SKIP_IF_CPU
    def test_fp8_small_values_handled(self):
        """FP8 should handle inputs with very small magnitudes."""
        Q, K, V = make_qkv(device=DEVICE)
        Q = Q * 1e-4
        K = K * 1e-4
        o = fa_v3(Q, K, V, use_fp8=True)
        assert not torch.isnan(o).any()
