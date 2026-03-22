"""
Unified Attention Test Suite: v1 vs v2 vs v3 vs Sparse vs Ring

This is the master test file that cross-validates ALL attention implementations
against each other and against the PyTorch reference. This is the test you
run before the interview to prove everything works.

Tests:
  1. All dense attention versions (v1, v2, v3, v3-FP8) match reference
  2. All dense versions match each other
  3. Sparse attention matches reference (with full-K equivalence test)
  4. Ring attention matches reference (for all chunk counts)
  5. Causal masking works across ALL variants
  6. Backward passes match across v1, v2, v3
  7. Scaling behavior: correctness holds at various (B, H, S, D)
  8. Edge cases: S=1, single head, large D

Run: pytest tests/test_all_attention.py -v
"""

import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reference.flash_attn_ref import flash_attention_reference
from reference.sparse_flash_attn_ref import sparse_flash_attention_reference
from reference.ring_attn_ref import ring_attention_reference
from kernels.flash_attn import flash_attention_forward as fa_v1
from kernels.flash_attn_v2 import flash_attention_v2_forward as fa_v2
from kernels.flash_attn_v3 import flash_attention_v3_forward as fa_v3
from kernels.sparse_flash_attn import sparse_flash_attention
from kernels.ring_attn import ring_attention


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


def make_coords(B=2, S=128, device=DEVICE, dtype=torch.float32):
    return torch.randn(B, S, 3, device=device, dtype=dtype)


ALL_DENSE_FNS = {
    'v1': fa_v1,
    'v2': fa_v2,
    'v3': fa_v3,
}


# ============================================================
# 1. All dense versions match PyTorch reference
# ============================================================

class TestDenseMatchesReference:
    """Every dense attention version must match standard PyTorch attention."""

    @SKIP_IF_CPU
    @pytest.mark.parametrize("name,fn", list(ALL_DENSE_FNS.items()))
    def test_forward_matches_reference(self, name, fn):
        Q, K, V = make_qkv()
        ref = flash_attention_reference(Q, K, V)
        out = fn(Q, K, V)
        torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3,
                                   msg=f"{name} forward doesn't match reference")

    @SKIP_IF_CPU
    @pytest.mark.parametrize("name,fn", list(ALL_DENSE_FNS.items()))
    def test_causal_matches_reference(self, name, fn):
        Q, K, V = make_qkv()
        ref = flash_attention_reference(Q, K, V, causal=True)
        out = fn(Q, K, V, causal=True)
        torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3,
                                   msg=f"{name} causal doesn't match reference")

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
        for name, fn in ALL_DENSE_FNS.items():
            out = fn(Q, K, V)
            torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3,
                                       msg=f"{name} shape ({B},{H},{S},{D}) mismatch")


# ============================================================
# 2. All dense versions match each other
# ============================================================

class TestDenseVersionsMatch:
    """v1, v2, v3 must produce identical output (same algorithm, different scheduling)."""

    @SKIP_IF_CPU
    def test_v1_v2_v3_match(self):
        Q, K, V = make_qkv()
        o1 = fa_v1(Q, K, V)
        o2 = fa_v2(Q, K, V)
        o3 = fa_v3(Q, K, V)
        torch.testing.assert_close(o1, o2, atol=5e-3, rtol=5e-3, msg="v1 vs v2")
        torch.testing.assert_close(o2, o3, atol=5e-3, rtol=5e-3, msg="v2 vs v3")
        torch.testing.assert_close(o1, o3, atol=5e-3, rtol=5e-3, msg="v1 vs v3")

    @SKIP_IF_CPU
    def test_v1_v2_v3_causal_match(self):
        Q, K, V = make_qkv()
        o1 = fa_v1(Q, K, V, causal=True)
        o2 = fa_v2(Q, K, V, causal=True)
        o3 = fa_v3(Q, K, V, causal=True)
        torch.testing.assert_close(o1, o2, atol=5e-3, rtol=5e-3, msg="v1 vs v2 causal")
        torch.testing.assert_close(o2, o3, atol=5e-3, rtol=5e-3, msg="v2 vs v3 causal")

    @SKIP_IF_CPU
    def test_v3_fp8_close_to_fp16(self):
        """FP8 introduces quantization error but should be close."""
        Q, K, V = make_qkv()
        o_fp16 = fa_v3(Q, K, V, use_fp8=False)
        o_fp8 = fa_v3(Q, K, V, use_fp8=True)
        torch.testing.assert_close(o_fp8, o_fp16, atol=0.15, rtol=0.15,
                                   msg="v3 FP8 too far from FP16")


# ============================================================
# 3. Sparse attention tests
# ============================================================

class TestSparseAttention:
    """Sparse attention with KNN mask from 3D coordinates."""

    @SKIP_IF_CPU
    def test_matches_reference(self):
        Q, K, V = make_qkv(S=64, D=32)
        coords = make_coords(S=64)
        k = 16
        ref = sparse_flash_attention_reference(Q, K, V, coords, k)
        out = sparse_flash_attention(Q, K, V, coords, k)
        torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)

    @SKIP_IF_CPU
    def test_full_k_equals_dense_attention(self):
        """When k=S (all points are neighbors), sparse = dense attention."""
        B, H, S, D = 1, 2, 32, 16
        Q, K, V = make_qkv(B=B, H=H, S=S, D=D)
        coords = make_coords(B=B, S=S)

        sparse_out = sparse_flash_attention(Q, K, V, coords, k=S)
        dense_out = flash_attention_reference(Q, K, V)
        torch.testing.assert_close(sparse_out, dense_out, atol=5e-3, rtol=5e-3,
                                   msg="Sparse with k=S should equal dense")

    @SKIP_IF_CPU
    def test_sparse_vs_dense_versions(self):
        """Sparse with k=S should also match v1, v2, v3."""
        B, H, S, D = 1, 2, 32, 16
        Q, K, V = make_qkv(B=B, H=H, S=S, D=D)
        coords = make_coords(B=B, S=S)

        sparse_out = sparse_flash_attention(Q, K, V, coords, k=S)
        for name, fn in ALL_DENSE_FNS.items():
            dense_out = fn(Q, K, V)
            torch.testing.assert_close(sparse_out, dense_out, atol=5e-3, rtol=5e-3,
                                       msg=f"Sparse(k=S) vs {name}")

    @SKIP_IF_CPU
    @pytest.mark.parametrize("k", [4, 8, 16, 32])
    def test_various_k_match_reference(self, k):
        Q, K, V = make_qkv(S=64, D=32)
        coords = make_coords(S=64)
        ref = sparse_flash_attention_reference(Q, K, V, coords, k)
        out = sparse_flash_attention(Q, K, V, coords, k)
        torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3,
                                   msg=f"Sparse k={k}")

    @SKIP_IF_CPU
    def test_clustered_3d_points(self):
        """Points in tight spatial clusters — tests real-world 3D use case."""
        B, H, S, D = 1, 4, 64, 32
        Q, K, V = make_qkv(B=B, H=H, S=S, D=D)

        # 4 tight clusters at corners of a cube
        coords = torch.zeros(B, S, 3, device=DEVICE)
        cluster_size = S // 4
        for i in range(4):
            start = i * cluster_size
            end = start + cluster_size
            center = torch.tensor([i % 2, (i // 2) % 2, 0], dtype=torch.float32, device=DEVICE) * 10
            coords[:, start:end] = center + torch.randn(B, cluster_size, 3, device=DEVICE) * 0.1

        ref = sparse_flash_attention_reference(Q, K, V, coords, k=8)
        out = sparse_flash_attention(Q, K, V, coords, k=8)
        torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)


# ============================================================
# 4. Ring attention tests
# ============================================================

class TestRingAttention:
    """Ring attention must produce identical results to standard attention."""

    @SKIP_IF_CPU
    def test_matches_standard_attention(self):
        Q, K, V = make_qkv(S=64)
        ring_out = ring_attention(Q, K, V, n_chunks=4)
        std_out = flash_attention_reference(Q, K, V)
        torch.testing.assert_close(ring_out, std_out, atol=5e-3, rtol=5e-3)

    @SKIP_IF_CPU
    def test_matches_all_dense_versions(self):
        """Ring attention should match v1, v2, v3."""
        Q, K, V = make_qkv(S=64)
        ring_out = ring_attention(Q, K, V, n_chunks=4)
        for name, fn in ALL_DENSE_FNS.items():
            dense_out = fn(Q, K, V)
            torch.testing.assert_close(ring_out, dense_out, atol=5e-3, rtol=5e-3,
                                       msg=f"Ring vs {name}")

    @SKIP_IF_CPU
    @pytest.mark.parametrize("n_chunks", [1, 2, 4, 8])
    def test_various_chunk_counts(self, n_chunks):
        S = 128  # divisible by all chunk counts
        Q, K, V = make_qkv(S=S)
        ring_out = ring_attention(Q, K, V, n_chunks=n_chunks)
        std_out = flash_attention_reference(Q, K, V)
        torch.testing.assert_close(ring_out, std_out, atol=5e-3, rtol=5e-3,
                                   msg=f"Ring n_chunks={n_chunks}")

    @SKIP_IF_CPU
    def test_causal_matches_standard(self):
        Q, K, V = make_qkv(S=64)
        ring_out = ring_attention(Q, K, V, n_chunks=4, causal=True)
        std_out = flash_attention_reference(Q, K, V, causal=True)
        torch.testing.assert_close(ring_out, std_out, atol=5e-3, rtol=5e-3)

    @SKIP_IF_CPU
    def test_causal_matches_dense_versions(self):
        Q, K, V = make_qkv(S=64)
        ring_out = ring_attention(Q, K, V, n_chunks=4, causal=True)
        for name, fn in ALL_DENSE_FNS.items():
            dense_out = fn(Q, K, V, causal=True)
            torch.testing.assert_close(ring_out, dense_out, atol=5e-3, rtol=5e-3,
                                       msg=f"Ring causal vs {name}")


# ============================================================
# 5. Backward pass comparison (v1 vs v2 vs v3)
# ============================================================

class TestBackwardAllVersions:
    """Gradients must match across all versions and against reference."""

    @SKIP_IF_CPU
    def test_all_gradients_match_reference(self):
        Q, K, V = make_qkv(B=1, H=2, S=64, D=32)
        dO = torch.randn_like(Q)

        # Reference gradients
        Qr, Kr, Vr = [t.clone().requires_grad_(True) for t in (Q, K, V)]
        flash_attention_reference(Qr, Kr, Vr).backward(dO)

        for name, fn in ALL_DENSE_FNS.items():
            Qi, Ki, Vi = [t.clone().requires_grad_(True) for t in (Q, K, V)]
            fn(Qi, Ki, Vi).backward(dO)
            torch.testing.assert_close(Qi.grad, Qr.grad, atol=5e-2, rtol=5e-2,
                                       msg=f"{name} dQ")
            torch.testing.assert_close(Ki.grad, Kr.grad, atol=5e-2, rtol=5e-2,
                                       msg=f"{name} dK")
            torch.testing.assert_close(Vi.grad, Vr.grad, atol=5e-2, rtol=5e-2,
                                       msg=f"{name} dV")

    @SKIP_IF_CPU
    def test_causal_gradients_match(self):
        Q, K, V = make_qkv(B=1, H=2, S=64, D=32)
        dO = torch.randn_like(Q)

        Qr, Kr, Vr = [t.clone().requires_grad_(True) for t in (Q, K, V)]
        flash_attention_reference(Qr, Kr, Vr, causal=True).backward(dO)

        for name, fn in ALL_DENSE_FNS.items():
            Qi, Ki, Vi = [t.clone().requires_grad_(True) for t in (Q, K, V)]
            fn(Qi, Ki, Vi, causal=True).backward(dO)
            torch.testing.assert_close(Qi.grad, Qr.grad, atol=5e-2, rtol=5e-2,
                                       msg=f"{name} causal dQ")
            torch.testing.assert_close(Ki.grad, Kr.grad, atol=5e-2, rtol=5e-2,
                                       msg=f"{name} causal dK")
            torch.testing.assert_close(Vi.grad, Vr.grad, atol=5e-2, rtol=5e-2,
                                       msg=f"{name} causal dV")

    @SKIP_IF_CPU
    def test_v1_v2_v3_gradients_match_each_other(self):
        Q, K, V = make_qkv(B=1, H=4, S=128, D=64)
        dO = torch.randn_like(Q)

        grads = {}
        for name, fn in ALL_DENSE_FNS.items():
            Qi, Ki, Vi = [t.clone().requires_grad_(True) for t in (Q, K, V)]
            fn(Qi, Ki, Vi).backward(dO)
            grads[name] = (Qi.grad.clone(), Ki.grad.clone(), Vi.grad.clone())

        # v1 vs v2
        for i, label in enumerate(['dQ', 'dK', 'dV']):
            torch.testing.assert_close(grads['v1'][i], grads['v2'][i],
                                       atol=5e-2, rtol=5e-2, msg=f"v1 vs v2 {label}")
            torch.testing.assert_close(grads['v2'][i], grads['v3'][i],
                                       atol=5e-2, rtol=5e-2, msg=f"v2 vs v3 {label}")


# ============================================================
# 6. Edge cases
# ============================================================

class TestEdgeCases:
    """Edge cases that should work across all implementations."""

    @SKIP_IF_CPU
    def test_seq_len_1(self):
        """S=1: output should equal V (only one position to attend to)."""
        B, H, S, D = 1, 2, 1, 32
        Q, K, V = make_qkv(B=B, H=H, S=S, D=D)
        for name, fn in ALL_DENSE_FNS.items():
            out = fn(Q, K, V)
            torch.testing.assert_close(out, V, atol=5e-3, rtol=5e-3,
                                       msg=f"{name} S=1")

    @SKIP_IF_CPU
    def test_single_head(self):
        Q, K, V = make_qkv(B=2, H=1, S=64, D=32)
        ref = flash_attention_reference(Q, K, V)
        for name, fn in ALL_DENSE_FNS.items():
            out = fn(Q, K, V)
            torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3,
                                       msg=f"{name} H=1")

    @SKIP_IF_CPU
    def test_causal_first_row_is_v0(self):
        """In causal mode, first query only attends to first key -> output = V[0]."""
        B, H, S, D = 1, 2, 64, 32
        Q, K, V = make_qkv(B=B, H=H, S=S, D=D)
        expected = V[:, :, 0:1, :]
        for name, fn in ALL_DENSE_FNS.items():
            out = fn(Q, K, V, causal=True)
            torch.testing.assert_close(out[:, :, 0:1, :], expected, atol=5e-3, rtol=5e-3,
                                       msg=f"{name} causal first row")

    @SKIP_IF_CPU
    def test_uniform_v_gives_uniform_output(self):
        """When V is all-ones, output should be all-ones (regardless of attention weights)."""
        B, H, S, D = 1, 2, 32, 16
        Q = torch.randn(B, H, S, D, device=DEVICE)
        K = torch.randn(B, H, S, D, device=DEVICE)
        V = torch.ones(B, H, S, D, device=DEVICE)
        for name, fn in ALL_DENSE_FNS.items():
            out = fn(Q, K, V)
            torch.testing.assert_close(out, V, atol=5e-3, rtol=5e-3,
                                       msg=f"{name} uniform V")


# ============================================================
# 7. The money test: everything against everything
# ============================================================

class TestEverythingAgainstEverything:
    """The ultimate cross-validation: dense v1/v2/v3, sparse(k=S), ring — all must agree."""

    @SKIP_IF_CPU
    def test_all_implementations_agree(self):
        """Dense v1, v2, v3, sparse(full), ring(4 chunks) all produce same output."""
        B, H, S, D = 1, 4, 64, 32
        Q, K, V = make_qkv(B=B, H=H, S=S, D=D)
        coords = make_coords(B=B, S=S)

        # Reference
        ref = flash_attention_reference(Q, K, V)

        # Dense versions
        o_v1 = fa_v1(Q, K, V)
        o_v2 = fa_v2(Q, K, V)
        o_v3 = fa_v3(Q, K, V)

        # Sparse with k=S (full attention)
        o_sparse = sparse_flash_attention(Q, K, V, coords, k=S)

        # Ring with 4 chunks
        o_ring = ring_attention(Q, K, V, n_chunks=4)

        # All must match reference
        torch.testing.assert_close(o_v1, ref, atol=5e-3, rtol=5e-3, msg="v1 vs ref")
        torch.testing.assert_close(o_v2, ref, atol=5e-3, rtol=5e-3, msg="v2 vs ref")
        torch.testing.assert_close(o_v3, ref, atol=5e-3, rtol=5e-3, msg="v3 vs ref")
        torch.testing.assert_close(o_sparse, ref, atol=5e-3, rtol=5e-3, msg="sparse vs ref")
        torch.testing.assert_close(o_ring, ref, atol=5e-3, rtol=5e-3, msg="ring vs ref")

    @SKIP_IF_CPU
    def test_all_causal_agree(self):
        """Causal: dense v1, v2, v3, ring — all must agree."""
        B, H, S, D = 1, 4, 64, 32
        Q, K, V = make_qkv(B=B, H=H, S=S, D=D)

        ref = flash_attention_reference(Q, K, V, causal=True)
        o_v1 = fa_v1(Q, K, V, causal=True)
        o_v2 = fa_v2(Q, K, V, causal=True)
        o_v3 = fa_v3(Q, K, V, causal=True)
        o_ring = ring_attention(Q, K, V, n_chunks=4, causal=True)

        torch.testing.assert_close(o_v1, ref, atol=5e-3, rtol=5e-3, msg="v1 causal vs ref")
        torch.testing.assert_close(o_v2, ref, atol=5e-3, rtol=5e-3, msg="v2 causal vs ref")
        torch.testing.assert_close(o_v3, ref, atol=5e-3, rtol=5e-3, msg="v3 causal vs ref")
        torch.testing.assert_close(o_ring, ref, atol=5e-3, rtol=5e-3, msg="ring causal vs ref")
