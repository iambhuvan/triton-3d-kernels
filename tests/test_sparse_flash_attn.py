"""Tests for K3: KNN-Sparse Flash Attention kernel."""

import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reference.sparse_flash_attn_ref import sparse_flash_attention_reference, build_knn_mask
from reference.flash_attn_ref import flash_attention_reference
from kernels.sparse_flash_attn import sparse_flash_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_sparse_attn_inputs(B=2, H=4, S=64, D=32, device='cuda', dtype=torch.float32):
    Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    V = torch.randn(B, H, S, D, device=device, dtype=dtype)
    # Random 3D coordinates
    coords = torch.randn(B, S, 3, device=device, dtype=dtype)
    return Q, K, V, coords


class TestSparseFlashAttn:
    def test_basic_shapes(self):
        Q, K, V, coords = make_sparse_attn_inputs()
        out = sparse_flash_attention(Q, K, V, coords, k=16)
        assert out.shape == Q.shape

    def test_matches_reference(self):
        Q, K, V, coords = make_sparse_attn_inputs()
        k = 16
        ref = sparse_flash_attention_reference(Q, K, V, coords, k)
        tri = sparse_flash_attention(Q, K, V, coords, k)
        torch.testing.assert_close(tri, ref, atol=5e-3, rtol=5e-3)

    def test_full_k_equals_dense(self):
        """When k=S (all neighbors), sparse attention = dense attention."""
        B, H, S, D = 1, 2, 32, 16
        Q, K, V, coords = make_sparse_attn_inputs(B=B, H=H, S=S, D=D)
        sparse_out = sparse_flash_attention(Q, K, V, coords, k=S)
        dense_out = flash_attention_reference(Q, K, V)
        torch.testing.assert_close(sparse_out, dense_out, atol=5e-3, rtol=5e-3)

    @pytest.mark.parametrize("k", [4, 8, 16, 32])
    def test_various_k(self, k):
        Q, K, V, coords = make_sparse_attn_inputs(S=64)
        ref = sparse_flash_attention_reference(Q, K, V, coords, k)
        tri = sparse_flash_attention(Q, K, V, coords, k)
        torch.testing.assert_close(tri, ref, atol=5e-3, rtol=5e-3)

    def test_clustered_coords(self):
        """Points in tight clusters should produce different results than random."""
        B, H, S, D = 1, 2, 64, 16
        Q = torch.randn(B, H, S, D, device='cuda')
        K = torch.randn(B, H, S, D, device='cuda')
        V = torch.randn(B, H, S, D, device='cuda')

        # Clustered: 4 groups at corners of a cube
        coords = torch.zeros(B, S, 3, device='cuda')
        for i in range(4):
            start = i * (S // 4)
            end = start + S // 4
            coords[:, start:end, :] = torch.randn(B, S // 4, 3, device='cuda') * 0.1 + i * 10

        out = sparse_flash_attention(Q, K, V, coords, k=8)
        assert out.shape == Q.shape


class TestKNNMask:
    def test_knn_mask_shape(self):
        coords = torch.randn(2, 32, 3, device='cuda')
        mask = build_knn_mask(coords, k=8)
        assert mask.shape == (2, 32, 32)
        assert mask.dtype == torch.bool

    def test_self_is_neighbor(self):
        """Each point should be its own nearest neighbor (distance 0)."""
        coords = torch.randn(1, 16, 3, device='cuda')
        mask = build_knn_mask(coords, k=1)
        # Diagonal should be True (self is nearest neighbor)
        assert torch.all(mask[0].diagonal())

    def test_k_neighbors_per_row(self):
        """Each row should have exactly k True entries."""
        coords = torch.randn(1, 32, 3, device='cuda')
        k = 8
        mask = build_knn_mask(coords, k=k)
        assert torch.all(mask.sum(dim=-1) == k)
