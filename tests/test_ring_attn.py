"""Tests for K7: Ring Attention kernel."""

import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reference.flash_attn_ref import flash_attention_reference
from reference.ring_attn_ref import ring_attention_reference
from kernels.ring_attn import ring_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_attn_inputs(B=2, H=4, S=128, D=32, device='cuda', dtype=torch.float32):
    Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    V = torch.randn(B, H, S, D, device=device, dtype=dtype)
    return Q, K, V


class TestRingAttnCorrectness:
    """Ring attention should produce IDENTICAL output to standard attention."""

    def test_basic_shapes(self):
        Q, K, V = make_attn_inputs()
        out = ring_attention(Q, K, V, n_chunks=4)
        assert out.shape == Q.shape

    def test_matches_standard_attention(self):
        """The whole point: ring attention = standard attention, just distributed."""
        Q, K, V = make_attn_inputs(S=64)
        ring_out = ring_attention(Q, K, V, n_chunks=4)
        std_out = flash_attention_reference(Q, K, V)
        torch.testing.assert_close(ring_out, std_out, atol=5e-3, rtol=5e-3)

    def test_matches_reference(self):
        Q, K, V = make_attn_inputs(S=64)
        ref = ring_attention_reference(Q, K, V, n_chunks=4)
        tri = ring_attention(Q, K, V, n_chunks=4)
        torch.testing.assert_close(tri, ref, atol=5e-3, rtol=5e-3)

    @pytest.mark.parametrize("n_chunks", [2, 4, 8])
    def test_various_chunk_counts(self, n_chunks):
        S = 128  # must be divisible by n_chunks
        Q, K, V = make_attn_inputs(S=S)
        ring_out = ring_attention(Q, K, V, n_chunks=n_chunks)
        std_out = flash_attention_reference(Q, K, V)
        torch.testing.assert_close(ring_out, std_out, atol=5e-3, rtol=5e-3)

    def test_single_chunk_equals_standard(self):
        """With n_chunks=1, ring attention = standard attention (no communication)."""
        Q, K, V = make_attn_inputs(S=64)
        ring_out = ring_attention(Q, K, V, n_chunks=1)
        std_out = flash_attention_reference(Q, K, V)
        torch.testing.assert_close(ring_out, std_out, atol=5e-3, rtol=5e-3)

    @pytest.mark.parametrize("B,H,S,D", [
        (1, 1, 32, 16),
        (4, 8, 128, 64),
        (2, 4, 256, 32),
    ])
    def test_various_shapes(self, B, H, S, D):
        Q, K, V = make_attn_inputs(B=B, H=H, S=S, D=D)
        ring_out = ring_attention(Q, K, V, n_chunks=4)
        std_out = flash_attention_reference(Q, K, V)
        torch.testing.assert_close(ring_out, std_out, atol=5e-3, rtol=5e-3)

    def test_causal_matches_standard(self):
        Q, K, V = make_attn_inputs(S=64)
        ring_out = ring_attention(Q, K, V, n_chunks=4, causal=True)
        std_out = flash_attention_reference(Q, K, V, causal=True)
        torch.testing.assert_close(ring_out, std_out, atol=5e-3, rtol=5e-3)
