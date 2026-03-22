"""Tests for Flash Attention v3 CUDA kernel (H100 Hopper).

Requires: H100 GPU (sm_90a) for WGMMA/TMA/warp specialization.
Skip gracefully on non-Hopper hardware.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch

# Skip entire module if no H100
def requires_hopper():
    if not torch.cuda.is_available():
        return pytest.mark.skip(reason="No CUDA GPU available")
    props = torch.cuda.get_device_properties(0)
    if props.major < 9:
        return pytest.mark.skip(
            reason=f"Requires H100 (sm_90), got {props.name} (sm_{props.major}{props.minor})"
        )
    return pytest.mark.skipif(False, reason="")


hopper = requires_hopper()


@hopper
class TestV3CudaForward:
    """Test CUDA v3 forward correctness against PyTorch reference."""

    def _reference(self, Q, K, V, causal=False):
        """Standard PyTorch attention."""
        scale = Q.shape[-1] ** -0.5
        S = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
        if causal:
            seq_len = Q.shape[2]
            mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
            S.masked_fill_(mask, float('-inf'))
        P = torch.softmax(S, dim=-1)
        O = torch.matmul(P, V.float())
        return O.half()

    def test_basic_forward(self):
        """Basic forward pass matches reference."""
        from kernels.flash_attn_v3_cuda import flash_attention_v3_cuda

        B, H, S, D = 2, 8, 128, 64
        Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)

        out = flash_attention_v3_cuda(Q, K, V)
        ref = self._reference(Q, K, V)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_causal(self):
        """Causal masking works correctly."""
        from kernels.flash_attn_v3_cuda import flash_attention_v3_cuda

        B, H, S, D = 1, 4, 64, 64
        Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)

        out = flash_attention_v3_cuda(Q, K, V, causal=True)
        ref = self._reference(Q, K, V, causal=True)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_various_seq_lengths(self):
        """Different sequence lengths."""
        from kernels.flash_attn_v3_cuda import flash_attention_v3_cuda

        for S in [64, 128, 256, 512]:
            Q = torch.randn(1, 4, S, 64, device='cuda', dtype=torch.float16)
            K = torch.randn(1, 4, S, 64, device='cuda', dtype=torch.float16)
            V = torch.randn(1, 4, S, 64, device='cuda', dtype=torch.float16)

            out = flash_attention_v3_cuda(Q, K, V)
            ref = self._reference(Q, K, V)

            torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2,
                                       msg=f"Failed for S={S}")

    def test_matches_triton_v1(self):
        """CUDA v3 matches Triton v1 output."""
        from kernels.flash_attn_v3_cuda import flash_attention_v3_cuda
        from kernels.flash_attn import flash_attention_forward

        B, H, S, D = 2, 8, 128, 64
        Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)

        # Triton v1 uses float32, CUDA v3 uses float16
        out_cuda = flash_attention_v3_cuda(Q, K, V)
        out_triton = flash_attention_forward(Q.float(), K.float(), V.float()).half()

        torch.testing.assert_close(out_cuda, out_triton, atol=2e-2, rtol=2e-2)

    def test_no_nan_inf(self):
        """Output contains no NaN or Inf."""
        from kernels.flash_attn_v3_cuda import flash_attention_v3_cuda

        B, H, S, D = 2, 8, 256, 64
        Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)

        out = flash_attention_v3_cuda(Q, K, V)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"


@hopper
class TestV3CudaMatchesAllVersions:
    """Cross-validate CUDA v3 against all other attention implementations."""

    def test_all_versions_agree(self):
        """v1 (Triton), v2 (Triton), v3 (Triton), v3 (CUDA) all match."""
        from kernels.flash_attn_v3_cuda import flash_attention_v3_cuda
        from kernels.flash_attn import flash_attention_forward as v1_fwd
        from kernels.flash_attn_v2 import flash_attention_v2_forward as v2_fwd
        from kernels.flash_attn_v3 import flash_attention_v3_forward as v3_triton_fwd

        B, H, S, D = 1, 4, 128, 64
        Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)

        # CUDA v3 (FP16 in/out)
        out_cuda = flash_attention_v3_cuda(Q, K, V)

        # Triton versions (FP32 compute)
        Q32, K32, V32 = Q.float(), K.float(), V.float()
        out_v1 = v1_fwd(Q32, K32, V32).half()
        out_v2 = v2_fwd(Q32, K32, V32).half()
        out_v3_triton = v3_triton_fwd(Q32, K32, V32).half()

        # All should be close (FP16 tolerance)
        tol = dict(atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(out_cuda, out_v1, **tol, msg="CUDA v3 vs Triton v1")
        torch.testing.assert_close(out_cuda, out_v2, **tol, msg="CUDA v3 vs Triton v2")
        torch.testing.assert_close(out_cuda, out_v3_triton, **tol, msg="CUDA v3 vs Triton v3")


@hopper
class TestV3CudaBenchmark:
    """Quick latency check (not full benchmark, just sanity)."""

    def test_perf_smoke(self):
        """Kernel runs without error at larger sizes."""
        from kernels.flash_attn_v3_cuda import flash_attention_v3_cuda

        B, H, S, D = 2, 32, 1024, 64
        Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)

        # Warmup
        for _ in range(3):
            _ = flash_attention_v3_cuda(Q, K, V)
        torch.cuda.synchronize()

        # Timed run
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            _ = flash_attention_v3_cuda(Q, K, V)
        end.record()
        torch.cuda.synchronize()

        ms = start.elapsed_time(end) / 10
        print(f"\n  CUDA v3 Flash Attention: {ms:.3f} ms "
              f"(B={B}, H={H}, S={S}, D={D})")
        assert ms > 0  # Sanity check
