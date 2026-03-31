"""
Benchmark: Flash Attention v2 vs PyTorch Reference

Measures:
  - Forward pass latency (ms) at various sequence lengths
  - Backward pass latency (ms)
  - Achieved TFLOPS

Usage:
  python bench/bench_flash_attn_versions.py
  python bench/bench_flash_attn_versions.py --fwd-only
  python bench/bench_flash_attn_versions.py --seq-lengths 256,512,1024,2048,4096
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from reference.flash_attn_ref import flash_attention_reference
from kernels.flash_attn_v2 import flash_attention_v2_forward as fa_v2
from utils.benchmark import benchmark_fn, roofline_analysis


def bench_forward(device, seq_lengths, B=2, H=16, D=64):
    """Benchmark forward pass."""
    print(f"\n{'='*80}")
    print(f"  FORWARD PASS BENCHMARK")
    print(f"  B={B}, H={H}, D={D}, device={device}")
    print(f"{'='*80}")

    header = f"  {'S':>6s} | {'PyTorch':>10s} | {'v2':>10s} | {'Speedup':>7s}"
    print(header)
    print(f"  {'-'*len(header)}")

    for S in seq_lengths:
        Q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        K = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        V = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        r_ref = benchmark_fn(flash_attention_reference, Q, K, V, warmup=5, n_runs=30)
        r_v2 = benchmark_fn(fa_v2, Q, K, V, warmup=5, n_runs=30)

        speedup = r_ref.median_ms / r_v2.median_ms if r_v2.median_ms > 0 else 0

        print(f"  {S:6d} | {r_ref.median_ms:8.3f}ms | {r_v2.median_ms:8.3f}ms | {speedup:6.2f}x")


def bench_backward(device, seq_lengths, B=2, H=16, D=64):
    """Benchmark backward pass."""
    print(f"\n{'='*80}")
    print(f"  BACKWARD PASS BENCHMARK")
    print(f"  B={B}, H={H}, D={D}, device={device}")
    print(f"{'='*80}")

    header = f"  {'S':>6s} | {'v2 bwd':>10s}"
    print(header)
    print(f"  {'-'*len(header)}")

    for S in seq_lengths:
        Q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        K = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        V = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        dO = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        def bwd_fn():
            Qi = Q.clone().requires_grad_(True)
            Ki = K.clone().requires_grad_(True)
            Vi = V.clone().requires_grad_(True)
            o = fa_v2(Qi, Ki, Vi)
            o.backward(dO)

        r = benchmark_fn(bwd_fn, warmup=3, n_runs=20)
        print(f"  {S:6d} | {r.median_ms:8.3f}ms")


def bench_tflops(device, seq_lengths, B=2, H=16, D=64):
    """Compute achieved TFLOPS."""
    print(f"\n{'='*80}")
    print(f"  ACHIEVED TFLOPS (Forward Only)")
    print(f"  B={B}, H={H}, D={D}")
    print(f"{'='*80}")

    header = f"  {'S':>6s} | {'v2 TFLOPS':>12s}"
    print(header)
    print(f"  {'-'*len(header)}")

    for S in seq_lengths:
        Q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        K = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        V = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        flops = 4 * B * H * S * S * D
        r = benchmark_fn(fa_v2, Q, K, V, warmup=5, n_runs=30)
        tflops = (flops / 1e12) / (r.median_ms / 1000)

        print(f"  {S:6d} | {tflops:10.2f}")


def main():
    parser = argparse.ArgumentParser(description="Flash Attention v2 benchmark")
    parser.add_argument("--fwd-only", action="store_true", help="Only benchmark forward pass")
    parser.add_argument("--seq-lengths", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    seq_lengths = [int(s) for s in args.seq_lengths.split(",")]

    if device == "cpu":
        print("WARNING: Running on CPU. Triton autotune requires CUDA.")
        return

    print(f"\n{'#'*80}")
    print(f"  Flash Attention v2 Benchmark")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"{'#'*80}")

    bench_forward(device, seq_lengths)
    bench_tflops(device, seq_lengths)

    if not args.fwd_only:
        bench_backward(device, seq_lengths)

    print(f"\n{'#'*80}")
    print(f"  Benchmark complete.")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
