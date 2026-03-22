"""
Comparative benchmark: Flash Attention v1 vs v2 vs v3

Measures:
  - Forward pass latency (ms) at various sequence lengths
  - Backward pass latency (ms)
  - Forward + backward combined
  - FP8 vs FP16 throughput (v3 only)
  - Achieved TFLOPS and memory bandwidth
  - Scaling behavior: how each version handles increasing S

Output: formatted tables + data for plotting.

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
from kernels.flash_attn import flash_attention_forward as fa_v1
from kernels.flash_attn_v2 import flash_attention_v2_forward as fa_v2
from kernels.flash_attn_v3 import flash_attention_v3_forward as fa_v3
from utils.benchmark import benchmark_fn, roofline_analysis


def bench_forward(device, seq_lengths, B=2, H=16, D=64):
    """Benchmark forward pass across versions and sequence lengths."""
    print(f"\n{'='*80}")
    print(f"  FORWARD PASS BENCHMARK")
    print(f"  B={B}, H={H}, D={D}, device={device}")
    print(f"{'='*80}")

    header = f"  {'S':>6s} | {'PyTorch':>10s} | {'v1':>10s} | {'v2':>10s} | {'v3':>10s} | {'v3-FP8':>10s} | {'v2/v1':>7s} | {'v3/v1':>7s}"
    print(header)
    print(f"  {'-'*len(header)}")

    for S in seq_lengths:
        Q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        K = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        V = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        results = {}

        # PyTorch reference
        r = benchmark_fn(flash_attention_reference, Q, K, V, warmup=5, n_runs=30)
        results['ref'] = r.median_ms

        # v1
        r = benchmark_fn(fa_v1, Q, K, V, warmup=5, n_runs=30)
        results['v1'] = r.median_ms

        # v2
        r = benchmark_fn(fa_v2, Q, K, V, warmup=5, n_runs=30)
        results['v2'] = r.median_ms

        # v3 (FP16)
        r = benchmark_fn(fa_v3, Q, K, V, warmup=5, n_runs=30)
        results['v3'] = r.median_ms

        # v3 (FP8)
        r = benchmark_fn(lambda: fa_v3(Q, K, V, use_fp8=True), warmup=5, n_runs=30)
        results['v3_fp8'] = r.median_ms

        # Speedups
        v2_vs_v1 = results['v1'] / results['v2'] if results['v2'] > 0 else 0
        v3_vs_v1 = results['v1'] / results['v3'] if results['v3'] > 0 else 0

        print(f"  {S:6d} | {results['ref']:8.3f}ms | {results['v1']:8.3f}ms | "
              f"{results['v2']:8.3f}ms | {results['v3']:8.3f}ms | {results['v3_fp8']:8.3f}ms | "
              f"{v2_vs_v1:6.2f}x | {v3_vs_v1:6.2f}x")

    return results


def bench_backward(device, seq_lengths, B=2, H=16, D=64):
    """Benchmark backward pass across versions."""
    print(f"\n{'='*80}")
    print(f"  BACKWARD PASS BENCHMARK")
    print(f"  B={B}, H={H}, D={D}, device={device}")
    print(f"{'='*80}")

    header = f"  {'S':>6s} | {'v1 bwd':>10s} | {'v2 bwd':>10s} | {'v3 bwd':>10s} | {'v2/v1':>7s} | {'v3/v1':>7s}"
    print(header)
    print(f"  {'-'*len(header)}")

    for S in seq_lengths:
        Q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        K = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        V = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        dO = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        results = {}

        for name, fa_fn in [('v1', fa_v1), ('v2', fa_v2), ('v3', fa_v3)]:
            def bwd_fn():
                Qi = Q.clone().requires_grad_(True)
                Ki = K.clone().requires_grad_(True)
                Vi = V.clone().requires_grad_(True)
                o = fa_fn(Qi, Ki, Vi)
                o.backward(dO)

            r = benchmark_fn(bwd_fn, warmup=3, n_runs=20)
            results[name] = r.median_ms

        v2_vs_v1 = results['v1'] / results['v2'] if results['v2'] > 0 else 0
        v3_vs_v1 = results['v1'] / results['v3'] if results['v3'] > 0 else 0

        print(f"  {S:6d} | {results['v1']:8.3f}ms | {results['v2']:8.3f}ms | "
              f"{results['v3']:8.3f}ms | {v2_vs_v1:6.2f}x | {v3_vs_v1:6.2f}x")


def bench_tflops(device, seq_lengths, B=2, H=16, D=64):
    """Compute achieved TFLOPS for each version."""
    print(f"\n{'='*80}")
    print(f"  ACHIEVED TFLOPS (Forward Only)")
    print(f"  B={B}, H={H}, D={D}")
    print(f"{'='*80}")

    header = f"  {'S':>6s} | {'v1 TFLOPS':>12s} | {'v2 TFLOPS':>12s} | {'v3 TFLOPS':>12s} | {'v3-FP8':>12s}"
    print(header)
    print(f"  {'-'*len(header)}")

    for S in seq_lengths:
        Q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        K = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        V = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        # Attention FLOPs: 2 * B * H * S * S * D (QK) + 2 * B * H * S * S * D (PV) = 4*B*H*S^2*D
        flops = 4 * B * H * S * S * D

        tflops = {}
        for name, fn in [('v1', lambda: fa_v1(Q, K, V)),
                          ('v2', lambda: fa_v2(Q, K, V)),
                          ('v3', lambda: fa_v3(Q, K, V)),
                          ('v3_fp8', lambda: fa_v3(Q, K, V, use_fp8=True))]:
            r = benchmark_fn(fn, warmup=5, n_runs=30)
            tflops[name] = (flops / 1e12) / (r.median_ms / 1000)

        print(f"  {S:6d} | {tflops['v1']:10.2f}   | {tflops['v2']:10.2f}   | "
              f"{tflops['v3']:10.2f}   | {tflops['v3_fp8']:10.2f}")


def bench_fp8_accuracy(device, B=2, H=8, S=512, D=64):
    """Measure FP8 quantization error across different input distributions."""
    print(f"\n{'='*80}")
    print(f"  FP8 QUANTIZATION ERROR ANALYSIS")
    print(f"  B={B}, H={H}, S={S}, D={D}")
    print(f"{'='*80}")

    distributions = {
        'Normal(0,1)': lambda: torch.randn(B, H, S, D, device=device, dtype=torch.float16),
        'Normal(0,0.1)': lambda: torch.randn(B, H, S, D, device=device, dtype=torch.float16) * 0.1,
        'Uniform(-1,1)': lambda: torch.rand(B, H, S, D, device=device, dtype=torch.float16) * 2 - 1,
        'Large Normal(0,10)': lambda: torch.randn(B, H, S, D, device=device, dtype=torch.float16) * 10,
    }

    header = f"  {'Distribution':>20s} | {'Max Abs Err':>12s} | {'Mean Rel Err':>12s} | {'Cosine Sim':>12s}"
    print(header)
    print(f"  {'-'*len(header)}")

    for dist_name, make_fn in distributions.items():
        Q = make_fn()
        K = make_fn()
        V = make_fn()

        o_fp16 = fa_v3(Q, K, V, use_fp8=False)
        o_fp8 = fa_v3(Q, K, V, use_fp8=True)

        max_abs_err = (o_fp8 - o_fp16).abs().max().item()
        mean_rel_err = ((o_fp8 - o_fp16).abs() / (o_fp16.abs() + 1e-8)).mean().item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            o_fp16.flatten().unsqueeze(0),
            o_fp8.flatten().unsqueeze(0),
        ).item()

        print(f"  {dist_name:>20s} | {max_abs_err:12.6f} | {mean_rel_err:12.6f} | {cosine_sim:12.6f}")


def main():
    parser = argparse.ArgumentParser(description="Flash Attention v1/v2/v3 benchmark")
    parser.add_argument("--fwd-only", action="store_true", help="Only benchmark forward pass")
    parser.add_argument("--seq-lengths", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    seq_lengths = [int(s) for s in args.seq_lengths.split(",")]

    if device == "cpu":
        print("WARNING: Running on CPU. Triton autotune requires CUDA.")
        print("         Results will reflect reference implementation speed only.")
        return

    print(f"\n{'#'*80}")
    print(f"  Flash Attention Version Comparison")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"{'#'*80}")

    # Forward benchmark
    bench_forward(device, seq_lengths)

    # TFLOPS analysis
    bench_tflops(device, seq_lengths)

    if not args.fwd_only:
        # Backward benchmark
        bench_backward(device, seq_lengths)

    # FP8 accuracy analysis
    bench_fp8_accuracy(device)

    print(f"\n{'#'*80}")
    print(f"  Benchmark complete.")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
