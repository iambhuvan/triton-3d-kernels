"""Benchmark: Flash Attention CUDA v3 (H100) vs Triton versions.

Compares:
  - Triton v1 (basic flash attention)
  - Triton v2 (autotuned, split backward)
  - Triton v3 (persistent + FP8)
  - CUDA v3  (WGMMA + TMA + warp specialization)
  - PyTorch reference

Requires H100 GPU.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from utils.benchmark import benchmark_fn, print_benchmark_result


def bench_all_versions():
    if not torch.cuda.is_available():
        print("No CUDA GPU — skipping")
        return

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} (sm_{props.major}{props.minor})")
    print(f"VRAM: {props.total_mem / 1024**3:.1f} GB")

    from reference.flash_attn_ref import flash_attention_reference
    from kernels.flash_attn import flash_attention_forward as v1_fwd
    from kernels.flash_attn_v2 import flash_attention_v2_forward as v2_fwd
    from kernels.flash_attn_v3 import flash_attention_v3_forward as v3_triton_fwd

    has_hopper = props.major >= 9
    if has_hopper:
        from kernels.flash_attn_v3_cuda import flash_attention_v3_cuda
    else:
        print("WARNING: Not H100 — CUDA v3 will be skipped\n")

    configs = [
        {"B": 2, "H": 8,  "S": 256,  "D": 64, "label": "Small"},
        {"B": 2, "H": 16, "S": 1024, "D": 64, "label": "Medium"},
        {"B": 1, "H": 32, "S": 4096, "D": 64, "label": "Large"},
    ]

    for cfg in configs:
        B, H, S, D = cfg["B"], cfg["H"], cfg["S"], cfg["D"]
        print(f"\n{'='*60}")
        print(f"  {cfg['label']}: B={B}, H={H}, S={S}, D={D}")
        print(f"{'='*60}")

        # FP32 inputs for Triton
        Q32 = torch.randn(B, H, S, D, device='cuda')
        K32 = torch.randn(B, H, S, D, device='cuda')
        V32 = torch.randn(B, H, S, D, device='cuda')

        # FP16 inputs for CUDA v3
        Q16 = Q32.half()
        K16 = K32.half()
        V16 = V32.half()

        results = {}

        # Reference
        r = benchmark_fn(flash_attention_reference, Q32, K32, V32, warmup=5, n_runs=50)
        r.name = "PyTorch ref"
        results["ref"] = r
        print(f"  PyTorch ref:     {r.median_ms:.3f} ms")

        # Triton v1
        r = benchmark_fn(v1_fwd, Q32, K32, V32, warmup=5, n_runs=50)
        r.name = "Triton v1"
        results["v1"] = r
        print(f"  Triton v1:       {r.median_ms:.3f} ms  ({results['ref'].median_ms / r.median_ms:.2f}x vs ref)")

        # Triton v2
        r = benchmark_fn(v2_fwd, Q32, K32, V32, warmup=5, n_runs=50)
        r.name = "Triton v2"
        results["v2"] = r
        print(f"  Triton v2:       {r.median_ms:.3f} ms  ({results['ref'].median_ms / r.median_ms:.2f}x vs ref)")

        # Triton v3
        r = benchmark_fn(v3_triton_fwd, Q32, K32, V32, warmup=5, n_runs=50)
        r.name = "Triton v3"
        results["v3_triton"] = r
        print(f"  Triton v3:       {r.median_ms:.3f} ms  ({results['ref'].median_ms / r.median_ms:.2f}x vs ref)")

        # CUDA v3 (H100 only)
        if has_hopper:
            r = benchmark_fn(flash_attention_v3_cuda, Q16, K16, V16, warmup=5, n_runs=50)
            r.name = "CUDA v3 (WGMMA)"
            results["v3_cuda"] = r
            print(f"  CUDA v3 (WGMMA): {r.median_ms:.3f} ms  ({results['ref'].median_ms / r.median_ms:.2f}x vs ref)")

        # FLOPS analysis
        flops = 4 * B * H * S * S * D
        fastest = min(r.median_ms for r in results.values())
        tflops = flops / (fastest * 1e-3) / 1e12
        print(f"\n  Best TFLOPS: {tflops:.1f}")


if __name__ == "__main__":
    bench_all_versions()
