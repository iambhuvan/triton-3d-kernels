"""Benchmark K1: Fused RoPE."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from reference.rope_ref import rope_reference, precompute_freqs
from kernels.rope import fused_rope
from utils.benchmark import benchmark_fn, print_benchmark_result


def bench_rope(device='cpu'):
    configs = [
        {"B": 4, "S": 512, "H": 32, "D": 128, "label": "LLaMA-7B-like"},
        {"B": 2, "S": 2048, "H": 32, "D": 128, "label": "Long sequence"},
        {"B": 8, "S": 256, "H": 16, "D": 64, "label": "Small model"},
    ]

    for cfg in configs:
        B, S, H, D = cfg["B"], cfg["S"], cfg["H"], cfg["D"]
        print(f"\n  Config: {cfg['label']} (B={B}, S={S}, H={H}, D={D})")

        x = torch.randn(B, S, H, D, device=device)
        cos, sin = precompute_freqs(D, S, device=device)

        ref_result = benchmark_fn(rope_reference, x, cos, sin, warmup=5, n_runs=50)
        tri_result = benchmark_fn(fused_rope, x, cos, sin, warmup=5, n_runs=50)
        ref_result.name = "PyTorch ref"
        tri_result.name = "Triton RoPE"
        print_benchmark_result(ref_result, tri_result)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bench_rope(device)
