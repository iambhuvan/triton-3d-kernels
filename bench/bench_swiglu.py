"""Benchmark K5: Fused SwiGLU."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from reference.swiglu_ref import swiglu_fused_part_reference
from kernels.swiglu import FusedSwiGLUFunction
from utils.benchmark import benchmark_fn, print_benchmark_result


def bench_swiglu(device='cpu'):
    configs = [
        {"B": 4, "S": 512, "D_ff": 4096, "label": "LLaMA-7B intermediate"},
        {"B": 2, "S": 2048, "D_ff": 11008, "label": "LLaMA-7B full"},
        {"B": 1, "S": 4096, "D_ff": 14336, "label": "LLaMA-13B"},
    ]

    for cfg in configs:
        B, S, D_ff = cfg["B"], cfg["S"], cfg["D_ff"]
        print(f"\n  Config: {cfg['label']} (B={B}, S={S}, D_ff={D_ff})")

        gate = torch.randn(B, S, D_ff, device=device)
        up = torch.randn(B, S, D_ff, device=device)

        ref_fn = lambda: swiglu_fused_part_reference(gate, up)
        tri_fn = lambda: FusedSwiGLUFunction.apply(gate, up)

        ref_result = benchmark_fn(ref_fn, warmup=5, n_runs=50)
        tri_result = benchmark_fn(tri_fn, warmup=5, n_runs=50)
        ref_result.name = "PyTorch ref"
        tri_result.name = "Triton SwiGLU"
        print_benchmark_result(ref_result, tri_result)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bench_swiglu(device)
