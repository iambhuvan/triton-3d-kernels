"""Benchmark K4: Batched Chamfer Distance."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from reference.chamfer_ref import chamfer_distance_reference
from kernels.chamfer import batched_chamfer_distance
from utils.benchmark import benchmark_fn, print_benchmark_result


def bench_chamfer(device='cpu'):
    configs = [
        {"B": 4, "N": 1024, "M": 1024, "label": "Small clouds"},
        {"B": 2, "N": 4096, "M": 4096, "label": "Medium clouds"},
        {"B": 1, "N": 8192, "M": 8192, "label": "Large clouds (3D gen typical)"},
    ]

    for cfg in configs:
        B, N, M = cfg["B"], cfg["N"], cfg["M"]
        print(f"\n  Config: {cfg['label']} (B={B}, N={N}, M={M})")

        pc1 = torch.randn(B, N, 3, device=device)
        pc2 = torch.randn(B, M, 3, device=device)

        # Skip large configs on CPU (too slow)
        if device == 'cpu' and N * M > 4096 * 4096:
            print("    Skipping on CPU (too slow)")
            continue

        ref_result = benchmark_fn(chamfer_distance_reference, pc1, pc2, warmup=3, n_runs=20)
        tri_result = benchmark_fn(batched_chamfer_distance, pc1, pc2, warmup=3, n_runs=20)
        ref_result.name = "PyTorch ref"
        tri_result.name = "Triton Chamfer"
        print_benchmark_result(ref_result, tri_result)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bench_chamfer(device)
