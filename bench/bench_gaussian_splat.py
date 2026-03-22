"""Benchmark K6: 3D Gaussian Splatting Rasterizer."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from reference.gaussian_splat_ref import gaussian_splat_reference
from kernels.gaussian_splat import gaussian_splat_forward
from tests.test_gaussian_splat import make_splat_inputs, make_camera_matrices
from utils.benchmark import benchmark_fn, print_benchmark_result


def bench_gaussian_splat(device='cpu'):
    configs = [
        {"N": 100, "H": 64, "W": 64, "label": "Tiny scene"},
        {"N": 1000, "H": 256, "W": 256, "label": "Small scene"},
        # Large configs are too slow for reference on CPU
    ]

    if device == "cuda":
        configs.append({"N": 10000, "H": 512, "W": 512, "label": "Medium scene"})

    for cfg in configs:
        N, H, W = cfg["N"], cfg["H"], cfg["W"]
        print(f"\n  Config: {cfg['label']} (N={N}, {H}x{W})")

        args = make_splat_inputs(N=N, img_h=H, img_w=W, device=device)

        ref_fn = lambda: gaussian_splat_reference(*args)
        tri_fn = lambda: gaussian_splat_forward(*args)

        ref_result = benchmark_fn(ref_fn, warmup=2, n_runs=10)
        tri_result = benchmark_fn(tri_fn, warmup=2, n_runs=10)
        ref_result.name = "PyTorch ref"
        tri_result.name = "Triton GS"
        print_benchmark_result(ref_result, tri_result)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bench_gaussian_splat(device)
