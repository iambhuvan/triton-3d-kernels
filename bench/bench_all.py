"""
Master benchmark script — runs all kernel benchmarks and generates summary.

Usage:
  python bench/bench_all.py                    # Run all
  python bench/bench_all.py --kernel rope      # Run one kernel
  python bench/bench_all.py --device cuda      # Specify device
"""

import argparse
import sys
import os
# Add both project root and bench/ directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from bench_rope import bench_rope
from bench_flash_attn import bench_flash_attn
from bench_chamfer import bench_chamfer
from bench_swiglu import bench_swiglu
from bench_gaussian_splat import bench_gaussian_splat
from bench_sparse_flash_attn import bench_sparse_flash_attn
from bench_ring_attn import bench_ring_attn


BENCHMARKS = {
    "rope": bench_rope,
    "flash_attn": bench_flash_attn,
    "chamfer": bench_chamfer,
    "swiglu": bench_swiglu,
    "gaussian_splat": bench_gaussian_splat,
    "sparse_flash_attn": bench_sparse_flash_attn,
    "ring_attn": bench_ring_attn,
}


def main():
    parser = argparse.ArgumentParser(description="Run kernel benchmarks")
    parser.add_argument("--kernel", type=str, default=None, choices=list(BENCHMARKS.keys()),
                        help="Run a specific kernel benchmark")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print(f"\n{'#'*60}")
    print(f"  triton-3d-kernels Benchmark Suite")
    print(f"  Device: {device}")
    if device == "cuda" and torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  CUDA: {torch.version.cuda}")
    print(f"{'#'*60}")

    if args.kernel:
        BENCHMARKS[args.kernel](device)
    else:
        for name, bench_fn in BENCHMARKS.items():
            print(f"\n{'='*60}")
            print(f"  Running: {name}")
            print(f"{'='*60}")
            bench_fn(device)

    print(f"\n{'#'*60}")
    print(f"  All benchmarks complete.")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
