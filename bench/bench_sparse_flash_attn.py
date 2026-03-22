"""Benchmark K3: KNN-Sparse Flash Attention."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from reference.sparse_flash_attn_ref import sparse_flash_attention_reference
from kernels.sparse_flash_attn import sparse_flash_attention
from kernels.flash_attn import flash_attention_forward
from utils.benchmark import benchmark_fn, print_benchmark_result, roofline_analysis


def bench_sparse_flash_attn(device='cpu'):
    # --- Part 1: Sparse Triton vs Reference across S and k ---
    configs = [
        {"B": 2, "H": 8, "S": 256,  "D": 64, "k": 8,  "label": "S=256, k=8"},
        {"B": 2, "H": 8, "S": 256,  "D": 64, "k": 16, "label": "S=256, k=16"},
        {"B": 2, "H": 8, "S": 256,  "D": 64, "k": 32, "label": "S=256, k=32"},
        {"B": 2, "H": 8, "S": 512,  "D": 64, "k": 8,  "label": "S=512, k=8"},
        {"B": 2, "H": 8, "S": 512,  "D": 64, "k": 16, "label": "S=512, k=16"},
        {"B": 2, "H": 8, "S": 512,  "D": 64, "k": 32, "label": "S=512, k=32"},
        {"B": 2, "H": 8, "S": 1024, "D": 64, "k": 8,  "label": "S=1024, k=8"},
        {"B": 2, "H": 8, "S": 1024, "D": 64, "k": 16, "label": "S=1024, k=16"},
        {"B": 2, "H": 8, "S": 1024, "D": 64, "k": 32, "label": "S=1024, k=32"},
        {"B": 2, "H": 8, "S": 2048, "D": 64, "k": 8,  "label": "S=2048, k=8"},
        {"B": 2, "H": 8, "S": 2048, "D": 64, "k": 16, "label": "S=2048, k=16"},
        {"B": 2, "H": 8, "S": 2048, "D": 64, "k": 32, "label": "S=2048, k=32"},
    ]

    print("\n  === Sparse Flash Attention: Triton vs Reference ===")

    for cfg in configs:
        B, H, S, D, k = cfg["B"], cfg["H"], cfg["S"], cfg["D"], cfg["k"]
        print(f"\n  Config: {cfg['label']} (B={B}, H={H}, S={S}, D={D}, k={k})")

        Q = torch.randn(B, H, S, D, device=device)
        K = torch.randn(B, H, S, D, device=device)
        V = torch.randn(B, H, S, D, device=device)
        coords = torch.randn(B, S, 3, device=device)

        ref_result = benchmark_fn(sparse_flash_attention_reference, Q, K, V, coords, k, warmup=5, n_runs=50)
        tri_result = benchmark_fn(sparse_flash_attention, Q, K, V, coords, k, warmup=5, n_runs=50)
        ref_result.name = "PyTorch ref"
        tri_result.name = "Triton SparseFlashAttn"
        print_benchmark_result(ref_result, tri_result)

        # Roofline: sparse attention is ~4*B*H*S*k*D flops (S*k instead of S^2)
        flops = 4 * B * H * S * k * D
        bytes_accessed = 4 * B * H * S * D * 4  # Q, K, V reads + O write (fp32)
        if device == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            if "A100" in gpu_name:
                roofline_analysis(tri_result.median_ms, flops, bytes_accessed, "A100")
            elif "H100" in gpu_name:
                roofline_analysis(tri_result.median_ms, flops, bytes_accessed, "H100")

    # --- Part 2: Sparse vs Dense attention (sparsity speedup) ---
    print("\n  === Sparsity Speedup: Sparse vs Dense Attention ===")

    sparsity_configs = [
        {"B": 2, "H": 8, "S": 512,  "D": 64, "k": 16, "label": "S=512, k=16"},
        {"B": 2, "H": 8, "S": 1024, "D": 64, "k": 16, "label": "S=1024, k=16"},
        {"B": 2, "H": 8, "S": 2048, "D": 64, "k": 16, "label": "S=2048, k=16"},
    ]

    for cfg in sparsity_configs:
        B, H, S, D, k = cfg["B"], cfg["H"], cfg["S"], cfg["D"], cfg["k"]
        print(f"\n  Config: {cfg['label']} (B={B}, H={H}, D={D}) — sparse k={k} vs dense S={S}")

        Q = torch.randn(B, H, S, D, device=device)
        K = torch.randn(B, H, S, D, device=device)
        V = torch.randn(B, H, S, D, device=device)
        coords = torch.randn(B, S, 3, device=device)

        dense_result = benchmark_fn(flash_attention_forward, Q, K, V, warmup=5, n_runs=50)
        sparse_result = benchmark_fn(sparse_flash_attention, Q, K, V, coords, k, warmup=5, n_runs=50)
        dense_result.name = "Dense FlashAttn"
        sparse_result.name = "Sparse FlashAttn"
        print_benchmark_result(dense_result, sparse_result)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bench_sparse_flash_attn(device)
