"""Benchmark K2: Flash Attention."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from reference.flash_attn_ref import flash_attention_reference
from kernels.flash_attn import flash_attention_forward
from utils.benchmark import benchmark_fn, print_benchmark_result, roofline_analysis


def bench_flash_attn(device='cpu'):
    configs = [
        {"B": 2, "H": 8, "S": 256, "D": 64, "label": "Small"},
        {"B": 2, "H": 16, "S": 1024, "D": 64, "label": "Medium"},
        {"B": 1, "H": 32, "S": 4096, "D": 128, "label": "Large (LLaMA-like)"},
    ]

    for cfg in configs:
        B, H, S, D = cfg["B"], cfg["H"], cfg["S"], cfg["D"]
        print(f"\n  Config: {cfg['label']} (B={B}, H={H}, S={S}, D={D})")

        Q = torch.randn(B, H, S, D, device=device)
        K = torch.randn(B, H, S, D, device=device)
        V = torch.randn(B, H, S, D, device=device)

        ref_result = benchmark_fn(flash_attention_reference, Q, K, V, warmup=5, n_runs=50)
        tri_result = benchmark_fn(flash_attention_forward, Q, K, V, warmup=5, n_runs=50)
        ref_result.name = "PyTorch ref"
        tri_result.name = "Triton FlashAttn"
        print_benchmark_result(ref_result, tri_result)

        # Roofline: attention is ~4*B*H*S^2*D flops
        flops = 4 * B * H * S * S * D
        bytes_accessed = 4 * B * H * S * D * 4  # Q, K, V reads + O write (fp32)
        if device == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            if "A100" in gpu_name:
                roofline_analysis(tri_result.median_ms, flops, bytes_accessed, "A100")
            elif "H100" in gpu_name:
                roofline_analysis(tri_result.median_ms, flops, bytes_accessed, "H100")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bench_flash_attn(device)
