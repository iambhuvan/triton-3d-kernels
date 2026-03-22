"""Benchmark K7: Ring Attention."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from reference.ring_attn_ref import ring_attention_reference
from kernels.ring_attn import ring_attention
from kernels.flash_attn import flash_attention_forward
from utils.benchmark import benchmark_fn, print_benchmark_result, roofline_analysis


def bench_ring_attn(device='cpu'):
    # --- Part 1: Ring Attention Triton vs Reference across S and n_chunks ---
    configs = [
        {"B": 2, "H": 8, "S": 256,  "D": 64, "n_chunks": 2, "label": "S=256, chunks=2"},
        {"B": 2, "H": 8, "S": 256,  "D": 64, "n_chunks": 4, "label": "S=256, chunks=4"},
        {"B": 2, "H": 8, "S": 256,  "D": 64, "n_chunks": 8, "label": "S=256, chunks=8"},
        {"B": 2, "H": 8, "S": 512,  "D": 64, "n_chunks": 2, "label": "S=512, chunks=2"},
        {"B": 2, "H": 8, "S": 512,  "D": 64, "n_chunks": 4, "label": "S=512, chunks=4"},
        {"B": 2, "H": 8, "S": 512,  "D": 64, "n_chunks": 8, "label": "S=512, chunks=8"},
        {"B": 2, "H": 8, "S": 1024, "D": 64, "n_chunks": 2, "label": "S=1024, chunks=2"},
        {"B": 2, "H": 8, "S": 1024, "D": 64, "n_chunks": 4, "label": "S=1024, chunks=4"},
        {"B": 2, "H": 8, "S": 1024, "D": 64, "n_chunks": 8, "label": "S=1024, chunks=8"},
        {"B": 2, "H": 8, "S": 2048, "D": 64, "n_chunks": 2, "label": "S=2048, chunks=2"},
        {"B": 2, "H": 8, "S": 2048, "D": 64, "n_chunks": 4, "label": "S=2048, chunks=4"},
        {"B": 2, "H": 8, "S": 2048, "D": 64, "n_chunks": 8, "label": "S=2048, chunks=8"},
        {"B": 2, "H": 8, "S": 4096, "D": 64, "n_chunks": 2, "label": "S=4096, chunks=2"},
        {"B": 2, "H": 8, "S": 4096, "D": 64, "n_chunks": 4, "label": "S=4096, chunks=4"},
        {"B": 2, "H": 8, "S": 4096, "D": 64, "n_chunks": 8, "label": "S=4096, chunks=8"},
    ]

    print("\n  === Ring Attention: Triton vs Reference ===")

    for cfg in configs:
        B, H, S, D, n_chunks = cfg["B"], cfg["H"], cfg["S"], cfg["D"], cfg["n_chunks"]
        print(f"\n  Config: {cfg['label']} (B={B}, H={H}, S={S}, D={D}, n_chunks={n_chunks})")

        Q = torch.randn(B, H, S, D, device=device)
        K = torch.randn(B, H, S, D, device=device)
        V = torch.randn(B, H, S, D, device=device)

        ref_result = benchmark_fn(ring_attention_reference, Q, K, V, n_chunks, warmup=5, n_runs=50)
        tri_result = benchmark_fn(ring_attention, Q, K, V, n_chunks, warmup=5, n_runs=50)
        ref_result.name = "PyTorch ref"
        tri_result.name = "Triton RingAttn"
        print_benchmark_result(ref_result, tri_result)

        # Roofline: ring attention is still O(S^2) compute, same as standard attention
        flops = 4 * B * H * S * S * D
        bytes_accessed = 4 * B * H * S * D * 4  # Q, K, V reads + O write (fp32)
        if device == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            if "A100" in gpu_name:
                roofline_analysis(tri_result.median_ms, flops, bytes_accessed, "A100")
            elif "H100" in gpu_name:
                roofline_analysis(tri_result.median_ms, flops, bytes_accessed, "H100")

    # --- Part 2: Ring Attention vs Standard Flash Attention ---
    # Ring attention trades memory for the same compute; expect similar speed
    print("\n  === Ring vs Standard Flash Attention (memory savings, not speed) ===")

    compare_configs = [
        {"B": 2, "H": 8, "S": 512,  "D": 64, "n_chunks": 4, "label": "S=512, chunks=4"},
        {"B": 2, "H": 8, "S": 1024, "D": 64, "n_chunks": 4, "label": "S=1024, chunks=4"},
        {"B": 2, "H": 8, "S": 2048, "D": 64, "n_chunks": 4, "label": "S=2048, chunks=4"},
        {"B": 2, "H": 8, "S": 4096, "D": 64, "n_chunks": 4, "label": "S=4096, chunks=4"},
    ]

    for cfg in compare_configs:
        B, H, S, D, n_chunks = cfg["B"], cfg["H"], cfg["S"], cfg["D"], cfg["n_chunks"]
        print(f"\n  Config: {cfg['label']} (B={B}, H={H}, D={D}) — ring vs standard")

        Q = torch.randn(B, H, S, D, device=device)
        K = torch.randn(B, H, S, D, device=device)
        V = torch.randn(B, H, S, D, device=device)

        std_result = benchmark_fn(flash_attention_forward, Q, K, V, warmup=5, n_runs=50)
        ring_result = benchmark_fn(ring_attention, Q, K, V, n_chunks, warmup=5, n_runs=50)
        std_result.name = "Standard FlashAttn"
        ring_result.name = "Ring Attention"
        print_benchmark_result(std_result, ring_result)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bench_ring_attn(device)
