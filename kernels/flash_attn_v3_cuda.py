"""
Flash Attention v3 — CUDA C++ Kernel Wrapper (H100 Hopper)

This wraps the real CUDA implementation that uses H100-specific hardware:
  - WGMMA (Warp Group Matrix Multiply Accumulate) for QK^T and PV matmuls
  - TMA (Tensor Memory Accelerator) for async GMEM→SMEM bulk copies
  - Warp Specialization: producer/consumer warp groups with named barriers
  - 2-stage software pipeline: prefetch next KV while computing current
  - Register reallocation: producers get fewer regs, consumers get more

These features are impossible in Triton — they require inline PTX / CUDA C++.

Usage:
    from kernels.flash_attn_v3_cuda import flash_attention_v3_cuda

    # Requires H100 GPU (sm_90a) and FP16 inputs
    O = flash_attention_v3_cuda(Q, K, V, causal=False)
"""

import os
import torch

# JIT compile the CUDA extension on first import
_cuda_module = None


def _get_cuda_module():
    """Lazy JIT compilation of the CUDA kernel."""
    global _cuda_module
    if _cuda_module is not None:
        return _cuda_module

    from torch.utils.cpp_extension import load

    cuda_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cuda')
    cuda_dir = os.path.normpath(cuda_dir)

    # Find CUDA include paths for driver API headers (cuda.h with CUtensorMap etc)
    extra_includes = [cuda_dir]
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and os.path.isdir(os.path.join(cuda_home, "include")):
        extra_includes.append(os.path.join(cuda_home, "include"))

    # We need to link libcuda (driver API) for cuTensorMapEncodeTiled
    extra_ldflags = ["-lcuda"]

    _cuda_module = load(
        name="flash_attn_v3_hopper",
        sources=[os.path.join(cuda_dir, "flash_attn_v3_hopper.cu")],
        extra_include_paths=extra_includes,
        extra_cuda_cflags=[
            "-O3",
            "-arch=sm_90a",
            "--use_fast_math",
            "--ptxas-options=-v",
        ],
        extra_cflags=["-O3"],
        extra_ldflags=extra_ldflags,
        verbose=True,
    )
    return _cuda_module


def flash_attention_v3_cuda(Q, K, V, causal=False):
    """Flash Attention v3 forward pass using CUDA C++ kernel with H100 features.

    This is the REAL v3 implementation using:
      - WGMMA for matrix multiplies (not just tl.dot)
      - TMA for async memory loads (not just tl.load)
      - Producer/consumer warp specialization (impossible in Triton)
      - Named barrier synchronization
      - Register reallocation between warp groups

    Args:
        Q, K, V: (B, H, S, D) tensors, FP16, contiguous
                 D must be 64 (matching kHeadDim in the kernel)
        causal: whether to apply causal mask

    Returns:
        O: (B, H, S, D) attention output, FP16
    """
    assert Q.is_cuda, "Requires CUDA tensor"
    assert Q.dtype == torch.float16, "Requires FP16 input"
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()

    # Check for H100
    props = torch.cuda.get_device_properties(Q.device)
    if props.major < 9:
        raise RuntimeError(
            f"Flash Attention v3 CUDA kernel requires H100 (sm_90a), "
            f"but got {props.name} (sm_{props.major}{props.minor}). "
            f"Use the Triton v3 kernel (flash_attn_v3.py) as fallback."
        )

    module = _get_cuda_module()
    return module.forward(Q, K, V, causal)


def is_hopper_available():
    """Check if H100 (Hopper, sm_90) is available."""
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major >= 9
