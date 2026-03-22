"""
Shared benchmarking utilities for all kernels.

Provides:
  - benchmark_fn: time a function with warmup, returns median/mean/min
  - print_benchmark_result: pretty-print comparison table
  - roofline_analysis: compute achieved FLOPS and bandwidth vs theoretical peak
"""

import torch
import time
from dataclasses import dataclass


@dataclass
class BenchResult:
    name: str
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    n_runs: int


def benchmark_fn(
    fn,
    *args,
    warmup: int = 10,
    n_runs: int = 100,
    sync: bool = True,
    **kwargs,
) -> BenchResult:
    """Benchmark a function with warmup and multiple runs.

    Args:
        fn: callable to benchmark
        *args, **kwargs: arguments to pass to fn
        warmup: number of warmup iterations
        n_runs: number of timed iterations
        sync: whether to torch.cuda.synchronize() (set True for GPU ops)

    Returns:
        BenchResult with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)

    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times.sort()
    median = times[len(times) // 2]
    mean = sum(times) / len(times)

    return BenchResult(
        name=fn.__name__ if hasattr(fn, '__name__') else str(fn),
        median_ms=median,
        mean_ms=mean,
        min_ms=times[0],
        max_ms=times[-1],
        n_runs=n_runs,
    )


def print_benchmark_result(ref_result: BenchResult, tri_result: BenchResult):
    """Pretty-print a comparison between reference and Triton implementation."""
    speedup = ref_result.median_ms / tri_result.median_ms if tri_result.median_ms > 0 else float('inf')
    print(f"\n{'='*60}")
    print(f"  Benchmark: {tri_result.name}")
    print(f"{'='*60}")
    print(f"  {'':20s} {'Reference':>12s} {'Triton':>12s} {'Speedup':>10s}")
    print(f"  {'-'*54}")
    print(f"  {'Median (ms)':20s} {ref_result.median_ms:12.3f} {tri_result.median_ms:12.3f} {speedup:9.2f}x")
    print(f"  {'Mean (ms)':20s} {ref_result.mean_ms:12.3f} {tri_result.mean_ms:12.3f}")
    print(f"  {'Min (ms)':20s} {ref_result.min_ms:12.3f} {tri_result.min_ms:12.3f}")
    print(f"  {'Runs':20s} {ref_result.n_runs:12d} {tri_result.n_runs:12d}")
    print(f"{'='*60}\n")


def roofline_analysis(
    time_ms: float,
    flops: int,
    bytes_accessed: int,
    device_name: str = "A100",
):
    """Compute roofline model metrics.

    Args:
        time_ms: measured execution time in milliseconds
        flops: total floating point operations
        bytes_accessed: total bytes read + written
        device_name: GPU name for peak specs

    Returns:
        dict with achieved TFLOPS, bandwidth, arithmetic intensity, and utilization
    """
    # Peak specs for common GPUs
    GPU_SPECS = {
        "A100": {"peak_tflops_fp32": 19.5, "peak_tflops_fp16": 312, "peak_bw_gb": 2039},
        "H100": {"peak_tflops_fp32": 51.2, "peak_tflops_fp16": 990, "peak_bw_gb": 3350},
        "V100": {"peak_tflops_fp32": 15.7, "peak_tflops_fp16": 125, "peak_bw_gb": 900},
        "L4":   {"peak_tflops_fp32": 30.3, "peak_tflops_fp16": 242, "peak_bw_gb": 300},
    }

    specs = GPU_SPECS.get(device_name, GPU_SPECS["A100"])
    time_s = time_ms / 1000

    achieved_tflops = (flops / 1e12) / time_s if time_s > 0 else 0
    achieved_bw_gb = (bytes_accessed / 1e9) / time_s if time_s > 0 else 0
    arithmetic_intensity = flops / bytes_accessed if bytes_accessed > 0 else 0

    # Utilization
    compute_util = achieved_tflops / specs["peak_tflops_fp16"] * 100
    memory_util = achieved_bw_gb / specs["peak_bw_gb"] * 100

    result = {
        "achieved_tflops": achieved_tflops,
        "achieved_bw_gb_s": achieved_bw_gb,
        "arithmetic_intensity": arithmetic_intensity,
        "compute_utilization_pct": compute_util,
        "memory_utilization_pct": memory_util,
        "bottleneck": "compute" if arithmetic_intensity > (specs["peak_tflops_fp16"] * 1e3 / specs["peak_bw_gb"]) else "memory",
    }

    print(f"\n  Roofline Analysis ({device_name})")
    print(f"  {'-'*40}")
    print(f"  Achieved TFLOPS:       {achieved_tflops:.2f}")
    print(f"  Achieved BW (GB/s):    {achieved_bw_gb:.1f}")
    print(f"  Arithmetic Intensity:  {arithmetic_intensity:.1f} FLOP/byte")
    print(f"  Compute Utilization:   {compute_util:.1f}%")
    print(f"  Memory Utilization:    {memory_util:.1f}%")
    print(f"  Bottleneck:            {result['bottleneck']}")

    return result
