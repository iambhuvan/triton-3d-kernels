# triton-3d-kernels — Benchmark Results

**Hardware:** NVIDIA H100 80GB HBM3 (via Modal)
**Stack:** PyTorch 2.10.0, Triton 3.6.0, CUDA 12.8, Python 3.11.12

---

## Flash Attention v2 (Triton)

**File:** `kernels/flash_attn_v2.py`

Key optimizations:
- Deferred normalization — accumulates unnormalized O, divides once at end
- Split backward — separate dK/dV and dQ kernels, no atomic_add
- Autotuned block sizes via `@triton.autotune`

### Forward Pass

| S | PyTorch | v2 | TFLOPS |
|---|---------|-----|--------|
| 128 | 0.061ms | 0.068ms | 2.9 |
| 256 | 0.059ms | 0.065ms | 10.9 |
| 512 | 0.063ms | 0.067ms | 32.8 |
| 1024 | 0.194ms | 0.127ms | 71.8 |
| 2048 | 0.737ms | 0.340ms | **100.8** |

### Backward Pass

| S | v2 bwd |
|---|--------|
| 128 | 0.358ms |
| 256 | 0.354ms |
| 512 | 0.389ms |
| 1024 | 0.587ms |
| 2048 | 1.676ms |

---

## TripoSR Integration

| Metric | Default (PyTorch SDPA) | My Triton v2 | Speedup |
|--------|----------------------|---------------|---------|
| Attention (16 self-attn layers) | 45.9 ms | 34.3 ms | **1.35x** |
| End-to-end forward | 94.4 ms | 82.8 ms | **1.14x** |
| Output correctness | — | cosine similarity = **1.000000** | — |
