# triton-3d-kernels

**Custom Triton GPU kernel for 3D generative AI** — Flash Attention v2 achieving 100.8 TFLOPS on H100, integrated into TripoSR's image-to-3D pipeline for 1.35x attention speedup.

> **Live Demo:** [bnallamo--triton-3d-demo-launch-gradio.modal.run](https://bnallamo--triton-3d-demo-launch-gradio.modal.run)
> Swap my Triton Flash Attention into TripoSR and see the speedup with identical output.

---

## Benchmark Results (NVIDIA H100 80GB HBM3)

| Kernel | Config | Time | TFLOPS |
|--------|--------|------|--------|
| **Flash Attention v2** | B=2, H=16, S=2048, D=64 | 0.340 ms | **100.8** |
| **TripoSR Attention** | 16 self-attn layers | 34.3 ms (vs 45.9 ms default) | **1.35x speedup** |
| **TripoSR End-to-End** | Image → scene codes | 82.8 ms (vs 94.4 ms default) | **1.14x speedup** |

Output correctness: **cosine similarity = 1.000000**

---

## Flash Attention v2 — Key Optimizations

My implementation (`kernels/flash_attn_v2.py`) achieves **100.8 TFLOPS** through:

1. **Deferred normalization** — accumulates unnormalized O throughout the inner loop, divides by l once at the end. Fewer non-matmul FLOPs = better throughput.

2. **Split backward (no atomics)** — two separate kernels:
   - `_bwd_dkdv`: outer loop over KV blocks, inner over Q → computes dK, dV directly
   - `_bwd_dq`: outer loop over Q blocks, inner over KV → computes dQ directly
   - Neither needs `atomic_add` — each program owns its output exclusively
   - Result: **1.4x faster backward** at long sequences vs v1-style atomic approach

3. **Autotuned** — `@triton.autotune` searches 6 configs of `BLOCK_Q`, `BLOCK_KV`, `num_warps`, `num_stages`

### Online Softmax (Core Algorithm)

Standard attention materializes the full S×S attention matrix in HBM — O(S²) memory. My kernel avoids this by streaming through Q and KV blocks with three running accumulators:

```
for each KV block:
    S = Q_block @ K_block^T * scale     # compute in SRAM
    m_new = max(m, rowmax(S))            # update running max
    P = exp(S - m_new)                   # numerically stable softmax
    alpha = exp(m_old - m_new)           # rescale factor
    l = alpha * l + rowsum(P)            # update running sum
    O = alpha * O + P @ V_block          # accumulate (unnormalized)

O = O / l                               # normalize once at end
```

This is O(N) memory and the same O(N²) compute, but much faster because Q blocks stay in SRAM while K/V stream through.

---

## TripoSR Integration

My kernel runs inside a **real 3D generative AI pipeline** (TripoSR image-to-3D):

```
Input Image → TripoSR (32 transformer layers) → 3D Mesh

    16x Cross-Attention: PyTorch SDPA (Q_len != KV_len)
    16x Self-Attention:  MY TRITON FLASH ATTENTION v2
         └→ swap_attention() monkey-patches the processor
         └→ flash_attention_v2_forward(q, k, v, causal=False)

    Result: 1.35x attention speedup, 1.14x end-to-end
    Correctness: cosine similarity = 1.000000
```

**Why "only" 1.35x?** PyTorch 2.x already dispatches `F.scaled_dot_product_attention` to NVIDIA's FlashAttention-2 CUDA kernel. Beating an already-optimized baseline with a hand-written Triton kernel demonstrates real GPU programming skill.

---

## Project Structure

```
triton-3d-kernels/
├── kernels/
│   └── flash_attn_v2.py          # Flash Attention v2 (100.8 TFLOPS, fwd + bwd)
├── reference/
│   └── flash_attn_ref.py         # PyTorch reference (for correctness validation)
├── tests/                        # Test suite
├── bench/                        # Benchmarks
├── demo/
│   ├── gradio_app.py             # Interactive Gradio demo
│   ├── triton_attn_processor.py  # Drop-in TripoSR attention replacement
│   ├── modal_demo.py             # Modal H100 deployment
│   └── triposr/                  # TripoSR model
└── README.md
```

## Setup & Usage

```bash
pip install -r requirements.txt
pytest tests/ -v
python bench/bench_flash_attn_versions.py
python demo/gradio_app.py                   # local GPU
python -m modal deploy demo/modal_demo.py   # Modal H100
```

---

*Built by Bhuvan Nallamo (CMU) — [bnallamo@andrew.cmu.edu](mailto:bnallamo@andrew.cmu.edu)*
