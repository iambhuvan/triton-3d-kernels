# triton-3d-kernels

**10 custom GPU kernels for 3D generative AI** — written in Triton and CUDA C++, targeting NVIDIA H100.

From high-level Triton abstractions down to raw Hopper PTX intrinsics (WGMMA, TMA, warp specialization), this project demonstrates GPU systems engineering across the full stack.

> **Live Demo:** [bnallamo--triton-3d-demo-launch-gradio.modal.run](https://bnallamo--triton-3d-demo-launch-gradio.modal.run)
> Swap our Triton Flash Attention into TripoSR's image-to-3D pipeline and see 1.35x attention speedup with identical output.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         triton-3d-kernels                                  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    TRANSFORMER KERNELS                               │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │  Flash Attn   │  │  Fused RoPE  │  │ Fused SwiGLU │               │   │
│  │  │  v1/v2/v3     │  │   3.4x ⚡    │  │   2.4x ⚡    │               │   │
│  │  │  100.8 TFLOPS │  │  (Triton)    │  │  (Triton)    │               │   │
│  │  │  (Triton)     │  └──────────────┘  └──────────────┘               │   │
│  │  └──────┬───────┘                                                    │   │
│  │         │                                                             │   │
│  │  ┌──────▼───────────────────────────────────────────────────────┐     │   │
│  │  │  Flash Attention v3 — CUDA C++ (H100-exclusive)             │     │   │
│  │  │                                                              │     │   │
│  │  │  WGMMA ─── 128-thread warp groups, 64-bit SMEM descriptors │     │   │
│  │  │  TMA ───── Async bulk tensor copies, hardware mbarrier     │     │   │
│  │  │  WARP SPEC ── Producer/consumer groups, register realloc   │     │   │
│  │  └──────────────────────────────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    3D-SPECIFIC KERNELS                               │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │ 3D Gaussian   │  │ Sparse Flash │  │   Chamfer    │               │   │
│  │  │  Splatting    │  │  Attention   │  │  Distance    │               │   │
│  │  │  (rasterizer) │  │  O(S·k)     │  │  2.3x ⚡     │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    DISTRIBUTED KERNEL                                │   │
│  │  ┌──────────────────────────────────────────────────┐               │   │
│  │  │  Ring Attention — sequence-parallel, multi-GPU   │               │   │
│  │  │  Online softmax merge across ring rotations      │               │   │
│  │  └──────────────────────────────────────────────────┘               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    LIVE DEMO (Modal H100)                            │   │
│  │                                                                      │   │
│  │  Input Image ──► TripoSR (32 attention layers) ──► 3D Mesh          │   │
│  │                       ↑                                              │   │
│  │              Our Triton kernel swapped in                            │   │
│  │              1.35x attention speedup                                 │   │
│  │              cosine similarity = 1.000000                            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Benchmark Results (NVIDIA H100 80GB HBM3)

| Kernel | Config | Our Kernel | PyTorch Ref | Speedup | TFLOPS |
|--------|--------|-----------|-------------|---------|--------|
| **Flash Attn v2** | B=2 H=16 S=2048 D=64 | 0.340 ms | — | — | **100.8** |
| **Flash Attn v1** | B=2 H=16 S=1024 D=64 | 0.151 ms | 0.468 ms | **3.1x** | 56.8 |
| **Flash Attn v1** | B=1 H=32 S=4096 D=128 | 3.664 ms | 9.160 ms | **2.5x** | 75.0 |
| **Fused RoPE** | B=2 S=2048 H=32 D=128 | 0.109 ms | 0.371 ms | **3.4x** | — |
| **Fused SwiGLU** | B=2 S=2048 D_ff=11008 | 0.211 ms | 0.507 ms | **2.4x** | — |
| **Chamfer Distance** | B=1 N=8192 | 0.973 ms | 2.238 ms | **2.3x** | — |
| **Gaussian Splatting** | N=10K, 512×512 | 1163 ms | 1512 ms | **1.3x** | — |
| **TripoSR Attention** | 32 layers, self-attn | 34.3 ms | 45.9 ms | **1.35x** | — |
| **TripoSR E2E Forward** | Image → scene codes | 82.8 ms | 94.4 ms | **1.14x** | — |

> **132/133 tests passing.** Every kernel validated against a pure-PyTorch reference implementation.

---

## Flash Attention — 4 Implementations, 2 Languages

The progression from v1→v3 mirrors the real Flash Attention papers (Dao 2022, 2023; Shah 2024):

```
v1 (Triton)          v2 (Triton)           v3-lite (Triton)       v3 (CUDA C++)
─────────────        ─────────────         ─────────────          ─────────────
Online softmax    →  Deferred norm      →  Persistent kernel  →  H100 PTX intrinsics
atomic_add dQ     →  Split backward     →  FP8 QK matmul     →  WGMMA + TMA
56.8 TFLOPS       →  100.8 TFLOPS      →  92.4 TFLOPS        →  (hardware-native)

Key insight:        Key insight:          Key insight:           Key insight:
Streaming m,l,O     Don't normalize O     One CTA loops over     Triton can't express
avoids S×S matrix   until the end;        all Q blocks;          WGMMA descriptors,
in HBM              separate dQ kernel    FP8 gives 2× FLOPs    TMA, or warp-spec

Backward: ✅        Backward: ✅          Backward: ✅           Forward only
(atomic_add dQ)     (no atomics, 1.4×     (same as v2)           (proves H100 mastery)
                     faster than v1)
```

### CUDA v3 — H100 Hopper Architecture Deep Dive

The CUDA kernel (`cuda/flash_attn_v3_hopper.cu`) directly programs three H100-exclusive hardware units:

```
┌─────────────────── CUDA Block (256 threads) ──────────────────┐
│                                                                 │
│  WG0: PRODUCER (threads 0-127)     WG1: CONSUMER (128-255)    │
│  ┌──────────────────────────┐      ┌──────────────────────────┐│
│  │  setmaxnreg.dec.sync 40 │      │  setmaxnreg.inc.sync 232││
│  │  (fewer regs → more for │      │  (more regs for WGMMA    ││
│  │   consumer side)         │      │   FP32 accumulators)     ││
│  │                          │      │                          ││
│  │  Loop:                   │      │  Loop:                   ││
│  │   TMA load K[stage]────────────►│   Wait mbarrier         ││
│  │   TMA load V[stage]────────────►│   WGMMA: S = Q @ K^T    ││
│  │   arrive_expect_tx()     │      │   Online softmax(S)     ││
│  │   Wait for consumer      │      │   WGMMA: O += P @ V     ││
│  │    to release stage      │      │   Release stage          ││
│  └──────────────────────────┘      └──────────────────────────┘│
│                                                                 │
│  Shared Memory (128B swizzled):                                │
│  ┌──────────┬──────────┬──────────┐                            │
│  │ Q tile   │ K[0]/K[1]│ V[0]/V[1]│  ← 2-stage pipeline      │
│  │ 64×64    │ 64×64 ×2 │ 64×64 ×2 │                           │
│  │ (8 KB)   │ (16 KB)  │ (16 KB)  │                           │
│  └──────────┴──────────┴──────────┘                            │
│                                                                 │
│  WGMMA Register Layout (m64n64k16, 32 regs/thread):           │
│    row = warp_id*16 + ((reg>>1)&1)*8 + (lane_id>>2)           │
│    col = (lane_id&3)*2 + (reg&1) + (reg>>2)*8                 │
│    (from CUTLASS CLayout_64x64)                                │
│                                                                 │
│  SMEM Descriptor (64-bit):                                     │
│    [13:0] = start_addr >> 4                                    │
│    [29:16] = leading byte offset >> 4                          │
│    [45:32] = stride byte offset >> 4                           │
│    [63:62] = swizzle_mode (1 = 128B swizzle)                   │
└─────────────────────────────────────────────────────────────────┘
```

**Inline PTX used:**
- `wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16` — Warp group matmul
- `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier` — TMA load
- `mbarrier.init`, `mbarrier.arrive.expect_tx`, `mbarrier.try_wait.parity` — Hardware barriers
- `setmaxnreg.dec.sync`, `setmaxnreg.inc.sync` — Register reallocation
- `bar.sync`, `bar.arrive` — Named barriers for warp specialization
- `fence.proxy.async` — Memory fence for async operations

---

## 3D-Specific Kernels

### Sparse Flash Attention (O(S·k) for Point Clouds)
```
Standard attention:  Every token attends to ALL S tokens     → O(S²)
Sparse attention:    Each token attends to k nearest neighbors → O(S·k)

At S=4096, k=16: skip 99.6% of KV blocks → 256× fewer compute pairs

Pipeline:
  3D coordinates → torch.cdist → KNN graph → block-level sparsity mask
  → Modified flash attention inner loop skips masked blocks
```

### 3D Gaussian Splatting Rasterizer
```
3D Gaussians → Project to 2D → Sort by depth → Tile-based rasterization

Standard 3DGS formula with proper Jacobian:
  Σ_2d = J @ W @ Σ_3d @ W^T @ J^T
  where J = [[fx/tz, 0, -fx·tx/tz²],      ← Perspective projection Jacobian
             [0, fy/tz, -fy·ty/tz²]]

Kernel: one Triton program per 16×16 screen tile
  For each pixel: front-to-back alpha blending over assigned Gaussians
  Early termination when transmittance < 0.001
```

### Chamfer Distance (3D Point Cloud Metric)
```
Input:  Two point clouds P1 (N pts), P2 (M pts)
Output: Average bidirectional nearest-neighbor distance

Tiled approach: never materializes full N×M distance matrix
  Block1: For each p1, find min distance to any p2 (streaming min)
  Block2: For each p2, find min distance to any p1 (streaming min)
  Result: mean(min_distances_1) + mean(min_distances_2)
```

---

## Live Demo — TripoSR Integration

Our kernels run inside a **real 3D generative AI pipeline** (TripoSR image-to-3D):

```
Input Image (PIL)
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  TripoSR Transformer (32 attention layers)           │
│                                                       │
│  16× Cross-Attention: Q=scene, K/V=image tokens      │
│       → PyTorch SDPA (seq lengths differ)            │
│                                                       │
│  16× Self-Attention: Q=K=V=scene tokens              │
│       → ⚡ OUR TRITON FLASH ATTENTION ⚡              │
│       → swap_attention() monkey-patches processor    │
│       → flash_attention_forward(q, k, v, causal=F)   │
│                                                       │
│  Result: 1.35x attention speedup, 1.14x E2E          │
│  Correctness: cosine similarity = 1.000000            │
└───────────────────────┬──────────────────────────────┘
                        │
                        ▼
          scene_codes → Marching Cubes → 3D Mesh (.obj)
```

**Why "only" 1.35x?** PyTorch 2.x already dispatches `F.scaled_dot_product_attention` to NVIDIA's FlashAttention-2 CUDA kernel. We're competing against an already-optimized baseline — beating it at all with a hand-written Triton kernel demonstrates real GPU programming skill.

---

## Project Structure

```
triton-3d-kernels/
├── kernels/                     # 10 Triton kernel implementations (2,831 LOC)
│   ├── flash_attn.py            # Flash Attention v1 (online softmax, fwd+bwd)
│   ├── flash_attn_v2.py         # Flash Attention v2 (deferred norm, split bwd)
│   ├── flash_attn_v3.py         # Flash Attention v3-lite (persistent, FP8)
│   ├── flash_attn_v3_cuda.py    # Flash Attention v3 CUDA wrapper
│   ├── rope.py                  # Fused RoPE (3.4x speedup)
│   ├── swiglu.py                # Fused SwiGLU MLP (2.4x speedup)
│   ├── gaussian_splat.py        # 3D Gaussian Splatting rasterizer
│   ├── sparse_flash_attn.py     # KNN-Sparse Attention for point clouds
│   ├── chamfer.py               # Batched Chamfer Distance (2.3x speedup)
│   └── ring_attn.py             # Ring Attention (distributed)
├── cuda/                        # CUDA C++ with inline PTX (~960 LOC)
│   ├── flash_attn_v3_hopper.cu  # Kernel: WGMMA + TMA + warp specialization
│   └── flash_attn_v3_hopper.cuh # PTX wrappers and constants
├── reference/                   # PyTorch reference implementations (for validation)
├── tests/                       # 133 tests (132 passing)
├── bench/                       # Benchmarks with roofline analysis
├── demo/                        # Live Gradio demo
│   ├── gradio_app.py            # 3-tab demo app
│   ├── triton_attn_processor.py # Drop-in TripoSR attention replacement
│   ├── modal_demo.py            # Modal H100 deployment
│   └── triposr/                 # TripoSR model (cloned)
└── README.md
```

## Setup & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run benchmarks
python bench/bench_all.py

# Launch local demo (requires GPU)
python demo/gradio_app.py

# Deploy to Modal H100
python -m modal deploy demo/modal_demo.py
```

## Key Design Decisions

- **Two abstraction levels** — Triton for 9 kernels (rapid iteration), CUDA C++ for 1 kernel (hardware-native). Shows ability to work at both.
- **Every kernel has a PyTorch reference** — correctness verified against naive-but-correct Python, not "does it not crash"
- **Online softmax everywhere** — Flash Attention, Ring Attention, and Sparse Attention all use streaming max+sum to avoid materializing S×S matrices
- **Real backward passes** — Flash Attention v1/v2/v3 and SwiGLU implement `torch.autograd.Function` with proper gradients
- **Honest about tradeoffs** — Sparse attention shows launch overhead at small S; Ring attention shows chunk management cost; FP8 shows accuracy vs speed tradeoff
- **Production integration** — Not just microbenchmarks: kernels swapped into TripoSR's full image-to-3D pipeline

---

*Built by Bhuvan Nallamo (CMU) — [bnallamo@andrew.cmu.edu](mailto:bnallamo@andrew.cmu.edu)*
