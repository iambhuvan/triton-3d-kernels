# Interview Guide — Meshy AI ML Systems Engineer Intern
**Tue Mar 24, 2026 · 5:30 PM EDT**

---

## 30-Second Elevator Pitch

> "I built 10 custom GPU kernels for 3D generative AI — Flash Attention through 4 versions from Triton to raw CUDA C++ with H100 PTX intrinsics, plus 3D-specific kernels like Gaussian Splatting, Sparse Attention for point clouds, and Chamfer Distance. To prove these aren't just microbenchmarks, I swapped my Triton Flash Attention kernel into TripoSR's image-to-3D pipeline and got 1.35x attention speedup with numerically identical output. Here's the live demo."

---

## Demo Walkthrough (5 minutes)

### Step 1: Image-to-3D Tab (2 min)
1. Upload a chair/shoe/object image
2. Click "Generate 3D & Compare"
3. **While it's loading, say:**
   > "This runs the same image through TripoSR twice — once with PyTorch's default attention, once with my Triton kernel swapped in. The model has 32 attention layers. I replace the 16 self-attention layers via a monkey-patching processor pattern."

4. **When results appear, point to:**
   - Side-by-side renders: "Same mesh — visually identical"
   - Cosine similarity: "1.000000 — numerically identical"
   - Timing table: "1.35x speedup on attention, 1.14x end-to-end"

5. **Why only 1.35x? Have this ready:**
   > "PyTorch 2.x already dispatches SDPA to NVIDIA's FlashAttention-2 CUDA kernel internally. I'm competing against an already highly-optimized baseline. Beating it at all with hand-written Triton proves I understand the GPU memory hierarchy — tiling, online softmax, register pressure."

### Step 2: 3DGS Tab (1 min)
1. Click "Render" with Colored Cube
2. **Say:**
   > "This is our tile-based Gaussian Splatting rasterizer, 100% written in Triton. It does the full pipeline — project 3D Gaussians to 2D using the perspective Jacobian, sort by depth, assign to screen tiles, then alpha-blend front-to-back per pixel."

3. Change scene to Sphere or Galaxy Spiral, click Render again

### Step 3: Kernel Overview Tab (1 min)
1. Scroll through benchmarks
2. **Highlight:**
   - "100.8 TFLOPS on Flash Attention v2"
   - "3.4x speedup on RoPE"
   - "132 out of 133 tests passing"
3. Point to Flash Attention evolution diagram

### Step 4: CUDA v3 (1 min — if asked)
**Have this in your back pocket if they ask about low-level GPU:**
> "The CUDA v3 kernel uses three H100 features that Triton can't express:
> 1. WGMMA — 128-thread warp groups with 64-bit SMEM descriptors
> 2. TMA — hardware tensor memory accelerator, single thread issues async bulk load
> 3. Warp specialization — producer warp group does TMA, consumer does WGMMA, with register reallocation between them
>
> I can show you the actual inline PTX in the source code."

---

## Likely Questions & Answers

### "Why did you build this?"
> "I wanted to understand GPU programming at every level — not just PyTorch, but the actual hardware. I started with Flash Attention because it's THE bottleneck in transformers, then extended to 3D-specific kernels because that's what Meshy needs. The CUDA v3 kernel was the ultimate challenge — proving I can work at the PTX level on H100."

### "What was the hardest part?"
> "The CUDA v3 kernel. Getting the WGMMA register layout right took days — each warp group of 128 threads has 32 FP32 accumulators, and the mapping from register index to matrix row/col follows a specific bit-encoding from CUTLASS's CLayout_64x64. I also had to figure out the 64-bit SMEM descriptor format — bits [63:62] encode the swizzle mode, and I initially had it wrong (used 3 instead of 1 for 128B swizzle)."

### "How does Flash Attention work?"
> "Standard attention materializes the full S×S attention matrix in HBM — that's O(S²) memory. Flash Attention avoids this by processing Q blocks one at a time, streaming K/V blocks through, and maintaining a running softmax via three accumulators: m (row max), l (row sum of exp), and O (unnormalized output). At the end, you divide O by l. This is O(N) memory and the same O(N²) compute, but much faster because it fits in SRAM."

### "What's the difference between v1 and v2?"
> "v1 normalizes O after every K block — unnecessary FLOPs. v2 defers normalization to the end, accumulating unnormalized O and applying 1/l once. For backward, v1 uses atomic_add to accumulate dQ from different K blocks. v2 splits this into two separate kernels — one for dK/dV, one for dQ — eliminating atomics entirely. Result: 1.4x faster backward."

### "Why Triton instead of CUDA?"
> "Triton lets me iterate 10x faster — I wrote 9 kernels in Triton. But for H100-specific features like WGMMA and TMA, you NEED CUDA C++ with inline PTX. I did both to show I can work at every level. In production you'd use Triton for most kernels and drop to CUDA only when you need hardware-specific features."

### "How would you optimize Meshy's pipeline?"
> "First, profile to find the bottleneck — likely attention in the transformer backbone. Then:
> 1. Swap in optimized attention (Flash Attention or even FlashDecoding for inference)
> 2. Fuse element-wise ops (RoPE, SwiGLU, LayerNorm) to reduce HBM traffic
> 3. For 3D-specific ops: sparse attention exploiting spatial locality, and optimized Gaussian Splatting rasterizer
> 4. Quantization (FP8) for 2x throughput on H100 with <6% accuracy loss
> 5. Sequence parallelism (Ring Attention) for long-context multi-view generation"

### "What would you do differently?"
> "I'd pre-compile the CUDA v3 kernel instead of JIT compiling on every Modal run — the 25-minute compilation time is a practical issue. I'd also implement the backward pass for the CUDA v3 kernel and add multi-head attention patterns (GQA, MQA) which are what production models actually use."

### "Tell me about the Gaussian Splatting kernel"
> "3DGS represents scenes as collections of 3D Gaussians. My rasterizer does the full pipeline:
> 1. Project 3D→2D using the standard Jacobian: Σ_2d = J @ W @ Σ_3d @ W^T @ J^T
> 2. Sort by depth for correct alpha compositing
> 3. Assign Gaussians to 16×16 screen tiles based on their bounding radius
> 4. Triton kernel: one program per tile, iterate pixels, alpha-blend front-to-back
> The key optimization is tile-based — adjacent pixels share the same Gaussian list, enabling coalesced loads."

### "What's your experience with distributed training?"
> "I implemented Ring Attention — splits the sequence across GPUs and rotates K/V blocks in a ring. The key insight is the online softmax merge: when you combine partial attention outputs from different K/V chunks, you need to rescale by the relative max values. I implemented both single-GPU simulation and the real torch.distributed version with isend/irecv."

---

## GitHub Repo Link
**https://github.com/iambhuvan/triton-3d-kernels**

## Live Demo Link
**https://bnallamo--triton-3d-demo-launch-gradio.modal.run**

## Backup Plan
If Modal is down or the demo fails:
1. Show the GitHub README (has all architecture diagrams)
2. Show RESULTS.md with benchmark numbers
3. Walk through code in IDE — show flash_attn.py, the CUDA kernel, triton_attn_processor.py
4. Show test results: `pytest tests/ -v` (132/133 passing)

---

## Pre-Interview Checklist
- [ ] Push latest code to GitHub
- [ ] Test the live demo URL (upload an image, check timing numbers)
- [ ] Test 3DGS tab (render a cube)
- [ ] Have repo open in VS Code for code walkthrough
- [ ] Have this guide open on a second screen
- [ ] Record backup demo video (just in case)
