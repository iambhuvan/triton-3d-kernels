# Interview Guide — Meshy AI ML Systems Engineer Intern
**Tue Mar 24, 2026 · 5:30 PM EDT**

---

## 30-Second Elevator Pitch

> "I built a custom Flash Attention v2 kernel in Triton that hits 100.8 TFLOPS on H100. To prove it works in production, I swapped it into TripoSR's image-to-3D pipeline and got 1.35x attention speedup with numerically identical output — cosine similarity of 1.000000. Here's the live demo."

---

## Demo Walkthrough (5 minutes)

### Step 1: Image-to-3D Tab (3 min)
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

### Step 2: Benchmark Tab (2 min)
- Show how TFLOPS scales with sequence length
- At S=2048: 100.8 TFLOPS (10% of H100's 989 FP16 peak — typical for memory-bound attention)

---

## Likely Questions & Answers

### "How does Flash Attention work?"
> "Standard attention materializes the full S×S attention matrix in HBM — that's O(S²) memory. Flash Attention avoids this by processing Q blocks one at a time, streaming K/V blocks through, and maintaining a running softmax via three accumulators: m (row max), l (row sum of exp), and O (unnormalized output). At the end, you divide O by l. This is O(N) memory and the same O(N²) compute, but much faster because it fits in SRAM."

### "What makes your v2 implementation fast?"
> "Two key optimizations. First, deferred normalization — instead of normalizing O after every K block, I accumulate unnormalized O and apply 1/l once at the end. Fewer non-matmul FLOPs in the inner loop. Second, the backward pass is split into two separate kernels — one for dK/dV, one for dQ — eliminating atomic_add entirely. Result: 1.4x faster backward at long sequences and 100.8 TFLOPS forward."

### "Why did you build this?"
> "I wanted to understand GPU programming beyond PyTorch — the actual memory hierarchy, tiling strategies, online algorithms. Flash Attention is the perfect target because it's THE bottleneck in transformers and the optimization is elegant: you can do O(N²) compute with O(N) memory by streaming. Then I integrated it into TripoSR to prove it works in a real 3D pipeline."

### "What was the hardest part?"
> "Getting the backward pass right. You have to recompute the attention matrix from Q, K, V and the stored log-sum-exp, then compute three gradients (dQ, dK, dV) efficiently. The key insight was splitting into two kernels — one where each program owns a KV block (for dK/dV) and one where each program owns a Q block (for dQ). This completely eliminates the atomic_add contention that makes v1-style backward slow."

### "Why Triton instead of CUDA?"
> "Triton gives you 80-90% of the performance with 10% of the development time. The autotuning is free — I just specify block sizes and warp counts to try, and Triton picks the best config. It also handles memory coalescing, shared memory allocation, and register pressure automatically. For most kernel work, Triton is the right tool."

### "How would you optimize Meshy's pipeline?"
> "First, profile to find the bottleneck — likely attention in the transformer backbone. Then:
> 1. Swap in optimized attention (Flash Attention or FlashDecoding for inference)
> 2. Fuse element-wise ops (RoPE, SwiGLU, LayerNorm) to reduce HBM traffic
> 3. Quantization (FP8) for 2x throughput on H100
> 4. For 3D-specific: sparse attention exploiting spatial locality
> 5. Sequence parallelism for long-context multi-view generation"

### "What would you do differently?"
> "I'd add multi-head attention patterns — GQA and MQA — since that's what production models actually use. I'd also implement FP8 quantization in the forward pass for 2x TFLOPS on H100. And I'd benchmark against FlashAttention-3 (the Dao AI Lab CUDA kernel) to see exactly where the gap is."

---

## GitHub Repo Link
**https://github.com/iambhuvan/triton-3d-kernels**

## Live Demo Link
**https://bnallamo--triton-3d-demo-launch-gradio.modal.run**

## Backup Plan
If Modal is down or the demo fails:
1. Show the GitHub README (has architecture diagrams)
2. Show RESULTS.md with benchmark numbers
3. Walk through code in IDE — show flash_attn_v2.py, triton_attn_processor.py
4. Show test results: `pytest tests/ -v`

---

## Pre-Interview Checklist
- [ ] Push latest code to GitHub
- [ ] Test the live demo URL (upload an image, check timing numbers)
- [ ] Have repo open in VS Code for code walkthrough
- [ ] Have this guide open on a second screen
- [ ] Record backup demo video (just in case)
