# Phase 2: Live Demo — Custom Kernels in a Real 3D Pipeline

**Goal:** Show our Triton/CUDA kernels powering actual 3D generative AI — not just benchmarks.
**Interview:** Meshy AI ML Systems Engineer Intern — Tue Mar 24, 2026 @ 5:30 PM EDT
**Deadline:** Mon Mar 23 evening (buffer for debugging)

---

## Architecture Overview

```
                          TripoSR (Image-to-3D)
                    ┌─────────────────────────────┐
  Input Image ───►  │  DINOv2 Encoder             │
                    │       ↓                      │
                    │  Transformer Backbone        │
                    │   ┌──────────────────────┐   │
                    │   │ Self-Attention ◄──────│───│──── OUR flash_attention_forward
                    │   │ Cross-Attention ◄─────│───│──── OUR flash_attention_forward
                    │   │ FFN                   │   │
                    │   └──────────────────────┘   │
                    │       ↓                      │
                    │  Triplane Features           │
                    │       ↓                      │
                    │  NeRF Decoder + Marching Cubes│
                    └──────────────┬───────────────┘
                                   ↓
                              3D Mesh (GLB)
                                   ↓
                         gr.Model3D (Gradio)


                     3DGS Viewer (Gaussian Splatting)
                    ┌──────────────────────────────┐
  Pre-trained ───►  │  Load .ply (positions, SH,   │
  .ply scene        │  scales, rotations, opacity) │
                    │       ↓                      │
                    │  Project 3D → 2D (camera)    │
                    │       ↓                      │
                    │  Compute 2D Covariances       │
                    │       ↓                      │
                    │  OUR gaussian_splat kernel ◄──│──── Renders every pixel
                    │       ↓                      │
                    └──────────────┬───────────────┘
                                   ↓
                         Rendered Images / Video
                                   ↓
                         gr.Gallery (Gradio)
```

---

## Task Breakdown

### TASK 1: Set Up TripoSR [~2 hours]

**What:** Get TripoSR running on Modal H100 with default attention.

**Steps:**
1. Clone TripoSR repo into our project:
   ```bash
   cd /Users/bhuvan/Desktop/triton-3d-kernels
   git clone https://github.com/VAST-AI-Research/TripoSR.git demo/triposr
   ```

2. Create `demo/requirements.txt` with TripoSR dependencies:
   ```
   omegaconf
   einops
   transformers
   trimesh
   rembg
   huggingface-hub
   gradio
   torchmcubes
   xatlas
   moderngl
   plyfile
   Pillow
   imageio
   imageio-ffmpeg
   ```

3. Write a minimal test script `demo/test_triposr.py`:
   - Load model: `TSR.from_pretrained("stabilityai/TripoSR")`
   - Run inference on a sample image (download one from web)
   - Export mesh to GLB
   - Print timing and verify mesh has vertices
   - This validates the pipeline works before we modify anything

4. Create `demo/modal_demo.py` with a Modal function that:
   - Installs all TripoSR deps in the image
   - Copies our code + TripoSR code
   - Runs the test script on H100
   - Verifies output mesh is valid

5. Run on Modal, confirm it works end-to-end.

**Success criteria:** TripoSR generates a valid 3D mesh from an image on Modal H100.

---

### TASK 2: Write TritonAttnProcessor [~2 hours]

**What:** A drop-in attention processor that calls our `flash_attention_forward` kernel.

**File:** `demo/triton_attn_processor.py`

**TripoSR's attention interface** (from `tsr/models/transformer/attention.py`):
```python
class AttnProcessor2_0:
    def __call__(
        self,
        attn: Attention,       # Module with .to_q, .to_k, .to_v, .to_out, .heads
        hidden_states: Tensor, # [B, S, C] — input to attention
        encoder_hidden_states: Optional[Tensor] = None,  # [B, S_enc, C] for cross-attn
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Projects Q, K, V via attn.to_q/k/v
        # Reshapes to [B, heads, S, head_dim]
        # Calls F.scaled_dot_product_attention(Q, K, V)
        # Reshapes back, applies attn.to_out
        ...
```

**Our processor must:**
1. Accept same signature
2. Project Q, K, V using attn's existing linear layers
3. Reshape from `[B, S, C]` → `[B, heads, S, head_dim]`
4. Call `flash_attention_forward(Q, K, V)` instead of `F.scaled_dot_product_attention`
5. Handle both self-attention (encoder_hidden_states=None) and cross-attention
6. Reshape back to `[B, S, C]`, apply `attn.to_out`

**Key considerations:**
- Our kernel expects float16 — TripoSR might run in float32. Add `.half()` conversion.
- Our kernel expects contiguous tensors — add `.contiguous()` calls.
- Cross-attention: K and V come from `encoder_hidden_states`, Q from `hidden_states`.
- Causal masking: TripoSR uses `is_causal=False` — pass accordingly.
- Head dim must be in {16, 32, 64, 128} — TripoSR uses D=64 per head, so this works.

**Also write:**
- `TritonAttnProcessorWithTiming` variant that records per-layer timing for benchmarking.
- `swap_attention(model)` helper function that patches all Attention modules.
- `restore_attention(model)` helper that restores default processors.

---

### TASK 3: Monkey-Patch and Verify Correctness [~1.5 hours]

**What:** Prove our kernel produces identical results to the default.

**File:** `demo/test_attention_swap.py`

**Steps:**
1. Load TripoSR model
2. Load a test image
3. Run inference with DEFAULT attention → save output mesh as `output_default.glb`
4. Call `swap_attention(model)` to patch in our kernel
5. Run inference with OUR attention → save output mesh as `output_triton.glb`
6. Compare:
   - Vertex positions: `torch.allclose(verts_default, verts_triton, atol=5e-3)`
   - Vertex colors: similar tolerance
   - Print max absolute difference
   - If mismatch > tolerance, log which layer diverges

**Edge cases to handle:**
- If TripoSR runs in float32 but our kernel only does float16:
  - Cast inputs to float16 before kernel, cast output back to float32
  - Accept slightly larger tolerance (atol=1e-2)
- If sequence length is not a multiple of BLOCK_SIZE:
  - Our kernel should handle this (it does — we have OOB masking)
- If batch size > 1:
  - Our kernel handles batched attention — should be fine

**Success criteria:** Both meshes are visually identical, vertex positions match within tolerance.

---

### TASK 4: Timing Comparison [~1 hour]

**What:** Measure real speedup of our kernel in the TripoSR pipeline.

**File:** `demo/benchmark_triposr.py`

**Measurements:**
1. **End-to-end inference time:**
   - Default attention: avg over 10 runs (skip first for warmup)
   - Our Triton attention: avg over 10 runs
   - Speedup = default_time / triton_time

2. **Attention-only time:**
   - Use `TritonAttnProcessorWithTiming` to measure just the attention calls
   - Sum across all layers
   - Compare to profiled default attention time
   - This isolates our kernel's contribution vs. the rest of the pipeline

3. **TFLOPS calculation:**
   - Count total attention FLOPs: `num_layers * 2 * B * H * S^2 * D` (for QK^T and PV)
   - Divide by measured time

4. **Output format:**
   ```
   TripoSR Inference Benchmark (NVIDIA H100 80GB HBM3)
   ============================================================
   Default (F.sdpa):     0.423s ± 0.012s
   Triton Flash Attn:    0.318s ± 0.008s
   Speedup:              1.33x

   Attention-only:
     Default:            0.089s (21% of inference)
     Triton:             0.034s (11% of inference)
     Attention speedup:  2.62x

   Achieved TFLOPS:      87.3 (our kernel)
   ```

---

### TASK 5: 3DGS Viewer with Our Kernel [~2.5 hours]

**What:** Load a pre-trained 3DGS scene and render it using our `gaussian_splat` kernel.

**File:** `demo/gaussian_viewer.py`

**Sub-steps:**

#### 5a. Find and download a pre-trained .ply scene
- Search HuggingFace for "gaussian splatting ply"
- Good candidates: garden scene, bicycle scene, room scene from Mip-NeRF360
- Download one .ply file (~50-200MB)
- Store in `demo/assets/scene.ply`

#### 5b. Write .ply loader
```python
def load_gaussians(ply_path):
    """Load 3DGS .ply file → dict of tensors on GPU."""
    # Returns:
    #   positions:  [N, 3]     float32
    #   sh_dc:      [N, 3]     float32 (base color)
    #   sh_rest:    [N, 45]    float32 (higher-order SH, optional)
    #   scales:     [N, 3]     float32 (log-space)
    #   rotations:  [N, 4]     float32 (quaternion)
    #   opacities:  [N, 1]     float32 (pre-sigmoid)
```

#### 5c. Write camera projection
```python
def make_camera(azimuth, elevation, radius, fov, W, H):
    """Create view + projection matrices for a camera orbit."""
    # Returns: view_matrix [4, 4], proj_matrix [4, 4], camera_pos [3]

def project_gaussians(positions, scales, rotations, view_matrix, proj_matrix, W, H):
    """Project 3D Gaussians to 2D screen space."""
    # 1. Transform positions to camera space: p_cam = view @ p_world
    # 2. Compute depths (for sorting)
    # 3. Compute 2D means: p_screen = proj @ p_cam (perspective divide)
    # 4. Compute 2D covariance from 3D cov + Jacobian of projection
    # 5. Sort by depth
    # Returns: means_2d [N, 2], covs_2d [N, 2, 2], depths [N], colors [N, 3], opacities [N]
```

#### 5d. Render with our kernel
```python
def render_scene(ply_path, azimuth=0, elevation=30, W=512, H=512):
    """Render one view of a 3DGS scene using our kernel."""
    gaussians = load_gaussians(ply_path)
    camera = make_camera(azimuth, elevation, ...)
    projected = project_gaussians(gaussians, camera)

    # Call OUR gaussian_splat_forward kernel
    image = gaussian_splat_forward(
        projected.means_2d,
        projected.covs_2d,
        projected.colors,
        projected.opacities,
        W, H
    )
    return image  # [H, W, 3] uint8
```

#### 5e. Generate rotating views
```python
def render_orbit(ply_path, n_frames=60, W=512, H=512):
    """Render a 360° orbit around the scene."""
    frames = []
    for i in range(n_frames):
        azimuth = i * (360 / n_frames)
        img = render_scene(ply_path, azimuth=azimuth)
        frames.append(img)

    # Save as video/GIF
    imageio.mimwrite("orbit.mp4", frames, fps=30)
    return frames
```

**Potential issues:**
- Our gaussian_splat kernel interface may not exactly match the projection output format
  → May need adapter code to reshape/reorder tensors
- SH evaluation: converting SH coefficients to RGB for a given view direction
  → Start with just DC (base color), add SH later if time permits
- Large scenes (millions of Gaussians) may need subsampling for real-time rendering
  → Cap at 100K-500K Gaussians for demo, or render offline

**Success criteria:** A rendered image of a recognizable 3D scene, produced entirely by our kernel.

---

### TASK 6: Build Gradio Demo App [~2 hours]

**What:** A polished interactive demo with multiple tabs.

**File:** `demo/app.py`

#### Tab 1: Image-to-3D (TripoSR + Our Flash Attention)
```
┌─────────────────────────────────────────────────────────┐
│  [Upload Image]            [Generate 3D]                │
│                                                          │
│  ┌──────────────┐          ┌──────────────────────────┐ │
│  │              │          │                          │ │
│  │  Input Image │          │  3D Mesh (interactive)   │ │
│  │              │          │  gr.Model3D              │ │
│  └──────────────┘          └──────────────────────────┘ │
│                                                          │
│  ⚡ Generated in 0.38s using Triton Flash Attention      │
│  📊 Attention speedup: 2.6x vs PyTorch default          │
│  🔧 Kernel: flash_attention_forward (Triton)             │
└─────────────────────────────────────────────────────────┘
```

#### Tab 2: 3D Gaussian Splatting Viewer
```
┌─────────────────────────────────────────────────────────┐
│  Scene: [Dropdown: garden / bicycle / room]              │
│  Camera Angle: [────●───────] 0° - 360°                 │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │                                                    │   │
│  │          Rendered View (512x512)                   │   │
│  │     "Every pixel computed by our Triton kernel"   │   │
│  │                                                    │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  [▶ Play Orbit Video]                                    │
│  Render time: 12.3ms per frame (81 FPS)                  │
│  Gaussians: 142,000 | Resolution: 512x512               │
└─────────────────────────────────────────────────────────┘
```

#### Tab 3: Kernel Dashboard
```
┌─────────────────────────────────────────────────────────┐
│  triton-3d-kernels — 10 Triton + 1 CUDA Kernel          │
│                                                          │
│  Kernel             │ Speedup │ Used In                  │
│  ─────────────────────────────────────────────           │
│  Flash Attention v1 │  3.1x   │ TripoSR (Tab 1)         │
│  Flash Attention v2 │  100 TF │ —                        │
│  Flash Attn v3 CUDA │  H100   │ WGMMA + TMA + WarpSpec  │
│  Gaussian Splat     │  1.3x   │ 3DGS Viewer (Tab 2)     │
│  RoPE               │  3.4x   │ —                        │
│  SwiGLU             │  2.4x   │ —                        │
│  Chamfer Distance   │  2.3x   │ —                        │
│  Sparse Flash Attn  │  —      │ —                        │
│  Ring Attention      │  4.5x   │ —                        │
│                                                          │
│  H100 Features (CUDA C++ kernel):                        │
│  ✅ WGMMA — wgmma.mma_async.sync.aligned.m64n64k16     │
│  ✅ TMA — cp.async.bulk.tensor.2d (hardware DMA)        │
│  ✅ Warp Specialization — setmaxnreg + named barriers   │
└─────────────────────────────────────────────────────────┘
```

---

### TASK 7: Interview Polish [~1 hour]

**What:** Make the demo presentation-ready.

**Details:**
1. **Side-by-side correctness proof:**
   - Render same 3D mesh from default vs. our kernel
   - Show they're visually identical
   - Display max vertex error: "Max error: 0.002mm"

2. **Live performance metrics:**
   - During TripoSR inference, show real-time progress
   - Display TFLOPS achieved by our kernel

3. **Code walkthrough prep:**
   - Highlight 3-4 key code snippets to show if interviewer asks:
     - WGMMA inline PTX (the most impressive)
     - TMA load with mbarrier
     - TritonAttnProcessor (the monkey-patch)
     - Gaussian splat kernel (tile-based rendering)

4. **Talking points document** (`demo/TALKING_POINTS.md`):
   - "Why I built this" → custom GPU kernels for 3D AI
   - "What's impossible in Triton" → WGMMA, TMA, warp specialization
   - "Real-world impact" → 2-3x attention speedup in actual 3D gen pipeline
   - "What I'd do at Meshy" → optimize your attention/rendering kernels

---

### TASK 8: Modal Deployment Script [~1 hour]

**What:** One-command launch of the full Gradio demo on H100.

**File:** `demo/modal_demo.py`

```python
# Usage: python -m modal serve demo/modal_demo.py
# → Opens a public Gradio URL on H100

@app.function(
    image=demo_image,  # All deps installed
    gpu="H100",
    timeout=3600,      # 1 hour for interview session
    allow_concurrent_inputs=5,
)
@modal.web_endpoint(method="GET")
def demo():
    # Launch Gradio app
    ...
```

**Image build includes:**
- PyTorch + Triton
- TripoSR + all its deps
- Pre-downloaded model weights (baked into image, no download during inference)
- Pre-downloaded 3DGS .ply scene
- Our triton-3d-kernels code

---

### TASK 9: End-to-End Test on Modal [~1 hour]

**What:** Verify everything works together on H100.

**Test checklist:**
- [ ] TripoSR loads and generates mesh with default attention
- [ ] Attention swap produces matching mesh
- [ ] Timing comparison shows speedup
- [ ] 3DGS scene loads from .ply
- [ ] Gaussian splat kernel renders valid images
- [ ] Orbit video generates correctly
- [ ] Gradio UI is responsive
- [ ] All three tabs work
- [ ] Modal URL is accessible from browser
- [ ] Test with 5 different input images
- [ ] Test with 2 different 3DGS scenes

---

### TASK 10: Record Backup Demo [~0.5 hours]

**What:** Screen recording + screenshots in case live demo fails.

**Capture:**
1. Screen recording of full demo flow (2-3 minutes):
   - Upload image → generate 3D mesh → rotate it
   - Switch to 3DGS tab → show rendered scene → orbit
   - Switch to dashboard → show kernel performance
2. Screenshots of key results:
   - Generated 3D mesh
   - Rendered 3DGS scene
   - Benchmark comparison table
   - WGMMA PTX code snippet
3. Save to `demo/assets/backup/`

---

## File Structure (Final)

```
triton-3d-kernels/
├── kernels/                    # Our 10 Triton kernels (Phase 1 — DONE)
├── cuda/                       # CUDA C++ Flash Attn v3 (Phase 1 — DONE)
├── tests/                      # 133 tests (Phase 1 — DONE)
├── bench/                      # Benchmarks (Phase 1 — DONE)
├── reference/                  # PyTorch reference implementations
├── demo/                       # Phase 2 — NEW
│   ├── triposr/                # TripoSR repo (git clone)
│   ├── assets/
│   │   ├── sample_images/      # Test input images
│   │   ├── scene.ply           # Pre-trained 3DGS scene
│   │   └── backup/             # Screenshots + video
│   ├── triton_attn_processor.py  # Our attention processor
│   ├── gaussian_viewer.py      # 3DGS viewer with our kernel
│   ├── test_triposr.py         # Validate TripoSR works
│   ├── test_attention_swap.py  # Correctness test
│   ├── benchmark_triposr.py    # Timing comparison
│   ├── app.py                  # Gradio demo (3 tabs)
│   ├── modal_demo.py           # Modal deployment
│   ├── requirements.txt        # Demo dependencies
│   └── TALKING_POINTS.md       # Interview prep notes
├── PHASE2_TODO.md              # This file
├── RESULTS.md                  # Phase 1 benchmark results
├── README.md                   # Project overview
└── modal_run.py                # Phase 1 Modal runner
```

---

## Timeline

| When | What | Hours |
|------|------|-------|
| **Sat Mar 21 evening** | Tasks 1-2: Set up TripoSR + write processor | 4h |
| **Sun Mar 22 morning** | Tasks 3-4: Verify correctness + benchmark | 2.5h |
| **Sun Mar 22 afternoon** | Task 5: 3DGS viewer | 2.5h |
| **Sun Mar 22 evening** | Task 6: Gradio app | 2h |
| **Mon Mar 23 morning** | Tasks 7-8: Polish + Modal deploy | 2h |
| **Mon Mar 23 afternoon** | Task 9: End-to-end test | 1h |
| **Mon Mar 23 evening** | Task 10: Record backup + final review | 0.5h |
| **Tue Mar 24 @ 5:30 PM** | **Interview** | — |

**Total: ~14.5 hours across 3 days**

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| TripoSR won't install on Modal | Pre-test in Task 1; fallback to Shap-E (simpler deps) |
| Our kernel shape mismatch | Task 2 handles reshaping; add `.contiguous()` and dtype casts |
| 3DGS rendering looks wrong | Start with solid colors (no SH), add view-dependent color later |
| Gradio 3D viewer doesn't work | Export mesh as GLB (best format support); render images as fallback |
| Modal times out | Bake model weights into image; increase timeout to 3600s |
| Live demo fails during interview | Task 10 captures backup video + screenshots |
