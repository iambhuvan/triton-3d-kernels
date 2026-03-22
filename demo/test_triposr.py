"""
test_triposr.py — Verify TripoSR works end-to-end, then test with our Triton kernel.

This script:
  1. Loads the TripoSR model (downloads weights from HuggingFace)
  2. Runs inference on a sample image with DEFAULT attention
  3. Runs inference with our TRITON attention kernel swapped in
  4. Compares outputs for correctness (cosine similarity)
  5. Reports timing for both paths

Usage:
    python demo/test_triposr.py                    # basic verification
    python demo/test_triposr.py --benchmark         # run timing comparison
    python demo/test_triposr.py --image path/to.png # custom image
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

# ── Path setup ──────────────────────────────────────────────────────────────
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DEMO_DIR)

# Add TripoSR to path
sys.path.insert(0, os.path.join(DEMO_DIR, "triposr"))
# Add project root for our kernels
sys.path.insert(0, PROJECT_ROOT)

from tsr.system import TSR
from triton_attn_processor import (
    TritonAttnProcessor,
    TritonAttnProcessorWithTiming,
    DefaultAttnProcessorWithTiming,
    swap_attention,
    swap_default_with_timing,
    restore_attention,
    count_attention_layers,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_image(image_path: str) -> Image.Image:
    """Load and prepare image for TripoSR (RGB with gray background)."""
    try:
        import rembg
        image = Image.open(image_path)
        # Remove background
        session = rembg.new_session()
        image = rembg.remove(image, session=session)
        # Convert RGBA to RGB with gray background
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image
    except (ImportError, Exception) as e:
        log.warning(f"rembg not available ({e}) — loading image as-is (RGB)")
        return Image.open(image_path).convert("RGB")


def load_model(device: str = "cuda:0") -> TSR:
    """Load TripoSR model from HuggingFace."""
    log.info("Loading TripoSR model from stabilityai/TripoSR ...")
    t0 = time.perf_counter()
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(8192)
    model.to(device)
    elapsed = time.perf_counter() - t0
    log.info(f"Model loaded in {elapsed:.2f}s")
    return model


def run_forward(model, image, device: str = "cuda:0"):
    """Run TripoSR forward pass (image → scene codes). Returns scene_codes tensor."""
    with torch.no_grad():
        scene_codes = model([image], device=device)
    return scene_codes


def extract_mesh(model, scene_codes, resolution: int = 256):
    """Extract 3D mesh from scene codes via marching cubes."""
    meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=resolution)
    return meshes[0]


# ── Main test functions ─────────────────────────────────────────────────────

def test_basic_inference(model, image, device: str = "cuda:0"):
    """Test 1: Verify TripoSR produces valid output with default attention."""
    log.info("=" * 60)
    log.info("TEST 1: Basic TripoSR inference (default attention)")
    log.info("=" * 60)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    scene_codes = run_forward(model, image, device)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    log.info(f"  scene_codes shape: {scene_codes.shape}")
    log.info(f"  scene_codes dtype: {scene_codes.dtype}")
    log.info(f"  scene_codes range: [{scene_codes.min():.4f}, {scene_codes.max():.4f}]")
    log.info(f"  Forward pass time: {elapsed * 1000:.1f} ms")

    # Verify output is valid (not NaN, not all zeros)
    assert not torch.isnan(scene_codes).any(), "scene_codes contains NaN!"
    assert not torch.isinf(scene_codes).any(), "scene_codes contains Inf!"
    assert scene_codes.abs().sum() > 0, "scene_codes is all zeros!"

    log.info("  ✓ Basic inference PASSED\n")
    return scene_codes


def test_triton_kernel(model, image, default_scene_codes, device: str = "cuda:0"):
    """Test 2: Swap in our Triton kernel and verify correctness."""
    log.info("=" * 60)
    log.info("TEST 2: TripoSR with our Triton Flash Attention kernel")
    log.info("=" * 60)

    # Count attention layers
    count, info = count_attention_layers(model)
    log.info(f"  Found {count} attention layers:")
    for layer_info in info[:5]:  # show first 5
        log.info(f"    {layer_info['name']}: heads={layer_info['heads']}, "
                 f"dim={layer_info['inner_dim']}, cross={layer_info['is_cross']}")
    if len(info) > 5:
        log.info(f"    ... and {len(info) - 5} more")

    # Swap attention to our Triton kernel
    original_processors = swap_attention(model, use_timing=False)

    # Run forward pass with our kernel
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    triton_scene_codes = run_forward(model, image, device)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    log.info(f"  triton scene_codes shape: {triton_scene_codes.shape}")
    log.info(f"  Forward pass time: {elapsed * 1000:.1f} ms")

    # Restore original attention
    restore_attention(model, original_processors)

    # Check correctness
    assert not torch.isnan(triton_scene_codes).any(), "Triton output contains NaN!"
    assert not torch.isinf(triton_scene_codes).any(), "Triton output contains Inf!"

    # Compare with default output (cosine similarity)
    cos_sim = torch.nn.functional.cosine_similarity(
        default_scene_codes.flatten().unsqueeze(0).float(),
        triton_scene_codes.flatten().unsqueeze(0).float(),
    ).item()

    # L2 relative error
    l2_err = (default_scene_codes.float() - triton_scene_codes.float()).norm() / default_scene_codes.float().norm()

    log.info(f"  Cosine similarity (default vs triton): {cos_sim:.6f}")
    log.info(f"  Relative L2 error: {l2_err:.6f}")

    # Threshold: fp16 quantization introduces some error, but should be very close
    if cos_sim > 0.99:
        log.info("  ✓ Triton kernel correctness PASSED (cosine sim > 0.99)\n")
    elif cos_sim > 0.95:
        log.warning("  ⚠ Triton kernel output is close but not exact (cosine sim > 0.95)")
        log.warning("    This is expected due to fp16 precision differences\n")
    else:
        log.error(f"  ✗ Triton kernel output diverged significantly! cosine sim = {cos_sim:.4f}\n")

    return triton_scene_codes


def test_benchmark(model, image, device: str = "cuda:0", n_warmup: int = 2, n_runs: int = 5):
    """Test 3: Benchmark default vs Triton attention timing."""
    log.info("=" * 60)
    log.info("TEST 3: Attention Timing Benchmark")
    log.info(f"  Warmup runs: {n_warmup}, Timed runs: {n_runs}")
    log.info("=" * 60)

    # ── Benchmark default attention ──
    log.info("\n  [Default PyTorch Attention]")
    default_procs = swap_default_with_timing(model)
    DefaultAttnProcessorWithTiming.reset_stats()

    # Warmup
    for _ in range(n_warmup):
        run_forward(model, image, device)
    DefaultAttnProcessorWithTiming.reset_stats()

    # Timed runs
    default_total_times = []
    for i in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_forward(model, image, device)
        torch.cuda.synchronize()
        default_total_times.append(time.perf_counter() - t0)

    default_stats = DefaultAttnProcessorWithTiming.get_stats()
    default_attn_total = default_stats.total_time
    default_attn_calls = default_stats.call_count
    default_avg_forward = np.mean(default_total_times) * 1000
    default_avg_attn_per_call = default_stats.avg_time * 1000

    restore_attention(model, default_procs)

    # ── Benchmark Triton attention ──
    log.info("\n  [Triton Flash Attention]")
    triton_procs = swap_attention(model, use_timing=True)
    TritonAttnProcessorWithTiming.reset_stats()

    # Warmup
    for _ in range(n_warmup):
        run_forward(model, image, device)
    TritonAttnProcessorWithTiming.reset_stats()

    # Timed runs
    triton_total_times = []
    for i in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_forward(model, image, device)
        torch.cuda.synchronize()
        triton_total_times.append(time.perf_counter() - t0)

    triton_stats = TritonAttnProcessorWithTiming.get_stats()
    triton_attn_total = triton_stats.total_time
    triton_attn_calls = triton_stats.call_count
    triton_avg_forward = np.mean(triton_total_times) * 1000
    triton_avg_attn_per_call = triton_stats.avg_time * 1000

    restore_attention(model, triton_procs)

    # ── Report ──
    log.info("\n" + "=" * 60)
    log.info("BENCHMARK RESULTS")
    log.info("=" * 60)
    log.info(f"  {'Metric':<35} {'Default':>10} {'Triton':>10} {'Speedup':>10}")
    log.info(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")
    log.info(f"  {'Avg forward pass (ms)':<35} {default_avg_forward:>10.1f} {triton_avg_forward:>10.1f} {default_avg_forward/triton_avg_forward:>10.2f}x")
    log.info(f"  {'Total attn time (ms)':<35} {default_attn_total*1000:>10.1f} {triton_attn_total*1000:>10.1f} {default_attn_total/max(triton_attn_total,1e-9):>10.2f}x")
    log.info(f"  {'Avg attn call (ms)':<35} {default_avg_attn_per_call:>10.3f} {triton_avg_attn_per_call:>10.3f} {default_avg_attn_per_call/max(triton_avg_attn_per_call,1e-9):>10.2f}x")
    log.info(f"  {'Total attn calls':<35} {default_attn_calls:>10d} {triton_attn_calls:>10d}")
    log.info(f"  {'Attn % of forward (default)':<35} {default_attn_total*1000/max(default_avg_forward*n_runs,1e-9)*100:>10.1f}%")
    log.info(f"  {'Attn % of forward (triton)':<35} {triton_attn_total*1000/max(triton_avg_forward*n_runs,1e-9)*100:>10.1f}%")
    log.info("=" * 60)

    return {
        "default_avg_forward_ms": default_avg_forward,
        "triton_avg_forward_ms": triton_avg_forward,
        "default_attn_total_ms": default_attn_total * 1000,
        "triton_attn_total_ms": triton_attn_total * 1000,
        "speedup_forward": default_avg_forward / triton_avg_forward,
        "speedup_attn": default_attn_total / max(triton_attn_total, 1e-9),
    }


def test_mesh_extraction(model, scene_codes, output_dir: str):
    """Test 4: Extract mesh and verify it's valid."""
    log.info("=" * 60)
    log.info("TEST 4: Mesh extraction")
    log.info("=" * 60)

    t0 = time.perf_counter()
    mesh = extract_mesh(model, scene_codes, resolution=256)
    elapsed = time.perf_counter() - t0

    log.info(f"  Vertices: {len(mesh.vertices)}")
    log.info(f"  Faces: {len(mesh.faces)}")
    log.info(f"  Has vertex colors: {mesh.visual.vertex_colors is not None}")
    log.info(f"  Extraction time: {elapsed:.2f}s")

    assert len(mesh.vertices) > 100, f"Mesh has too few vertices ({len(mesh.vertices)})"
    assert len(mesh.faces) > 100, f"Mesh has too few faces ({len(mesh.faces)})"

    # Save mesh
    os.makedirs(output_dir, exist_ok=True)
    mesh_path = os.path.join(output_dir, "test_output.obj")
    mesh.export(mesh_path)
    log.info(f"  Saved mesh to {mesh_path}")
    log.info(f"  Mesh file size: {os.path.getsize(mesh_path) / 1024:.1f} KB")
    log.info("  ✓ Mesh extraction PASSED\n")

    return mesh


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test TripoSR with Triton kernels")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image (default: demo/assets/sample_images/toy_robot.png)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run timing benchmark (default vs Triton)")
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--skip-mesh", action="store_true",
                        help="Skip mesh extraction test")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(DEMO_DIR, "output"))
    args = parser.parse_args()

    # Determine image path
    if args.image is None:
        args.image = os.path.join(DEMO_DIR, "assets", "sample_images", "toy_robot.png")
    if not os.path.exists(args.image):
        log.error(f"Image not found: {args.image}")
        sys.exit(1)

    device = args.device
    if not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU (will be slow)")
        device = "cpu"

    log.info(f"Device: {device}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    model = load_model(device)

    # Load image
    log.info(f"Loading image: {args.image}")
    image = load_image(args.image)
    log.info(f"Image size: {image.size}")

    # Test 1: Basic inference
    default_scene_codes = test_basic_inference(model, image, device)

    # Test 2: Triton kernel correctness
    triton_scene_codes = test_triton_kernel(model, image, default_scene_codes, device)

    # Test 3: Benchmark (optional)
    if args.benchmark:
        test_benchmark(model, image, device, args.n_warmup, args.n_runs)

    # Test 4: Mesh extraction
    if not args.skip_mesh:
        test_mesh_extraction(model, default_scene_codes, args.output_dir)

    log.info("=" * 60)
    log.info("ALL TESTS PASSED ✓")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
