"""
modal_demo.py — Run the TripoSR + Triton Attention demo on Modal H100.

This script:
  1. Builds a Modal image with all TripoSR + Triton dependencies
  2. Runs the test_triposr.py verification script on H100
  3. Optionally runs the benchmark comparison
  4. Optionally launches the Gradio demo (future)

Usage:
    modal run demo/modal_demo.py                    # verify + test
    modal run demo/modal_demo.py --benchmark        # verify + benchmark
    modal run demo/modal_demo.py::test_triposr      # just verification
    modal run demo/modal_demo.py::benchmark_attn    # just benchmark
"""

import os
import modal

# Resolve project root (parent of demo/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

app = modal.App("triton-3d-demo")

# ── Modal image with all dependencies ───────────────────────────────────────
#
# Key considerations:
#   - TripoSR needs: omegaconf, einops, transformers, trimesh, rembg, huggingface-hub
#   - torchmcubes needs compilation from source (marching cubes for mesh extraction)
#   - My kernels need: triton
#   - We pre-download model weights into the image to avoid re-downloading each run
#

demo_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")  # for rembg/opencv deps
    .pip_install(
        # PyTorch + Triton
        "torch",
        "triton",
        # TripoSR dependencies
        "omegaconf==2.3.0",
        "einops==0.7.0",
        "transformers==4.35.0",
        "trimesh==4.0.5",
        "rembg[gpu]",
        "huggingface-hub",
        "imageio[ffmpeg]",
        "xatlas==0.0.9",
        "moderngl==5.10.0",
        "plyfile",
        "Pillow",
        "numpy",
        "gradio",
    )
    # Install CUDA toolkit (nvcc + headers) — needed by torchmcubes for compilation
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends wget gnupg2 software-properties-common",
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        # Install CUDA 12-6 (compatible with PyTorch's bundled CUDA 12.x runtime)
        "apt-get install -y --no-install-recommends cuda-toolkit-12-6",
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
    })
    # Install torchmcubes from source (needs nvcc for CUDA kernels)
    .pip_install("git+https://github.com/tatsy/torchmcubes.git")
    # Copy the project (PROJECT_ROOT resolves to absolute path)
    .add_local_dir(
        PROJECT_ROOT,
        "/root/triton-3d-kernels",
        copy=True,
        ignore=["__pycache__", ".git", "*.pyc", ".triton", "demo/output", "demo/triposr/.git"],
    )
    # Pre-download TripoSR model weights into the image (avoids download each run)
    .run_commands(
        "python -c \""
        "from huggingface_hub import hf_hub_download; "
        "hf_hub_download('stabilityai/TripoSR', 'config.yaml'); "
        "hf_hub_download('stabilityai/TripoSR', 'model.ckpt'); "
        "print('Model weights cached successfully')"
        "\"",
    )
)


@app.function(
    image=demo_image,
    gpu="H100",
    timeout=600,
    # TripoSR model is ~1.5GB, needs decent RAM
    memory=16384,
)
def test_triposr():
    """Verify TripoSR works with default + Triton attention on H100."""
    import subprocess
    import os

    os.chdir("/root/triton-3d-kernels")

    # GPU info
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    print(f"GPU: {result.stdout.strip()}")

    # Python/Triton versions
    result = subprocess.run(
        ["python", "-c",
         "import torch; import triton; "
         "print(f'PyTorch {torch.__version__}, Triton {triton.__version__}, CUDA {torch.version.cuda}')"],
        capture_output=True, text=True,
    )
    print(result.stdout.strip())

    print("\n" + "=" * 60)
    print("  TRIPOSR + TRITON ATTENTION TEST")
    print("=" * 60 + "\n")

    # Run test script (basic verification + correctness check, no benchmark)
    result = subprocess.run(
        ["python", "demo/test_triposr.py", "--skip-mesh"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        timeout=300,
    )
    print(result.stdout)

    if result.returncode != 0:
        print(f"\n*** TEST FAILED (exit code {result.returncode}) ***")
    else:
        print("\n*** TEST PASSED ***")

    return result.returncode


@app.function(
    image=demo_image,
    gpu="H100",
    timeout=900,
    memory=16384,
)
def benchmark_attn():
    """Run attention benchmark: default PyTorch vs my Triton kernel."""
    import subprocess
    import os

    os.chdir("/root/triton-3d-kernels")

    # GPU info
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    print(f"GPU: {result.stdout.strip()}")

    print("\n" + "=" * 60)
    print("  ATTENTION BENCHMARK: Default vs Triton")
    print("=" * 60 + "\n")

    # Run test script with benchmark flag
    result = subprocess.run(
        ["python", "demo/test_triposr.py", "--benchmark", "--n-warmup", "3", "--n-runs", "10"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        timeout=600,
    )
    print(result.stdout)

    if result.returncode != 0:
        print(f"\n*** BENCHMARK FAILED (exit code {result.returncode}) ***")
    else:
        print("\n*** BENCHMARK COMPLETE ***")

    return result.returncode


@app.function(
    image=demo_image,
    gpu="H100",
    timeout=600,
    memory=16384,
)
def full_pipeline():
    """Run full pipeline: test + mesh extraction + save output."""
    import subprocess
    import os

    os.chdir("/root/triton-3d-kernels")

    print("=" * 60)
    print("  FULL PIPELINE: Image → 3D Mesh")
    print("=" * 60 + "\n")

    # Run all tests including mesh extraction
    result = subprocess.run(
        ["python", "demo/test_triposr.py"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        timeout=300,
    )
    print(result.stdout)

    # Check if mesh was created
    mesh_path = "/root/triton-3d-kernels/demo/output/test_output.obj"
    if os.path.exists(mesh_path):
        size_kb = os.path.getsize(mesh_path) / 1024
        print(f"\n✓ Mesh generated: {mesh_path} ({size_kb:.1f} KB)")
    else:
        print("\n✗ No mesh file generated")

    return result.returncode




@app.function(
    image=demo_image,
    gpu="H100",
    timeout=3600,
    memory=16384,
)
@modal.concurrent(max_inputs=5)
@modal.web_server(port=7860, startup_timeout=120)
def launch_gradio():
    """Launch Gradio demo on Modal H100 — accessible via public URL."""
    import subprocess
    import os

    os.chdir("/root/triton-3d-kernels")

    # Launch gradio app
    subprocess.Popen(
        ["python", "demo/gradio_app.py"],
        env={**os.environ, "GRADIO_SERVER_NAME": "0.0.0.0", "GRADIO_SERVER_PORT": "7860"},
    )


@app.local_entrypoint()
def main(benchmark: bool = False):
    """Run TripoSR verification, optionally with benchmark."""
    print("Launching TripoSR + Triton demo on Modal H100...\n")

    # Step 1: Verify correctness
    rc = test_triposr.remote()
    if rc != 0:
        print("\n*** VERIFICATION FAILED — fix before continuing ***")
        return

    print("\n*** VERIFICATION PASSED ***\n")

    # Step 2: Benchmark (optional)
    if benchmark:
        print("Running attention benchmark...\n")
        benchmark_attn.remote()

    print("\nDone! ✓")
