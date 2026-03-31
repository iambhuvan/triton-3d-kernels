"""Run triton-3d-kernels tests and benchmarks on Modal H100."""

import modal

app = modal.App("triton-3d-kernels")

triton_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "triton", "pytest")
    .add_local_dir(".", "/root/triton-3d-kernels", copy=True,
                   ignore=["__pycache__", ".git", "*.pyc", ".triton"])
)


@app.function(
    image=triton_image,
    gpu="H100",
    timeout=600,
)
def run_tests():
    """Run Triton-based tests on H100."""
    import subprocess
    import os

    os.chdir("/root/triton-3d-kernels")

    # GPU info
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True)
    print(f"GPU: {result.stdout.strip()}")

    # Python/Triton versions
    result = subprocess.run(["python", "-c",
        "import torch; import triton; "
        "print(f'PyTorch {torch.__version__}, Triton {triton.__version__}, CUDA {torch.version.cuda}')"
    ], capture_output=True, text=True)
    print(result.stdout.strip())

    print("\n" + "="*60)
    print("  RUNNING TESTS")
    print("="*60 + "\n")

    result = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])

    return result.returncode


@app.function(
    image=triton_image,
    gpu="H100",
    timeout=600,
)
def run_benchmarks():
    """Run benchmarks on H100."""
    import subprocess
    import os

    os.chdir("/root/triton-3d-kernels")

    print("="*60)
    print("  BENCHMARKS ON H100")
    print("="*60)

    result = subprocess.run(
        ["python", "bench/bench_flash_attn_versions.py"],
        capture_output=True, text=True,
        timeout=300,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])

    return result.returncode


@app.local_entrypoint()
def main():
    """Run tests first, then benchmarks if tests pass."""
    print("Launching on Modal H100...\n")

    rc = run_tests.remote()
    if rc != 0:
        print("\n*** TESTS FAILED ***")
        return

    print("\n*** ALL TESTS PASSED ***\n")

    run_benchmarks.remote()
