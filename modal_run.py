"""Run triton-3d-kernels tests and benchmarks on Modal H100."""

import modal

app = modal.App("triton-3d-kernels")

# Base image with PyTorch + Triton (no CUDA compiler needed for Triton tests)
triton_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "triton", "pytest")
    .add_local_dir(".", "/root/triton-3d-kernels", copy=True,
                   ignore=["__pycache__", ".git", "*.pyc", ".triton"])
)

# Image with full CUDA toolkit for compiling .cu files (nvcc, cuda.h, driver API headers).
# PyTorch ships CUDA runtime libs but NOT nvcc or driver API headers
# (cuda.h with CUtensorMap, cuTensorMapEncodeTiled etc).
# We install the full CUDA toolkit via apt which gives us nvcc, headers, and stubs.
cuda_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "triton", "pytest", "ninja")
    .run_commands(
        # Add NVIDIA package repo and install CUDA toolkit (nvcc + headers + stubs)
        "apt-get update && apt-get install -y --no-install-recommends wget gnupg2 software-properties-common",
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        # Install just the compiler toolkit (not the full driver) — includes nvcc, cuda.h, stubs
        "apt-get install -y --no-install-recommends cuda-toolkit-12-4",
        # Set CUDA_HOME so torch.utils.cpp_extension can find nvcc and headers
        "echo 'export CUDA_HOME=/usr/local/cuda' >> /etc/profile.d/cuda.sh",
        "echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /etc/profile.d/cuda.sh",
        # Verify nvcc works
        "export PATH=/usr/local/cuda/bin:$PATH && nvcc --version",
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
    })
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

    # Run Triton tests (exclude CUDA v3 which needs separate compilation)
    print("\n" + "="*60)
    print("  RUNNING TRITON TESTS")
    print("="*60 + "\n")

    result = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-v", "--tb=short",
         "--ignore=tests/test_flash_attn_v3_cuda.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])

    return result.returncode


@app.function(
    image=cuda_image,
    gpu="H100",
    timeout=1800,
)
def run_cuda_v3_tests():
    """Compile and test the CUDA C++ Flash Attention v3 kernel on H100."""
    import subprocess
    import os

    os.chdir("/root/triton-3d-kernels")

    # GPU info
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True)
    print(f"GPU: {result.stdout.strip()}")

    # Ensure CUDA_HOME is set for torch.utils.cpp_extension
    if not os.environ.get("CUDA_HOME"):
        for candidate in ["/usr/local/cuda", "/usr/local/cuda-12"]:
            if os.path.isdir(candidate):
                os.environ["CUDA_HOME"] = candidate
                break

    cuda_home = os.environ.get("CUDA_HOME", "")
    if cuda_home:
        os.environ["PATH"] = os.path.join(cuda_home, "bin") + ":" + os.environ.get("PATH", "")

    # Check nvcc is available
    result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: nvcc not found!")
        print(f"  CUDA_HOME={cuda_home}")
        print(f"  PATH={os.environ.get('PATH', '')}")
        print(f"  stderr: {result.stderr}")
        return 1
    else:
        print(f"nvcc: {result.stdout.strip().splitlines()[-1]}")
        print(f"CUDA_HOME: {cuda_home}")

    print("\n" + "="*60)
    print("  COMPILING & TESTING CUDA V3 FLASH ATTENTION")
    print("="*60 + "\n")

    # Run the CUDA v3 tests — JIT compilation happens on first import
    # Use combined stdout+stderr so we see nvcc output and errors interleaved
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/test_flash_attn_v3_cuda.py",
         "-v", "--tb=long", "-x", "-s"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        timeout=1500,
    )
    print(result.stdout[-10000:])

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
        ["python", "bench/bench_all.py"],
        capture_output=True, text=True,
        timeout=300,
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
def run_flash_attn_versions():
    """Run Flash Attention version comparison benchmark."""
    import subprocess
    import os

    os.chdir("/root/triton-3d-kernels")

    print("="*60)
    print("  FLASH ATTENTION VERSION COMPARISON (H100)")
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

    # Run Triton tests
    rc = run_tests.remote()
    if rc != 0:
        print("\n*** TRITON TESTS FAILED — fix before continuing ***")
        return

    print("\n*** TRITON TESTS PASSED ***\n")

    # Compile and run CUDA v3 kernel tests
    print("Compiling and testing CUDA v3 Flash Attention kernel...\n")
    try:
        rc = run_cuda_v3_tests.remote()
        if rc != 0:
            print("\n*** CUDA V3 TESTS FAILED ***")
            print("(Triton tests passed — CUDA kernel needs debugging)")
        else:
            print("\n*** ALL TESTS PASSED (Triton + CUDA v3) ***\n")
    except Exception as e:
        print(f"\n*** CUDA V3 TESTS ERROR: {e} ***")
        print("(Continuing to benchmarks...)")

    # Run benchmarks
    run_benchmarks.remote()
    run_flash_attn_versions.remote()
