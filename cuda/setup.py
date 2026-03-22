"""Build script for Flash Attention v3 CUDA kernel (H100 Hopper)."""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="flash_attn_v3_hopper",
    ext_modules=[
        CUDAExtension(
            name="flash_attn_v3_hopper",
            sources=["flash_attn_v3_hopper.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-arch=sm_90a",        # H100 Hopper (required for WGMMA/TMA)
                    "--use_fast_math",
                    "-lineinfo",           # For profiling
                    "--ptxas-options=-v",   # Show register/SMEM usage
                ],
            },
            libraries=["cuda"],            # Link libcuda for cuTensorMapEncodeTiled
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
