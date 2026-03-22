/*
 * Flash Attention v3 — H100 Hopper CUDA Kernel
 *
 * This header provides inline PTX wrappers for three H100-specific features:
 *   1. WGMMA  — Warp Group Matrix Multiply Accumulate (4 warps = 128 threads)
 *   2. TMA    — Tensor Memory Accelerator (async bulk GMEM→SMEM copies)
 *   3. Named Barriers + Warp Specialization (producer/consumer warp groups)
 *
 * These are hardware features exposed only through PTX, not available in Triton.
 *
 * Register Layout Reference (WGMMA m64n64k16 f32):
 *   row = warp_id*16 + ((reg>>1)&1)*8 + (lane_id>>2)
 *   col = (lane_id&3)*2 + (reg&1) + (reg>>2)*8
 *   Source: CUTLASS CuTe mma_traits_sm90_gmma.hpp, CLayout_64xN
 *
 * References:
 *   - Shah et al. "FlashAttention-3: Fast and Accurate Attention with Asynchrony
 *     and Low-precision" (2024)
 *   - NVIDIA PTX ISA 8.3+ (sm_90a)
 *   - Dao-AILab/flash-attention hopper/ directory
 *   - CUTLASS 3.x SM90 GMMA traits
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// ============================================================
// Compile-time constants
// ============================================================

constexpr int kBlockM = 64;   // Q tile rows
constexpr int kBlockN = 64;   // KV tile rows
constexpr int kHeadDim = 64;  // Head dimension
constexpr int kStages = 2;    // Pipeline depth

constexpr int kWarpSize = 32;
constexpr int kWarpsPerWarpGroup = 4;
constexpr int kThreadsPerWarpGroup = kWarpSize * kWarpsPerWarpGroup; // 128

constexpr int kNumWarpGroups = 2;  // 1 producer + 1 consumer
constexpr int kNumThreads = kNumWarpGroups * kThreadsPerWarpGroup; // 256

// WGMMA tile shape
constexpr int kWgmmaTileK = 16;

// Bytes per KV tile (for TMA transaction tracking)
constexpr int kTileBytes = kBlockN * kHeadDim * sizeof(__half);  // 64*64*2 = 8192
constexpr int kQTileBytes = kBlockM * kHeadDim * sizeof(__half); // 64*64*2 = 8192


// ============================================================
// Helper: Get SMEM address as uint32_t (for PTX operands)
// ============================================================
// All PTX instructions with .shared:: state space require 32-bit SMEM addresses.
// This converts a generic C++ pointer to shared memory into the correct format.

__device__ __forceinline__ uint32_t smem_ptr_to_uint(const void* ptr) {
    uint32_t addr;
    asm volatile(
        "{\n"
        ".reg .u64 u64addr;\n"
        "cvta.to.shared.u64 u64addr, %1;\n"
        "cvt.u32.u64 %0, u64addr;\n"
        "}\n"
        : "=r"(addr)
        : "l"(ptr)
    );
    return addr;
}


// ============================================================
// 1. WGMMA — Warp Group Matrix Multiply Accumulate
// ============================================================

namespace wgmma {

__device__ __forceinline__ void fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void wait_group() {
    static_assert(N >= 0 && N <= 7, "wait_group N must be 0..7");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void fence_operand(float& reg) {
    asm volatile("" : "+f"(reg) :: "memory");
}

// Build a 64-bit SMEM descriptor for WGMMA
//
// Descriptor layout (64 bits) per PTX ISA 8.3:
//   [13:0]   = start_address >> 4  (16B-aligned SMEM address)
//   [29:16]  = leading_dim_byte_offset >> 4  (row stride, but ignored with swizzle)
//   [45:32]  = stride_byte_offset >> 4  (stride between core matrices = 8 rows * row_stride)
//   [63:62]  = swizzle_mode: 0=none, 1=128B, 2=64B, 3=32B
__device__ __forceinline__ uint64_t make_smem_desc(uint32_t smem_addr,
                                                     int leading_dim_bytes,
                                                     int stride_bytes) {
    uint64_t desc = 0;
    desc |= (uint64_t)((smem_addr & 0x3FFFF) >> 4) & 0x3FFF;          // [13:0]
    desc |= ((uint64_t)(leading_dim_bytes >> 4) & 0x3FFF) << 16;       // [29:16]
    desc |= ((uint64_t)(stride_bytes >> 4) & 0x3FFF) << 32;            // [45:32]
    desc |= (uint64_t)(1) << 62;  // swizzle mode 1 = 128B swizzle
    return desc;
}

// WGMMA m64n64k16 FP16 x FP16 -> FP32 (SS mode: both operands in SMEM)
// 32 float accumulator registers per thread.
__device__ __forceinline__ void mma_m64n64k16_f16_ss(
    float d[32],
    uint64_t desc_a,
    uint64_t desc_b,
    bool accumulate
) {
    uint32_t scale_D = accumulate ? 1 : 0;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, p, 1, 1, 0, 1;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(desc_a), "l"(desc_b), "r"(scale_D)
        : "memory"
    );
}

// ---- WGMMA register ↔ matrix coordinate helpers ----
// CUTLASS CLayout_64x64 mapping (verified against all 4096 elements):
//   row = warp_id*16 + ((reg>>1)&1)*8 + (lane_id>>2)
//   col = (lane_id&3)*2 + (reg&1) + (reg>>2)*8
__device__ __forceinline__ int accum_row(int warp_id, int lane_id, int reg) {
    return warp_id * 16 + ((reg >> 1) & 1) * 8 + (lane_id >> 2);
}
__device__ __forceinline__ int accum_col(int lane_id, int reg) {
    return (lane_id & 3) * 2 + (reg & 1) + (reg >> 2) * 8;
}

}  // namespace wgmma


// ============================================================
// 2. TMA — Tensor Memory Accelerator
// ============================================================

namespace tma {

// Host-side: create TMA tensor map descriptor via CUDA driver API
CUtensorMap create_tensor_map_2d(
    void* global_ptr,
    int dim0, int dim1,
    int stride0,
    int box_dim0, int box_dim1
);

// Device: issue TMA copy GMEM → SMEM (single thread calls this)
// All addresses are 32-bit SMEM addresses obtained via smem_ptr_to_uint()
__device__ __forceinline__ void copy_2d_global_to_shared(
    const CUtensorMap* tensor_map,
    uint32_t smem_dst,          // SMEM destination (uint32_t from smem_ptr_to_uint)
    uint32_t mbar_smem,         // mbarrier SMEM address (uint32_t)
    int coord0, int coord1
) {
    uint64_t tmap = reinterpret_cast<uint64_t>(tensor_map);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4}], [%2];\n"
        :
        : "r"(smem_dst), "l"(tmap), "r"(mbar_smem), "r"(coord0), "r"(coord1)
        : "memory"
    );
}

}  // namespace tma


// ============================================================
// 3. mbarrier + Named Barriers
// ============================================================
// mbarrier: hardware barrier for TMA completion tracking
// Named barriers: producer/consumer warp group synchronization

namespace barrier {

// All mbarrier operations use 32-bit SMEM addresses (from smem_ptr_to_uint)

__device__ __forceinline__ void mbarrier_init(uint32_t mbar_smem, uint32_t arrival_count) {
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :
        : "r"(mbar_smem), "r"(arrival_count)
        : "memory"
    );
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint32_t mbar_smem, uint32_t tx_bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_smem), "r"(tx_bytes)
        : "memory"
    );
}

__device__ __forceinline__ bool mbarrier_try_wait(uint32_t mbar_smem, uint32_t phase) {
    uint32_t ready;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.u32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(ready)
        : "r"(mbar_smem), "r"(phase)
        : "memory"
    );
    return ready != 0;
}

__device__ __forceinline__ void mbarrier_wait(uint32_t mbar_smem, uint32_t phase) {
    while (!mbarrier_try_wait(mbar_smem, phase)) {
        asm volatile("nanosleep.u32 64;\n" ::: "memory");
    }
}

// Named barriers (bar.arrive / bar.sync)
__device__ __forceinline__ void named_barrier_arrive(int barrier_id, int num_threads) {
    asm volatile("bar.arrive %0, %1;\n" :: "r"(barrier_id), "r"(num_threads) : "memory");
}
__device__ __forceinline__ void named_barrier_sync(int barrier_id, int num_threads) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(barrier_id), "r"(num_threads) : "memory");
}

// Elect one thread from a warp (returns true for lane 0)
__device__ __forceinline__ bool elect_one() {
    uint32_t pred;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, 0xFFFFFFFF;\n"
        "selp.u32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(pred) : :
    );
    return pred != 0;
}

enum BarrierID : int {
    kQueryReady   = 0,
    kKVReady      = 1,
    kStageRelease = 2,
};

}  // namespace barrier


// ============================================================
// 4. Shared Memory Layout
// ============================================================

struct __align__(128) SharedMemory {
    __half q[kBlockM][kHeadDim];              // 8 KB
    __half k[kStages][kBlockN][kHeadDim];     // 16 KB
    __half v[kStages][kBlockN][kHeadDim];     // 16 KB
    __half p_smem[kBlockM][kBlockN];          // 8 KB  (softmax P in FP16 for WGMMA PV)
    uint64_t mbar_k[kStages];
    uint64_t mbar_v[kStages];
    uint64_t mbar_q;
    // Total: ~48 KB
};


// ============================================================
// 5. Register reallocation (setmaxnreg)
// ============================================================

namespace regs {
template <int N>
__device__ __forceinline__ void increase() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(N));
}
template <int N>
__device__ __forceinline__ void decrease() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(N));
}
}  // namespace regs


// ============================================================
// Forward declaration
// ============================================================

__global__ void __launch_bounds__(kNumThreads)
flash_attn_v3_fwd_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    float* __restrict__ LSE,
    const CUtensorMap* __restrict__ tma_q,
    const CUtensorMap* __restrict__ tma_k,
    const CUtensorMap* __restrict__ tma_v,
    int B, int H, int S, int D,
    float scale,
    bool is_causal
);
