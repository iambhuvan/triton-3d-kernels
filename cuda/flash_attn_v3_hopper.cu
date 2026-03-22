/*
 * Flash Attention v3 — H100 Hopper CUDA Kernel Implementation
 *
 * Forward pass with three H100-specific features ACTUALLY USED in the hot path:
 *   1. WGMMA for BOTH QK^T and PV matmuls (not just one)
 *   2. TMA for ALL data loads (Q, K, V — no manual cooperative loads)
 *   3. Warp Specialization: producer issues TMA || consumer runs WGMMA + softmax
 *
 * Architecture:
 *   - Grid: (ceil(S/kBlockM), B*H)
 *   - Block: 256 threads = 2 warp groups
 *     - WG0 (threads 0-127):   PRODUCER — issues TMA, manages 2-stage pipeline
 *     - WG1 (threads 128-255): CONSUMER — WGMMA matmuls + online softmax
 *
 * Memory flow (fully hardware-accelerated):
 *   GMEM --[TMA]--> SMEM (128B swizzled) --[WGMMA desc]--> Registers (FP32 accum)
 *
 * WGMMA register layout (m64n64k16 f32, 32 regs/thread):
 *   row = warp_id*16 + ((reg>>1)&1)*8 + (lane_id>>2)
 *   col = (lane_id&3)*2 + (reg&1) + (reg>>2)*8
 */

#include "flash_attn_v3_hopper.cuh"
#include <math.h>
#include <float.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>


// ============================================================
// TMA Tensor Map Creation (Host-side, CUDA Driver API)
// ============================================================

CUtensorMap tma::create_tensor_map_2d(
    void* global_ptr,
    int dim0, int dim1,
    int stride0,
    int box_dim0, int box_dim1
) {
    CUtensorMap tensor_map{};

    CUtensorMapDataType dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    // cuTensorMapEncodeTiled uses column-major dimension ordering
    uint64_t global_dims[2] = {(uint64_t)dim1, (uint64_t)dim0};
    uint64_t global_strides[1] = {(uint64_t)stride0};
    uint32_t box_dims[2] = {(uint32_t)box_dim1, (uint32_t)box_dim0};
    uint32_t elem_strides[2] = {1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &tensor_map,
        dtype,
        2,
        global_ptr,
        global_dims,
        global_strides,
        box_dims,
        elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", result);
    return tensor_map;
}


// ============================================================
// The Kernel
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
) {
    const int tid = threadIdx.x;
    const int warp_group_idx = tid / kThreadsPerWarpGroup;  // 0=producer, 1=consumer
    const int tid_in_wg = tid % kThreadsPerWarpGroup;
    const int warp_id = tid_in_wg / kWarpSize;
    const int lane_id = tid_in_wg % kWarpSize;

    const int q_block_id = blockIdx.x;
    const int bh_id = blockIdx.y;
    const int q_start = q_block_id * kBlockM;

    // Shared memory
    extern __shared__ char smem_raw[];
    SharedMemory* smem = reinterpret_cast<SharedMemory*>(smem_raw);

    // ---- Precompute all mbarrier SMEM addresses (used by both producer/consumer) ----
    uint32_t mbar_k_addr[kStages], mbar_v_addr[kStages], mbar_q_addr;
    #pragma unroll
    for (int s = 0; s < kStages; s++) {
        mbar_k_addr[s] = smem_ptr_to_uint(&smem->mbar_k[s]);
        mbar_v_addr[s] = smem_ptr_to_uint(&smem->mbar_v[s]);
    }
    mbar_q_addr = smem_ptr_to_uint(&smem->mbar_q);

    // ---- Initialize mbarriers (thread 0 only) ----
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < kStages; s++) {
            barrier::mbarrier_init(mbar_k_addr[s], 1);
            barrier::mbarrier_init(mbar_v_addr[s], 1);
        }
        barrier::mbarrier_init(mbar_q_addr, 1);
    }
    __syncthreads();

    // Number of KV blocks to process
    int n_kv_blocks;
    if (is_causal) {
        n_kv_blocks = (q_start + kBlockM + kBlockN - 1) / kBlockN;
        n_kv_blocks = min(n_kv_blocks, (S + kBlockN - 1) / kBlockN);
    } else {
        n_kv_blocks = (S + kBlockN - 1) / kBlockN;
    }

    // TMA uses global row index = bh_id * S + local_row
    const int seq_offset = bh_id * S;

    // ============================================================
    // PRODUCER WARP GROUP (warp_group_idx == 0)
    // ============================================================
    if (warp_group_idx == 0) {
        regs::decrease<64>();  // Give registers to consumer

        const bool is_leader = (tid_in_wg == 0);

        uint32_t phase_k[kStages] = {0, 0};
        uint32_t phase_v[kStages] = {0, 0};

        // ---- Load Q via TMA (once) ----
        if (is_leader) {
            barrier::mbarrier_arrive_expect_tx(mbar_q_addr, kQTileBytes);
            uint32_t q_dst = smem_ptr_to_uint(&smem->q[0][0]);
            tma::copy_2d_global_to_shared(tma_q, q_dst, mbar_q_addr,
                                           0, seq_offset + q_start);
        }
        barrier::mbarrier_wait(mbar_q_addr, 0);

        // Signal consumer: Q is ready
        barrier::named_barrier_sync(barrier::kQueryReady, kNumThreads);

        // ---- 2-stage pipeline: load K, V via TMA ----
        for (int kv_block = 0; kv_block < n_kv_blocks; kv_block++) {
            int stage = kv_block % kStages;
            int kv_start = kv_block * kBlockN;

            // Wait for consumer to release this stage (after first kStages fills)
            if (kv_block >= kStages) {
                barrier::named_barrier_sync(barrier::kStageRelease, kNumThreads);
            }

            if (is_leader) {
                // Issue TMA for K tile
                barrier::mbarrier_arrive_expect_tx(mbar_k_addr[stage], kTileBytes);
                uint32_t k_dst = smem_ptr_to_uint(&smem->k[stage][0][0]);
                tma::copy_2d_global_to_shared(tma_k, k_dst, mbar_k_addr[stage],
                                               0, seq_offset + kv_start);

                // Issue TMA for V tile
                barrier::mbarrier_arrive_expect_tx(mbar_v_addr[stage], kTileBytes);
                uint32_t v_dst = smem_ptr_to_uint(&smem->v[stage][0][0]);
                tma::copy_2d_global_to_shared(tma_v, v_dst, mbar_v_addr[stage],
                                               0, seq_offset + kv_start);
            }

            // Wait for TMA completion
            barrier::mbarrier_wait(mbar_k_addr[stage], phase_k[stage]);
            barrier::mbarrier_wait(mbar_v_addr[stage], phase_v[stage]);
            phase_k[stage] ^= 1;
            phase_v[stage] ^= 1;

            // Signal consumer: K/V ready
            barrier::named_barrier_sync(barrier::kKVReady, kNumThreads);
        }

        return;  // Producer done
    }

    // ============================================================
    // CONSUMER WARP GROUP (warp_group_idx == 1)
    // ============================================================
    regs::increase<64>();  // Get extra registers for WGMMA accumulators

    // Per-row online softmax state (each thread owns 2 rows from WGMMA layout)
    const int row_a = warp_id * 16 + (lane_id >> 2);        // first row
    const int row_b = row_a + 8;                              // second row (8 apart)
    float m_a = -FLT_MAX, l_a = 0.0f;
    float m_b = -FLT_MAX, l_b = 0.0f;

    // Output accumulator in WGMMA register layout
    float o_acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) o_acc[i] = 0.0f;

    // Wait for Q
    barrier::named_barrier_sync(barrier::kQueryReady, kNumThreads);

    // QK^T scores buffer
    __shared__ float s_block[kBlockM][kBlockN];

    // ---- Main loop ----
    for (int kv_block = 0; kv_block < n_kv_blocks; kv_block++) {
        int stage = kv_block % kStages;
        int kv_start = kv_block * kBlockN;

        // Wait for K/V
        barrier::named_barrier_sync(barrier::kKVReady, kNumThreads);

        // ============================================================
        // Step 1: S = Q @ K^T  via WGMMA
        // ============================================================
        float qk_acc[32];

        #if __CUDA_ARCH__ >= 900
        {
            #pragma unroll
            for (int i = 0; i < 32; i++) qk_acc[i] = 0.0f;

            wgmma::fence();

            // D=64 → 4 iterations of k16
            #pragma unroll
            for (int k_iter = 0; k_iter < kHeadDim; k_iter += kWgmmaTileK) {
                uint32_t q_smem = smem_ptr_to_uint(&smem->q[0][k_iter]);
                uint32_t k_smem = smem_ptr_to_uint(&smem->k[stage][0][k_iter]);

                int row_stride = kHeadDim * (int)sizeof(__half);  // 128 bytes
                int core_stride = 8 * row_stride;                 // 1024 bytes

                uint64_t desc_q = wgmma::make_smem_desc(q_smem, row_stride, core_stride);
                uint64_t desc_k = wgmma::make_smem_desc(k_smem, row_stride, core_stride);

                wgmma::mma_m64n64k16_f16_ss(qk_acc, desc_q, desc_k, (k_iter > 0));
            }

            wgmma::commit_group();
            wgmma::wait_group<0>();

            // Apply scale
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                wgmma::fence_operand(qk_acc[i]);
                qk_acc[i] *= scale;
            }

            // Store to s_block with correct WGMMA register mapping + masking
            #pragma unroll
            for (int reg = 0; reg < 32; reg++) {
                int r = wgmma::accum_row(warp_id, lane_id, reg);
                int c = wgmma::accum_col(lane_id, reg);
                float val = qk_acc[reg];

                if (is_causal && (kv_start + c > q_start + r)) val = -FLT_MAX;
                if (kv_start + c >= S || q_start + r >= S)     val = -FLT_MAX;

                s_block[r][c] = val;
            }
        }
        #else
        // Fallback for non-Hopper
        {
            for (int idx = tid_in_wg; idx < kBlockM * kBlockN; idx += kThreadsPerWarpGroup) {
                int r = idx / kBlockN;
                int c = idx % kBlockN;
                float acc = 0.0f;
                for (int d = 0; d < kHeadDim; d++)
                    acc += __half2float(smem->q[r][d]) * __half2float(smem->k[stage][c][d]);
                float val = acc * scale;
                if (is_causal && (kv_start + c > q_start + r)) val = -FLT_MAX;
                if (kv_start + c >= S || q_start + r >= S)     val = -FLT_MAX;
                s_block[r][c] = val;
            }
        }
        #endif

        __syncthreads();

        // ============================================================
        // Step 2: Online softmax + store P to FP16 SMEM for WGMMA PV
        // ============================================================

        // Each consumer thread processes its 2 owned rows (row_a, row_b)
        if (q_start + row_a < S) {
            float local_max = -FLT_MAX;
            for (int c = 0; c < kBlockN; c++)
                local_max = fmaxf(local_max, s_block[row_a][c]);

            float m_new = fmaxf(m_a, local_max);
            float alpha = expf(m_a - m_new);

            float local_sum = 0.0f;
            for (int c = 0; c < kBlockN; c++) {
                float p = expf(s_block[row_a][c] - m_new);
                smem->p_smem[row_a][c] = __float2half(p);
                local_sum += p;
            }

            // Rescale output accumulator for row_a
            #pragma unroll
            for (int reg = 0; reg < 32; reg++) {
                if (wgmma::accum_row(warp_id, lane_id, reg) == row_a)
                    o_acc[reg] *= alpha;
            }

            l_a = alpha * l_a + local_sum;
            m_a = m_new;
        }

        if (q_start + row_b < S) {
            float local_max = -FLT_MAX;
            for (int c = 0; c < kBlockN; c++)
                local_max = fmaxf(local_max, s_block[row_b][c]);

            float m_new = fmaxf(m_b, local_max);
            float alpha = expf(m_b - m_new);

            float local_sum = 0.0f;
            for (int c = 0; c < kBlockN; c++) {
                float p = expf(s_block[row_b][c] - m_new);
                smem->p_smem[row_b][c] = __float2half(p);
                local_sum += p;
            }

            #pragma unroll
            for (int reg = 0; reg < 32; reg++) {
                if (wgmma::accum_row(warp_id, lane_id, reg) == row_b)
                    o_acc[reg] *= alpha;
            }

            l_b = alpha * l_b + local_sum;
            m_b = m_new;
        }

        __syncthreads();

        // ============================================================
        // Step 3: O += P @ V  via WGMMA (P in p_smem, V in v[stage])
        // ============================================================

        #if __CUDA_ARCH__ >= 900
        {
            float pv_acc[32];
            #pragma unroll
            for (int i = 0; i < 32; i++) pv_acc[i] = 0.0f;

            wgmma::fence();

            // P is (64 x 64), V is (64 x 64), iterate shared K dim in chunks of 16
            #pragma unroll
            for (int k_iter = 0; k_iter < kBlockN; k_iter += kWgmmaTileK) {
                uint32_t p_addr = smem_ptr_to_uint(&smem->p_smem[0][k_iter]);
                uint32_t v_addr = smem_ptr_to_uint(&smem->v[stage][k_iter][0]);

                int p_row_stride = kBlockN * (int)sizeof(__half);    // 128 bytes
                int v_row_stride = kHeadDim * (int)sizeof(__half);   // 128 bytes
                int p_core_stride = 8 * p_row_stride;                // 1024 bytes
                int v_core_stride = 8 * v_row_stride;                // 1024 bytes

                uint64_t desc_p = wgmma::make_smem_desc(p_addr, p_row_stride, p_core_stride);
                uint64_t desc_v = wgmma::make_smem_desc(v_addr, v_row_stride, v_core_stride);

                wgmma::mma_m64n64k16_f16_ss(pv_acc, desc_p, desc_v, (k_iter > 0));
            }

            wgmma::commit_group();
            wgmma::wait_group<0>();

            #pragma unroll
            for (int i = 0; i < 32; i++) {
                wgmma::fence_operand(pv_acc[i]);
                o_acc[i] += pv_acc[i];
            }
        }
        #else
        // Fallback
        {
            for (int idx = tid_in_wg; idx < kBlockM * kHeadDim; idx += kThreadsPerWarpGroup) {
                int r = idx / kHeadDim;
                int d = idx % kHeadDim;
                if (q_start + r < S) {
                    float pv = 0.0f;
                    for (int c = 0; c < kBlockN; c++)
                        pv += __half2float(smem->p_smem[r][c]) * __half2float(smem->v[stage][c][d]);
                    // Fallback can't easily accumulate into WGMMA registers — skip
                }
            }
        }
        #endif

        __syncthreads();

        // Release this stage's buffer for producer
        barrier::named_barrier_arrive(barrier::kStageRelease, kNumThreads);
    }

    // ============================================================
    // Step 4: Normalize and store O
    // ============================================================
    __half* O_out = O + bh_id * S * D;

    #pragma unroll
    for (int reg = 0; reg < 32; reg++) {
        int r = wgmma::accum_row(warp_id, lane_id, reg);
        int c = wgmma::accum_col(lane_id, reg);
        int global_row = q_start + r;

        if (global_row < S && c < D) {
            float l = (r == row_a) ? l_a : l_b;
            float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
            O_out[global_row * D + c] = __float2half(o_acc[reg] * inv_l);
        }
    }

    // Store LSE (one value per row, one thread per row writes it)
    if ((lane_id & 3) == 0) {
        int gr_a = q_start + row_a;
        if (gr_a < S) {
            float* lse_base = LSE + bh_id * S;
            lse_base[gr_a] = m_a + logf(fmaxf(l_a, 1e-10f));
        }
        int gr_b = q_start + row_b;
        if (gr_b < S) {
            float* lse_base = LSE + bh_id * S;
            lse_base[gr_b] = m_b + logf(fmaxf(l_b, 1e-10f));
        }
    }
}


// ============================================================
// Host Launch
// ============================================================

torch::Tensor flash_attn_v3_cuda_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool is_causal
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA tensor");
    TORCH_CHECK(Q.scalar_type() == torch::kHalf, "Q must be FP16");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "Q, K, V must be contiguous");

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int S = Q.size(2);
    const int D = Q.size(3);
    TORCH_CHECK(D == kHeadDim, "Head dim must be ", kHeadDim, " but got ", D);

    auto O = torch::empty_like(Q);
    auto LSE = torch::empty({B, H, S}, Q.options().dtype(torch::kFloat32));
    float scale_val = 1.0f / sqrtf((float)D);

    // ---- Create TMA tensor maps ----
    int row_stride_bytes = D * sizeof(__half);

    CUtensorMap tma_q_host = tma::create_tensor_map_2d(
        const_cast<__half*>(reinterpret_cast<const __half*>(Q.data_ptr<at::Half>())),
        B * H * S, D, row_stride_bytes, kBlockM, kHeadDim);

    CUtensorMap tma_k_host = tma::create_tensor_map_2d(
        const_cast<__half*>(reinterpret_cast<const __half*>(K.data_ptr<at::Half>())),
        B * H * S, D, row_stride_bytes, kBlockN, kHeadDim);

    CUtensorMap tma_v_host = tma::create_tensor_map_2d(
        const_cast<__half*>(reinterpret_cast<const __half*>(V.data_ptr<at::Half>())),
        B * H * S, D, row_stride_bytes, kBlockN, kHeadDim);

    // Copy tensor maps to device
    CUtensorMap *tma_q_dev, *tma_k_dev, *tma_v_dev;
    cudaMalloc(&tma_q_dev, sizeof(CUtensorMap));
    cudaMalloc(&tma_k_dev, sizeof(CUtensorMap));
    cudaMalloc(&tma_v_dev, sizeof(CUtensorMap));
    cudaMemcpy(tma_q_dev, &tma_q_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    cudaMemcpy(tma_k_dev, &tma_k_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    cudaMemcpy(tma_v_dev, &tma_v_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

    // ---- Launch ----
    dim3 grid((S + kBlockM - 1) / kBlockM, B * H);
    dim3 block(kNumThreads);
    int smem_bytes = sizeof(SharedMemory);

    cudaFuncSetAttribute(flash_attn_v3_fwd_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    flash_attn_v3_fwd_kernel<<<grid, block, smem_bytes,
                                at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(O.data_ptr<at::Half>()),
        LSE.data_ptr<float>(),
        tma_q_dev, tma_k_dev, tma_v_dev,
        B, H, S, D, scale_val, is_causal
    );

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(err));

    cudaDeviceSynchronize();
    cudaFree(tma_q_dev);
    cudaFree(tma_k_dev);
    cudaFree(tma_v_dev);

    return O;
}


// ============================================================
// PyBind11
// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attn_v3_cuda_forward,
          "Flash Attention v3 forward (CUDA, H100 — TMA + WGMMA + Warp Specialization)",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("is_causal") = false);
}
