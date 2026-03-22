"""
K4: Batched Chamfer Distance — Triton Kernel

Architecture:
  The key insight: never materialize the full (N, M) distance matrix.
  Instead, tile over blocks and maintain a running minimum.

  Kernel 1 — dist1 (pc1 -> pc2):
    Grid: (B, cdiv(N, BLOCK_N))
    Each program handles BLOCK_N points from pc1:
      1. Load pc1_tile: (BLOCK_N, 3) — stays in registers
      2. Initialize min_dist = +inf: (BLOCK_N,)
      3. For each pc2 block (inner loop):
         a. Load pc2_tile: (BLOCK_M, 3)
         b. Compute pairwise squared distances: (BLOCK_N, BLOCK_M)
            For each (i, j): dist = sum_d (pc1[i,d] - pc2[j,d])^2
         c. Take min across M dimension: tile_min = dist.min(dim=1)
         d. Update running min: min_dist = min(min_dist, tile_min)
      4. Store min_dist: (BLOCK_N,)

  Kernel 2 — dist2 (pc2 -> pc1): symmetric, swap roles.

  Memory: O(BLOCK_N * BLOCK_M) per program instead of O(N * M) total.
  For N=M=8192, BLOCK=128: 64KB in SRAM vs 256MB in HBM.
"""

import torch
import triton
import triton.language as tl


# ============================================================
# Triton Kernel: min distances from pc1 -> pc2
# ============================================================

@triton.jit
def _chamfer_dist_kernel(
    # Pointers — source is what we iterate FOR, target is what we search IN
    SRC_ptr, TGT_ptr, DIST_ptr,
    # Dimensions
    N_src, N_tgt,
    # Strides for source: (B, N_src, 3)
    stride_sb, stride_sn, stride_sd,
    # Strides for target: (B, N_tgt, 3)
    stride_tb, stride_tn, stride_td,
    # Stride for output: (B, N_src)
    stride_db, stride_dn,
    # Block sizes
    BLOCK_SRC: tl.constexpr,
    BLOCK_TGT: tl.constexpr,
):
    """Compute min squared distance from each source point to nearest target point.

    Grid: (B, cdiv(N_src, BLOCK_SRC))

    For each source block:
      - Load BLOCK_SRC points (each is 3D)
      - Iterate over ALL target blocks
      - For each target block, compute (BLOCK_SRC, BLOCK_TGT) distances
      - Track running minimum per source point
    """
    # Map program IDs
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)

    # Source point indices for this program
    src_offsets = block_id * BLOCK_SRC + tl.arange(0, BLOCK_SRC)
    src_mask = src_offsets < N_src

    # Load source points: (BLOCK_SRC, 3)
    # Base pointer for this batch
    src_base = SRC_ptr + batch_id * stride_sb

    # Load x, y, z coordinates of source points
    src_x = tl.load(src_base + src_offsets * stride_sn + 0 * stride_sd, mask=src_mask, other=0.0)
    src_y = tl.load(src_base + src_offsets * stride_sn + 1 * stride_sd, mask=src_mask, other=0.0)
    src_z = tl.load(src_base + src_offsets * stride_sn + 2 * stride_sd, mask=src_mask, other=0.0)

    # Initialize running minimum distances to +inf
    min_dist = tl.full([BLOCK_SRC], value=float('inf'), dtype=tl.float32)

    # Iterate over all target blocks
    tgt_base = TGT_ptr + batch_id * stride_tb
    n_tgt_blocks = tl.cdiv(N_tgt, BLOCK_TGT)

    for tgt_block_id in range(0, n_tgt_blocks):
        # Target point indices
        tgt_offsets = tgt_block_id * BLOCK_TGT + tl.arange(0, BLOCK_TGT)
        tgt_mask = tgt_offsets < N_tgt

        # Load target points: (BLOCK_TGT,) per coordinate
        tgt_x = tl.load(tgt_base + tgt_offsets * stride_tn + 0 * stride_td, mask=tgt_mask, other=0.0)
        tgt_y = tl.load(tgt_base + tgt_offsets * stride_tn + 1 * stride_td, mask=tgt_mask, other=0.0)
        tgt_z = tl.load(tgt_base + tgt_offsets * stride_tn + 2 * stride_td, mask=tgt_mask, other=0.0)

        # Compute pairwise squared distances: (BLOCK_SRC, BLOCK_TGT)
        # For each source point i, target point j:
        #   dist[i,j] = (sx[i] - tx[j])^2 + (sy[i] - ty[j])^2 + (sz[i] - tz[j])^2
        #
        # src_x is (BLOCK_SRC,), tgt_x is (BLOCK_TGT,)
        # We broadcast: src_x[:, None] - tgt_x[None, :] -> (BLOCK_SRC, BLOCK_TGT)
        dx = src_x[:, None] - tgt_x[None, :]  # (BLOCK_SRC, BLOCK_TGT)
        dy = src_y[:, None] - tgt_y[None, :]
        dz = src_z[:, None] - tgt_z[None, :]
        dist_sq = dx * dx + dy * dy + dz * dz  # (BLOCK_SRC, BLOCK_TGT)

        # Mask out-of-bounds target points with +inf
        dist_sq = tl.where(tgt_mask[None, :], dist_sq, float('inf'))

        # Min across target dimension: (BLOCK_SRC,)
        block_min = tl.min(dist_sq, axis=1)

        # Update running minimum
        min_dist = tl.minimum(min_dist, block_min)

    # Store results
    out_base = DIST_ptr + batch_id * stride_db
    tl.store(out_base + src_offsets * stride_dn, min_dist, mask=src_mask)


# ============================================================
# Python wrapper
# ============================================================

def batched_chamfer_distance(
    pc1: torch.Tensor,
    pc2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute batched chamfer distance.

    Args:
        pc1: (B, N, 3) first point cloud
        pc2: (B, M, 3) second point cloud

    Returns:
        loss: (B,) chamfer distance per batch element
        dist1: (B, N) min squared distance from each pc1 point to pc2
        dist2: (B, M) min squared distance from each pc2 point to pc1
    """
    assert pc1.is_contiguous() and pc2.is_contiguous()
    assert pc1.shape[-1] == 3 and pc2.shape[-1] == 3
    B, N, _ = pc1.shape
    _, M, _ = pc2.shape

    dist1 = torch.empty(B, N, device=pc1.device, dtype=pc1.dtype)
    dist2 = torch.empty(B, M, device=pc1.device, dtype=pc1.dtype)

    BLOCK_SRC = 128
    BLOCK_TGT = 128

    # Kernel 1: dist1 — for each point in pc1, find nearest in pc2
    grid1 = (B, triton.cdiv(N, BLOCK_SRC))
    _chamfer_dist_kernel[grid1](
        pc1, pc2, dist1,
        N, M,
        pc1.stride(0), pc1.stride(1), pc1.stride(2),
        pc2.stride(0), pc2.stride(1), pc2.stride(2),
        dist1.stride(0), dist1.stride(1),
        BLOCK_SRC=BLOCK_SRC,
        BLOCK_TGT=BLOCK_TGT,
    )

    # Kernel 2: dist2 — for each point in pc2, find nearest in pc1 (swap roles)
    grid2 = (B, triton.cdiv(M, BLOCK_SRC))
    _chamfer_dist_kernel[grid2](
        pc2, pc1, dist2,
        M, N,
        pc2.stride(0), pc2.stride(1), pc2.stride(2),
        pc1.stride(0), pc1.stride(1), pc1.stride(2),
        dist2.stride(0), dist2.stride(1),
        BLOCK_SRC=BLOCK_SRC,
        BLOCK_TGT=BLOCK_TGT,
    )

    # Chamfer distance = mean(dist1) + mean(dist2) per batch
    loss = dist1.mean(dim=1) + dist2.mean(dim=1)

    return loss, dist1, dist2
