"""
K4: Batched Chamfer Distance — Reference Implementation

Chamfer distance is THE standard metric for comparing two point clouds in 3D.
It measures how close two sets of points are by finding, for each point in
set A, its nearest neighbor in set B, and vice versa.

Math:
  Given point clouds P1 of shape (B, N, 3) and P2 of shape (B, M, 3):

  CD(P1, P2) = (1/N) * sum_{p in P1} min_{q in P2} ||p - q||^2
             + (1/M) * sum_{q in P2} min_{p in P1} ||q - p||^2

  The first term: for each point in P1, find nearest point in P2
  The second term: for each point in P2, find nearest point in P1

Why this kernel matters:
  Chamfer distance requires computing an (N, M) pairwise distance matrix per
  batch element. For point clouds with N=M=8192 (common in 3D generation),
  this is a 67M element matrix PER SAMPLE. The naive implementation
  materializes this entire matrix in HBM.

  A tiled Triton kernel can:
    1. Compute distance tiles in SRAM
    2. Take the min within each tile
    3. Reduce across tiles — never materializing the full N×M matrix
    4. Achieve 40x+ speedup over naive PyTorch
"""

import torch


def chamfer_distance_reference(
    pc1: torch.Tensor,
    pc2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute batched chamfer distance between two point clouds.

    Args:
        pc1: (B, N, 3) first point cloud
        pc2: (B, M, 3) second point cloud

    Returns:
        loss: (B,) chamfer distance per batch element
        dist1: (B, N) min distance from each point in pc1 to pc2
        dist2: (B, M) min distance from each point in pc2 to pc1
    """
    # Pairwise squared distances: (B, N, M)
    # ||p - q||^2 = ||p||^2 + ||q||^2 - 2 * p @ q^T
    diff = pc1.unsqueeze(2) - pc2.unsqueeze(1)  # (B, N, M, 3)
    dist_matrix = (diff ** 2).sum(dim=-1)  # (B, N, M)

    # For each point in pc1, find min distance to pc2
    dist1, idx1 = dist_matrix.min(dim=2)  # (B, N), (B, N)

    # For each point in pc2, find min distance to pc1
    dist2, idx2 = dist_matrix.min(dim=1)  # (B, M), (B, M)

    # Chamfer distance = mean of both directions
    loss = dist1.mean(dim=1) + dist2.mean(dim=1)  # (B,)

    return loss, dist1, dist2
