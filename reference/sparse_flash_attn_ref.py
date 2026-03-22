"""
K3: KNN-Sparse Flash Attention for 3D — Reference Implementation

In 3D generative models, each token represents a point/patch in 3D space.
Unlike language tokens which attend to all other tokens, 3D tokens have
spatial locality — a point only needs to attend to its spatial neighbors.

Key insight: We can exploit this by building a KNN graph over 3D coordinates
and masking attention to only the K nearest neighbors per query token.
This reduces effective attention from O(S^2) to O(S*K) where K << S.

Math:
  Given 3D coordinates (B, S, 3) for each token:
    1. Compute pairwise distances or use a KNN algorithm
    2. For each query token, find its K nearest neighbors
    3. Build a sparse attention mask: mask[i,j] = 1 if j is in KNN(i)
    4. Apply standard attention but only over unmasked positions

  S_sparse[i,j] = (Q[i] @ K[j]^T / sqrt(d))  if j in KNN(i), else -inf
  P = softmax(S_sparse, dim=-1)
  O = P @ V

Why this kernel matters:
  3D models process thousands of spatial tokens. Full O(S^2) attention is
  wasteful because distant points contain little useful information.
  Sparse attention reduces compute AND memory while preserving quality,
  since the geometric prior (nearby points matter more) is well-founded.
"""

import torch
import torch.nn.functional as F


def build_knn_mask(
    coords: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Build a KNN attention mask from 3D coordinates.

    Args:
        coords: (B, S, 3) 3D coordinates for each token
        k: number of nearest neighbors per token

    Returns:
        mask: (B, S, S) boolean mask, True where attention is ALLOWED
    """
    # Pairwise L2 distances: (B, S, S)
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a@b^T
    dists = torch.cdist(coords, coords, p=2)  # (B, S, S)

    # For each query (row), find the K nearest neighbors
    # topk returns the K smallest distances
    _, topk_indices = dists.topk(k, dim=-1, largest=False)  # (B, S, K)

    # Build boolean mask
    mask = torch.zeros_like(dists, dtype=torch.bool)
    batch_idx = torch.arange(coords.shape[0], device=coords.device)[:, None, None]
    query_idx = torch.arange(coords.shape[1], device=coords.device)[None, :, None]
    mask[batch_idx, query_idx, topk_indices] = True

    return mask


def sparse_flash_attention_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    coords: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Attention with KNN sparsity mask from 3D coordinates.

    Args:
        Q: (B, H, S, D) queries
        K: (B, H, S, D) keys
        V: (B, H, S, D) values
        coords: (B, S, 3) 3D spatial coordinates per token
        k: number of nearest neighbors

    Returns:
        O: (B, H, S, D) sparse attention output
    """
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)

    # Build KNN mask: (B, S, S) -> (B, 1, S, S) for broadcasting over heads
    knn_mask = build_knn_mask(coords, k)  # (B, S, S)
    knn_mask = knn_mask.unsqueeze(1)  # (B, 1, S, S)

    # Standard attention with sparse mask
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, S, S)
    scores.masked_fill_(~knn_mask, float('-inf'))

    attn = torch.softmax(scores, dim=-1)
    # Replace NaN from all-masked rows (shouldn't happen if K >= 1, but safety)
    attn = attn.nan_to_num(0.0)

    out = torch.matmul(attn, V)
    return out
