"""
K3: KNN-Sparse Flash Attention for 3D — Triton Kernel

Architecture:
  Extends Flash Attention with a sparse mask from 3D spatial KNN.

  Preprocessing (Python, before kernel launch):
    1. Build KNN graph from 3D coordinates
    2. Convert to block-level sparsity: for each Q block, list active KV blocks
    3. Build within-block fine-grained masks

  Triton kernel:
    Grid: (cdiv(S, BLOCK_Q), B * H)
    Same as Flash Attention, but inner loop only visits KV blocks that
    contain at least one KNN neighbor. Within each block, applies fine-grained
    mask to zero out non-neighbor pairs.

  This reduces compute from O(S^2) to O(S * K) where K = number of neighbors.
  For 3D models with S=4096 tokens and K=64 neighbors, that's a 64x reduction.
"""

import torch
import triton
import triton.language as tl


# ============================================================
# Block sparsity preprocessing
# ============================================================

def build_knn_mask(
    coords: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Build KNN attention mask from 3D coordinates.

    Args:
        coords: (B, S, 3) 3D coordinates per token
        k: number of nearest neighbors

    Returns:
        mask: (B, S, S) boolean, True where attention is ALLOWED
    """
    dists = torch.cdist(coords, coords, p=2)
    _, topk_indices = dists.topk(k, dim=-1, largest=False)

    mask = torch.zeros_like(dists, dtype=torch.bool)
    B, S, _ = coords.shape
    batch_idx = torch.arange(B, device=coords.device)[:, None, None]
    query_idx = torch.arange(S, device=coords.device)[None, :, None]
    mask[batch_idx, query_idx, topk_indices] = True
    return mask


def precompute_block_sparsity(
    knn_mask: torch.Tensor,
    block_q: int,
    block_kv: int,
):
    """Convert fine-grained KNN mask to block-level data structures for the kernel.

    Args:
        knn_mask: (B, S, S) boolean mask
        block_q: query block size
        block_kv: key/value block size

    Returns:
        active_blocks: (B, n_q_blocks, max_active) int32 — indices of active KV blocks per Q block
        block_mask:    (B, n_q_blocks, max_active, block_q, block_kv) bool — fine-grained masks
        n_active:      (B, n_q_blocks) int32 — number of active KV blocks per Q block
    """
    B, S, _ = knn_mask.shape
    n_q_blocks = (S + block_q - 1) // block_q
    n_kv_blocks = (S + block_kv - 1) // block_kv

    # Reshape mask into block structure: (B, n_q_blocks, block_q, n_kv_blocks, block_kv)
    # Pad S to be divisible by block sizes
    S_pad_q = n_q_blocks * block_q
    S_pad_kv = n_kv_blocks * block_kv

    padded_mask = torch.zeros(B, S_pad_q, S_pad_kv, dtype=torch.bool, device=knn_mask.device)
    padded_mask[:, :S, :S] = knn_mask

    block_view = padded_mask.reshape(B, n_q_blocks, block_q, n_kv_blocks, block_kv)
    # Which KV blocks have any active entry per Q block: (B, n_q_blocks, n_kv_blocks)
    block_active = block_view.any(dim=(2, 4))

    # For each Q block, gather indices of active KV blocks
    max_active = block_active.sum(dim=-1).max().item()
    max_active = max(max_active, 1)  # at least 1

    active_blocks = torch.zeros(B, n_q_blocks, max_active, dtype=torch.int32, device=knn_mask.device)
    n_active = torch.zeros(B, n_q_blocks, dtype=torch.int32, device=knn_mask.device)
    block_masks = torch.zeros(B, n_q_blocks, max_active, block_q, block_kv,
                              dtype=torch.bool, device=knn_mask.device)

    for b in range(B):
        for qi in range(n_q_blocks):
            active_idx = block_active[b, qi].nonzero(as_tuple=True)[0]
            n = active_idx.shape[0]
            n_active[b, qi] = n
            active_blocks[b, qi, :n] = active_idx.int()
            for j in range(n):
                kvi = active_idx[j]
                block_masks[b, qi, j] = block_view[b, qi, :, kvi, :]

    return active_blocks, block_masks, n_active


# ============================================================
# Triton Kernel
# ============================================================

@triton.jit
def _sparse_flash_attn_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Block sparsity data
    ACTIVE_BLOCKS_ptr,  # (B, n_q_blocks, max_active) — which KV blocks to visit
    BLOCK_MASKS_ptr,    # (B, n_q_blocks, max_active, BLOCK_Q, BLOCK_KV) — fine masks
    N_ACTIVE_ptr,       # (B, n_q_blocks) — count of active blocks
    # Dims
    S, D: tl.constexpr,
    H,
    scale,
    # Block sparsity dims
    n_q_blocks, max_active,
    # Block sizes
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """Sparse Flash Attention kernel.

    Same online softmax as Flash Attention, but only visits active KV blocks
    and applies fine-grained masks within each block.
    """
    q_block_id = tl.program_id(0)
    bh_id = tl.program_id(1)
    batch_id = bh_id // H
    head_id = bh_id % H

    # QKV base for this (batch, head)
    qkv_base = bh_id * S * D

    # Q block
    q_start = q_block_id * BLOCK_Q
    q_offsets = q_start + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < S
    d_range = tl.arange(0, D)

    # Load Q block: (BLOCK_Q, D)
    q = tl.load(
        Q_ptr + qkv_base + q_offsets[:, None] * D + d_range[None, :],
        mask=q_mask[:, None], other=0.0,
    )

    # Online softmax accumulators
    m_i = tl.full([BLOCK_Q], value=float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_Q, D], dtype=tl.float32)

    # Load number of active KV blocks for this Q block
    n_act = tl.load(N_ACTIVE_ptr + batch_id * n_q_blocks + q_block_id)

    # Iterate only over active KV blocks
    for act_idx in range(0, max_active):
        # Only process if we haven't exceeded active block count
        if act_idx < n_act:
            # Which KV block to visit
            kv_block_id = tl.load(
                ACTIVE_BLOCKS_ptr + batch_id * n_q_blocks * max_active +
                q_block_id * max_active + act_idx
            )

            kv_start = kv_block_id * BLOCK_KV
            kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
            kv_mask = kv_offsets < S

            # Load K, V: (BLOCK_KV, D)
            k = tl.load(
                K_ptr + qkv_base + kv_offsets[:, None] * D + d_range[None, :],
                mask=kv_mask[:, None], other=0.0,
            )
            v = tl.load(
                V_ptr + qkv_base + kv_offsets[:, None] * D + d_range[None, :],
                mask=kv_mask[:, None], other=0.0,
            )

            # Attention scores: (BLOCK_Q, BLOCK_KV)
            s_block = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * scale

            # Load fine-grained mask for this (q_block, active_kv_block)
            # mask shape: (BLOCK_Q, BLOCK_KV)
            mask_base = (batch_id * n_q_blocks * max_active * BLOCK_Q * BLOCK_KV +
                         q_block_id * max_active * BLOCK_Q * BLOCK_KV +
                         act_idx * BLOCK_Q * BLOCK_KV)
            q_range = tl.arange(0, BLOCK_Q)
            kv_range = tl.arange(0, BLOCK_KV)
            fine_mask = tl.load(
                BLOCK_MASKS_ptr + mask_base + q_range[:, None] * BLOCK_KV + kv_range[None, :],
            )

            # Apply KNN mask: non-neighbor positions get -inf
            s_block = tl.where(fine_mask, s_block, float('-inf'))
            # Also mask out-of-bounds
            s_block = tl.where(kv_mask[None, :], s_block, float('-inf'))

            # Online softmax update (same as Flash Attention)
            m_block = tl.max(s_block, axis=1)
            m_new = tl.maximum(m_i, m_block)
            alpha = tl.exp(m_i - m_new)
            p_block = tl.exp(s_block - m_new[:, None])
            l_new = alpha * l_i + tl.sum(p_block, axis=1)
            o_i = alpha[:, None] * o_i + tl.dot(p_block.to(tl.float32), v.to(tl.float32))
            m_i = m_new
            l_i = l_new

    # Final normalization
    o_i = o_i / tl.maximum(l_i[:, None], 1e-8)

    # Store
    tl.store(
        O_ptr + qkv_base + q_offsets[:, None] * D + d_range[None, :],
        o_i,
        mask=q_mask[:, None],
    )


# ============================================================
# Python wrapper
# ============================================================

def sparse_flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    coords: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """KNN-Sparse Flash Attention.

    Args:
        Q: (B, H, S, D) queries
        K: (B, H, S, D) keys
        V: (B, H, S, D) values
        coords: (B, S, 3) 3D coordinates per token
        k: number of nearest neighbors

    Returns:
        O: (B, H, S, D) sparse attention output
    """
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
    B, H, S, D = Q.shape

    BLOCK_Q = 64
    BLOCK_KV = 64

    # Step 1: Build KNN mask
    knn_mask = build_knn_mask(coords, k)

    # Step 2: Precompute block sparsity
    active_blocks, block_masks, n_active = precompute_block_sparsity(
        knn_mask, BLOCK_Q, BLOCK_KV
    )
    max_active = active_blocks.shape[2]
    n_q_blocks = active_blocks.shape[1]

    # Convert bool mask to int8 for Triton
    block_masks_int = block_masks.to(torch.int8).contiguous()

    O = torch.empty_like(Q)
    scale = 1.0 / (D ** 0.5)

    grid = (n_q_blocks, B * H)
    _sparse_flash_attn_kernel[grid](
        Q, K, V, O,
        active_blocks, block_masks_int, n_active,
        S, D, H,
        scale,
        n_q_blocks, max_active,
        BLOCK_Q=BLOCK_Q,
        BLOCK_KV=BLOCK_KV,
    )

    return O
