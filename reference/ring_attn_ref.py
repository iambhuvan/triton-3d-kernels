"""
K7: Ring Attention — Reference Implementation

Ring Attention distributes long sequences across multiple GPUs by partitioning
the sequence into chunks and passing KV blocks around in a ring topology.

The idea:
  - Split sequence of length S into P chunks (one per GPU), each of length S/P
  - Each GPU holds its local Q chunk and iterates over all KV chunks
  - KV chunks are passed around the ring using send/recv
  - Each GPU computes local attention against the current KV chunk and
    accumulates using the online softmax trick (same as Flash Attention)

Math (on each GPU p):
  Q_p = Q[p * S/P : (p+1) * S/P]     # local queries, stays fixed

  Initialize: O_p = 0, l_p = 0, m_p = -inf

  For step in 0..P-1:
    source = (p - step) % P
    K_src, V_src = recv_from_ring()    # or use local if step=0

    # Local flash attention block
    S_block = Q_p @ K_src^T / sqrt(d)
    m_new = max(m_p, S_block.max(dim=-1))
    P_block = exp(S_block - m_new)
    l_new = exp(m_p - m_new) * l_p + P_block.sum(dim=-1)
    O_p = (exp(m_p - m_new) * l_p / l_new) * O_p + (P_block / l_new) @ V_src
    m_p, l_p = m_new, l_new

    send_kv_to_next()

  This is mathematically equivalent to full attention but uses O(S/P) memory
  per GPU instead of O(S).

Why this kernel matters:
  3D generative models can have very long token sequences (thousands of 3D points).
  Ring attention enables scaling to sequences that don't fit in a single GPU's memory.
  It combines kernel-level optimization (tiled attention) with distributed systems
  (NCCL ring communication), showing both low-level and systems-level thinking.

Note: This reference implementation simulates the ring on a single device by
  explicitly chunking and iterating. The real implementation uses torch.distributed.
"""

import torch
import torch.nn.functional as F


def ring_attention_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    n_chunks: int,
    causal: bool = False,
) -> torch.Tensor:
    """Simulate ring attention on a single device.

    This splits Q, K, V into chunks and processes them using the ring
    attention algorithm to verify correctness against standard attention.

    Args:
        Q: (B, H, S, D) queries
        K: (B, H, S, D) keys
        V: (B, H, S, D) values
        n_chunks: number of ring steps (simulated GPUs)
        causal: if True, apply causal masking

    Returns:
        O: (B, H, S, D) attention output (should match standard attention)
    """
    B, H, S, D = Q.shape
    assert S % n_chunks == 0, f"Sequence length {S} must be divisible by n_chunks {n_chunks}"
    chunk_size = S // n_chunks

    # Split Q, K, V into chunks along sequence dimension
    Q_chunks = Q.split(chunk_size, dim=2)  # list of (B, H, S/P, D)
    K_chunks = K.split(chunk_size, dim=2)
    V_chunks = V.split(chunk_size, dim=2)

    output_chunks = []

    for p in range(n_chunks):
        Q_local = Q_chunks[p]  # (B, H, chunk_size, D)
        scale = 1.0 / (D ** 0.5)

        # Online softmax accumulators
        O_acc = torch.zeros_like(Q_local)
        l_acc = torch.zeros(B, H, chunk_size, 1, device=Q.device, dtype=Q.dtype)
        m_acc = torch.full((B, H, chunk_size, 1), float('-inf'), device=Q.device, dtype=Q.dtype)

        # Iterate over KV chunks in ring order
        for step in range(n_chunks):
            kv_idx = (p + step) % n_chunks  # simulates ring recv
            K_block = K_chunks[kv_idx]
            V_block = V_chunks[kv_idx]

            # Compute attention scores for this block
            S_block = torch.matmul(Q_local, K_block.transpose(-2, -1)) * scale  # (B,H,cs,cs)

            # Causal masking: Q at positions [p*cs : (p+1)*cs], K at [kv*cs : (kv+1)*cs]
            if causal:
                q_start = p * chunk_size
                k_start = kv_idx * chunk_size
                cs = chunk_size
                # Build mask: q_pos >= k_pos (causal = can attend to past)
                q_pos = torch.arange(q_start, q_start + cs, device=Q.device)
                k_pos = torch.arange(k_start, k_start + cs, device=Q.device)
                causal_mask = q_pos[:, None] < k_pos[None, :]  # True where BLOCKED
                S_block.masked_fill_(causal_mask[None, None, :, :], float('-inf'))

            # Online softmax update
            m_block = S_block.max(dim=-1, keepdim=True).values  # (B, H, cs, 1)
            m_new = torch.maximum(m_acc, m_block)

            # Rescale old accumulator
            exp_old = torch.exp(m_acc - m_new)
            # New block probabilities
            P_block = torch.exp(S_block - m_new)

            l_new = exp_old * l_acc + P_block.sum(dim=-1, keepdim=True)

            # Update output: rescale old output + add new contribution
            O_acc = (exp_old * l_acc / (l_new + 1e-8)) * O_acc + \
                    torch.matmul(P_block, V_block) / (l_new + 1e-8)

            m_acc = m_new
            l_acc = l_new

        output_chunks.append(O_acc)

    # Concatenate chunks back
    return torch.cat(output_chunks, dim=2)
