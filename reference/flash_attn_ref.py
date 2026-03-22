"""
K2: Flash Attention — Reference Implementation

Flash Attention computes exact attention without materializing the full S×S
attention matrix in HBM. Instead, it tiles over blocks of Q, K, V and
accumulates the output in SRAM using the online softmax trick.

Standard Attention Math:
  S = Q @ K^T / sqrt(d)          # (B, H, S, S) — this is the memory bottleneck
  P = softmax(S, dim=-1)         # (B, H, S, S)
  O = P @ V                      # (B, H, S, D)

Memory: O(S^2) for the attention matrix
Flash Attention: O(S) by never materializing S×S

The backward pass requires recomputation of the attention matrix from Q, K, V
(stored in HBM) rather than caching it — trading compute for memory.

Key insight from Dao et al.: the online softmax trick allows computing
softmax(Q @ K^T) block by block without needing the full row.
  - Track running max m_i and running sum l_i per query row
  - Rescale accumulated output when a new block has a larger max

Why this kernel matters:
  Attention is the dominant cost in transformer training. Flash Attention
  reduces memory from O(S^2) to O(S) and improves wall-clock time by
  reducing HBM reads/writes. Essential for long-sequence 3D token models.
"""

import torch
import torch.nn.functional as F


def flash_attention_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Standard attention — the ground truth Flash Attention must match.

    Args:
        Q: (B, H, S, D) queries
        K: (B, H, S, D) keys
        V: (B, H, S, D) values
        causal: if True, apply causal mask (upper triangle = -inf)

    Returns:
        O: (B, H, S, D) attention output
    """
    d = Q.shape[-1]
    scale = 1.0 / (d ** 0.5)

    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, S, S)

    # Apply causal mask
    if causal:
        S = Q.shape[-2]
        mask = torch.triu(torch.ones(S, S, device=Q.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float('-inf'))

    # Softmax + matmul
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)

    return out


def flash_attention_reference_with_lse(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
):
    """Attention that also returns log-sum-exp (needed for backward pass).

    Returns:
        O:   (B, H, S, D) attention output
        lse: (B, H, S) log-sum-exp per query row = log(sum(exp(scores - max)))  + max
    """
    d = Q.shape[-1]
    scale = 1.0 / (d ** 0.5)

    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if causal:
        S = Q.shape[-2]
        mask = torch.triu(torch.ones(S, S, device=Q.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float('-inf'))

    # log-sum-exp for backward
    row_max = scores.max(dim=-1, keepdim=True).values
    stable_scores = scores - row_max
    lse = row_max.squeeze(-1) + torch.log(torch.exp(stable_scores).sum(dim=-1))

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)

    return out, lse
