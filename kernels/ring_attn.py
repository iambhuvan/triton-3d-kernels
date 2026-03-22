"""
K7: Ring Attention — Distributed + Triton

Architecture:
  Combines two layers:
    1. LOCAL: Triton Flash Attention kernel for attention on local Q vs received KV
    2. DISTRIBUTED: torch.distributed send/recv for KV block rotation in a ring

  Each GPU holds Q_local (fixed) and iterates through KV blocks from all GPUs.
  Online softmax accumulation merges results across ring steps.

  Single-GPU simulation mode:
    Splits Q, K, V into chunks and iterates as if on separate GPUs.
    Uses the same online softmax merging logic.

  Multi-GPU mode (torch.distributed):
    Each rank holds one chunk. Uses non-blocking send/recv to overlap
    communication with compute:
      - Step t: compute attention with KV_current, async send KV to next rank
      - Step t+1: receive new KV, compute attention, send, ...
"""

import torch
import triton
import triton.language as tl


# ============================================================
# Local attention Triton kernel (one ring step)
# ============================================================

@triton.jit
def _ring_local_attn_kernel(
    Q_ptr, K_ptr, V_ptr,
    # Running accumulators (read + write)
    O_ptr, M_ptr, L_ptr,
    # Layout
    S_local, D: tl.constexpr,
    S_kv,   # KV chunk may differ in size
    scale,
    # Causal masking: global position offsets
    Q_OFFSET,   # global start position of this Q chunk
    KV_OFFSET,  # global start position of this KV chunk
    IS_CAUSAL: tl.constexpr,
    # Block sizes
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """Compute one ring step's local attention and merge with running state.

    Grid: (cdiv(S_local, BLOCK_Q), B * H)

    This is a modified Flash Attention forward that reads/updates existing
    O, m, l accumulators instead of initializing from scratch.
    """
    q_block_id = tl.program_id(0)
    bh_id = tl.program_id(1)

    # Base offsets
    q_base = Q_ptr + bh_id * S_local * D
    k_base = K_ptr + bh_id * S_kv * D
    v_base = V_ptr + bh_id * S_kv * D
    o_base = O_ptr + bh_id * S_local * D
    m_base = M_ptr + bh_id * S_local
    l_base = L_ptr + bh_id * S_local

    q_start = q_block_id * BLOCK_Q
    q_offsets = q_start + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < S_local
    d_range = tl.arange(0, D)

    # Global positions for causal masking
    q_global = Q_OFFSET + q_offsets  # global seq positions of Q rows

    # Load Q block
    q = tl.load(q_base + q_offsets[:, None] * D + d_range[None, :],
                mask=q_mask[:, None], other=0.0)

    # Load current accumulators
    m_i = tl.load(m_base + q_offsets, mask=q_mask, other=float('-inf'))
    l_i = tl.load(l_base + q_offsets, mask=q_mask, other=0.0)
    o_i = tl.load(o_base + q_offsets[:, None] * D + d_range[None, :],
                  mask=q_mask[:, None], other=0.0)

    # Iterate over KV blocks from this ring step's chunk
    n_kv_blocks = tl.cdiv(S_kv, BLOCK_KV)

    for kv_block_id in range(0, n_kv_blocks):
        kv_start = kv_block_id * BLOCK_KV
        kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offsets < S_kv

        # Global positions for causal masking
        kv_global = KV_OFFSET + kv_offsets

        k = tl.load(k_base + kv_offsets[:, None] * D + d_range[None, :],
                     mask=kv_mask[:, None], other=0.0)
        v = tl.load(v_base + kv_offsets[:, None] * D + d_range[None, :],
                     mask=kv_mask[:, None], other=0.0)

        # S = Q @ K^T * scale
        s_block = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * scale
        s_block = tl.where(kv_mask[None, :], s_block, float('-inf'))

        # Causal mask: q_global_pos < kv_global_pos -> blocked
        if IS_CAUSAL:
            causal_mask = q_global[:, None] < kv_global[None, :]
            s_block = tl.where(causal_mask, float('-inf'), s_block)

        # Online softmax update
        m_block = tl.max(s_block, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p_block = tl.exp(s_block - m_new[:, None])
        l_new = alpha * l_i + tl.sum(p_block, axis=1)
        o_i = alpha[:, None] * o_i + tl.dot(p_block.to(tl.float32), v.to(tl.float32))
        m_i = m_new
        l_i = l_new

    # Store updated accumulators
    tl.store(o_base + q_offsets[:, None] * D + d_range[None, :],
             o_i, mask=q_mask[:, None])
    tl.store(m_base + q_offsets, m_i, mask=q_mask)
    tl.store(l_base + q_offsets, l_i, mask=q_mask)


# ============================================================
# Ring orchestration
# ============================================================

def _run_ring_step_triton(Q_local, K_chunk, V_chunk, O_acc, m_acc, l_acc, scale,
                          q_offset=0, kv_offset=0, causal=False):
    """Run one ring step using the Triton local attention kernel."""
    if Q_local.dim() == 4:
        B, H, S_local, D = Q_local.shape
        Q_flat = Q_local.reshape(B * H, S_local, D).contiguous()
        K_flat = K_chunk.reshape(B * H, -1, D).contiguous()
        V_flat = V_chunk.reshape(B * H, -1, D).contiguous()
        O_flat = O_acc.reshape(B * H, S_local, D).contiguous()
        m_flat = m_acc.reshape(B * H, S_local).contiguous()
        l_flat = l_acc.reshape(B * H, S_local).contiguous()
    else:
        raise ValueError("Expected 4D tensors")

    S_kv = K_flat.shape[1]
    BLOCK_Q = 64
    BLOCK_KV = 64

    grid = (triton.cdiv(S_local, BLOCK_Q), B * H)
    _ring_local_attn_kernel[grid](
        Q_flat, K_flat, V_flat,
        O_flat, m_flat, l_flat,
        S_local, D, S_kv,
        scale,
        q_offset, kv_offset,
        IS_CAUSAL=causal,
        BLOCK_Q=BLOCK_Q,
        BLOCK_KV=BLOCK_KV,
    )

    # Reshape back — clone() avoids RuntimeError when reshape returns a view
    # that shares memory with the destination tensor of copy_().
    O_acc.copy_(O_flat.reshape(B, H, S_local, D).clone())
    m_acc.copy_(m_flat.reshape(B, H, S_local).clone())
    l_acc.copy_(l_flat.reshape(B, H, S_local).clone())


def ring_attention_single_gpu(Q, K, V, n_chunks, causal=False):
    """Simulate ring attention on a single device.

    Splits Q, K, V into chunks and runs the ring protocol,
    using Triton kernels for local attention at each step.
    """
    B, H, S, D = Q.shape
    assert S % n_chunks == 0
    chunk_size = S // n_chunks
    scale = 1.0 / (D ** 0.5)

    Q_chunks = Q.split(chunk_size, dim=2)
    K_chunks = K.split(chunk_size, dim=2)
    V_chunks = V.split(chunk_size, dim=2)

    output_chunks = []

    for p in range(n_chunks):
        Q_local = Q_chunks[p]

        # Initialize accumulators
        O_acc = torch.zeros_like(Q_local)
        m_acc = torch.full((B, H, chunk_size), float('-inf'), device=Q.device, dtype=Q.dtype)
        l_acc = torch.zeros(B, H, chunk_size, device=Q.device, dtype=Q.dtype)

        # Ring iteration: visit all KV chunks
        for step in range(n_chunks):
            kv_idx = (p + step) % n_chunks
            K_block = K_chunks[kv_idx]
            V_block = V_chunks[kv_idx]

            if causal:
                # Apply causal masking logic:
                # Q positions: [p*cs, (p+1)*cs), KV positions: [kv*cs, (kv+1)*cs)
                # Skip entirely if all Q positions < all KV positions
                q_end = (p + 1) * chunk_size
                kv_start = kv_idx * chunk_size
                if kv_start >= q_end:
                    continue

            # Run one ring step (Triton kernel) with global offsets for causal masking
            q_offset = p * chunk_size
            kv_offset = kv_idx * chunk_size
            _run_ring_step_triton(Q_local, K_block, V_block, O_acc, m_acc, l_acc, scale,
                                  q_offset=q_offset, kv_offset=kv_offset, causal=causal)

        # Final normalization: O = O / l
        O_final = O_acc / l_acc.unsqueeze(-1).clamp(min=1e-8)
        output_chunks.append(O_final)

    return torch.cat(output_chunks, dim=2)


def ring_attention_distributed(Q, K, V, causal=False):
    """Real distributed ring attention using torch.distributed.

    Each rank holds its local Q, K, V chunk.
    KV blocks are rotated around the ring using send/recv.

    Must be launched with torchrun / torch.distributed.launch.
    """
    import torch.distributed as dist

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    B, H, S_local, D = Q.shape
    scale = 1.0 / (D ** 0.5)

    # Initialize accumulators
    O_acc = torch.zeros_like(Q)
    m_acc = torch.full((B, H, S_local), float('-inf'), device=Q.device, dtype=Q.dtype)
    l_acc = torch.zeros(B, H, S_local, device=Q.device, dtype=Q.dtype)

    # Current KV block (starts as local)
    K_curr = K.clone()
    V_curr = V.clone()

    # Buffers for async recv
    K_recv = torch.empty_like(K)
    V_recv = torch.empty_like(V)

    for step in range(world_size):
        # Async send current KV to next rank, recv from previous
        send_rank = (rank + 1) % world_size
        recv_rank = (rank - 1) % world_size

        if step < world_size - 1:
            # Non-blocking communication (overlap with compute)
            send_k = dist.isend(K_curr, send_rank)
            send_v = dist.isend(V_curr, send_rank)
            recv_k = dist.irecv(K_recv, recv_rank)
            recv_v = dist.irecv(V_recv, recv_rank)

        # Compute local attention for this step (Triton kernel)
        _run_ring_step_triton(Q, K_curr, V_curr, O_acc, m_acc, l_acc, scale)

        if step < world_size - 1:
            # Wait for communication to complete
            send_k.wait()
            send_v.wait()
            recv_k.wait()
            recv_v.wait()
            # Swap buffers
            K_curr, K_recv = K_recv, K_curr
            V_curr, V_recv = V_recv, V_curr

    # Final normalization
    O = O_acc / l_acc.unsqueeze(-1).clamp(min=1e-8)
    return O


# ============================================================
# Public API
# ============================================================

def ring_attention(Q, K, V, n_chunks, causal=False):
    """Ring Attention.

    In single-GPU mode, simulates the ring by splitting Q, K, V into chunks.
    In distributed mode (torch.distributed initialized), uses real communication.

    Args:
        Q, K, V: (B, H, S, D)
        n_chunks: number of ring participants (ignored in distributed mode)
        causal: whether to apply causal masking

    Returns:
        O: (B, H, S, D)
    """
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return ring_attention_distributed(Q, K, V, causal)
    except ImportError:
        pass

    return ring_attention_single_gpu(Q, K, V, n_chunks, causal)
