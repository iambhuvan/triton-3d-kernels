"""
TritonAttnProcessor — Drop-in replacement for TripoSR's AttnProcessor2_0
that uses my custom Triton Flash Attention v2 kernel.

This replaces PyTorch's F.scaled_dot_product_attention with my
flash_attention_v2_forward kernel from kernels/flash_attn_v2.py.

Usage:
    from demo.triton_attn_processor import swap_attention, restore_attention

    model = TSR.from_pretrained(...)
    swap_attention(model)     # Now uses my Triton kernel
    model([image], device)    # Inference with my kernel
    restore_attention(model)  # Revert to default
"""

import time
from typing import Optional, Dict, List
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

import sys
import os

# Add project root to path so we can import my kernels
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kernels.flash_attn_v2 import flash_attention_v2_forward

# Also add triposr to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "triposr"))
from tsr.models.transformer.attention import Attention, AttnProcessor2_0


class TritonAttnProcessor:
    """
    Attention processor that uses my custom Triton Flash Attention kernel
    instead of PyTorch's F.scaled_dot_product_attention.

    Follows the same interface as TripoSR's AttnProcessor2_0.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        residual = hidden_states

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        # Group norm (if applicable)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        # Q/K/V projections (using the model's existing linear layers)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            # Self-attention: K, V come from hidden_states
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Reshape to (B, heads, S, head_dim) — matches my kernel's expected input
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # ============================================================
        # HERE IS THE SWAP: my Triton kernel instead of F.sdpa
        # ============================================================
        #
        # My kernel requires:
        #   - Input shape: (B, H, S, D) — already in this format
        #   - dtype: float16 (Triton's tl.dot requires float16/bfloat16)
        #   - Contiguous tensors
        #   - D (head_dim) must be in {16, 32, 64, 128}
        #   - Q and K must have the SAME sequence length
        #     (my kernel uses a single S for both Q and K strides)
        #
        # For cross-attention (Q_len != KV_len), we fall back to PyTorch SDPA.
        # Self-attention (the bulk of computation) uses my Triton kernel.

        orig_dtype = query.dtype
        is_cross_attn = (query.shape[2] != key.shape[2])

        if is_cross_attn:
            # Cross-attention: Q_len != KV_len — use PyTorch SDPA
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            # Self-attention: use my Triton Flash Attention kernel
            q_f16 = query.contiguous().half()
            k_f16 = key.contiguous().half()
            v_f16 = value.contiguous().half()

            # Call my kernel (causal=False for TripoSR — it's not autoregressive)
            hidden_states = flash_attention_v2_forward(q_f16, k_f16, v_f16, causal=False)

            # Cast back to original dtype
            hidden_states = hidden_states.to(orig_dtype)

        # Reshape back to (B, S, heads * head_dim)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )

        # Output projection + dropout (using model's existing layers)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


@dataclass
class TimingStats:
    """Tracks per-layer attention timing."""
    call_times: List[float] = field(default_factory=list)
    total_time: float = 0.0
    call_count: int = 0

    def record(self, elapsed: float):
        self.call_times.append(elapsed)
        self.total_time += elapsed
        self.call_count += 1

    def reset(self):
        self.call_times.clear()
        self.total_time = 0.0
        self.call_count = 0

    @property
    def avg_time(self):
        return self.total_time / max(self.call_count, 1)


class TritonAttnProcessorWithTiming(TritonAttnProcessor):
    """
    Same as TritonAttnProcessor, but records timing for each attention call.
    Used for benchmarking the attention portion of inference.
    """

    # Class-level shared stats so we can collect across all layers
    stats = TimingStats()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        torch.cuda.synchronize()
        start = time.perf_counter()

        result = super().__call__(
            attn, hidden_states, encoder_hidden_states, attention_mask
        )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        TritonAttnProcessorWithTiming.stats.record(elapsed)

        return result

    @classmethod
    def reset_stats(cls):
        cls.stats.reset()

    @classmethod
    def get_stats(cls):
        return cls.stats


class DefaultAttnProcessorWithTiming(AttnProcessor2_0):
    """
    Default PyTorch attention with timing instrumentation.
    Used to measure the baseline for comparison.
    """

    stats = TimingStats()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        torch.cuda.synchronize()
        start = time.perf_counter()

        result = super().__call__(
            attn, hidden_states, encoder_hidden_states, attention_mask
        )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        DefaultAttnProcessorWithTiming.stats.record(elapsed)

        return result

    @classmethod
    def reset_stats(cls):
        cls.stats.reset()

    @classmethod
    def get_stats(cls):
        return cls.stats


# ============================================================
# Helper functions to swap attention in a model
# ============================================================

def swap_attention(model, use_timing=False):
    """
    Replace all Attention processors in the model with my Triton kernel.

    Args:
        model: TripoSR model (TSR instance)
        use_timing: if True, use the timing variant for benchmarking

    Returns:
        dict mapping module name → original processor (for restore)
    """
    processor_cls = TritonAttnProcessorWithTiming if use_timing else TritonAttnProcessor
    original_processors = {}
    count = 0

    for name, module in model.named_modules():
        if isinstance(module, Attention):
            original_processors[name] = module.processor
            module.set_processor(processor_cls())
            count += 1

    print(f"[TritonAttnProcessor] Swapped {count} attention layers")
    return original_processors


def swap_default_with_timing(model):
    """
    Replace all Attention processors with the timed default processor.
    Used to measure baseline attention time for fair comparison.
    """
    original_processors = {}
    count = 0

    for name, module in model.named_modules():
        if isinstance(module, Attention):
            original_processors[name] = module.processor
            module.set_processor(DefaultAttnProcessorWithTiming())
            count += 1

    print(f"[DefaultAttnWithTiming] Instrumented {count} attention layers")
    return original_processors


def restore_attention(model, original_processors):
    """
    Restore original attention processors.

    Args:
        model: TripoSR model
        original_processors: dict from swap_attention()
    """
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, Attention) and name in original_processors:
            module.set_processor(original_processors[name])
            count += 1

    print(f"[restore] Restored {count} attention layers to default")


def count_attention_layers(model):
    """Count how many Attention modules exist in the model."""
    count = 0
    info = []
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            count += 1
            info.append({
                "name": name,
                "heads": module.heads,
                "inner_dim": module.inner_dim,
                "is_cross": module.cross_attention_dim != module.query_dim,
            })
    return count, info
