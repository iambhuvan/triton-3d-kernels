from .rope import fused_rope
from .flash_attn import flash_attention_forward, flash_attention_backward
from .flash_attn_v2 import flash_attention_v2_forward, flash_attention_v2_backward
from .flash_attn_v3 import flash_attention_v3_forward, flash_attention_v3_backward
from .sparse_flash_attn import sparse_flash_attention
from .chamfer import batched_chamfer_distance
from .swiglu import fused_swiglu
from .gaussian_splat import gaussian_splat_forward
from .ring_attn import ring_attention

# CUDA v3 (H100 only — requires JIT compilation)
try:
    from .flash_attn_v3_cuda import flash_attention_v3_cuda, is_hopper_available
except Exception:
    flash_attention_v3_cuda = None
    is_hopper_available = lambda: False
