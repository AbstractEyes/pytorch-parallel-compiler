"""
TinyFlux: A /12 scaled Flux architecture for experimentation.
OPTIMIZED VERSION - Flash Attention, vectorized RoPE, caching

Architecture:
  - hidden: 256 (3072/12)
  - num_heads: 2 (24/12)
  - head_dim: 128 (preserved for RoPE compatibility)
  - in_channels: 16 (Flux VAE output channels)
  - double_layers: 3
  - single_layers: 3

Optimizations:
  - Flash Attention (F.scaled_dot_product_attention)
  - Vectorized RoPE with precomputed frequencies
  - Vectorized img_ids creation (no Python loops)
  - Caching for img_ids and RoPE embeddings
  - Precomputed sinusoidal embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class TinyFluxConfig:
    """Configuration for TinyFlux model."""
    # Core dimensions
    hidden_size: int = 256
    num_attention_heads: int = 2
    attention_head_dim: int = 128  # Preserved for RoPE

    # Input/output (Flux VAE has 16 channels)
    in_channels: int = 16
    patch_size: int = 1

    # Text encoder interfaces
    joint_attention_dim: int = 768  # flan-t5-base output dim
    pooled_projection_dim: int = 768  # CLIP-L pooled dim

    # Layers
    num_double_layers: int = 3
    num_single_layers: int = 3

    # MLP
    mlp_ratio: float = 4.0

    # RoPE (must sum to head_dim)
    axes_dims_rope: Tuple[int, int, int] = (16, 56, 56)

    # Misc
    guidance_embeds: bool = True

    def __post_init__(self):
        assert self.num_attention_heads * self.attention_head_dim == self.hidden_size, \
            f"heads ({self.num_attention_heads}) * head_dim ({self.attention_head_dim}) != hidden ({self.hidden_size})"
        assert sum(self.axes_dims_rope) == self.attention_head_dim, \
            f"RoPE dims {self.axes_dims_rope} must sum to head_dim {self.attention_head_dim}"


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding - OPTIMIZED with precomputed frequencies."""

    def __init__(self, dim: int, axes_dims: Tuple[int, int, int], theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.axes_dims = axes_dims
        self.theta = theta

        # Precompute frequencies for each axis (no loop at runtime)
        for i, axis_dim in enumerate(axes_dims):
            freqs = 1.0 / (theta ** (torch.arange(0, axis_dim, 2).float() / axis_dim))
            self.register_buffer(f'freqs_{i}', freqs)

    def forward(self, ids: torch.Tensor, dtype: torch.dtype = None) -> torch.Tensor:
        """
        ids: (B, N, 3) - temporal, height, width indices
        Returns: (B, N, dim) rotary embeddings
        """
        B, N, _ = ids.shape
        output_dtype = dtype if dtype is not None else ids.dtype

        # Extract positions for each axis
        pos0 = ids[:, :, 0:1].float()  # (B, N, 1)
        pos1 = ids[:, :, 1:2].float()
        pos2 = ids[:, :, 2:3].float()

        # Compute angles (broadcasting: (B, N, 1) * (axis_dim/2,) -> (B, N, axis_dim/2))
        angles0 = pos0 * self.freqs_0
        angles1 = pos1 * self.freqs_1
        angles2 = pos2 * self.freqs_2

        # Stack sin/cos and flatten for each axis
        emb0 = torch.stack([angles0.cos(), angles0.sin()], dim=-1).flatten(-2)
        emb1 = torch.stack([angles1.cos(), angles1.sin()], dim=-1).flatten(-2)
        emb2 = torch.stack([angles2.cos(), angles2.sin()], dim=-1).flatten(-2)

        return torch.cat([emb0, emb1, emb2], dim=-1).to(output_dtype)


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to input tensor."""
    # x: (B, heads, N, head_dim)
    # rope: (B, N, head_dim)
    B, H, N, D = x.shape

    rope = rope.to(x.dtype).unsqueeze(1)  # (B, 1, N, D)

    # Split into pairs
    x_pairs = x.reshape(B, H, N, D // 2, 2)
    rope_pairs = rope.reshape(B, 1, N, D // 2, 2)

    cos = rope_pairs[..., 0]
    sin = rope_pairs[..., 1]

    x0 = x_pairs[..., 0]
    x1 = x_pairs[..., 1]

    out0 = x0 * cos - x1 * sin
    out1 = x1 * cos + x0 * sin

    return torch.stack([out0, out1], dim=-1).flatten(-2)


class MLPEmbedder(nn.Module):
    """MLP for embedding scalars - OPTIMIZED with precomputed basis."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        # Precompute sinusoidal basis
        half_dim = 128
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer('sin_basis', emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use precomputed basis
        emb = x.unsqueeze(-1) * self.sin_basis.to(x.dtype)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(emb)


class AdaLayerNormZero(nn.Module):
    """AdaLN-Zero for double-stream blocks."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        self.norm = RMSNorm(hidden_size)

    def forward(
            self, x: torch.Tensor, emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb_out = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb_out.chunk(6, dim=-1)
        x = self.norm(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    """AdaLN-Zero for single-stream blocks."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.norm = RMSNorm(hidden_size)

    def forward(
            self, x: torch.Tensor, emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb_out = self.linear(self.silu(emb))
        shift, scale, gate = emb_out.chunk(3, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x, gate


class Attention(nn.Module):
    """Multi-head attention - OPTIMIZED with Flash Attention."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
            self,
            x: torch.Tensor,
            rope: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, _ = x.shape
        dtype = x.dtype

        if rope is not None:
            rope = rope.to(dtype)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3 x (B, heads, N, head_dim)

        if rope is not None:
            q = apply_rope(q, rope)
            k = apply_rope(k, rope)

        # Flash Attention - faster and memory efficient
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.out_proj(out)


class JointAttention(nn.Module):
    """Joint attention - OPTIMIZED with Flash Attention."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.txt_qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.img_qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)

        self.txt_out = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.img_out = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
            self,
            txt: torch.Tensor,
            img: torch.Tensor,
            rope: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = txt.shape
        _, N, _ = img.shape

        dtype = img.dtype
        txt = txt.to(dtype)
        if rope is not None:
            rope = rope.to(dtype)

        # Compute Q, K, V for both streams
        txt_qkv = self.txt_qkv(txt).reshape(B, L, 3, self.num_heads, self.head_dim)
        img_qkv = self.img_qkv(img).reshape(B, N, 3, self.num_heads, self.head_dim)

        txt_q, txt_k, txt_v = txt_qkv.permute(2, 0, 3, 1, 4)
        img_q, img_k, img_v = img_qkv.permute(2, 0, 3, 1, 4)

        # Apply RoPE to image only
        if rope is not None:
            img_q = apply_rope(img_q, rope)
            img_k = apply_rope(img_k, rope)

        # Concatenate for joint attention
        k = torch.cat([txt_k, img_k], dim=2)
        v = torch.cat([txt_v, img_v], dim=2)

        # Flash Attention for both streams
        txt_out = F.scaled_dot_product_attention(txt_q, k, v, scale=self.scale)
        img_out = F.scaled_dot_product_attention(img_q, k, v, scale=self.scale)

        txt_out = txt_out.transpose(1, 2).reshape(B, L, -1)
        img_out = img_out.transpose(1, 2).reshape(B, N, -1)

        return self.txt_out(txt_out), self.img_out(img_out)


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, mlp_hidden)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(mlp_hidden, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DoubleStreamBlock(nn.Module):
    """Double-stream transformer block (MMDiT style)."""

    def __init__(self, config: TinyFluxConfig):
        super().__init__()
        hidden = config.hidden_size
        heads = config.num_attention_heads
        head_dim = config.attention_head_dim

        self.img_norm1 = AdaLayerNormZero(hidden)
        self.txt_norm1 = AdaLayerNormZero(hidden)
        self.attn = JointAttention(hidden, heads, head_dim)
        self.img_norm2 = RMSNorm(hidden)
        self.txt_norm2 = RMSNorm(hidden)
        self.img_mlp = MLP(hidden, config.mlp_ratio)
        self.txt_mlp = MLP(hidden, config.mlp_ratio)

    def forward(
            self,
            txt: torch.Tensor,
            img: torch.Tensor,
            vec: torch.Tensor,
            rope: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_normed, img_gate_msa, img_shift_mlp, img_scale_mlp, img_gate_mlp = self.img_norm1(img, vec)
        txt_normed, txt_gate_msa, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = self.txt_norm1(txt, vec)

        txt_attn_out, img_attn_out = self.attn(txt_normed, img_normed, rope)

        txt = txt + txt_gate_msa.unsqueeze(1) * txt_attn_out
        img = img + img_gate_msa.unsqueeze(1) * img_attn_out

        txt_mlp_in = self.txt_norm2(txt) * (1 + txt_scale_mlp.unsqueeze(1)) + txt_shift_mlp.unsqueeze(1)
        img_mlp_in = self.img_norm2(img) * (1 + img_scale_mlp.unsqueeze(1)) + img_shift_mlp.unsqueeze(1)

        txt = txt + txt_gate_mlp.unsqueeze(1) * self.txt_mlp(txt_mlp_in)
        img = img + img_gate_mlp.unsqueeze(1) * self.img_mlp(img_mlp_in)

        return txt, img


class SingleStreamBlock(nn.Module):
    """Single-stream transformer block."""

    def __init__(self, config: TinyFluxConfig):
        super().__init__()
        hidden = config.hidden_size
        heads = config.num_attention_heads
        head_dim = config.attention_head_dim

        self.norm = AdaLayerNormZeroSingle(hidden)
        self.attn = Attention(hidden, heads, head_dim)
        self.mlp = MLP(hidden, config.mlp_ratio)
        self.norm2 = RMSNorm(hidden)

    def forward(
            self,
            txt: torch.Tensor,
            img: torch.Tensor,
            vec: torch.Tensor,
            txt_rope: Optional[torch.Tensor] = None,
            img_rope: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        L = txt.shape[1]

        x = torch.cat([txt, img], dim=1)

        if img_rope is not None:
            B, N, D = img_rope.shape
            txt_rope_zeros = torch.zeros(B, L, D, device=img_rope.device, dtype=img_rope.dtype)
            rope = torch.cat([txt_rope_zeros, img_rope], dim=1)
        else:
            rope = None

        x_normed, gate = self.norm(x, vec)
        x = x + gate.unsqueeze(1) * self.attn(x_normed, rope)
        x = x + self.mlp(self.norm2(x))

        txt, img = x.split([L, x.shape[1] - L], dim=1)
        return txt, img


# Global cache for img_ids (they don't change for same resolution)
_IMG_IDS_CACHE: Dict[Tuple, torch.Tensor] = {}


class TinyFlux(nn.Module):
    """
    TinyFlux: A scaled-down Flux diffusion transformer.
    OPTIMIZED with Flash Attention, vectorized ops, and caching.
    """

    def __init__(self, config: Optional[TinyFluxConfig] = None):
        super().__init__()
        self.config = config or TinyFluxConfig()
        cfg = self.config

        # Input projections
        self.img_in = nn.Linear(cfg.in_channels, cfg.hidden_size)
        self.txt_in = nn.Linear(cfg.joint_attention_dim, cfg.hidden_size)

        # Conditioning projections
        self.time_in = MLPEmbedder(cfg.hidden_size)
        self.vector_in = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.pooled_projection_dim, cfg.hidden_size)
        )
        if cfg.guidance_embeds:
            self.guidance_in = MLPEmbedder(cfg.hidden_size)

        # RoPE
        self.rope = RotaryEmbedding(cfg.attention_head_dim, cfg.axes_dims_rope)

        # Transformer blocks
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(cfg) for _ in range(cfg.num_double_layers)
        ])
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(cfg) for _ in range(cfg.num_single_layers)
        ])

        # Output
        self.final_norm = RMSNorm(cfg.hidden_size)
        self.final_linear = nn.Linear(cfg.hidden_size, cfg.in_channels)

        # RoPE cache
        self._rope_cache: Dict[Tuple, torch.Tensor] = {}

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        def _init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_init)
        nn.init.zeros_(self.final_linear.weight)

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            pooled_projections: torch.Tensor,
            timestep: torch.Tensor,
            img_ids: torch.Tensor,
            guidance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        # Input projections
        img = self.img_in(hidden_states)
        txt = self.txt_in(encoder_hidden_states)

        # Conditioning vector
        vec = self.time_in(timestep)
        vec = vec + self.vector_in(pooled_projections)
        if self.config.guidance_embeds and guidance is not None:
            vec = vec + self.guidance_in(guidance)

        # RoPE for image positions
        img_rope = self.rope(img_ids, dtype=img.dtype)

        # Double-stream blocks
        for block in self.double_blocks:
            txt, img = block(txt, img, vec, img_rope)

        # Single-stream blocks
        for block in self.single_blocks:
            txt, img = block(txt, img, vec, img_rope=img_rope)

        # Output
        img = self.final_norm(img)
        img = self.final_linear(img)

        return img

    @staticmethod
    def create_img_ids(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Create image position IDs - VECTORIZED (no Python loops)."""
        global _IMG_IDS_CACHE

        # Check cache first
        cache_key = (batch_size, height, width, device)
        if cache_key in _IMG_IDS_CACHE:
            return _IMG_IDS_CACHE[cache_key]

        # Vectorized creation using meshgrid
        h_ids = torch.arange(height, device=device, dtype=torch.float32)
        w_ids = torch.arange(width, device=device, dtype=torch.float32)

        grid_h, grid_w = torch.meshgrid(h_ids, w_ids, indexing='ij')

        # Stack: (H*W, 3) with [temporal=0, height, width]
        img_ids = torch.stack([
            torch.zeros(height * width, device=device),  # temporal
            grid_h.flatten(),
            grid_w.flatten(),
        ], dim=-1)

        # Expand for batch
        img_ids = img_ids.unsqueeze(0).expand(batch_size, -1, -1)

        # Cache it
        _IMG_IDS_CACHE[cache_key] = img_ids

        return img_ids

    def count_parameters(self) -> dict:
        """Count parameters by component."""
        counts = {}
        counts['img_in'] = sum(p.numel() for p in self.img_in.parameters())
        counts['txt_in'] = sum(p.numel() for p in self.txt_in.parameters())
        counts['time_in'] = sum(p.numel() for p in self.time_in.parameters())
        counts['vector_in'] = sum(p.numel() for p in self.vector_in.parameters())
        if hasattr(self, 'guidance_in'):
            counts['guidance_in'] = sum(p.numel() for p in self.guidance_in.parameters())
        counts['double_blocks'] = sum(p.numel() for p in self.double_blocks.parameters())
        counts['single_blocks'] = sum(p.numel() for p in self.single_blocks.parameters())
        counts['final'] = sum(p.numel() for p in self.final_norm.parameters()) + \
                          sum(p.numel() for p in self.final_linear.parameters())
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts


def test_tiny_flux():
    """Quick test of the optimized model."""
    print("=" * 60)
    print("TinyFlux OPTIMIZED Model Test")
    print("=" * 60)

    config = TinyFluxConfig()
    print(f"\nConfig:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_heads: {config.num_attention_heads}")
    print(f"  head_dim: {config.attention_head_dim}")

    model = TinyFlux(config)

    counts = model.count_parameters()
    print(f"\nParameters: {counts['total']:,} ({counts['total'] / 1e6:.2f}M)")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    batch_size = 4
    latent_h, latent_w = 64, 64
    num_patches = latent_h * latent_w
    text_len = 77

    hidden_states = torch.randn(batch_size, num_patches, config.in_channels, device=device)
    encoder_hidden_states = torch.randn(batch_size, text_len, config.joint_attention_dim, device=device)
    pooled_projections = torch.randn(batch_size, config.pooled_projection_dim, device=device)
    timestep = torch.rand(batch_size, device=device)
    img_ids = TinyFlux.create_img_ids(batch_size, latent_h, latent_w, device)
    guidance = torch.ones(batch_size, device=device) * 3.5

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, guidance)

    # Benchmark
    if device == 'cuda':
        torch.cuda.synchronize()
        import time
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                output = model(hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, guidance)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 10
        print(f"\nAverage forward pass: {elapsed*1000:.2f}ms")

    print(f"Output shape: {output.shape}")
    print("\n✓ Forward pass successful!")


#if __name__ == "__main__":
#    test_tiny_flux()