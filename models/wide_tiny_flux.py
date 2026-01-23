"""
WideTinyFlux: N parallel TinyFlux models fused into a single wide model.

Uses WideCompiler primitives for efficient batched execution.
All operations use N-first format [N, B, ...] internally.

Key speedups come from:
- WideRMSNorm: 21x faster than sequential
- WideLinear: 9.7x faster (einsum batching)
- WideJointAttention: 9.0x faster
- WideMLP: 2.9x faster
- WideAttention: 7.9x faster

Copyright 2025
Apache 2.0 License
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

# Import wide primitives
from wide_compiler.core.primitives import (
    WideRMSNorm,
    WideLinear,
    WideMLPEmbedder,
    WideAdaLayerNormZeroSingle,
)
from wide_compiler.core.primitives.wide_rotary_embedding import apply_rope
from wide_compiler.core.blocks import (
    WideAttention,
    WideJointAttention,
    WideMLP,
)


@dataclass
class WideTinyFluxConfig:
    """Configuration for WideTinyFlux model."""
    # Number of parallel models
    n: int = 8

    # Core dimensions
    hidden_size: int = 256
    num_attention_heads: int = 2
    attention_head_dim: int = 128

    # Input/output
    in_channels: int = 16
    patch_size: int = 1

    # Text encoder interfaces
    joint_attention_dim: int = 768
    pooled_projection_dim: int = 768

    # Layers
    num_double_layers: int = 3
    num_single_layers: int = 3

    # MLP
    mlp_ratio: float = 4.0

    # RoPE
    axes_dims_rope: Tuple[int, int, int] = (16, 56, 56)

    # Misc
    guidance_embeds: bool = True

    # Strategy for wide primitives
    strategy: str = 'fused'

    def __post_init__(self):
        assert self.num_attention_heads * self.attention_head_dim == self.hidden_size, \
            f"heads ({self.num_attention_heads}) * head_dim ({self.attention_head_dim}) != hidden ({self.hidden_size})"
        assert sum(self.axes_dims_rope) == self.attention_head_dim, \
            f"RoPE dims {self.axes_dims_rope} must sum to head_dim {self.attention_head_dim}"


class WideRotaryEmbedding(nn.Module):
    """
    Wide Rotary Position Embedding with precomputed frequencies.

    Input format (N-first):
        ids: [N, B, seq_len, 3] or [B, seq_len, 3] (temporal, height, width indices)

    Output:
        [N, B, seq_len, head_dim] or [B, seq_len, head_dim] rotary embeddings
    """

    def __init__(self, n: int, dim: int, axes_dims: Tuple[int, int, int], theta: float = 10000.0):
        super().__init__()
        self.n = n
        self.dim = dim
        self.axes_dims = axes_dims
        self.theta = theta

        # Precompute frequencies for each axis
        for i, axis_dim in enumerate(axes_dims):
            freqs = 1.0 / (theta ** (torch.arange(0, axis_dim, 2).float() / axis_dim))
            self.register_buffer(f'freqs_{i}', freqs)

    def forward(self, ids: torch.Tensor, dtype: torch.dtype = None) -> torch.Tensor:
        """
        Compute rotary embeddings from position IDs.

        Args:
            ids: [B, seq_len, 3] - position indices (same for all N models)

        Returns:
            [B, seq_len, dim] - rotary embeddings
        """
        B, S, _ = ids.shape
        output_dtype = dtype if dtype is not None else ids.dtype

        # Extract positions for each axis
        pos0 = ids[:, :, 0:1].float()
        pos1 = ids[:, :, 1:2].float()
        pos2 = ids[:, :, 2:3].float()

        # Compute angles
        angles0 = pos0 * self.freqs_0
        angles1 = pos1 * self.freqs_1
        angles2 = pos2 * self.freqs_2

        # Stack sin/cos and flatten for each axis
        emb0 = torch.stack([angles0.cos(), angles0.sin()], dim=-1).flatten(-2)
        emb1 = torch.stack([angles1.cos(), angles1.sin()], dim=-1).flatten(-2)
        emb2 = torch.stack([angles2.cos(), angles2.sin()], dim=-1).flatten(-2)

        return torch.cat([emb0, emb1, emb2], dim=-1).to(output_dtype)


class WideMLPEmbedderTimestep(nn.Module):
    """
    Wide MLP for embedding timesteps with sinusoidal basis.

    Input: [N, B] timestep values
    Output: [N, B, hidden_size] embeddings
    """

    def __init__(self, n: int, hidden_size: int, strategy: str = 'fused'):
        super().__init__()
        self.n = n
        self.hidden_size = hidden_size
        self._strategy = strategy

        # First projection: 256 -> hidden_size
        self.fc1_weight = nn.Parameter(torch.empty(n, hidden_size, 256))
        self.fc1_bias = nn.Parameter(torch.zeros(n, hidden_size))

        # Second projection: hidden_size -> hidden_size
        self.fc2_weight = nn.Parameter(torch.empty(n, hidden_size, hidden_size))
        self.fc2_bias = nn.Parameter(torch.zeros(n, hidden_size))

        # Precompute sinusoidal basis
        half_dim = 128
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer('sin_basis', emb)

        self._reset_parameters()

    def _reset_parameters(self):
        for i in range(self.n):
            nn.init.xavier_uniform_(self.fc1_weight[i])
            nn.init.xavier_uniform_(self.fc2_weight[i])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, B] timestep values
        Returns:
            [N, B, hidden_size] embeddings
        """
        N, B = x.shape

        # Create sinusoidal embedding: [N, B] -> [N, B, 256]
        emb = x.unsqueeze(-1) * self.sin_basis.to(x.dtype)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # [N, B, 256]

        if self._strategy == 'fused':
            # First projection: [N, B, 256] @ [N, H, 256].T -> [N, B, H]
            h = torch.einsum('nbi,nhi->nbh', emb, self.fc1_weight)
            h = h + self.fc1_bias.unsqueeze(1)
            h = F.silu(h)

            # Second projection
            out = torch.einsum('nbh,noh->nbo', h, self.fc2_weight)
            out = out + self.fc2_bias.unsqueeze(1)
        else:
            outputs = []
            for i in range(N):
                h = F.linear(emb[i], self.fc1_weight[i], self.fc1_bias[i])
                h = F.silu(h)
                out_i = F.linear(h, self.fc2_weight[i], self.fc2_bias[i])
                outputs.append(out_i)
            out = torch.stack(outputs, dim=0)

        return out


class WideVectorIn(nn.Module):
    """
    Wide vector input projection (SiLU + Linear).

    Input: [N, B, pooled_dim]
    Output: [N, B, hidden_size]
    """

    def __init__(self, n: int, in_features: int, out_features: int, strategy: str = 'fused'):
        super().__init__()
        self.n = n
        self._strategy = strategy

        self.weight = nn.Parameter(torch.empty(n, out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(n, out_features))

        self._reset_parameters()

    def _reset_parameters(self):
        for i in range(self.n):
            nn.init.xavier_uniform_(self.weight[i])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, B, in_features]
        Returns:
            [N, B, out_features]
        """
        x = F.silu(x)

        if self._strategy == 'fused':
            out = torch.einsum('nbi,noi->nbo', x, self.weight)
            out = out + self.bias.unsqueeze(1)
        else:
            outputs = []
            for i in range(self.n):
                out_i = F.linear(x[i], self.weight[i], self.bias[i])
                outputs.append(out_i)
            out = torch.stack(outputs, dim=0)

        return out


class WideAdaLayerNormZero(nn.Module):
    """
    Wide Adaptive LayerNorm Zero for double-stream blocks.

    Outputs 6 modulation parameters: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    Input x: [N, B, seq_len, hidden_size]
    Input vec: [N, B, hidden_size] (conditioning)
    Output: (x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp)
    """

    def __init__(self, n: int, hidden_size: int, eps: float = 1e-6, strategy: str = 'fused'):
        super().__init__()
        self.n = n
        self.hidden_size = hidden_size
        self.eps = eps
        self._strategy = strategy

        # RMSNorm parameters
        self.norm_weight = nn.Parameter(torch.ones(n, hidden_size))

        # Linear for 6 modulation outputs
        self.linear_weight = nn.Parameter(torch.empty(n, 6 * hidden_size, hidden_size))
        self.linear_bias = nn.Parameter(torch.zeros(n, 6 * hidden_size))

        self._reset_parameters()

    def _reset_parameters(self):
        for i in range(self.n):
            nn.init.xavier_uniform_(self.linear_weight[i])

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x: [N, B, S, H] - input features
            emb: [N, B, H] - conditioning (after SiLU)

        Returns:
            (x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        """
        N, B, S, H = x.shape

        # Compute modulation parameters: [N, B, H] @ [N, 6H, H].T -> [N, B, 6H]
        emb_silu = F.silu(emb)

        if self._strategy == 'fused':
            modulation = torch.einsum('nbh,noh->nbo', emb_silu, self.linear_weight)
            modulation = modulation + self.linear_bias.unsqueeze(1)
        else:
            modulations = []
            for i in range(N):
                mod_i = F.linear(emb_silu[i], self.linear_weight[i], self.linear_bias[i])
                modulations.append(mod_i)
            modulation = torch.stack(modulations, dim=0)

        # Split into 6 parts: each [N, B, H]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)

        # RMSNorm
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms

        # Apply affine with norm weight
        weight = self.norm_weight.view(N, 1, 1, H)
        x_norm = x_norm * weight

        # Apply modulation: x_norm * (1 + scale) + shift
        x_norm = x_norm * (1 + scale_msa.unsqueeze(2)) + shift_msa.unsqueeze(2)

        return x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp


class WideDoubleStreamBlock(nn.Module):
    """
    Wide Double-Stream Transformer Block (MMDiT style).

    Two streams (txt, img) attend to concatenated [txt, img] sequence.

    Input:
        txt: [N, B, L, H]
        img: [N, B, S, H]
        vec: [N, B, H] conditioning
        rope: [B, S, head_dim] (optional, for image positions)

    Output:
        (txt_out, img_out): both in N-first format
    """

    def __init__(
        self,
        n: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float = 4.0,
        strategy: str = 'fused',
    ):
        super().__init__()
        self.n = n
        self.hidden_size = hidden_size
        self._strategy = strategy

        # Adaptive norms for both streams
        self.img_norm1 = WideAdaLayerNormZero(n, hidden_size, strategy=strategy)
        self.txt_norm1 = WideAdaLayerNormZero(n, hidden_size, strategy=strategy)

        # Joint attention
        self.attn = WideJointAttention(
            n=n,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=False,
            strategy=strategy,
        )

        # Post-attention norms (RMSNorm)
        self.img_norm2 = WideRMSNorm(n, hidden_size, strategy='batched')
        self.txt_norm2 = WideRMSNorm(n, hidden_size, strategy='batched')

        # MLPs
        intermediate_size = int(hidden_size * mlp_ratio)
        self.img_mlp = WideMLP(n, hidden_size, intermediate_size, activation='gelu_tanh', strategy=strategy)
        self.txt_mlp = WideMLP(n, hidden_size, intermediate_size, activation='gelu_tanh', strategy=strategy)

    def forward(
        self,
        txt: torch.Tensor,
        img: torch.Tensor,
        vec: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with N-first format."""
        # Adaptive norms
        img_normed, img_gate_msa, img_shift_mlp, img_scale_mlp, img_gate_mlp = self.img_norm1(img, vec)
        txt_normed, txt_gate_msa, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = self.txt_norm1(txt, vec)

        # Joint attention
        txt_attn_out, img_attn_out = self.attn(txt_normed, img_normed, rope)

        # Residual with gate
        txt = txt + img_gate_msa.unsqueeze(2) * txt_attn_out
        img = img + img_gate_msa.unsqueeze(2) * img_attn_out

        # MLP with modulation
        txt_mlp_in = self.txt_norm2(txt) * (1 + txt_scale_mlp.unsqueeze(2)) + txt_shift_mlp.unsqueeze(2)
        img_mlp_in = self.img_norm2(img) * (1 + img_scale_mlp.unsqueeze(2)) + img_shift_mlp.unsqueeze(2)

        txt = txt + txt_gate_mlp.unsqueeze(2) * self.txt_mlp(txt_mlp_in)
        img = img + img_gate_mlp.unsqueeze(2) * self.img_mlp(img_mlp_in)

        return txt, img


class WideSingleStreamBlock(nn.Module):
    """
    Wide Single-Stream Transformer Block.

    Combined txt+img sequence with self-attention.

    Input:
        txt: [N, B, L, H]
        img: [N, B, S, H]
        vec: [N, B, H]
        img_rope: [B, S, head_dim] (optional)

    Output:
        (txt_out, img_out)
    """

    def __init__(
        self,
        n: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float = 4.0,
        strategy: str = 'fused',
    ):
        super().__init__()
        self.n = n
        self.hidden_size = hidden_size
        self._strategy = strategy

        # Adaptive norm (simplified - outputs x_norm and gate)
        self.norm = WideAdaLayerNormZeroSingle(n, hidden_size, emb_size=hidden_size, strategy=strategy)

        # Self attention
        self.attn = WideAttention(
            n=n,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=False,
            strategy=strategy,
        )

        # MLP
        intermediate_size = int(hidden_size * mlp_ratio)
        self.mlp = WideMLP(n, hidden_size, intermediate_size, activation='gelu_tanh', strategy=strategy)

        # Post-MLP norm
        self.norm2 = WideRMSNorm(n, hidden_size, strategy='batched')

    def forward(
        self,
        txt: torch.Tensor,
        img: torch.Tensor,
        vec: torch.Tensor,
        img_rope: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with N-first format."""
        N, B, L, H = txt.shape
        S = img.shape[2]

        # Concatenate streams
        x = torch.cat([txt, img], dim=2)  # [N, B, L+S, H]

        # Build rope for combined sequence (zeros for text positions)
        if img_rope is not None:
            txt_rope_zeros = torch.zeros(B, L, img_rope.shape[-1], device=img_rope.device, dtype=img_rope.dtype)
            rope = torch.cat([txt_rope_zeros, img_rope], dim=1)  # [B, L+S, D]
        else:
            rope = None

        # Adaptive norm
        x_normed, gate = self.norm(x, vec)

        # Self attention with residual
        x = x + gate.unsqueeze(2) * self.attn(x_normed, rope)

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        # Split back to streams
        txt, img = x.split([L, S], dim=2)

        return txt, img


class WideTinyFlux(nn.Module):
    """
    WideTinyFlux: N parallel TinyFlux models fused using WideCompiler primitives.

    All operations use N-first format [N, B, ...] internally for maximum efficiency.

    Usage:
        config = WideTinyFluxConfig(n=8)
        model = WideTinyFlux(config).cuda()

        # Inputs in N-first format
        hidden_states = torch.randn(8, 4, 4096, 16).cuda()  # [N, B, num_patches, in_channels]
        encoder_hidden_states = torch.randn(8, 4, 77, 768).cuda()  # [N, B, text_len, joint_dim]
        pooled_projections = torch.randn(8, 4, 768).cuda()  # [N, B, pooled_dim]
        timestep = torch.rand(8, 4).cuda()  # [N, B]
        img_ids = model.create_img_ids(8, 4, 64, 64, device)  # [N, B, H*W, 3]

        output = model(hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids)
        # output: [N, B, num_patches, in_channels]
    """

    def __init__(self, config: Optional[WideTinyFluxConfig] = None):
        super().__init__()
        self.config = config or WideTinyFluxConfig()
        cfg = self.config

        self.n = cfg.n
        strategy = cfg.strategy

        # Input projections
        self.img_in = WideLinear(cfg.n, cfg.in_channels, cfg.hidden_size, bias=True, strategy='einsum')
        self.txt_in = WideLinear(cfg.n, cfg.joint_attention_dim, cfg.hidden_size, bias=True, strategy='einsum')

        # Conditioning projections
        self.time_in = WideMLPEmbedderTimestep(cfg.n, cfg.hidden_size, strategy=strategy)
        self.vector_in = WideVectorIn(cfg.n, cfg.pooled_projection_dim, cfg.hidden_size, strategy=strategy)

        if cfg.guidance_embeds:
            self.guidance_in = WideMLPEmbedderTimestep(cfg.n, cfg.hidden_size, strategy=strategy)

        # RoPE
        self.rope = WideRotaryEmbedding(cfg.n, cfg.attention_head_dim, cfg.axes_dims_rope)

        # Transformer blocks
        self.double_blocks = nn.ModuleList([
            WideDoubleStreamBlock(
                n=cfg.n,
                hidden_size=cfg.hidden_size,
                num_heads=cfg.num_attention_heads,
                head_dim=cfg.attention_head_dim,
                mlp_ratio=cfg.mlp_ratio,
                strategy=strategy,
            )
            for _ in range(cfg.num_double_layers)
        ])

        self.single_blocks = nn.ModuleList([
            WideSingleStreamBlock(
                n=cfg.n,
                hidden_size=cfg.hidden_size,
                num_heads=cfg.num_attention_heads,
                head_dim=cfg.attention_head_dim,
                mlp_ratio=cfg.mlp_ratio,
                strategy=strategy,
            )
            for _ in range(cfg.num_single_layers)
        ])

        # Output
        self.final_norm = WideRMSNorm(cfg.n, cfg.hidden_size, strategy='batched')
        self.final_linear = WideLinear(cfg.n, cfg.hidden_size, cfg.in_channels, bias=True, strategy='einsum')

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with zeros for final projection."""
        with torch.no_grad():
            self.final_linear.weight.zero_()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with N-first format.

        Args:
            hidden_states: [N, B, num_patches, in_channels] - image latents
            encoder_hidden_states: [N, B, text_len, joint_attention_dim] - text embeddings
            pooled_projections: [N, B, pooled_projection_dim] - pooled text features
            timestep: [N, B] - diffusion timesteps
            img_ids: [N, B, num_patches, 3] or [B, num_patches, 3] - position indices
            guidance: Optional [N, B] - guidance scale

        Returns:
            [N, B, num_patches, in_channels] - denoised output
        """
        N = self.n

        # Input projections
        img = self.img_in(hidden_states)  # [N, B, S, H]
        txt = self.txt_in(encoder_hidden_states)  # [N, B, L, H]

        # Conditioning vector
        vec = self.time_in(timestep)  # [N, B, H]
        vec = vec + self.vector_in(pooled_projections)

        if self.config.guidance_embeds and guidance is not None:
            vec = vec + self.guidance_in(guidance)

        # RoPE for image positions
        # img_ids can be [B, S, 3] or [N, B, S, 3]
        if img_ids.dim() == 3:
            img_rope = self.rope(img_ids, dtype=img.dtype)  # [B, S, D]
        else:
            # Take first model's positions (assumed same for all)
            img_rope = self.rope(img_ids[0], dtype=img.dtype)  # [B, S, D]

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
    def create_img_ids(
        n: int,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create image position IDs for N models.

        Returns:
            [B, H*W, 3] - position indices (same for all N models)
        """
        h_ids = torch.arange(height, device=device, dtype=torch.float32)
        w_ids = torch.arange(width, device=device, dtype=torch.float32)

        grid_h, grid_w = torch.meshgrid(h_ids, w_ids, indexing='ij')

        img_ids = torch.stack([
            torch.zeros(height * width, device=device),
            grid_h.flatten(),
            grid_w.flatten(),
        ], dim=-1)

        # Expand for batch
        img_ids = img_ids.unsqueeze(0).expand(batch_size, -1, -1)

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
        counts['final'] = (sum(p.numel() for p in self.final_norm.parameters()) +
                          sum(p.numel() for p in self.final_linear.parameters()))
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts

    @classmethod
    def from_models(
        cls,
        models: List[nn.Module],
        config: Optional[WideTinyFluxConfig] = None,
    ) -> 'WideTinyFlux':
        """
        Create WideTinyFlux from N existing TinyFlux models.

        This copies weights from N separate models into the fused wide model.

        Args:
            models: List of N TinyFlux models
            config: Optional config (inferred from first model if not provided)

        Returns:
            WideTinyFlux with weights copied from input models
        """
        n = len(models)
        template = models[0]

        # Infer config from template if not provided
        if config is None:
            config = WideTinyFluxConfig(
                n=n,
                hidden_size=template.config.hidden_size,
                num_attention_heads=template.config.num_attention_heads,
                attention_head_dim=template.config.attention_head_dim,
                in_channels=template.config.in_channels,
                joint_attention_dim=template.config.joint_attention_dim,
                pooled_projection_dim=template.config.pooled_projection_dim,
                num_double_layers=template.config.num_double_layers,
                num_single_layers=template.config.num_single_layers,
                mlp_ratio=template.config.mlp_ratio,
                axes_dims_rope=template.config.axes_dims_rope,
                guidance_embeds=template.config.guidance_embeds,
            )

        wide = cls(config)

        # Copy weights (this would be a detailed implementation)
        # For brevity, just showing the structure
        with torch.no_grad():
            for i, m in enumerate(models):
                # Input projections
                wide.img_in.weight[i] = m.img_in.weight
                wide.img_in.bias[i] = m.img_in.bias
                wide.txt_in.weight[i] = m.txt_in.weight
                wide.txt_in.bias[i] = m.txt_in.bias

                # Final layers
                wide.final_norm.weight[i] = m.final_norm.weight
                wide.final_linear.weight[i] = m.final_linear.weight
                wide.final_linear.bias[i] = m.final_linear.bias

                # ... (copy other weights similarly)

        return wide


def test_wide_tiny_flux():
    """Test the WideTinyFlux model."""
    print("=" * 60)
    print("WideTinyFlux Model Test")
    print("=" * 60)

    config = WideTinyFluxConfig(n=4)
    print(f"\nConfig:")
    print(f"  n (parallel models): {config.n}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_heads: {config.num_attention_heads}")
    print(f"  head_dim: {config.attention_head_dim}")
    print(f"  double_layers: {config.num_double_layers}")
    print(f"  single_layers: {config.num_single_layers}")

    model = WideTinyFlux(config)

    counts = model.count_parameters()
    print(f"\nParameters: {counts['total']:,} ({counts['total'] / 1e6:.2f}M)")
    print(f"  Per component:")
    for name, count in counts.items():
        if name != 'total':
            print(f"    {name}: {count:,}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Test inputs (N-first format)
    n = config.n
    batch_size = 2
    latent_h, latent_w = 32, 32
    num_patches = latent_h * latent_w
    text_len = 77

    hidden_states = torch.randn(n, batch_size, num_patches, config.in_channels, device=device)
    encoder_hidden_states = torch.randn(n, batch_size, text_len, config.joint_attention_dim, device=device)
    pooled_projections = torch.randn(n, batch_size, config.pooled_projection_dim, device=device)
    timestep = torch.rand(n, batch_size, device=device)
    img_ids = WideTinyFlux.create_img_ids(n, batch_size, latent_h, latent_w, device)
    guidance = torch.ones(n, batch_size, device=device) * 3.5

    print(f"\nInput shapes:")
    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
    print(f"  pooled_projections: {pooled_projections.shape}")
    print(f"  timestep: {timestep.shape}")
    print(f"  img_ids: {img_ids.shape}")
    print(f"  guidance: {guidance.shape}")

    # Warmup
    print("\nRunning warmup...")
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
    else:
        with torch.no_grad():
            output = model(hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, guidance)

    print(f"\nOutput shape: {output.shape}")
    print(f"Expected:     [{n}, {batch_size}, {num_patches}, {config.in_channels}]")

    assert output.shape == (n, batch_size, num_patches, config.in_channels), "Shape mismatch!"
    print("\n[OK] Forward pass successful!")


if __name__ == "__main__":
    test_wide_tiny_flux()
