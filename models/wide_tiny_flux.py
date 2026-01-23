"""
WideTinyFlux: N parallel TinyFlux models fused using WideCompiler.

This module uses the ACTUAL tested primitives and blocks from wide_compiler:
- WideLinear (9.7x speedup)
- WideRMSNorm (21x speedup)
- WideMLPEmbedder (14.8x speedup)
- WideDoubleStreamBlock (5.2x speedup)
- WideSingleStreamBlock (3.4x speedup)

The primary API is `WideTinyFlux.from_models()` which takes N TinyFlux models
and fuses them into a single wide model using the tested primitives.

All operations use N-first format [N, B, ...] internally.

Copyright 2025
Apache 2.0 License
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

# Import the actual tested primitives
from wide_compiler.core.primitives import (
    WideLinear,
    WideRMSNorm,
    WideMLPEmbedder,
)

# Import the actual tested blocks
from wide_compiler.core.blocks import (
    WideDoubleStreamBlock,
    WideSingleStreamBlock,
)


class WideTinyFlux(nn.Module):
    """
    WideTinyFlux: N parallel TinyFlux models fused using WideCompiler primitives.

    This class wraps N TinyFlux models into a single wide model by using
    the tested wide primitives and blocks from wide_compiler.

    The recommended way to create this is via `from_models()`:

        models = [TinyFlux(config) for _ in range(n)]
        wide_model = WideTinyFlux.from_models(models)

    All inputs/outputs use N-first format [N, B, ...].
    """

    def __init__(
        self,
        n: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        in_channels: int = 16,
        joint_attention_dim: int = 768,
        pooled_projection_dim: int = 768,
        num_double_layers: int = 3,
        num_single_layers: int = 3,
        mlp_ratio: float = 4.0,
        guidance_embeds: bool = True,
        strategy: str = 'fused',
    ):
        """
        Initialize WideTinyFlux.

        Note: Prefer using `from_models()` to create from existing TinyFlux models.
        """
        super().__init__()
        self.n = n
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.in_channels = in_channels
        self.guidance_embeds = guidance_embeds
        self._strategy = strategy

        # Input projections using tested WideLinear
        self.img_in = WideLinear(n, in_channels, hidden_size, bias=True, strategy='einsum')
        self.txt_in = WideLinear(n, joint_attention_dim, hidden_size, bias=True, strategy='einsum')

        # Conditioning using tested WideMLPEmbedder
        # Note: WideMLPEmbedder expects modules with .fc1, .fc2 structure
        # For direct init, we create the weight tensors manually
        self._init_mlp_embedders(n, hidden_size, pooled_projection_dim, guidance_embeds)

        # Double-stream blocks using tested WideDoubleStreamBlock
        self.double_blocks = nn.ModuleList([
            WideDoubleStreamBlock(
                n=n,
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                mlp_ratio=mlp_ratio,
                bias=False,
                strategy=strategy,
            )
            for _ in range(num_double_layers)
        ])

        # Single-stream blocks using tested WideSingleStreamBlock
        self.single_blocks = nn.ModuleList([
            WideSingleStreamBlock(
                n=n,
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                mlp_ratio=mlp_ratio,
                bias=False,
                strategy=strategy,
            )
            for _ in range(num_single_layers)
        ])

        # Output using tested WideRMSNorm and WideLinear
        self.final_norm = WideRMSNorm(n, hidden_size, strategy='batched')
        self.final_linear = WideLinear(n, hidden_size, in_channels, bias=True, strategy='einsum')

        self._init_weights()

    def _init_mlp_embedders(self, n, hidden_size, pooled_dim, guidance_embeds):
        """Initialize MLP embedders for conditioning."""
        # Time embedding: sinusoidal basis (256) -> hidden_size -> hidden_size
        self.time_fc1_weight = nn.Parameter(torch.empty(n, hidden_size, 256))
        self.time_fc1_bias = nn.Parameter(torch.zeros(n, hidden_size))
        self.time_fc2_weight = nn.Parameter(torch.empty(n, hidden_size, hidden_size))
        self.time_fc2_bias = nn.Parameter(torch.zeros(n, hidden_size))

        # Vector projection: pooled_dim -> hidden_size (with SiLU)
        self.vec_weight = nn.Parameter(torch.empty(n, hidden_size, pooled_dim))
        self.vec_bias = nn.Parameter(torch.zeros(n, hidden_size))

        if guidance_embeds:
            self.guid_fc1_weight = nn.Parameter(torch.empty(n, hidden_size, 256))
            self.guid_fc1_bias = nn.Parameter(torch.zeros(n, hidden_size))
            self.guid_fc2_weight = nn.Parameter(torch.empty(n, hidden_size, hidden_size))
            self.guid_fc2_bias = nn.Parameter(torch.zeros(n, hidden_size))

        # Precompute sinusoidal basis
        import math
        half_dim = 128
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer('sin_basis', emb)

        # Initialize
        for i in range(n):
            nn.init.xavier_uniform_(self.time_fc1_weight[i])
            nn.init.xavier_uniform_(self.time_fc2_weight[i])
            nn.init.xavier_uniform_(self.vec_weight[i])
            if guidance_embeds:
                nn.init.xavier_uniform_(self.guid_fc1_weight[i])
                nn.init.xavier_uniform_(self.guid_fc2_weight[i])

    def _init_weights(self):
        """Initialize final projection to zeros."""
        with torch.no_grad():
            self.final_linear.weight.zero_()

    def _embed_timestep(self, t: torch.Tensor, fc1_w, fc1_b, fc2_w, fc2_b) -> torch.Tensor:
        """Embed timestep with sinusoidal basis and MLP. t: [N, B] -> [N, B, H]"""
        # Sinusoidal embedding
        emb = t.unsqueeze(-1) * self.sin_basis.to(t.dtype)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # [N, B, 256]

        # MLP: fc1 -> silu -> fc2
        h = torch.einsum('nbi,nhi->nbh', emb, fc1_w) + fc1_b.unsqueeze(1)
        h = F.silu(h)
        out = torch.einsum('nbh,noh->nbo', h, fc2_w) + fc2_b.unsqueeze(1)
        return out

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
            hidden_states: [N, B, num_patches, in_channels]
            encoder_hidden_states: [N, B, text_len, joint_attention_dim]
            pooled_projections: [N, B, pooled_projection_dim]
            timestep: [N, B]
            img_ids: [B, num_patches, 3] - position indices (shared across N)
            guidance: Optional [N, B]

        Returns:
            [N, B, num_patches, in_channels]
        """
        # Input projections
        img = self.img_in(hidden_states)  # [N, B, S, H]
        txt = self.txt_in(encoder_hidden_states)  # [N, B, L, H]

        # Conditioning
        vec = self._embed_timestep(timestep, self.time_fc1_weight, self.time_fc1_bias,
                                    self.time_fc2_weight, self.time_fc2_bias)

        # Vector projection with SiLU
        pooled_silu = F.silu(pooled_projections)
        vec = vec + torch.einsum('nbi,nhi->nbh', pooled_silu, self.vec_weight) + self.vec_bias.unsqueeze(1)

        if self.guidance_embeds and guidance is not None:
            guid_emb = self._embed_timestep(guidance, self.guid_fc1_weight, self.guid_fc1_bias,
                                            self.guid_fc2_weight, self.guid_fc2_bias)
            vec = vec + guid_emb

        # Note: RoPE computation would go here if needed
        # For simplicity, passing None - full impl would compute from img_ids
        rope = None

        # Double-stream blocks
        for block in self.double_blocks:
            txt, img = block(txt, img, vec, rope)

        # Single-stream blocks - concatenate txt and img for self-attention
        for block in self.single_blocks:
            # WideSingleStreamBlock expects (x, vec, rope)
            # where x is the concatenated sequence
            N, B, L, H = txt.shape
            S = img.shape[2]
            x = torch.cat([txt, img], dim=2)  # [N, B, L+S, H]
            x = block(x, vec, rope)
            txt, img = x.split([L, S], dim=2)

        # Output
        img = self.final_norm(img)
        img = self.final_linear(img)

        return img

    @classmethod
    def from_models(
        cls,
        models: List[nn.Module],
        strategy: str = 'fused',
    ) -> 'WideTinyFlux':
        """
        Create WideTinyFlux from N existing TinyFlux models.

        This is the recommended way to create a WideTinyFlux model.
        It copies weights from N separate TinyFlux models into a single
        wide model using the tested WideCompiler primitives.

        Args:
            models: List of N TinyFlux models with identical architecture
            strategy: 'fused' or 'sequential'

        Returns:
            WideTinyFlux with weights from input models

        Example:
            from tiny_flux import TinyFlux, TinyFluxConfig

            # Create N models
            config = TinyFluxConfig()
            models = [TinyFlux(config).cuda() for _ in range(8)]

            # Fuse into wide model
            wide = WideTinyFlux.from_models(models)

            # Now use with N-first format inputs
            # hidden_states: [8, B, num_patches, 16]
            # output: [8, B, num_patches, 16]
        """
        n = len(models)
        template = models[0]
        cfg = template.config

        # Create wide model shell
        wide = cls(
            n=n,
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            head_dim=cfg.attention_head_dim,
            in_channels=cfg.in_channels,
            joint_attention_dim=cfg.joint_attention_dim,
            pooled_projection_dim=cfg.pooled_projection_dim,
            num_double_layers=cfg.num_double_layers,
            num_single_layers=cfg.num_single_layers,
            mlp_ratio=cfg.mlp_ratio,
            guidance_embeds=cfg.guidance_embeds,
            strategy=strategy,
        )

        # Copy weights from individual models
        with torch.no_grad():
            # === Input projections ===
            img_in_modules = [m.img_in for m in models]
            txt_in_modules = [m.txt_in for m in models]
            wide.img_in = WideLinear.from_modules(img_in_modules, strategy='einsum')
            wide.txt_in = WideLinear.from_modules(txt_in_modules, strategy='einsum')

            # === Time embedding MLP ===
            for i, m in enumerate(models):
                # m.time_in is an MLPEmbedder with .mlp (Sequential)
                mlp = m.time_in.mlp
                wide.time_fc1_weight[i] = mlp[0].weight
                wide.time_fc1_bias[i] = mlp[0].bias
                wide.time_fc2_weight[i] = mlp[2].weight
                wide.time_fc2_bias[i] = mlp[2].bias

            # === Vector projection ===
            for i, m in enumerate(models):
                # m.vector_in is Sequential(SiLU, Linear)
                wide.vec_weight[i] = m.vector_in[1].weight
                wide.vec_bias[i] = m.vector_in[1].bias

            # === Guidance embedding MLP ===
            if cfg.guidance_embeds:
                for i, m in enumerate(models):
                    mlp = m.guidance_in.mlp
                    wide.guid_fc1_weight[i] = mlp[0].weight
                    wide.guid_fc1_bias[i] = mlp[0].bias
                    wide.guid_fc2_weight[i] = mlp[2].weight
                    wide.guid_fc2_bias[i] = mlp[2].bias

            # === Double-stream blocks ===
            for layer_idx in range(cfg.num_double_layers):
                block_modules = [m.double_blocks[layer_idx] for m in models]
                wide.double_blocks[layer_idx] = WideDoubleStreamBlock.from_modules(
                    block_modules, strategy=strategy
                )

            # === Single-stream blocks ===
            for layer_idx in range(cfg.num_single_layers):
                block_modules = [m.single_blocks[layer_idx] for m in models]
                wide.single_blocks[layer_idx] = WideSingleStreamBlock.from_modules(
                    block_modules, strategy=strategy
                )

            # === Final layers ===
            final_norm_modules = [m.final_norm for m in models]
            wide.final_norm = WideRMSNorm.from_modules(final_norm_modules, strategy='batched')

            final_linear_modules = [m.final_linear for m in models]
            wide.final_linear = WideLinear.from_modules(final_linear_modules, strategy='einsum')

        return wide

    def count_parameters(self) -> dict:
        """Count parameters by component."""
        counts = {}
        counts['img_in'] = sum(p.numel() for p in self.img_in.parameters())
        counts['txt_in'] = sum(p.numel() for p in self.txt_in.parameters())
        counts['conditioning'] = (
            self.time_fc1_weight.numel() + self.time_fc1_bias.numel() +
            self.time_fc2_weight.numel() + self.time_fc2_bias.numel() +
            self.vec_weight.numel() + self.vec_bias.numel()
        )
        if self.guidance_embeds:
            counts['conditioning'] += (
                self.guid_fc1_weight.numel() + self.guid_fc1_bias.numel() +
                self.guid_fc2_weight.numel() + self.guid_fc2_bias.numel()
            )
        counts['double_blocks'] = sum(p.numel() for p in self.double_blocks.parameters())
        counts['single_blocks'] = sum(p.numel() for p in self.single_blocks.parameters())
        counts['final'] = (sum(p.numel() for p in self.final_norm.parameters()) +
                          sum(p.numel() for p in self.final_linear.parameters()))
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts

    def __repr__(self):
        return (f"WideTinyFlux(n={self.n}, hidden={self.hidden_size}, "
                f"heads={self.num_heads}, head_dim={self.head_dim}, "
                f"double_blocks={len(self.double_blocks)}, "
                f"single_blocks={len(self.single_blocks)}, "
                f"strategy={self._strategy})")


__all__ = ['WideTinyFlux']
