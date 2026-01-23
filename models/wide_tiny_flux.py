"""
WideTinyFlux: N parallel TinyFlux models fused using WideCompiler.

Uses ONLY the tested primitives and blocks from wide_compiler:
- WideLinear (primitives) - for all linear projections
- WideRMSNorm (primitives) - for normalization
- WideDoubleStreamBlock (blocks) - for double-stream transformer
- WideSingleStreamBlock (blocks) - for single-stream transformer

NO LOOPS in forward pass - all operations are batched via einsum.

The primary API is `WideTinyFlux.from_models()` which fuses N TinyFlux models.

All operations use N-first format [N, B, ...] internally.

Copyright 2025
Apache 2.0 License
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

# Import the actual tested primitives
from wide_compiler.core.primitives import (
    WideLinear,
    WideRMSNorm,
)

# Import the actual tested blocks
from wide_compiler.core.blocks import (
    WideDoubleStreamBlock,
    WideSingleStreamBlock,
)


class WideTinyFlux(nn.Module):
    """
    WideTinyFlux: N parallel TinyFlux models fused using WideCompiler primitives.

    NO LOOPS in forward pass - uses batched einsum operations throughout.

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
        super().__init__()
        self.n = n
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.in_channels = in_channels
        self.guidance_embeds = guidance_embeds
        self._strategy = strategy

        # === Input projections using WideLinear ===
        self.img_in = WideLinear(n, in_channels, hidden_size, bias=True, strategy='einsum')
        self.txt_in = WideLinear(n, joint_attention_dim, hidden_size, bias=True, strategy='einsum')

        # === Time embedding: sinusoidal(256) -> hidden -> hidden ===
        # Using WideLinear for each projection (no loops!)
        self.time_fc1 = WideLinear(n, 256, hidden_size, bias=True, strategy='einsum')
        self.time_fc2 = WideLinear(n, hidden_size, hidden_size, bias=True, strategy='einsum')

        # === Vector projection: pooled_dim -> hidden (with SiLU before) ===
        self.vector_proj = WideLinear(n, pooled_projection_dim, hidden_size, bias=True, strategy='einsum')

        # === Guidance embedding (same structure as time) ===
        if guidance_embeds:
            self.guidance_fc1 = WideLinear(n, 256, hidden_size, bias=True, strategy='einsum')
            self.guidance_fc2 = WideLinear(n, hidden_size, hidden_size, bias=True, strategy='einsum')

        # === Precompute sinusoidal basis (shared, not per-model) ===
        half_dim = 128
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer('sin_basis', emb)

        # === Double-stream blocks using WideDoubleStreamBlock ===
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

        # === Single-stream blocks using WideSingleStreamBlock ===
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

        # === Output using WideRMSNorm and WideLinear ===
        self.final_norm = WideRMSNorm(n, hidden_size, strategy='batched')
        self.final_linear = WideLinear(n, hidden_size, in_channels, bias=True, strategy='einsum')

        # Zero-init final projection
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
        Forward pass with N-first format. NO LOOPS - all batched operations.

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
        # === Input projections (WideLinear - batched einsum) ===
        img = self.img_in(hidden_states)  # [N, B, S, H]
        txt = self.txt_in(encoder_hidden_states)  # [N, B, L, H]

        # === Time embedding (batched, no loops) ===
        # Sinusoidal: [N, B] -> [N, B, 256]
        t_emb = timestep.unsqueeze(-1) * self.sin_basis  # [N, B, 128]
        t_emb = torch.cat([t_emb.sin(), t_emb.cos()], dim=-1)  # [N, B, 256]

        # MLP: WideLinear handles N-first batching
        vec = self.time_fc1(t_emb)  # [N, B, H]
        vec = F.silu(vec)
        vec = self.time_fc2(vec)  # [N, B, H]

        # === Vector projection (batched) ===
        pooled_silu = F.silu(pooled_projections)  # [N, B, pooled_dim]
        vec = vec + self.vector_proj(pooled_silu)  # [N, B, H]

        # === Guidance embedding (batched, no loops) ===
        if self.guidance_embeds and guidance is not None:
            g_emb = guidance.unsqueeze(-1) * self.sin_basis
            g_emb = torch.cat([g_emb.sin(), g_emb.cos()], dim=-1)
            g_vec = self.guidance_fc1(g_emb)
            g_vec = F.silu(g_vec)
            g_vec = self.guidance_fc2(g_vec)
            vec = vec + g_vec

        # RoPE would be computed here from img_ids if needed
        rope = None

        # === Double-stream blocks (WideDoubleStreamBlock - no loops internally) ===
        for block in self.double_blocks:
            txt, img = block(txt, img, vec, rope)

        # === Single-stream blocks (WideSingleStreamBlock - no loops internally) ===
        for block in self.single_blocks:
            N, B, L, H = txt.shape
            S = img.shape[2]
            x = torch.cat([txt, img], dim=2)  # [N, B, L+S, H]
            x = block(x, vec, rope)
            txt, img = x.split([L, S], dim=2)

        # === Output (WideRMSNorm + WideLinear - batched) ===
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

        Uses WideLinear.from_modules() and block.from_modules() to copy weights.
        """
        n = len(models)
        template = models[0]
        cfg = template.config

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

        with torch.no_grad():
            # === Input projections ===
            wide.img_in = WideLinear.from_modules([m.img_in for m in models], strategy='einsum')
            wide.txt_in = WideLinear.from_modules([m.txt_in for m in models], strategy='einsum')

            # === Time embedding MLPs ===
            # TinyFlux has time_in.mlp = Sequential(Linear, SiLU, Linear)
            time_fc1_modules = [m.time_in.mlp[0] for m in models]
            time_fc2_modules = [m.time_in.mlp[2] for m in models]
            wide.time_fc1 = WideLinear.from_modules(time_fc1_modules, strategy='einsum')
            wide.time_fc2 = WideLinear.from_modules(time_fc2_modules, strategy='einsum')

            # === Vector projection ===
            # TinyFlux has vector_in = Sequential(SiLU, Linear)
            vector_proj_modules = [m.vector_in[1] for m in models]
            wide.vector_proj = WideLinear.from_modules(vector_proj_modules, strategy='einsum')

            # === Guidance embedding ===
            if cfg.guidance_embeds:
                guid_fc1_modules = [m.guidance_in.mlp[0] for m in models]
                guid_fc2_modules = [m.guidance_in.mlp[2] for m in models]
                wide.guidance_fc1 = WideLinear.from_modules(guid_fc1_modules, strategy='einsum')
                wide.guidance_fc2 = WideLinear.from_modules(guid_fc2_modules, strategy='einsum')

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
            wide.final_norm = WideRMSNorm.from_modules([m.final_norm for m in models], strategy='batched')
            wide.final_linear = WideLinear.from_modules([m.final_linear for m in models], strategy='einsum')

        return wide

    def __repr__(self):
        return (f"WideTinyFlux(n={self.n}, hidden={self.hidden_size}, "
                f"heads={self.num_heads}, head_dim={self.head_dim}, "
                f"double_blocks={len(self.double_blocks)}, "
                f"single_blocks={len(self.single_blocks)}, "
                f"strategy={self._strategy})")


__all__ = ['WideTinyFlux']
