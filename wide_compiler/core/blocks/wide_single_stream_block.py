"""
WideSingleStreamBlock - N parallel Flux-style single-stream transformer blocks.

Complete transformer block with adaptive normalization and gated modulation.
Used in the later layers of Flux where text and image are already fused.

Architecture:
    x -> adaptive norm (modulated by vec)
      -> self-attention with RoPE
      -> gated MLP (with modulation gate)
      -> residual connection

Expected speedup: 8-12x (dominated by attention and MLP)

Strategies:
- 'fused': Fused operations throughout (FASTEST)
- 'sequential': N separate blocks (baseline)

Input/Output Format (v0.7.0):
- x:    [N, B, S, hidden_size]  (N-first, fused sequence)
- vec:  [N, B, emb_size]        (N-first, conditioning vector)
- RoPE: Optional [B, S, head_dim]
- Output: [N, B, S, hidden_size]

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .wide_attention import WideAttention
from .wide_mlp import WideMLP
from ..primitives.wide_ada_layer_norm_zero_single import WideAdaLayerNormZeroSingle


class WideSingleStreamBlock(nn.Module):
    """
    N parallel single-stream transformer blocks (Flux-style).

    Single fused sequence with:
    - Adaptive normalization conditioned on vec
    - Self-attention with optional RoPE
    - Gated MLP (gate computed from vec)
    - Residual connections
    """

    def __init__(
        self,
        n: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float = 4.0,
        bias: bool = False,
        strategy: str = 'fused',
    ):
        super().__init__()
        self.n = n
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.mlp_ratio = mlp_ratio
        self._strategy = strategy

        intermediate_size = int(hidden_size * mlp_ratio)

        # Adaptive norm with zero-init (modulated by vec, returns norm + gate)
        self.norm = WideAdaLayerNormZeroSingle(
            n=n,
            hidden_size=hidden_size,
            conditioning_size=hidden_size,
            strategy=strategy,
        )

        # Self-attention
        self.attn = WideAttention(
            n=n,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            strategy=strategy,
        )

        # MLP
        self.mlp = WideMLP(
            n=n,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation='gelu',
            bias=bias,
            strategy=strategy,
        )

    @property
    def strategy(self) -> str:
        return self._strategy

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        rope: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with N-first format.

        Args:
            x: [N, B, S, hidden_size] - fused sequence
            vec: [N, B, emb_size] - conditioning vector (timestep/text embedding)
            rope: Optional [B, S, head_dim] - RoPE

        Returns:
            [N, B, S, hidden_size]
        """
        if self._strategy == 'fused':
            return self._forward_fused(x, vec, rope)
        else:
            return self._forward_sequential(x, vec, rope)

    def _forward_fused(
        self,
        x: Tensor,
        vec: Tensor,
        rope: Optional[Tensor],
    ) -> Tensor:
        """Fused single-stream block."""
        # Adaptive norm with gate
        x_norm, gate = self.norm(x, vec)

        # Self-attention
        attn_out = self.attn(x_norm, rope=rope)

        # MLP
        mlp_out = self.mlp(x_norm)

        # Gated residual: x + gate * (attn + mlp)
        # gate: [N, B, H] -> expand to [N, B, 1, H] for broadcasting
        gate = gate.unsqueeze(2)
        out = x + gate * (attn_out + mlp_out)

        return out

    def _forward_sequential(
        self,
        x: Tensor,
        vec: Tensor,
        rope: Optional[Tensor],
    ) -> Tensor:
        """Sequential single-stream block (baseline)."""
        N, B, S, H = x.shape
        outputs = []

        for i in range(N):
            x_i = x[i]  # [B, S, H]
            vec_i = vec[i]  # [B, E]

            # Adaptive norm with gate (need to add N dim for norm module)
            x_norm_i, gate_i = self.norm._forward_sequential(
                x_i.unsqueeze(0), vec_i.unsqueeze(0)
            )
            x_norm_i = x_norm_i.squeeze(0)  # [B, S, H]
            gate_i = gate_i.squeeze(0)  # [B, H]

            # Self-attention
            attn_out_i = self.attn._forward_sequential(x_norm_i.unsqueeze(0), rope, None)
            attn_out_i = attn_out_i.squeeze(0)  # [B, S, H]

            # MLP
            mlp_out_i = self.mlp._forward_sequential(x_norm_i.unsqueeze(0))
            mlp_out_i = mlp_out_i.squeeze(0)  # [B, S, H]

            # Gated residual
            gate_i = gate_i.unsqueeze(1)  # [B, 1, H]
            out_i = x_i + gate_i * (attn_out_i + mlp_out_i)

            outputs.append(out_i)

        return torch.stack(outputs, dim=0)

    @classmethod
    def from_modules(cls, modules: List[nn.Module], strategy: str = 'fused') -> 'WideSingleStreamBlock':
        """
        Create from N existing SingleStreamBlock modules.

        Expects modules with:
        - .norm: adaptive layer norm (AdaLayerNormZeroSingle)
        - .attn: self-attention module
        - .mlp: MLP module
        - .num_heads, .head_dim: attention config
        """
        n = len(modules)
        t = modules[0]

        hidden_size = t.norm.hidden_size
        num_heads = t.attn.num_heads
        head_dim = t.attn.head_dim

        # Detect MLP ratio
        mlp_ratio = 4.0  # Default
        if hasattr(t, 'mlp'):
            mlp = t.mlp
            if isinstance(mlp, nn.Sequential):
                fc1 = mlp[0]
            else:
                fc1 = getattr(mlp, 'fc1', getattr(mlp, 'linear1', None))
            if fc1 is not None:
                mlp_ratio = fc1.out_features / fc1.in_features

        wide = cls(
            n=n,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            bias=(hasattr(t.attn, 'qkv_bias') and t.attn.qkv_bias is not None),
            strategy=strategy,
        )

        # Copy sub-modules
        with torch.no_grad():
            norm_modules = [m.norm for m in modules]
            attn_modules = [m.attn for m in modules]
            mlp_modules = [m.mlp for m in modules]

            wide.norm = WideAdaLayerNormZeroSingle.from_modules(norm_modules, strategy=strategy)
            wide.attn = WideAttention.from_modules(attn_modules, strategy=strategy)
            wide.mlp = WideMLP.from_modules(mlp_modules, strategy=strategy)

        return wide

    def __repr__(self):
        return (f"WideSingleStreamBlock({self.n}x[hidden={self.hidden_size}, "
                f"heads={self.num_heads}, head_dim={self.head_dim}, "
                f"mlp_ratio={self.mlp_ratio}], strategy={self._strategy})")


__all__ = ['WideSingleStreamBlock']
