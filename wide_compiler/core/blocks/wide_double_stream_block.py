"""
WideDoubleStreamBlock - N parallel Flux-style double-stream transformer blocks.

Complete MMDiT block with joint attention over text and image streams.
This is the core building block of Flux and similar multi-modal diffusion models.

Architecture (per stream):
    txt, img -> adaptive norm (modulated by vec)
             -> joint attention (both streams attend to txt+img)
             -> MLP
             -> residual connection

Expected speedup: 8-12x (dominated by attention and MLP)

Strategies:
- 'fused': Fused operations throughout (FASTEST)
- 'sequential': N separate blocks (baseline)

Input/Output Format (v0.7.0):
- txt:  [N, B, L, hidden_size]  (N-first, text sequence)
- img:  [N, B, S, hidden_size]  (N-first, image sequence)
- vec:  [N, B, emb_size]        (N-first, conditioning vector)
- RoPE: Optional [B, S, head_dim] (applied to image only)
- Output: (txt_out, img_out) both [N, B, *, hidden_size]

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .wide_joint_attention import WideJointAttention
from .wide_mlp import WideMLP


class WideDoubleStreamBlock(nn.Module):
    """
    N parallel double-stream transformer blocks (Flux-style MMDiT).

    Two streams (text and image) with:
    - Adaptive normalization conditioned on vec
    - Joint attention (both attend to concatenated sequence)
    - Feed-forward MLPs
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

        # Text stream adaptive norm (simple scale + shift from vec)
        self.txt_norm_weight = nn.Parameter(torch.ones(n, hidden_size))
        self.txt_norm_bias = nn.Parameter(torch.zeros(n, hidden_size))
        self.txt_mod_weight = nn.Parameter(torch.empty(n, 2 * hidden_size, hidden_size))
        self.txt_mod_bias = nn.Parameter(torch.zeros(n, 2 * hidden_size))

        # Image stream adaptive norm
        self.img_norm_weight = nn.Parameter(torch.ones(n, hidden_size))
        self.img_norm_bias = nn.Parameter(torch.zeros(n, hidden_size))
        self.img_mod_weight = nn.Parameter(torch.empty(n, 2 * hidden_size, hidden_size))
        self.img_mod_bias = nn.Parameter(torch.zeros(n, 2 * hidden_size))

        # Joint attention
        self.attn = WideJointAttention(
            n=n,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            strategy=strategy,
        )

        # Text MLP
        self.txt_mlp = WideMLP(
            n=n,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation='gelu',
            bias=bias,
            strategy=strategy,
        )

        # Image MLP
        self.img_mlp = WideMLP(
            n=n,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation='gelu',
            bias=bias,
            strategy=strategy,
        )

        self.eps = 1e-6
        self._reset_parameters()

    @property
    def strategy(self) -> str:
        return self._strategy

    def _reset_parameters(self):
        """Initialize weights."""
        for i in range(self.n):
            nn.init.xavier_uniform_(self.txt_mod_weight[i])
            nn.init.xavier_uniform_(self.img_mod_weight[i])

    def forward(
        self,
        txt: Tensor,
        img: Tensor,
        vec: Tensor,
        rope: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with N-first format.

        Args:
            txt: [N, B, L, hidden_size] - text tokens
            img: [N, B, S, hidden_size] - image tokens
            vec: [N, B, emb_size] - conditioning vector (timestep/text embedding)
            rope: Optional [B, S, head_dim] - RoPE for image only

        Returns:
            (txt_out, img_out):
                txt_out: [N, B, L, hidden_size]
                img_out: [N, B, S, hidden_size]
        """
        if self._strategy == 'fused':
            return self._forward_fused(txt, img, vec, rope)
        else:
            return self._forward_sequential(txt, img, vec, rope)

    def _forward_fused(
        self,
        txt: Tensor,
        img: Tensor,
        vec: Tensor,
        rope: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Fused double-stream block."""
        N, B, L, H = txt.shape
        _, _, S, _ = img.shape

        # === Text stream adaptive norm ===
        # Compute modulation parameters from vec
        txt_mod = torch.einsum('nbe,nme->nbm', vec, self.txt_mod_weight)
        txt_mod = txt_mod + self.txt_mod_bias.unsqueeze(1)
        txt_scale, txt_shift = txt_mod.chunk(2, dim=-1)  # Each [N, B, H]

        # LayerNorm
        txt_mean = txt.mean(dim=-1, keepdim=True)
        txt_var = txt.var(dim=-1, keepdim=True, unbiased=False)
        txt_norm = (txt - txt_mean) / torch.sqrt(txt_var + self.eps)

        # Apply learned norm params
        weight_txt = self.txt_norm_weight.view(N, 1, 1, H)
        bias_txt = self.txt_norm_bias.view(N, 1, 1, H)
        txt_norm = txt_norm * weight_txt + bias_txt

        # Apply adaptive modulation
        txt_scale = txt_scale.unsqueeze(2)  # [N, B, 1, H]
        txt_shift = txt_shift.unsqueeze(2)
        txt_norm = txt_norm * (1 + txt_scale) + txt_shift

        # === Image stream adaptive norm ===
        img_mod = torch.einsum('nbe,nme->nbm', vec, self.img_mod_weight)
        img_mod = img_mod + self.img_mod_bias.unsqueeze(1)
        img_scale, img_shift = img_mod.chunk(2, dim=-1)

        img_mean = img.mean(dim=-1, keepdim=True)
        img_var = img.var(dim=-1, keepdim=True, unbiased=False)
        img_norm = (img - img_mean) / torch.sqrt(img_var + self.eps)

        weight_img = self.img_norm_weight.view(N, 1, 1, H)
        bias_img = self.img_norm_bias.view(N, 1, 1, H)
        img_norm = img_norm * weight_img + bias_img

        img_scale = img_scale.unsqueeze(2)
        img_shift = img_shift.unsqueeze(2)
        img_norm = img_norm * (1 + img_scale) + img_shift

        # === Joint attention ===
        txt_attn, img_attn = self.attn(txt_norm, img_norm, rope)

        # === MLPs with residual ===
        txt_out = txt + txt_attn + self.txt_mlp(txt_norm)
        img_out = img + img_attn + self.img_mlp(img_norm)

        return txt_out, img_out

    def _forward_sequential(
        self,
        txt: Tensor,
        img: Tensor,
        vec: Tensor,
        rope: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Sequential double-stream block (baseline)."""
        N, B, L, H = txt.shape
        _, _, S, _ = img.shape

        txt_outputs = []
        img_outputs = []

        for i in range(N):
            txt_i = txt[i]  # [B, L, H]
            img_i = img[i]  # [B, S, H]
            vec_i = vec[i]  # [B, E]

            # Text adaptive norm
            txt_mod_i = F.linear(vec_i, self.txt_mod_weight[i].T, self.txt_mod_bias[i])
            txt_scale_i, txt_shift_i = txt_mod_i.chunk(2, dim=-1)

            txt_mean_i = txt_i.mean(dim=-1, keepdim=True)
            txt_var_i = txt_i.var(dim=-1, keepdim=True, unbiased=False)
            txt_norm_i = (txt_i - txt_mean_i) / torch.sqrt(txt_var_i + self.eps)
            txt_norm_i = txt_norm_i * self.txt_norm_weight[i] + self.txt_norm_bias[i]
            txt_norm_i = txt_norm_i * (1 + txt_scale_i.unsqueeze(1)) + txt_shift_i.unsqueeze(1)

            # Image adaptive norm
            img_mod_i = F.linear(vec_i, self.img_mod_weight[i].T, self.img_mod_bias[i])
            img_scale_i, img_shift_i = img_mod_i.chunk(2, dim=-1)

            img_mean_i = img_i.mean(dim=-1, keepdim=True)
            img_var_i = img_i.var(dim=-1, keepdim=True, unbiased=False)
            img_norm_i = (img_i - img_mean_i) / torch.sqrt(img_var_i + self.eps)
            img_norm_i = img_norm_i * self.img_norm_weight[i] + self.img_norm_bias[i]
            img_norm_i = img_norm_i * (1 + img_scale_i.unsqueeze(1)) + img_shift_i.unsqueeze(1)

            # Joint attention (need to add batch dim back for sub-modules)
            txt_norm_i = txt_norm_i.unsqueeze(0)  # [1, B, L, H]
            img_norm_i = img_norm_i.unsqueeze(0)  # [1, B, S, H]
            txt_attn_i, img_attn_i = self.attn._forward_sequential(txt_norm_i, img_norm_i, rope)
            txt_attn_i = txt_attn_i.squeeze(0)  # [B, L, H]
            img_attn_i = img_attn_i.squeeze(0)  # [B, S, H]

            # MLPs with residual
            txt_norm_i = txt_norm_i.squeeze(0)
            img_norm_i = img_norm_i.squeeze(0)

            txt_mlp_i = self.txt_mlp._forward_sequential(txt_norm_i.unsqueeze(0)).squeeze(0)
            img_mlp_i = self.img_mlp._forward_sequential(img_norm_i.unsqueeze(0)).squeeze(0)

            txt_out_i = txt_i + txt_attn_i + txt_mlp_i
            img_out_i = img_i + img_attn_i + img_mlp_i

            txt_outputs.append(txt_out_i)
            img_outputs.append(img_out_i)

        return torch.stack(txt_outputs, dim=0), torch.stack(img_outputs, dim=0)

    @classmethod
    def from_modules(cls, modules: List[nn.Module], strategy: str = 'fused') -> 'WideDoubleStreamBlock':
        """
        Create from N existing DoubleStreamBlock modules.

        Expects modules with:
        - .txt_norm, .img_norm: normalization layers
        - .txt_mod, .img_mod: modulation linear layers
        - .attn: joint attention module
        - .txt_mlp, .img_mlp: MLP modules
        - .num_heads, .head_dim: attention config
        """
        n = len(modules)
        t = modules[0]

        hidden_size = t.txt_norm.normalized_shape[0]
        num_heads = t.attn.num_heads
        head_dim = t.attn.head_dim

        # Detect MLP ratio
        mlp_ratio = 4.0  # Default
        if hasattr(t, 'txt_mlp'):
            mlp = t.txt_mlp
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
            bias=(t.txt_mod.bias is not None),
            strategy=strategy,
        )

        # Copy weights
        with torch.no_grad():
            for i, m in enumerate(modules):
                # Text norm
                wide.txt_norm_weight[i] = m.txt_norm.weight
                wide.txt_norm_bias[i] = m.txt_norm.bias

                # Image norm
                wide.img_norm_weight[i] = m.img_norm.weight
                wide.img_norm_bias[i] = m.img_norm.bias

                # Text modulation
                wide.txt_mod_weight[i] = m.txt_mod.weight
                if m.txt_mod.bias is not None:
                    wide.txt_mod_bias[i] = m.txt_mod.bias

                # Image modulation
                wide.img_mod_weight[i] = m.img_mod.weight
                if m.img_mod.bias is not None:
                    wide.img_mod_bias[i] = m.img_mod.bias

            # Copy attention and MLP modules
            attn_modules = [m.attn for m in modules]
            txt_mlp_modules = [m.txt_mlp for m in modules]
            img_mlp_modules = [m.img_mlp for m in modules]

            wide.attn = WideJointAttention.from_modules(attn_modules, strategy=strategy)
            wide.txt_mlp = WideMLP.from_modules(txt_mlp_modules, strategy=strategy)
            wide.img_mlp = WideMLP.from_modules(img_mlp_modules, strategy=strategy)

        return wide

    def __repr__(self):
        return (f"WideDoubleStreamBlock({self.n}x[hidden={self.hidden_size}, "
                f"heads={self.num_heads}, head_dim={self.head_dim}, "
                f"mlp_ratio={self.mlp_ratio}], strategy={self._strategy})")


__all__ = ['WideDoubleStreamBlock']
