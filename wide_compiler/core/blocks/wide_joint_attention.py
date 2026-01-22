"""
WideJointAttention - N parallel Joint Attention blocks (Flux-style MMDiT).

Joint attention over two streams (text and image) where both streams attend
to the concatenated sequence of text + image tokens. This is the core of
multi-modal diffusion transformers (MMDiT) like Flux.

Architecture:
    txt, img -> separate QKV projections
             -> optional RoPE on image Q, K
             -> concatenate K, V from both streams
             -> Flash Attention for txt_q and img_q
             -> separate output projections

Expected speedup: 10-15x (dominated by attention computation)

Strategies:
- 'fused': Fused QKV projections + batched Flash Attention (FASTEST)
- 'sequential': N separate joint attention blocks (baseline)

Input/Output Format (v0.7.0):
- txt:  [N, B, L, hidden_size]  (N-first, text sequence)
- img:  [N, B, S, hidden_size]  (N-first, image sequence)
- RoPE: Optional [B, S, head_dim] (applied to image only)
- Output: (txt_out, img_out) both [N, B, *, hidden_size]

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..primitives.wide_rotary_embedding import apply_rope


class WideJointAttention(nn.Module):
    """
    N parallel Joint Attention blocks for multi-modal transformers.

    Two-stream attention where both text and image attend to the
    concatenated [text, image] sequence.
    """

    def __init__(
        self,
        n: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        bias: bool = False,
        strategy: str = 'fused',
    ):
        super().__init__()
        self.n = n
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self._strategy = strategy

        qkv_dim = 3 * num_heads * head_dim

        # Text QKV projection
        self.txt_qkv_weight = nn.Parameter(torch.empty(n, qkv_dim, hidden_size))
        if bias:
            self.txt_qkv_bias = nn.Parameter(torch.zeros(n, qkv_dim))
        else:
            self.register_parameter('txt_qkv_bias', None)

        # Image QKV projection
        self.img_qkv_weight = nn.Parameter(torch.empty(n, qkv_dim, hidden_size))
        if bias:
            self.img_qkv_bias = nn.Parameter(torch.zeros(n, qkv_dim))
        else:
            self.register_parameter('img_qkv_bias', None)

        # Output projections
        out_in_dim = num_heads * head_dim
        self.txt_out_weight = nn.Parameter(torch.empty(n, hidden_size, out_in_dim))
        self.img_out_weight = nn.Parameter(torch.empty(n, hidden_size, out_in_dim))

        if bias:
            self.txt_out_bias = nn.Parameter(torch.zeros(n, hidden_size))
            self.img_out_bias = nn.Parameter(torch.zeros(n, hidden_size))
        else:
            self.register_parameter('txt_out_bias', None)
            self.register_parameter('img_out_bias', None)

        self._reset_parameters()

    @property
    def strategy(self) -> str:
        return self._strategy

    def _reset_parameters(self):
        """Initialize weights."""
        for i in range(self.n):
            nn.init.xavier_uniform_(self.txt_qkv_weight[i])
            nn.init.xavier_uniform_(self.img_qkv_weight[i])
            nn.init.xavier_uniform_(self.txt_out_weight[i])
            nn.init.xavier_uniform_(self.img_out_weight[i])

    def forward(
        self,
        txt: Tensor,
        img: Tensor,
        rope: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with N-first format.

        Args:
            txt: [N, B, L, hidden_size] - text tokens
            img: [N, B, S, hidden_size] - image tokens
            rope: Optional [B, S, head_dim] - RoPE for image only

        Returns:
            (txt_out, img_out):
                txt_out: [N, B, L, hidden_size]
                img_out: [N, B, S, hidden_size]
        """
        if self._strategy == 'fused':
            return self._forward_fused(txt, img, rope)
        else:
            return self._forward_sequential(txt, img, rope)

    def _forward_fused(
        self,
        txt: Tensor,
        img: Tensor,
        rope: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Fused joint attention."""
        N, B, L, H = txt.shape
        _, _, S, _ = img.shape
        dtype = img.dtype

        txt = txt.to(dtype)
        if rope is not None:
            rope = rope.to(dtype)

        # Text QKV projection: [N, B, L, H] @ [N, 3*heads*head_dim, H].T
        txt_qkv = torch.einsum('nblh,nqh->nblq', txt, self.txt_qkv_weight)
        if self.txt_qkv_bias is not None:
            txt_qkv = txt_qkv + self.txt_qkv_bias.view(N, 1, 1, -1)

        # Image QKV projection: [N, B, S, H] @ [N, 3*heads*head_dim, H].T
        img_qkv = torch.einsum('nbsh,nqh->nbsq', img, self.img_qkv_weight)
        if self.img_qkv_bias is not None:
            img_qkv = img_qkv + self.img_qkv_bias.view(N, 1, 1, -1)

        # Reshape and split: [N, B, *, 3, num_heads, head_dim]
        txt_qkv = txt_qkv.reshape(N, B, L, 3, self.num_heads, self.head_dim)
        img_qkv = img_qkv.reshape(N, B, S, 3, self.num_heads, self.head_dim)

        # Split Q, K, V: each [N, B, num_heads, *, head_dim]
        txt_q, txt_k, txt_v = txt_qkv.permute(3, 0, 1, 4, 2, 5)
        img_q, img_k, img_v = img_qkv.permute(3, 0, 1, 4, 2, 5)

        # Apply RoPE to image only
        if rope is not None:
            img_q_flat = img_q.reshape(N * B * self.num_heads, S, self.head_dim)
            img_k_flat = img_k.reshape(N * B * self.num_heads, S, self.head_dim)

            img_q_flat = apply_rope(img_q_flat, rope)
            img_k_flat = apply_rope(img_k_flat, rope)

            img_q = img_q_flat.reshape(N, B, self.num_heads, S, self.head_dim)
            img_k = img_k_flat.reshape(N, B, self.num_heads, S, self.head_dim)

        # Concatenate K, V for joint attention: [N, B, num_heads, L+S, head_dim]
        k = torch.cat([txt_k, img_k], dim=3)
        v = torch.cat([txt_v, img_v], dim=3)

        # Reshape for batched attention: [N*B, num_heads, *, head_dim]
        txt_q = txt_q.reshape(N * B, self.num_heads, L, self.head_dim)
        img_q = img_q.reshape(N * B, self.num_heads, S, self.head_dim)
        k = k.reshape(N * B, self.num_heads, L + S, self.head_dim)
        v = v.reshape(N * B, self.num_heads, L + S, self.head_dim)

        # Flash Attention for both streams
        txt_out = F.scaled_dot_product_attention(txt_q, k, v, scale=self.scale)
        img_out = F.scaled_dot_product_attention(img_q, k, v, scale=self.scale)

        # Reshape: [N*B, num_heads, *, head_dim] -> [N, B, *, num_heads*head_dim]
        txt_out = txt_out.reshape(N, B, L, self.num_heads * self.head_dim)
        img_out = img_out.reshape(N, B, S, self.num_heads * self.head_dim)

        # Output projections
        txt_out = torch.einsum('nbld,nhd->nblh', txt_out, self.txt_out_weight)
        if self.txt_out_bias is not None:
            txt_out = txt_out + self.txt_out_bias.view(N, 1, 1, -1)

        img_out = torch.einsum('nbsd,nhd->nbsh', img_out, self.img_out_weight)
        if self.img_out_bias is not None:
            img_out = img_out + self.img_out_bias.view(N, 1, 1, -1)

        return txt_out, img_out

    def _forward_sequential(
        self,
        txt: Tensor,
        img: Tensor,
        rope: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Sequential joint attention (baseline)."""
        N, B, L, H = txt.shape
        _, _, S, _ = img.shape
        dtype = img.dtype

        txt = txt.to(dtype)
        if rope is not None:
            rope = rope.to(dtype)

        txt_outputs = []
        img_outputs = []

        for i in range(N):
            txt_i = txt[i]  # [B, L, H]
            img_i = img[i]  # [B, S, H]

            # QKV projections
            txt_qkv = F.linear(txt_i, self.txt_qkv_weight[i].T,
                              self.txt_qkv_bias[i] if self.txt_qkv_bias is not None else None)
            img_qkv = F.linear(img_i, self.img_qkv_weight[i].T,
                              self.img_qkv_bias[i] if self.img_qkv_bias is not None else None)

            # Reshape and split
            txt_qkv = txt_qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
            img_qkv = img_qkv.reshape(B, S, 3, self.num_heads, self.head_dim)

            txt_q, txt_k, txt_v = txt_qkv.permute(2, 0, 3, 1, 4)
            img_q, img_k, img_v = img_qkv.permute(2, 0, 3, 1, 4)

            # Apply RoPE to image
            if rope is not None:
                img_q_flat = img_q.reshape(B * self.num_heads, S, self.head_dim)
                img_k_flat = img_k.reshape(B * self.num_heads, S, self.head_dim)

                img_q_flat = apply_rope(img_q_flat, rope)
                img_k_flat = apply_rope(img_k_flat, rope)

                img_q = img_q_flat.reshape(B, self.num_heads, S, self.head_dim)
                img_k = img_k_flat.reshape(B, self.num_heads, S, self.head_dim)

            # Concatenate K, V
            k = torch.cat([txt_k, img_k], dim=2)
            v = torch.cat([txt_v, img_v], dim=2)

            # Flash Attention
            txt_out_i = F.scaled_dot_product_attention(txt_q, k, v, scale=self.scale)
            img_out_i = F.scaled_dot_product_attention(img_q, k, v, scale=self.scale)

            txt_out_i = txt_out_i.transpose(1, 2).reshape(B, L, -1)
            img_out_i = img_out_i.transpose(1, 2).reshape(B, S, -1)

            # Output projections
            txt_out_i = F.linear(txt_out_i, self.txt_out_weight[i].T,
                                self.txt_out_bias[i] if self.txt_out_bias is not None else None)
            img_out_i = F.linear(img_out_i, self.img_out_weight[i].T,
                                self.img_out_bias[i] if self.img_out_bias is not None else None)

            txt_outputs.append(txt_out_i)
            img_outputs.append(img_out_i)

        return torch.stack(txt_outputs, dim=0), torch.stack(img_outputs, dim=0)

    @classmethod
    def from_modules(cls, modules: List[nn.Module], strategy: str = 'fused') -> 'WideJointAttention':
        """
        Create from N existing JointAttention modules.

        Expects modules with:
        - .txt_qkv, .img_qkv (Linear): QKV projections
        - .txt_out, .img_out (Linear): output projections
        - .num_heads, .head_dim: attention config
        """
        n = len(modules)
        t = modules[0]

        hidden_size = t.txt_qkv.in_features
        num_heads = t.num_heads
        head_dim = t.head_dim

        wide = cls(
            n=n,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=(t.txt_qkv.bias is not None),
            strategy=strategy,
        )

        wide = wide.to(device=t.txt_qkv.weight.device, dtype=t.txt_qkv.weight.dtype)

        # Copy weights
        with torch.no_grad():
            for i, m in enumerate(modules):
                wide.txt_qkv_weight[i] = m.txt_qkv.weight
                wide.img_qkv_weight[i] = m.img_qkv.weight
                wide.txt_out_weight[i] = m.txt_out.weight
                wide.img_out_weight[i] = m.img_out.weight

                if m.txt_qkv.bias is not None:
                    wide.txt_qkv_bias[i] = m.txt_qkv.bias
                    wide.img_qkv_bias[i] = m.img_qkv.bias
                    wide.txt_out_bias[i] = m.txt_out.bias
                    wide.img_out_bias[i] = m.img_out.bias

        return wide

    def __repr__(self):
        return (f"WideJointAttention({self.n}x[hidden={self.hidden_size}, "
                f"heads={self.num_heads}, head_dim={self.head_dim}], "
                f"strategy={self._strategy})")


__all__ = ['WideJointAttention']
