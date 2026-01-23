"""
WideRotaryEmbedding - N parallel Rotary Position Embeddings with caching.

Rotary Position Embedding (RoPE) encodes positional information by rotating
the query and key representations in the complex plane. More efficient and
effective than absolute position embeddings.

Two modes:
1. WideRotaryEmbedding: Standard batched - all models use same position indices
2. WideRotaryEmbeddingShared: Per-model positions - different position ranges per model

Common in modern transformers (LLaMA, Mistral, Flux, etc.)

Expected speedup: 8-12x (mostly memory bandwidth bound)

Strategies:
- 'batched': Batched rotation (FASTEST)
- 'sequential': N separate applications (baseline)

Input/Output Format (v0.7.0):
- Input:  [N, B, num_heads, seq_len, head_dim]  (N-first)
- Output: [N, B, num_heads, seq_len, head_dim]  (N-first, rotated)

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math


def apply_rope(x: Tensor, rope: Tensor) -> Tensor:
    """
    Apply rotary embeddings to input tensor (efficient pair-wise rotation).

    Args:
        x: [*, num_heads, seq_len, head_dim] - query or key tensor
        rope: [B, seq_len, head_dim] or [1, seq_len, head_dim] - precomputed [cos, sin] pairs

    Returns:
        Rotated tensor with same shape as x
    """
    # x: (*, H, N, D)
    # rope: (B, N, D) where D has [cos, sin] pairs

    *batch_dims, H, N, D = x.shape
    B = rope.shape[0]

    # Expand rope to match batch and heads: (B, 1, N, D)
    rope = rope.to(x.dtype).unsqueeze(1)

    # Split into pairs: (*, H, N, D//2, 2)
    x_pairs = x.reshape(*batch_dims, H, N, D // 2, 2)
    rope_pairs = rope.reshape(B, 1, N, D // 2, 2)

    cos = rope_pairs[..., 0]  # (B, 1, N, D//2)
    sin = rope_pairs[..., 1]  # (B, 1, N, D//2)

    x0 = x_pairs[..., 0]  # (*, H, N, D//2)
    x1 = x_pairs[..., 1]  # (*, H, N, D//2)

    # Rotation: [cos -sin] [x0]
    #           [sin  cos] [x1]
    out0 = x0 * cos - x1 * sin
    out1 = x1 * cos + x0 * sin

    return torch.stack([out0, out1], dim=-1).flatten(-2)


class WideRotaryEmbedding(nn.Module):
    """
    N parallel Rotary Position Embeddings (standard batched mode).

    All N models use the same position indices. This is the standard use case
    for batching multiple forward passes.

    Features:
    - Precomputes and caches frequency tables up to max_seq_len
    - Zero recomputation on repeated sequence lengths
    - Efficient pair-wise rotation implementation
    """

    def __init__(
        self,
        n: int,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        strategy: str = 'batched',
        enable_cache: bool = True,
    ):
        super().__init__()
        self.n = n
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self._strategy = strategy
        self.enable_cache = enable_cache

        # Precompute inverse frequencies: [dim//2]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Cache for precomputed RoPE tensors
        self._rope_cache = {}

        # Precompute full table if caching enabled
        if enable_cache:
            self._precompute_rope_table(max_seq_len)

    @property
    def strategy(self) -> str:
        return self._strategy

    def _precompute_rope_table(self, seq_len: int):
        """Precompute and cache RoPE frequency table."""
        if seq_len in self._rope_cache:
            return

        # Generate positions: [seq_len]
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)

        # Compute frequencies: [seq_len, dim//2]
        freqs = torch.outer(positions, self.inv_freq)

        # Create cos and sin pairs: [seq_len, dim//2] -> [seq_len, dim]
        # Structure: [cos0, cos1, ..., cosD/2, sin0, sin1, ..., sinD/2]
        # Then interleave to: [cos0, sin0, cos1, sin1, ...]
        cos = freqs.cos()
        sin = freqs.sin()

        # Interleave: [seq_len, dim//2, 2] -> [seq_len, dim]
        rope = torch.stack([cos, sin], dim=-1).flatten(-2)  # [seq_len, dim]

        # Add batch dimension: [1, seq_len, dim]
        rope = rope.unsqueeze(0)

        self._rope_cache[seq_len] = rope

    def get_rope(self, seq_len: int, device: torch.device) -> Tensor:
        """Get cached or compute RoPE frequencies."""
        if self.enable_cache and seq_len <= self.max_seq_len:
            if seq_len not in self._rope_cache:
                self._precompute_rope_table(seq_len)
            rope = self._rope_cache[seq_len]
            return rope.to(device)

        # Compute on-the-fly for sequences longer than max_seq_len
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=device)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        cos = freqs.cos()
        sin = freqs.sin()
        rope = torch.stack([cos, sin], dim=-1).flatten(-2).unsqueeze(0)
        return rope

    def forward(self, x: Tensor, seq_len: Optional[int] = None) -> Tensor:
        """
        Apply RoPE to input tensor.

        Args:
            x: [N, B, num_heads, seq_len, head_dim] - query or key

        Returns:
            [N, B, num_heads, seq_len, head_dim] - rotated tensor
        """
        N, B, H, S, D = x.shape
        if seq_len is None:
            seq_len = S

        # Get cached RoPE frequencies: [1, seq_len, dim]
        rope = self.get_rope(seq_len, x.device)

        if self._strategy == 'batched':
            return self._forward_batched(x, rope)
        else:
            return self._forward_sequential(x, rope)

    def _forward_batched(self, x: Tensor, rope: Tensor) -> Tensor:
        """Batched RoPE application."""
        # x: [N, B, H, S, D]
        # rope: [1, S, D]

        N, B, H, S, D = x.shape

        # Reshape to [N*B, H, S, D] for batched rotation
        x_flat = x.reshape(N * B, H, S, D)

        # Apply rotation
        x_rotated = apply_rope(x_flat, rope)

        # Reshape back: [N, B, H, S, D]
        return x_rotated.reshape(N, B, H, S, D)

    def _forward_sequential(self, x: Tensor, rope: Tensor) -> Tensor:
        """Sequential RoPE application (baseline)."""
        N, B, H, S, D = x.shape
        outputs = []

        for i in range(N):
            x_i = x[i]  # [B, H, S, D]
            x_i_rotated = apply_rope(x_i, rope)
            outputs.append(x_i_rotated)

        return torch.stack(outputs, dim=0)

    def __repr__(self):
        return (f"WideRotaryEmbedding({self.n}x[dim={self.dim}, "
                f"max_len={self.max_seq_len}, cache={self.enable_cache}], "
                f"strategy={self._strategy})")


class WideRotaryEmbeddingShared(nn.Module):
    """
    N parallel Rotary Position Embeddings with per-model position ranges.

    Each of the N models can have different position indices, allowing for
    encoding different positional ranges. Useful for:
    - Dividing long sequences among models
    - Encoding different scale/resolution positions
    - Multi-resolution positional encoding

    Features:
    - Precomputes full position table for all N models
    - Caches different position ranges
    - Efficient indexing into precomputed table
    """

    def __init__(
        self,
        n: int,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        strategy: str = 'batched',
        enable_cache: bool = True,
    ):
        super().__init__()
        self.n = n
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self._strategy = strategy
        self.enable_cache = enable_cache

        # Precompute inverse frequencies: [dim//2]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Precompute full position table if caching enabled
        # This covers positions [0, N * max_seq_len] for all models
        if enable_cache:
            self._precompute_full_table()

    @property
    def strategy(self) -> str:
        return self._strategy

    def _precompute_full_table(self):
        """Precompute full RoPE table for all possible positions."""
        max_pos = self.n * self.max_seq_len

        # Generate all positions: [max_pos]
        positions = torch.arange(max_pos, dtype=self.inv_freq.dtype, device=self.inv_freq.device)

        # Compute frequencies: [max_pos, dim//2]
        freqs = torch.outer(positions, self.inv_freq)

        # Create cos and sin pairs
        cos = freqs.cos()
        sin = freqs.sin()

        # Interleave: [max_pos, dim]
        rope = torch.stack([cos, sin], dim=-1).flatten(-2)

        # Cache: [max_pos, dim]
        self.register_buffer('_rope_table', rope, persistent=False)

    def get_rope_for_offsets(
        self,
        seq_len: int,
        position_offsets: Tensor,
        device: torch.device
    ) -> Tensor:
        """
        Get RoPE frequencies for N models with different position offsets.

        Returns:
            [N, 1, seq_len, dim] - RoPE frequencies for each model
        """
        N = position_offsets.shape[0]

        if self.enable_cache and hasattr(self, '_rope_table'):
            # Use precomputed table - vectorized indexing
            # Convert offsets to indices: [N] -> [N, seq_len]
            offsets = position_offsets.long().unsqueeze(1)  # [N, 1]
            seq_indices = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
            indices = offsets + seq_indices  # [N, seq_len]

            # Gather from table: [N, seq_len, dim]
            rope = self._rope_table[indices.flatten()].reshape(N, seq_len, -1).to(device)

            return rope.unsqueeze(1)  # [N, 1, seq_len, dim]

        # Compute on-the-fly (batched)
        # Create position matrix: [N, seq_len]
        offsets = position_offsets.long().unsqueeze(1)  # [N, 1]
        seq_indices = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)  # [1, seq_len]
        positions = offsets + seq_indices  # [N, seq_len]

        # Compute frequencies: [N, seq_len, dim//2]
        positions_float = positions.to(dtype=self.inv_freq.dtype)
        inv_freq_expanded = self.inv_freq.to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, dim//2]
        freqs = positions_float.unsqueeze(-1) * inv_freq_expanded  # [N, seq_len, dim//2]

        # Create cos and sin pairs
        cos = freqs.cos()  # [N, seq_len, dim//2]
        sin = freqs.sin()  # [N, seq_len, dim//2]

        # Interleave: [N, seq_len, dim]
        rope = torch.stack([cos, sin], dim=-1).flatten(-2)

        return rope.unsqueeze(1)  # [N, 1, seq_len, dim]

    def forward(
        self,
        x: Tensor,
        position_offsets: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply RoPE with per-model position offsets.

        Args:
            x: [N, B, num_heads, seq_len, head_dim]
            position_offsets: [N] tensor of position offsets for each model
                            If None, uses [0, seq_len, 2*seq_len, ...]

        Returns:
            [N, B, num_heads, seq_len, head_dim] - rotated with per-model positions
        """
        N, B, H, S, D = x.shape

        # Default offsets: divide position space among N models
        if position_offsets is None:
            position_offsets = torch.arange(N, device=x.device) * S

        # Get RoPE frequencies for each model: [N, 1, S, D]
        rope = self.get_rope_for_offsets(S, position_offsets, x.device)

        if self._strategy == 'batched':
            return self._forward_batched(x, rope)
        else:
            return self._forward_sequential(x, rope)

    def _forward_batched(self, x: Tensor, rope: Tensor) -> Tensor:
        """Batched RoPE with per-model offsets."""
        # x: [N, B, H, S, D]
        # rope: [N, 1, S, D]

        N, B, H, S, D = x.shape

        # Reshape to [N*B, H, S, D] for batched rotation
        x_flat = x.reshape(N * B, H, S, D)

        # Expand rope to match: [N, 1, S, D] -> [N*B, S, D]
        # Each model's rope repeats B times
        rope_expanded = rope.repeat(1, B, 1, 1).reshape(N * B, S, D)

        # Apply batched rotation
        x_rotated = apply_rope(x_flat, rope_expanded)

        # Reshape back: [N, B, H, S, D]
        return x_rotated.reshape(N, B, H, S, D)

    def _forward_sequential(self, x: Tensor, rope: Tensor) -> Tensor:
        """Sequential RoPE (same as batched for this case)."""
        return self._forward_batched(x, rope)

    def __repr__(self):
        return (f"WideRotaryEmbeddingShared({self.n}x[dim={self.dim}, "
                f"max_len={self.max_seq_len}, cache={self.enable_cache}], "
                f"strategy={self._strategy})")


class WideRotaryEmbedding3D(nn.Module):
    """
    N parallel 3-axis Rotary Position Embeddings for Flux-style models.

    Encodes positions along 3 axes (typically time, height, width) for image/video.
    Each axis has its own frequency band.

    Input: img_ids [B, num_patches, 3] - shared across all N models
    Output: [B, num_patches, dim] - Shared across N models (blocks will broadcast)

    This is different from standard RoPE:
    - Standard RoPE: 1D positions → [seq_len, dim]
    - 3D RoPE (Flux): 3D positions [t, h, w] → [num_patches, dim]

    Strategies:
    - 'batched': Single computation shared across N (FASTEST)
    - 'sequential': N identical copies (baseline, for validation)
    """

    def __init__(
        self,
        n: int,
        dim: int,
        axes_dims: Tuple[int, int, int] = (16, 56, 56),
        theta: float = 10000.0,
        strategy: str = 'batched',
    ):
        """
        Initialize 3-axis RoPE.

        Args:
            n: Number of parallel models
            dim: Total embedding dimension (must equal sum of axes_dims)
            axes_dims: Dimensions for each axis (time, height, width)
            theta: Base for frequency computation
            strategy: 'batched' or 'sequential'
        """
        super().__init__()
        self.n = n
        self.dim = dim
        self.axes_dims = axes_dims
        self.theta = theta
        self._strategy = strategy

        # Precompute frequencies for each axis (no runtime loops)
        # Each axis produces axis_dim frequencies, concatenated to produce dim total
        for i, axis_dim in enumerate(axes_dims):
            freqs = 1.0 / (theta ** (torch.arange(0, axis_dim, 2).float() / axis_dim))
            self.register_buffer(f'freqs_{i}', freqs, persistent=False)

    @property
    def strategy(self) -> str:
        return self._strategy

    def forward(self, ids: Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
        """
        Compute 3-axis rotary embeddings.

        Args:
            ids: [B, num_patches, 3] - position indices (t, h, w) for each patch
            dtype: Output dtype

        Returns:
            [B, num_patches, dim] - RoPE embeddings (shared across all N models)
        """
        if self._strategy == 'batched':
            return self._forward_batched(ids, dtype)
        return self._forward_sequential(ids, dtype)

    def _forward_batched(self, ids: Tensor, dtype: torch.dtype) -> Tensor:
        """Batched 3-axis RoPE (shared across N models)."""
        B, num_patches, _ = ids.shape

        # Extract positions for each axis
        pos0 = ids[:, :, 0:1].float()  # [B, num_patches, 1]
        pos1 = ids[:, :, 1:2].float()
        pos2 = ids[:, :, 2:3].float()

        # Compute angles for each axis (broadcasting)
        # [B, num_patches, 1] * [axis_dim/2] -> [B, num_patches, axis_dim/2]
        angles0 = pos0 * self.freqs_0
        angles1 = pos1 * self.freqs_1
        angles2 = pos2 * self.freqs_2

        # Stack sin/cos and flatten for each axis
        # [B, num_patches, axis_dim/2, 2] -> [B, num_patches, axis_dim]
        emb0 = torch.stack([angles0.cos(), angles0.sin()], dim=-1).flatten(-2)
        emb1 = torch.stack([angles1.cos(), angles1.sin()], dim=-1).flatten(-2)
        emb2 = torch.stack([angles2.cos(), angles2.sin()], dim=-1).flatten(-2)

        # Concatenate all axes: [B, num_patches, dim]
        rope = torch.cat([emb0, emb1, emb2], dim=-1).to(dtype)

        # Return as-is: [B, num_patches, dim]
        # All N models share the same rope (blocks will broadcast)
        return rope

    def _forward_sequential(self, ids: Tensor, dtype: torch.dtype) -> Tensor:
        """Sequential (same as batched since rope is shared across N)."""
        return self._forward_batched(ids, dtype)

    @classmethod
    def from_modules(cls, modules: List[nn.Module], strategy: str = 'batched') -> 'WideRotaryEmbedding3D':
        """
        Create from N existing RotaryEmbedding modules.

        Expects modules with:
        - .dim attribute
        - .axes_dims attribute
        - .theta attribute
        """
        n = len(modules)
        t = modules[0]

        wide = cls(
            n=n,
            dim=t.dim,
            axes_dims=t.axes_dims,
            theta=t.theta,
            strategy=strategy,
        )

        # Copy frequency buffers from first module (all should be identical)
        with torch.no_grad():
            for i in range(len(t.axes_dims)):
                if hasattr(t, f'freqs_{i}'):
                    wide_freqs = getattr(wide, f'freqs_{i}')
                    module_freqs = getattr(t, f'freqs_{i}')
                    wide_freqs.copy_(module_freqs)

        return wide

    def __repr__(self):
        return (f"WideRotaryEmbedding3D({self.n}x[dim={self.dim}, "
                f"axes={self.axes_dims}, theta={self.theta}], "
                f"strategy={self._strategy})")


__all__ = ['WideRotaryEmbedding', 'WideRotaryEmbeddingShared', 'WideRotaryEmbedding3D', 'apply_rope']
