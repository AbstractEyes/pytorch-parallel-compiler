"""
WideAttention - N parallel Flash Attention blocks with optional RoPE.

Multi-head attention using Flash Attention (SDPA) with optional Rotary
Position Embeddings. This is a composite block that combines:
- QKV projection (WideLinear)
- Optional RoPE application
- Flash Attention (scaled_dot_product_attention)
- Output projection (WideLinear)

Common in modern transformers (LLaMA, Mistral, Flux).

Expected speedup: 10-15x (dominated by attention computation)

Strategies:
- 'fused': Fused QKV projection + batched Flash Attention (FASTEST)
- 'sequential': N separate attention blocks (baseline)

Input/Output Format (v0.7.0):
- Input:  [N, B, seq_len, hidden_size]  (N-first)
- RoPE:   Optional [B, seq_len, head_dim] or None
- Mask:   Optional attention mask
- Output: [N, B, seq_len, hidden_size]  (N-first)

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Optional, Tuple, Callable

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..primitives.wide_rotary_embedding import apply_rope


class WideAttention(nn.Module):
    """
    N parallel Multi-Head Attention blocks with Flash Attention and optional RoPE.

    Architecture:
        x -> QKV projection -> split to Q, K, V
          -> optional RoPE on Q, K
          -> Flash Attention (SDPA)
          -> output projection
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

        # QKV projection: hidden_size -> 3 * num_heads * head_dim
        qkv_dim = 3 * num_heads * head_dim
        self.qkv_weight = nn.Parameter(torch.empty(n, qkv_dim, hidden_size))
        if bias:
            self.qkv_bias = nn.Parameter(torch.zeros(n, qkv_dim))
        else:
            self.register_parameter('qkv_bias', None)

        # Output projection: num_heads * head_dim -> hidden_size
        out_in_dim = num_heads * head_dim
        self.out_weight = nn.Parameter(torch.empty(n, hidden_size, out_in_dim))
        if bias:
            self.out_bias = nn.Parameter(torch.zeros(n, hidden_size))
        else:
            self.register_parameter('out_bias', None)

        self._reset_parameters()

    @property
    def strategy(self) -> str:
        return self._strategy

    def _reset_parameters(self):
        """Initialize weights."""
        for i in range(self.n):
            nn.init.xavier_uniform_(self.qkv_weight[i])
            nn.init.xavier_uniform_(self.out_weight[i])

    def forward(
        self,
        x: Tensor,
        rope: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with N-first format.

        Args:
            x: [N, B, seq_len, hidden_size]
            rope: Optional [B, seq_len, head_dim] or [seq_len, head_dim]
            mask: Optional attention mask

        Returns:
            [N, B, seq_len, hidden_size]
        """
        if self._strategy == 'fused':
            return self._forward_fused(x, rope, mask)
        else:
            return self._forward_sequential(x, rope, mask)

    def _forward_fused(
        self,
        x: Tensor,
        rope: Optional[Tensor],
        mask: Optional[Tensor],
    ) -> Tensor:
        """Fused attention via batched operations."""
        N, B, S, H = x.shape
        dtype = x.dtype

        if rope is not None:
            rope = rope.to(dtype)

        # QKV projection: [N, B, S, H] @ [N, 3*heads*head_dim, H].T -> [N, B, S, 3*heads*head_dim]
        qkv = torch.einsum('nbsh,nqh->nbsq', x, self.qkv_weight)
        if self.qkv_bias is not None:
            qkv = qkv + self.qkv_bias.view(N, 1, 1, -1)

        # Reshape: [N, B, S, 3, num_heads, head_dim]
        qkv = qkv.reshape(N, B, S, 3, self.num_heads, self.head_dim)

        # Split Q, K, V: each [N, B, num_heads, S, head_dim]
        q, k, v = qkv.permute(3, 0, 1, 4, 2, 5)

        # Apply RoPE if provided
        if rope is not None:
            # Reshape for rope: [N*B*num_heads, S, head_dim]
            q_flat = q.reshape(N * B * self.num_heads, S, self.head_dim)
            k_flat = k.reshape(N * B * self.num_heads, S, self.head_dim)

            q_flat = apply_rope(q_flat, rope)
            k_flat = apply_rope(k_flat, rope)

            q = q_flat.reshape(N, B, self.num_heads, S, self.head_dim)
            k = k_flat.reshape(N, B, self.num_heads, S, self.head_dim)

        # Flash Attention: [N*B, num_heads, S, head_dim]
        q = q.reshape(N * B, self.num_heads, S, self.head_dim)
        k = k.reshape(N * B, self.num_heads, S, self.head_dim)
        v = v.reshape(N * B, self.num_heads, S, self.head_dim)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)

        # Reshape: [N*B, num_heads, S, head_dim] -> [N, B, S, num_heads*head_dim]
        out = out.reshape(N, B, S, self.num_heads * self.head_dim)

        # Output projection: [N, B, S, num_heads*head_dim] @ [N, hidden, num_heads*head_dim].T
        out = torch.einsum('nbsd,nhd->nbsh', out, self.out_weight)
        if self.out_bias is not None:
            out = out + self.out_bias.view(N, 1, 1, -1)

        return out

    def _forward_sequential(
        self,
        x: Tensor,
        rope: Optional[Tensor],
        mask: Optional[Tensor],
    ) -> Tensor:
        """Sequential attention (baseline)."""
        N, B, S, H = x.shape
        dtype = x.dtype
        outputs = []

        if rope is not None:
            rope = rope.to(dtype)

        for i in range(N):
            x_i = x[i]  # [B, S, H]

            # QKV projection
            qkv = F.linear(x_i, self.qkv_weight[i].T, self.qkv_bias[i] if self.qkv_bias is not None else None)
            qkv = qkv.reshape(B, S, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3 x [B, num_heads, S, head_dim]

            # Apply RoPE
            if rope is not None:
                q_flat = q.reshape(B * self.num_heads, S, self.head_dim)
                k_flat = k.reshape(B * self.num_heads, S, self.head_dim)

                q_flat = apply_rope(q_flat, rope)
                k_flat = apply_rope(k_flat, rope)

                q = q_flat.reshape(B, self.num_heads, S, self.head_dim)
                k = k_flat.reshape(B, self.num_heads, S, self.head_dim)

            # Flash Attention
            out_i = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)
            out_i = out_i.transpose(1, 2).reshape(B, S, -1)

            # Output projection
            out_i = F.linear(out_i, self.out_weight[i].T, self.out_bias[i] if self.out_bias is not None else None)

            outputs.append(out_i)

        return torch.stack(outputs, dim=0)

    @classmethod
    def from_modules(cls, modules: List[nn.Module], strategy: str = 'fused') -> 'WideAttention':
        """
        Create from N existing Attention modules.

        Expects modules with:
        - .qkv (Linear): QKV projection
        - .out_proj (Linear): output projection
        - .num_heads, .head_dim: attention config
        """
        n = len(modules)
        t = modules[0]

        hidden_size = t.qkv.in_features
        num_heads = t.num_heads
        head_dim = t.head_dim

        wide = cls(
            n=n,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=(t.qkv.bias is not None),
            strategy=strategy,
        )

        wide = wide.to(device=t.qkv.weight.device, dtype=t.qkv.weight.dtype)

        # Copy weights
        with torch.no_grad():
            for i, m in enumerate(modules):
                wide.qkv_weight[i] = m.qkv.weight
                if m.qkv.bias is not None:
                    wide.qkv_bias[i] = m.qkv.bias

                wide.out_weight[i] = m.out_proj.weight
                if m.out_proj.bias is not None:
                    wide.out_bias[i] = m.out_proj.bias

        return wide

    def __repr__(self):
        return (f"WideAttention({self.n}x[hidden={self.hidden_size}, "
                f"heads={self.num_heads}, head_dim={self.head_dim}], "
                f"strategy={self._strategy})")

    # =========================================================================
    # BENCHMARK INTERFACE
    # =========================================================================

    BENCHMARK_STRATEGIES = ['baseline', 'fused', 'sequential']
    BENCHMARK_SWEEPS = {}
    _SWEEPS_INITIALIZED = False

    @classmethod
    def _init_benchmark_sweeps(cls):
        """Initialize sweep configs (called once)."""
        if cls._SWEEPS_INITIALIZED:
            return
        cls._SWEEPS_INITIALIZED = True

        try:
            from ..benchmark.benchmark_schema import SweepParams
        except ImportError:
            return

        cls.BENCHMARK_SWEEPS = {
            'quick': SweepParams(
                n_values=[4, 8, 16, 32],
                batch_sizes=[4],
                seq_lengths=[128],
                d_model=[256],
                n_heads=[8],
            ),
            'full': SweepParams(
                n_values=[2, 4, 8, 16, 32, 64],
                batch_sizes=[2, 4, 8],
                seq_lengths=[64, 128, 256],
                d_model=[256, 512, 768],
                n_heads=[8, 16],
            ),
            'ci': SweepParams(
                n_values=[4, 8],
                batch_sizes=[4],
                seq_lengths=[64],
                d_model=[256],
                n_heads=[8],
            ),
        }

    @classmethod
    def benchmark_job(cls, preset: str = 'full'):
        """Create benchmark job for WideAttention block."""
        cls._init_benchmark_sweeps()
        from ..benchmark.benchmark_schema import BenchmarkJob

        sweep = cls.BENCHMARK_SWEEPS.get(preset, cls.BENCHMARK_SWEEPS['full'])

        return BenchmarkJob(
            name=f'attention_block_{preset}',
            primitive='attention_block',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
        )

    @staticmethod
    def _bench_model(d_model=256, n_heads=8, **kwargs):
        """Create a single Attention module."""
        head_dim = d_model // n_heads

        class Attention(nn.Module):
            def __init__(self, hidden_size, num_heads, head_dim):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.head_dim = head_dim
                self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
                self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

            def forward(self, x, rope=None, mask=None):
                # Simple self-attention for benchmarking
                B, S, D = x.shape
                qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
                q, k, v = qkv.permute(2, 0, 3, 1, 4)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                out = out.transpose(1, 2).reshape(B, S, -1)
                return self.out_proj(out)

        return Attention(d_model, n_heads, head_dim)

    @staticmethod
    def _bench_input(n: int, device: str, batch_sizes: int, seq_lengths: int, d_model: int, **kwargs):
        """Create input for Attention benchmark."""
        # Returns (x, rope, mask) tuple
        x = torch.randn(batch_sizes, seq_lengths, d_model, device=device)
        return (x, None, None)

    @classmethod
    def _bench_wide(cls, modules: List[nn.Module], strategy: str):
        """Create wide version for given strategy."""
        if strategy == 'baseline':
            return None
        return cls.from_modules(modules, strategy=strategy)


__all__ = ['WideAttention']
