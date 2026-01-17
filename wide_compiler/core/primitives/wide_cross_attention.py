"""
WideMultiheadCrossAttention - N parallel Cross-Attention layers.

Cross-attention has separate query and key/value sources.
Critical for encoder-decoder architectures, CLIP, cross-modal models.

Expected speedup: 8-12x (similar to WideAttention)

Strategies:
- 'fused': Batched SDPA with separate Q and KV sources (FASTEST)
- 'sequential': N separate cross-attention ops (baseline)

Input/Output Format (v0.6.0):
- Query:  [N, B, Tq, D]     (N-first)
- Key:    [N, B, Tkv, D]    (N-first)
- Value:  [N, B, Tkv, D]    (N-first)
- Output: [N, B, Tq, D]     (N-first)

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Optional, Tuple, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class WideMultiheadCrossAttention(nn.Module):
    """
    N parallel Multi-Head Cross-Attention modules.

    Cross-attention: Query from one source, Key/Value from another.
    Common in decoder layers where decoder queries attend to encoder outputs.
    """

    BENCHMARK_STRATEGIES = ['baseline', 'fused', 'sequential']

    def __init__(
        self,
        n: int,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        strategy: str = 'fused',
    ):
        super().__init__()

        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.n = n
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        self.has_bias = bias
        self._strategy = strategy
        self._use_fused = (strategy == 'fused')

        # Query projection: [N, D, D]
        self.q_weight = nn.Parameter(torch.empty(n, d_model, d_model))
        # Key projection: [N, D, D]
        self.k_weight = nn.Parameter(torch.empty(n, d_model, d_model))
        # Value projection: [N, D, D]
        self.v_weight = nn.Parameter(torch.empty(n, d_model, d_model))

        if bias:
            self.q_bias = nn.Parameter(torch.empty(n, d_model))
            self.k_bias = nn.Parameter(torch.empty(n, d_model))
            self.v_bias = nn.Parameter(torch.empty(n, d_model))
        else:
            self.register_parameter('q_bias', None)
            self.register_parameter('k_bias', None)
            self.register_parameter('v_bias', None)

        # Output projection: [N, D, D]
        self.out_weight = nn.Parameter(torch.empty(n, d_model, d_model))
        if bias:
            self.out_bias = nn.Parameter(torch.empty(n, d_model))
        else:
            self.register_parameter('out_bias', None)

        self._reset_parameters()

    @property
    def strategy(self) -> str:
        return self._strategy

    def _reset_parameters(self):
        """Initialize like nn.Linear."""
        for i in range(self.n):
            nn.init.xavier_uniform_(self.q_weight[i])
            nn.init.xavier_uniform_(self.k_weight[i])
            nn.init.xavier_uniform_(self.v_weight[i])
            nn.init.xavier_uniform_(self.out_weight[i])

            if self.q_bias is not None:
                nn.init.zeros_(self.q_bias[i])
                nn.init.zeros_(self.k_bias[i])
                nn.init.zeros_(self.v_bias[i])
            if self.out_bias is not None:
                nn.init.zeros_(self.out_bias[i])

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """
        Forward pass with N-first format.

        Args:
            query: [N, B, Tq, D]
            key:   [N, B, Tkv, D]
            value: [N, B, Tkv, D]
            attn_mask: Optional attention mask
            is_causal: Use causal masking

        Returns:
            [N, B, Tq, D] cross-attended output
        """
        if self._use_fused:
            return self._forward_fused(query, key, value, attn_mask, is_causal)
        else:
            return self._forward_sequential(query, key, value, attn_mask, is_causal)

    def _forward_fused(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor],
        is_causal: bool,
    ) -> Tensor:
        """Fused cross-attention via batched SDPA."""
        N, B, Tq, D = query.shape
        _, _, Tkv, _ = key.shape

        # Project Q from query: [N, B, Tq, D] -> [N, B, Tq, D]
        q = torch.einsum('nbtd,ndo->nbto', query, self.q_weight)
        if self.q_bias is not None:
            q = q + self.q_bias.view(N, 1, 1, D)

        # Project K from key: [N, B, Tkv, D] -> [N, B, Tkv, D]
        k = torch.einsum('nbtd,ndo->nbto', key, self.k_weight)
        if self.k_bias is not None:
            k = k + self.k_bias.view(N, 1, 1, D)

        # Project V from value: [N, B, Tkv, D] -> [N, B, Tkv, D]
        v = torch.einsum('nbtd,ndo->nbto', value, self.v_weight)
        if self.v_bias is not None:
            v = v + self.v_bias.view(N, 1, 1, D)

        # Reshape for multi-head: [N, B, T, D] -> [N*B, H, T, D/H]
        q = q.view(N * B, Tq, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(N * B, Tkv, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(N * B, Tkv, self.n_heads, self.d_head).transpose(1, 2)

        # Cross-attention via SDPA
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape back: [N*B, H, Tq, D/H] -> [N, B, Tq, D]
        attn_out = attn_out.transpose(1, 2).contiguous().view(N, B, Tq, D)

        # Output projection
        out = torch.einsum('nbtd,ndo->nbto', attn_out, self.out_weight)
        if self.out_bias is not None:
            out = out + self.out_bias.view(N, 1, 1, D)

        return out

    def _forward_sequential(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor],
        is_causal: bool,
    ) -> Tensor:
        """Sequential cross-attention (baseline)."""
        N, B, Tq, D = query.shape
        _, _, Tkv, _ = key.shape

        outputs = []
        for i in range(N):
            # Project (weights are stored transposed for einsum, so transpose back for F.linear)
            q_i = F.linear(query[i], self.q_weight[i].T, self.q_bias[i] if self.q_bias is not None else None)
            k_i = F.linear(key[i], self.k_weight[i].T, self.k_bias[i] if self.k_bias is not None else None)
            v_i = F.linear(value[i], self.v_weight[i].T, self.v_bias[i] if self.v_bias is not None else None)

            # Reshape for multi-head: [B, T, D] -> [B, H, T, D/H]
            q_i = q_i.view(B, Tq, self.n_heads, self.d_head).transpose(1, 2)
            k_i = k_i.view(B, Tkv, self.n_heads, self.d_head).transpose(1, 2)
            v_i = v_i.view(B, Tkv, self.n_heads, self.d_head).transpose(1, 2)

            # Cross-attention
            attn_i = F.scaled_dot_product_attention(
                q_i, k_i, v_i,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )

            # Reshape back and project
            attn_i = attn_i.transpose(1, 2).contiguous().view(B, Tq, D)
            out_i = F.linear(attn_i, self.out_weight[i].T, self.out_bias[i] if self.out_bias is not None else None)

            outputs.append(out_i)

        return torch.stack(outputs, dim=0)

    @classmethod
    def from_modules(cls, modules: List[nn.MultiheadAttention], strategy: str = 'fused') -> 'WideMultiheadCrossAttention':
        """
        Create from N MultiheadAttention modules.

        Note: PyTorch's MultiheadAttention doesn't distinguish self vs cross-attention
        in the constructor, so we can reuse the same modules.
        """
        n = len(modules)
        t = modules[0]

        wide = cls(
            n=n,
            d_model=t.embed_dim,
            n_heads=t.num_heads,
            dropout=t.dropout,
            bias=(t.in_proj_bias is not None),
            strategy=strategy,
        )

        wide = wide.to(device=t.in_proj_weight.device, dtype=t.in_proj_weight.dtype)

        # Copy weights
        with torch.no_grad():
            for i, m in enumerate(modules):
                d = m.embed_dim

                # Split in_proj_weight: [3D, D] -> Q, K, V each [D, D]
                # PyTorch stores weights as [out_features, in_features]
                # We transpose for einsum: 'nbtd,ndo->nbto' expects weight[n, in, out]
                w_q, w_k, w_v = m.in_proj_weight.chunk(3, dim=0)
                wide.q_weight[i] = w_q.T  # [D, D] -> [D, D] transposed for einsum
                wide.k_weight[i] = w_k.T
                wide.v_weight[i] = w_v.T

                if m.in_proj_bias is not None:
                    b_q, b_k, b_v = m.in_proj_bias.chunk(3, dim=0)
                    wide.q_bias[i] = b_q
                    wide.k_bias[i] = b_k
                    wide.v_bias[i] = b_v

                wide.out_weight[i] = m.out_proj.weight.T  # Transpose for einsum
                if m.out_proj.bias is not None:
                    wide.out_bias[i] = m.out_proj.bias

        return wide

    def __repr__(self):
        return (f"WideMultiheadCrossAttention({self.n}x[d={self.d_model}, "
                f"heads={self.n_heads}], strategy={self._strategy})")

    # =========================================================================
    # BENCHMARK INTERFACE
    # =========================================================================
    # Note: For benchmarking, we use self-attention mode (same tensor for Q/K/V)
    # to simplify the interface. This still measures cross-attention performance
    # accurately since the compute pattern is identical.

    BENCHMARK_SWEEPS = {
        'quick': {'n_values': [4, 8, 16, 32]},
        'full': {'n_values': [2, 4, 8, 16, 32, 64]},
        'ci': {'n_values': [4, 8]},
    }

    @classmethod
    def benchmark_job(cls, preset: str = 'full'):
        """Create benchmark job for WideMultiheadCrossAttention."""
        from ..benchmark.benchmark_schema import BenchmarkJob, SweepParams

        sweep_config = cls.BENCHMARK_SWEEPS.get(preset, cls.BENCHMARK_SWEEPS['full'])
        sweep = SweepParams(
            n_values=sweep_config['n_values'],
            batch_sizes=[8],
            seq_lengths=[64],    # Use same length for simplicity (can vary in factory)
            d_model=[256],
            n_heads=[8],
        )

        return BenchmarkJob(
            name=f'cross_attention_{preset}',
            primitive='cross_attention',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
        )

    @staticmethod
    def _bench_model(d_model=256, n_heads=8, **kwargs):
        """
        Create a single MultiheadAttention module wrapped for benchmarking.
        Wrapper accepts single input and uses it for Q/K/V (self-attention mode).
        """
        mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

        # Wrapper to convert single-input interface to Q/K/V interface
        class SelfAttentionWrapper(nn.Module):
            def __init__(self, mha):
                super().__init__()
                self.mha = mha

            def forward(self, x):
                # Use same input for Q, K, V (self-attention)
                return self.mha(x, x, x)[0]  # Return only output, not attention weights

        return SelfAttentionWrapper(mha)

    @staticmethod
    def _bench_input(n: int, device: str, batch_sizes: int, seq_lengths=64, d_model=256, **kwargs):
        """
        Create input for cross-attention benchmark.
        For simplicity in benchmarking, use same tensor for Q/K/V (self-attention mode).
        Returns single tensor [B, T, D].
        """
        return torch.randn(batch_sizes, seq_lengths, d_model, device=device)

    @classmethod
    def _bench_wide(cls, modules: List[nn.Module], strategy: str):
        """
        Create wide version for given strategy.
        Unwraps SelfAttentionWrapper to get actual MHA modules.
        """
        if strategy == 'baseline':
            return None

        # Unwrap the benchmark wrappers to get actual MHA modules
        mha_modules = [m.mha for m in modules]
        wide = cls.from_modules(mha_modules, strategy=strategy)

        # Wrap the wide model to accept single input
        class WideSelfAttentionWrapper(nn.Module):
            def __init__(self, wide_cross_attn):
                super().__init__()
                self.wide = wide_cross_attn

            def forward(self, x):
                # x is already packed [N, B, T, D]
                # Use same for Q/K/V
                return self.wide(x, x, x)

        return WideSelfAttentionWrapper(wide)


__all__ = ['WideMultiheadCrossAttention']
