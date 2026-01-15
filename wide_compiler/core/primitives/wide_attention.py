"""
WideAttention - N parallel Multi-Head Attention fused into a single module.

Key insight: Reshape [N, B, H, T, D] -> [N*B, H, T, D] to leverage
F.scaled_dot_product_attention (Flash Attention) for all N models in one call.

Strategies:
- 'fused': Batched SDPA - treats N as extra batch dim (FASTEST)
- 'sequential': N separate attention ops (baseline, exact)

Expected speedup: 2-8x depending on N and sequence length.

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Dict, Any, Union, Optional, Tuple
from enum import Enum

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class AttentionStrategy(Enum):
    FUSED = 'fused'           # Batched SDPA - fastest
    SEQUENTIAL = 'sequential' # N separate ops - baseline
    AUTO = 'auto'


class WideAttention(nn.Module):
    """
    N parallel Multi-Head Attention modules fused into single operations.

    Calling conventions:

    1. Direct self-attention (channel-packed):
       - attn(x) where x: [B, T, N*D]
       - Returns: Tensor [B, T, N*D]

    2. MHA-style self-attention (sequence-packed from TracedWideModel):
       - attn(query, key, value) where each is [B, N*T, D]
       - Auto-repacks to channel format internally
       - Returns: Tuple[Tensor [B, N*T, D], None]

    3. Cross-attention:
       - attn(query, key, value) where query: [B, N*Tq, D], key/value: [B, N*Tkv, D]
       - Q projected from query, K from key, V from value
       - Returns: Tuple[Tensor [B, N*Tq, D], None]

    Internally:
    1. Auto-detect packing format from input shape
    2. Separate Q projection from query, KV projection from key
    3. Reshape to [N*B, H, T, D_head] for batched attention
    4. Run F.scaled_dot_product_attention (Flash Attention)
    5. Reshape and apply output projection
    6. Repack to original format if needed

    Strategies:
        'fused': Reshape N models into batch dim, single SDPA call
            - Gets Flash Attention / memory-efficient attention for free
            - Fastest for N >= 2
        'sequential': N separate attention computations
            - Baseline, exact but slower
    """

    def __init__(
        self,
        n: int,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        strategy: Union[str, AttentionStrategy] = 'auto',
    ):
        super().__init__()

        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.n = n
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        self.has_bias = bias

        # Parse strategy
        if isinstance(strategy, str):
            strategy = AttentionStrategy(strategy)
        if strategy == AttentionStrategy.AUTO:
            strategy = AttentionStrategy.FUSED

        self._strategy = strategy
        self._use_fused = (strategy == AttentionStrategy.FUSED)

        # Separate Q and KV projections to support cross-attention
        # Q projection: [N, D, D]
        self.q_weight = nn.Parameter(torch.empty(n, d_model, d_model))
        # KV projection: [N, 2*D, D] (K and V concatenated)
        self.kv_weight = nn.Parameter(torch.empty(n, 2 * d_model, d_model))

        if bias:
            self.q_bias = nn.Parameter(torch.empty(n, d_model))
            self.kv_bias = nn.Parameter(torch.empty(n, 2 * d_model))
        else:
            self.register_parameter('q_bias', None)
            self.register_parameter('kv_bias', None)

        # Output projection: [N, D, D]
        self.out_weight = nn.Parameter(torch.empty(n, d_model, d_model))
        if bias:
            self.out_bias = nn.Parameter(torch.empty(n, d_model))
        else:
            self.register_parameter('out_bias', None)

        self._reset_parameters()

    @property
    def strategy(self) -> AttentionStrategy:
        return self._strategy

    def _reset_parameters(self):
        """Initialize like nn.Linear."""
        for i in range(self.n):
            # Q projection
            nn.init.xavier_uniform_(self.q_weight[i])
            if self.q_bias is not None:
                nn.init.zeros_(self.q_bias[i])
            # KV projection
            nn.init.xavier_uniform_(self.kv_weight[i].view(2, self.d_model, self.d_model))
            if self.kv_bias is not None:
                nn.init.zeros_(self.kv_bias[i])
            # Output
            nn.init.xavier_uniform_(self.out_weight[i])
            if self.out_bias is not None:
                nn.init.zeros_(self.out_bias[i])

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, None]]:
        """
        Forward pass with N-first format.

        Input:  [N, B, T, D] N-first format for query/key/value
        Output: [N, B, T, D] or Tuple[[N, B, T, D], None]

        For self-attention, key and value default to query.
        """
        N, D = self.n, self.d_model

        # Determine if MHA-style call (3 args) or direct call (1 arg)
        mha_mode = key is not None

        # Handle self-attention: key/value default to query
        if key is None:
            key = query
        if value is None:
            value = key

        # Run attention (inputs already N-first)
        if self._use_fused:
            out = self._forward_fused(query, key, value, attn_mask, is_causal)
        else:
            out = self._forward_sequential(query, key, value, attn_mask, is_causal)

        # Return appropriate type
        if mha_mode:
            return out, None
        else:
            return out

    def _forward_fused(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor],
        is_causal: bool,
    ) -> Tensor:
        """
        Fused attention using batched SDPA.

        Input:  [N, B, Tq, D] query, [N, B, Tkv, D] key/value (N-first)
        Output: [N, B, Tq, D]
        """
        N, D, H, Dh = self.n, self.d_model, self.n_heads, self.d_head
        _, B, Tq, _ = query.shape
        Tkv = key.size(2)

        # Inputs already N-first: [N, B, T, D]

        # Q projection: [N, B, Tq, D] @ [N, D, D] -> [N, B, Tq, D]
        q = torch.einsum('nbtd,nod->nbto', query, self.q_weight)
        if self.q_bias is not None:
            q = q + self.q_bias.view(N, 1, 1, D)

        # K projection from key: [N, B, Tkv, D] @ [N, D, D] -> [N, B, Tkv, D]
        k_weight = self.kv_weight[:, :D, :]  # [N, D, D]
        k = torch.einsum('nbtd,nod->nbto', key, k_weight)

        # V projection from value: [N, B, Tkv, D] @ [N, D, D] -> [N, B, Tkv, D]
        v_weight = self.kv_weight[:, D:, :]  # [N, D, D]
        v = torch.einsum('nbtd,nod->nbto', value, v_weight)

        if self.kv_bias is not None:
            k = k + self.kv_bias[:, :D].view(N, 1, 1, D)
            v = v + self.kv_bias[:, D:].view(N, 1, 1, D)

        # Reshape for multi-head attention
        # [N, B, T, D] -> [N, B, H, T, Dh] -> [N*B, H, T, Dh]
        q = q.view(N, B, Tq, H, Dh).permute(0, 1, 3, 2, 4).reshape(N * B, H, Tq, Dh)
        k = k.view(N, B, Tkv, H, Dh).permute(0, 1, 3, 2, 4).reshape(N * B, H, Tkv, Dh)
        v = v.view(N, B, Tkv, H, Dh).permute(0, 1, 3, 2, 4).reshape(N * B, H, Tkv, Dh)

        # Scaled dot-product attention (Flash Attention when available)
        dropout_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )  # [N*B, H, Tq, Dh]

        # Reshape back: [N*B, H, Tq, Dh] -> [N, B, Tq, D]
        attn_out = attn_out.view(N, B, H, Tq, Dh).permute(0, 1, 3, 2, 4).reshape(N, B, Tq, D)

        # Output projection: [N, B, Tq, D] @ [N, D, D] -> [N, B, Tq, D]
        out = torch.einsum('nbtd,nod->nbto', attn_out, self.out_weight)
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
        """
        N separate attention computations - baseline.

        Input:  [N, B, Tq, D] query, [N, B, Tkv, D] key/value (N-first)
        Output: [N, B, Tq, D]
        """
        N, D, H, Dh = self.n, self.d_model, self.n_heads, self.d_head
        _, B, Tq, _ = query.shape
        Tkv = key.size(2)

        # Split KV weight into K and V parts
        k_weight = self.kv_weight[:, :D, :]  # [N, D, D]
        v_weight = self.kv_weight[:, D:, :]  # [N, D, D]

        outputs = []
        dropout_p = self.dropout if self.training else 0.0

        for i in range(N):
            qi = query[i]  # [B, Tq, D]
            ki = key[i]    # [B, Tkv, D]
            vi = value[i]  # [B, Tkv, D]

            # Q projection
            q = F.linear(qi, self.q_weight[i],
                        self.q_bias[i] if self.q_bias is not None else None)

            # K projection from key
            k_bias = self.kv_bias[i, :D] if self.kv_bias is not None else None
            k = F.linear(ki, k_weight[i], k_bias)

            # V projection from value
            v_bias = self.kv_bias[i, D:] if self.kv_bias is not None else None
            v = F.linear(vi, v_weight[i], v_bias)

            # Reshape for multi-head: [B, T, D] -> [B, H, T, Dh]
            q = q.view(B, Tq, H, Dh).permute(0, 2, 1, 3)
            k = k.view(B, Tkv, H, Dh).permute(0, 2, 1, 3)
            v = v.view(B, Tkv, H, Dh).permute(0, 2, 1, 3)

            # SDPA
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )  # [B, H, Tq, Dh]

            # Reshape: [B, H, Tq, Dh] -> [B, Tq, D]
            attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, Tq, D)

            # Output projection
            out = F.linear(attn_out, self.out_weight[i],
                          self.out_bias[i] if self.out_bias is not None else None)
            outputs.append(out)

        # Stack: [N, B, Tq, D]
        return torch.stack(outputs, dim=0)

    @classmethod
    def from_modules(
        cls,
        modules: List[nn.MultiheadAttention],
        strategy: Union[str, AttentionStrategy] = 'auto',
    ) -> 'WideAttention':
        """
        Create from N existing MultiheadAttention modules.

        Supports both self-attention and cross-attention patterns.
        """
        n = len(modules)
        t = modules[0]

        wide = cls(
            n=n,
            d_model=t.embed_dim,
            n_heads=t.num_heads,
            dropout=t.dropout,
            bias=t.in_proj_bias is not None,
            strategy=strategy,
        )

        # Copy to same device/dtype
        wide = wide.to(device=t.in_proj_weight.device, dtype=t.in_proj_weight.dtype)
        D = t.embed_dim

        with torch.no_grad():
            for i, m in enumerate(modules):
                # in_proj_weight is [3*D, D] = [Q; K; V] stacked
                # in_proj_bias is [3*D]
                in_w = m.in_proj_weight  # [3D, D]

                # Split into Q [D, D] and KV [2D, D]
                q_w = in_w[:D, :]         # [D, D]
                kv_w = in_w[D:, :]        # [2D, D]

                wide.q_weight[i] = q_w
                wide.kv_weight[i] = kv_w

                if m.in_proj_bias is not None:
                    in_b = m.in_proj_bias  # [3D]
                    wide.q_bias[i] = in_b[:D]
                    wide.kv_bias[i] = in_b[D:]

                # out_proj
                wide.out_weight[i] = m.out_proj.weight
                if m.out_proj.bias is not None:
                    wide.out_bias[i] = m.out_proj.bias

        return wide

    def __repr__(self):
        return (
            f"WideAttention({self.n}x[d={self.d_model}, h={self.n_heads}], "
            f"strategy={self._strategy.value})"
        )

    # =========================================================================
    # BENCHMARK INTERFACE
    # =========================================================================

    BENCHMARK_STRATEGIES = ['baseline', 'fused', 'sequential']

    @classmethod
    def _get_sweep_params_class(cls):
        try:
            from ..benchmark.benchmark_schema import SweepParams
            return SweepParams
        except ImportError:
            pass
        try:
            from wide_compiler.core.benchmark.benchmark_schema import SweepParams
            return SweepParams
        except ImportError:
            pass
        return None

    @classmethod
    def _get_benchmark_job_class(cls):
        try:
            from ..benchmark.benchmark_schema import BenchmarkJob
            return BenchmarkJob
        except ImportError:
            pass
        try:
            from wide_compiler.core.benchmark.benchmark_schema import BenchmarkJob
            return BenchmarkJob
        except ImportError:
            pass
        return None

    BENCHMARK_SWEEPS: Dict[str, Any] = {}
    _SWEEPS_INITIALIZED = False

    @classmethod
    def _init_benchmark_sweeps(cls):
        if cls._SWEEPS_INITIALIZED:
            return
        cls._SWEEPS_INITIALIZED = True

        SweepParams = cls._get_sweep_params_class()
        if SweepParams is None:
            return

        cls.BENCHMARK_SWEEPS = {
            'quick': SweepParams(
                n_values=[4, 8, 16, 32],
                batch_sizes=[8],
                d_model=[256],
                n_heads=[8],
                seq_lengths=[128],
            ),
            'full': SweepParams(
                n_values=[2, 4, 8, 16, 32, 64],
                batch_sizes=[4, 8, 16],
                d_model=[256, 512, 768],
                n_heads=[4, 8, 12],
                seq_lengths=[64, 128, 256, 512],
            ),
            'ci': SweepParams(
                n_values=[4, 16],
                batch_sizes=[8],
                d_model=[256],
                n_heads=[8],
                seq_lengths=[128],
            ),
        }

    @classmethod
    def benchmark_job(cls, preset: str = 'full', **overrides) -> Any:
        cls._init_benchmark_sweeps()

        BenchmarkJob = cls._get_benchmark_job_class()
        if BenchmarkJob is None:
            raise ImportError("Could not import BenchmarkJob")

        sweep = cls.BENCHMARK_SWEEPS.get(preset)
        if sweep is None:
            raise ValueError(f"Unknown preset '{preset}'")

        if overrides:
            sweep = sweep.with_overrides(**overrides)

        return BenchmarkJob(
            name=f'attention_{preset}',
            primitive='attention',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
            pack_fn=cls._bench_pack,
            unpack_fn=cls._bench_unpack,
        )

    @staticmethod
    def _bench_model(d_model: int, n_heads: int, **_) -> nn.Module:
        """Create single attention module for benchmarking."""
        # Wrap MHA to accept single input (self-attention)
        class SelfAttentionWrapper(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

            def forward(self, x):
                out, _ = self.mha(x, x, x)
                return out

        return SelfAttentionWrapper(d_model, n_heads)

    @staticmethod
    def _bench_input(n: int, batch_sizes: int, d_model: int, seq_lengths: int, device: str = 'cpu', **_) -> Tensor:
        return torch.randn(batch_sizes, seq_lengths, d_model, device=device)

    @classmethod
    def _bench_wide(cls, modules: List[nn.Module], strategy: str) -> 'WideAttention':
        """Create WideAttention from wrapper modules."""
        strat_map = {
            'fused': AttentionStrategy.FUSED,
            'sequential': AttentionStrategy.SEQUENTIAL,
        }
        strat = strat_map.get(strategy, AttentionStrategy.FUSED)

        # Extract MHA from wrappers
        mha_modules = [m.mha for m in modules]
        return cls.from_modules(mha_modules, strategy=strat)

    @staticmethod
    def _bench_pack(inputs: List[Tensor]) -> Tensor:
        # inputs: list of [B, T, D]
        return torch.cat(inputs, dim=-1)  # [B, T, N*D]

    @staticmethod
    def _bench_unpack(output: Tensor, n: int) -> List[Tensor]:
        B, T, ND = output.shape
        D = ND // n
        return [output[..., i*D:(i+1)*D] for i in range(n)]


__all__ = ['WideAttention', 'AttentionStrategy']