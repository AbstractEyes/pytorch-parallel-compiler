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

    Supports two calling conventions:

    1. Direct (channel-packed):
       - Input:  [B, T, N*D]
       - Output: [B, T, N*D]
       - Usage:  out = wide_attn(x)

    2. MHA-compatible (sequence-packed):
       - Input:  [B, N*T, D]  (from TracedWideModel)
       - Output: Tuple[Tensor [B, N*T, D], None]
       - Usage:  out, _ = wide_attn(query, key, value)
       - Auto-repacks internally to channel format

    Internally:
    1. Auto-detect packing format from input shape
    2. Project Q, K, V using fused linear (or per-model weights)
    3. Reshape to [N*B, H, T, D_head] for batched attention
    4. Run F.scaled_dot_product_attention (Flash Attention)
    5. Reshape back and project output
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

        # QKV projection weights: [N, 3*D, D] for fused QKV
        # Stored as [N, 3, D, D] for clarity then reshaped
        self.qkv_weight = nn.Parameter(torch.empty(n, 3 * d_model, d_model))
        if bias:
            self.qkv_bias = nn.Parameter(torch.empty(n, 3 * d_model))
        else:
            self.register_parameter('qkv_bias', None)

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
        import math
        for i in range(self.n):
            # QKV
            nn.init.xavier_uniform_(self.qkv_weight[i].view(3, self.d_model, self.d_model))
            if self.qkv_bias is not None:
                nn.init.zeros_(self.qkv_bias[i])
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
        Forward pass with auto-detection of input format.

        Supports two calling conventions:

        1. Direct (channel-packed): forward(x) where x is [B, T, N*D]
           - Used when calling WideAttention directly
           - Returns: Tensor [B, T, N*D]

        2. MHA-compatible (sequence-packed): forward(query, key, value)
           where inputs are [B, N*T, D]
           - Used by TracedWideModel when replacing nn.MultiheadAttention
           - Auto-repacks to channel format internally
           - Returns: Tuple[Tensor, None] to match MHA signature

        Args:
            query: Input tensor. Shape depends on calling convention:
                   - Direct: [B, T, N*D] (channel-packed)
                   - MHA: [B, N*T, D] (sequence-packed)
            key: For MHA signature, usually same as query for self-attention
            value: For MHA signature, usually same as query for self-attention
            attn_mask: Optional attention mask [T, T] or [B, T, T]
            is_causal: If True, apply causal mask (for decoder)
            need_weights: Ignored (for MHA compatibility)

        Returns:
            Direct call: Tensor [B, T, N*D]
            MHA call: Tuple[Tensor [B, N*T, D], None]
        """
        # Detect calling convention
        mha_mode = key is not None or value is not None
        x = query

        # Auto-detect packing format from shape
        B = x.size(0)
        last_dim = x.size(-1)
        N, D = self.n, self.d_model

        # Determine if we need to repack
        # Channel-packed: [B, T, N*D] - last dim is N*D
        # Sequence-packed: [B, N*T, D] - last dim is D (from MHA trace)
        needs_repack = (last_dim == D) and (x.size(1) % N == 0)

        if needs_repack:
            # Sequence-packed [B, N*T, D] -> Channel-packed [B, T, N*D]
            NT = x.size(1)
            T = NT // N
            x = x.view(B, N, T, D)
            x = x.permute(0, 2, 1, 3).contiguous()  # [B, T, N, D]
            x = x.view(B, T, N * D)  # [B, T, N*D]
        else:
            T = x.size(1)

        # Run attention
        if self._use_fused:
            out = self._forward_fused(x, attn_mask, is_causal)
        else:
            out = self._forward_sequential(x, attn_mask, is_causal)

        # Repack output if needed
        if needs_repack:
            # Channel-packed [B, T, N*D] -> Sequence-packed [B, N*T, D]
            out = out.view(B, T, N, D)
            out = out.permute(0, 2, 1, 3).contiguous()  # [B, N, T, D]
            out = out.view(B, N * T, D)  # [B, N*T, D]

        # Return appropriate type
        if mha_mode:
            return out, None  # Match nn.MultiheadAttention signature
        else:
            return out

    def _forward_fused(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        is_causal: bool,
    ) -> Tensor:
        """
        Fused attention using batched SDPA.

        Reshape N parallel models into batch dimension to leverage
        Flash Attention / memory-efficient attention.
        """
        B, T, ND = x.shape
        N, D, H, Dh = self.n, self.d_model, self.n_heads, self.d_head

        # Reshape input: [B, T, N*D] -> [N, B, T, D]
        x = x.view(B, T, N, D).permute(2, 0, 1, 3)  # [N, B, T, D]

        # QKV projection via einsum: [N, B, T, D] @ [N, D, 3D] -> [N, B, T, 3D]
        qkv = torch.einsum('nbtd,nod->nbto', x, self.qkv_weight)
        if self.qkv_bias is not None:
            qkv = qkv + self.qkv_bias.view(N, 1, 1, 3 * D)

        # Split Q, K, V: [N, B, T, 3D] -> 3x [N, B, T, D]
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head: [N, B, T, D] -> [N, B, H, T, Dh] -> [N*B, H, T, Dh]
        q = q.view(N, B, T, H, Dh).permute(0, 1, 3, 2, 4).reshape(N * B, H, T, Dh)
        k = k.view(N, B, T, H, Dh).permute(0, 1, 3, 2, 4).reshape(N * B, H, T, Dh)
        v = v.view(N, B, T, H, Dh).permute(0, 1, 3, 2, 4).reshape(N * B, H, T, Dh)

        # Scaled dot-product attention (Flash Attention when available)
        # [N*B, H, T, Dh]
        dropout_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        # Reshape back: [N*B, H, T, Dh] -> [N, B, T, D]
        attn_out = attn_out.view(N, B, H, T, Dh).permute(0, 1, 3, 2, 4).reshape(N, B, T, D)

        # Output projection: [N, B, T, D] @ [N, D, D] -> [N, B, T, D]
        out = torch.einsum('nbtd,nod->nbto', attn_out, self.out_weight)
        if self.out_bias is not None:
            out = out + self.out_bias.view(N, 1, 1, D)

        # Reshape output: [N, B, T, D] -> [B, T, N*D]
        out = out.permute(1, 2, 0, 3).reshape(B, T, N * D)

        return out

    def _forward_sequential(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        is_causal: bool,
    ) -> Tensor:
        """N separate attention computations - baseline."""
        B, T, ND = x.shape
        N, D, H, Dh = self.n, self.d_model, self.n_heads, self.d_head

        # Reshape input: [B, T, N*D] -> [B, T, N, D]
        x = x.view(B, T, N, D)

        outputs = []
        dropout_p = self.dropout if self.training else 0.0

        for i in range(N):
            xi = x[:, :, i, :]  # [B, T, D]

            # QKV projection
            qkv = F.linear(xi, self.qkv_weight[i],
                          self.qkv_bias[i] if self.qkv_bias is not None else None)
            q, k, v = qkv.chunk(3, dim=-1)  # 3x [B, T, D]

            # Reshape for multi-head: [B, T, D] -> [B, H, T, Dh]
            q = q.view(B, T, H, Dh).permute(0, 2, 1, 3)
            k = k.view(B, T, H, Dh).permute(0, 2, 1, 3)
            v = v.view(B, T, H, Dh).permute(0, 2, 1, 3)

            # SDPA
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

            # Reshape: [B, H, T, Dh] -> [B, T, D]
            attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, D)

            # Output projection
            out = F.linear(attn_out, self.out_weight[i],
                          self.out_bias[i] if self.out_bias is not None else None)
            outputs.append(out)

        # Concatenate: [B, T, N*D]
        return torch.cat(outputs, dim=-1)

    @classmethod
    def from_modules(
        cls,
        modules: List[nn.MultiheadAttention],
        strategy: Union[str, AttentionStrategy] = 'auto',
    ) -> 'WideAttention':
        """
        Create from N existing MultiheadAttention modules.

        Note: Assumes modules use self-attention (q=k=v input).
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

        with torch.no_grad():
            for i, m in enumerate(modules):
                # in_proj_weight is [3*D, D], in_proj_bias is [3*D]
                wide.qkv_weight[i] = m.in_proj_weight
                if m.in_proj_bias is not None:
                    wide.qkv_bias[i] = m.in_proj_bias
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