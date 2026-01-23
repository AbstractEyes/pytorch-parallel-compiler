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

from typing import List, Optional, Tuple

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
            emb_size=hidden_size,
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

        # Gated residual: x + attn*gate + mlp (NOT x + gate*(attn+mlp))
        # gate: [N, B, H] -> expand to [N, B, 1, H] for broadcasting
        gate = gate.unsqueeze(2)
        out = x + attn_out * gate + mlp_out

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

            # Adaptive norm with gate - manually compute using weights at index i
            x_norm_i, gate_i = self._compute_norm_i(i, x_i, vec_i)

            # Self-attention - manually compute using weights at index i
            attn_out_i = self._compute_attn_i(i, x_norm_i, rope)

            # MLP - manually compute using weights at index i
            mlp_out_i = self._compute_mlp_i(i, x_norm_i)

            # Gated residual: x + attn*gate + mlp (NOT x + gate*(attn+mlp))
            gate_i = gate_i.unsqueeze(1)  # [B, 1, H]
            out_i = x_i + attn_out_i * gate_i + mlp_out_i

            outputs.append(out_i)

        return torch.stack(outputs, dim=0)

    def _compute_norm_i(self, i: int, x: Tensor, vec: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute adaptive norm for index i using manual weight access."""
        # LayerNorm
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.norm.eps)

        # Apply learned norm params
        x_norm = x_norm * self.norm.norm_weight[i] + self.norm.norm_bias[i]

        # Compute gate from vec
        gate = F.linear(vec, self.norm.linear_weight[i], self.norm.linear_bias[i])

        return x_norm, gate

    def _compute_attn_i(self, i: int, x: Tensor, rope: Optional[Tensor]) -> Tensor:
        """Compute self-attention for index i using manual weight access."""
        B, S, H = x.shape

        # QKV projection
        qkv = F.linear(x, self.attn.qkv_weight[i], self.attn.qkv_bias[i] if self.attn.qkv_bias is not None else None)
        qkv = qkv.reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # Apply RoPE if provided
        if rope is not None:
            from wide_compiler.core.primitives.wide_rotary_embedding import apply_rope
            # q/k are [B, num_heads, S, head_dim] - already correct format for apply_rope
            q = apply_rope(q, rope)
            k = apply_rope(k, rope)

        # Self-attention
        scale = self.head_dim ** -0.5
        attn_out = F.scaled_dot_product_attention(q, k, v, scale=scale)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, -1)

        # Output projection
        out = F.linear(attn_out, self.attn.out_weight[i], self.attn.out_bias[i] if self.attn.out_bias is not None else None)
        return out

    def _compute_mlp_i(self, i: int, x: Tensor) -> Tensor:
        """Compute MLP forward for index i using manual weight access."""
        # FC1
        h = F.linear(x, self.mlp.fc1_weight[i], self.mlp.fc1_bias[i] if self.mlp.fc1_bias is not None else None)
        # Activation
        h = self.mlp._activation(h)
        # FC2
        out = F.linear(h, self.mlp.fc2_weight[i], self.mlp.fc2_bias[i] if self.mlp.fc2_bias is not None else None)
        return out

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

    # =========================================================================
    # Benchmark interface
    # =========================================================================

    BENCHMARK_STRATEGIES = ['baseline', 'fused', 'sequential']
    BENCHMARK_SWEEPS = {}
    _SWEEPS_INITIALIZED = False

    @classmethod
    def _get_sweep_params_class(cls):
        """Get SweepParams class."""
        try:
            from ..benchmark.benchmark_schema import SweepParams
            return SweepParams
        except ImportError:
            try:
                from wide_compiler.core.benchmark.benchmark_schema import SweepParams
                return SweepParams
            except ImportError:
                return None

    @classmethod
    def _init_benchmark_sweeps(cls):
        """Initialize sweep configs."""
        if cls._SWEEPS_INITIALIZED:
            return
        cls._SWEEPS_INITIALIZED = True

        SweepParams = cls._get_sweep_params_class()
        if SweepParams is None:
            return

        cls.BENCHMARK_SWEEPS = {
            'quick': SweepParams(
                n_values=[4, 8],
                batch_sizes=[4],
                seq_lengths=[256],
                d_model=[256],
                n_heads=[8],
            ),
            'ci': SweepParams(
                n_values=[4],
                batch_sizes=[4],
                seq_lengths=[128],
                d_model=[256],
                n_heads=[8],
            ),
        }

    @classmethod
    def benchmark_job(cls, preset='quick'):
        """Create benchmark job for WideSingleStreamBlock."""
        cls._init_benchmark_sweeps()

        try:
            from ..benchmark.benchmark_schema import BenchmarkJob
        except ImportError:
            from wide_compiler.core.benchmark.benchmark_schema import BenchmarkJob

        if preset not in cls.BENCHMARK_SWEEPS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(cls.BENCHMARK_SWEEPS.keys())}")

        sweep = cls.BENCHMARK_SWEEPS[preset]

        return BenchmarkJob(
            name=f'single_stream_block_{preset}',
            primitive='single_stream_block',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
        )

    @staticmethod
    def _bench_model(d_model, n_heads, **kwargs):
        """Create a single SingleStreamBlock module."""
        hidden_size = d_model if isinstance(d_model, int) else d_model
        num_heads = n_heads if isinstance(n_heads, int) else n_heads
        head_dim = hidden_size // num_heads

        class SimpleNorm(nn.Module):
            """Simple norm matching expected structure for from_modules."""
            def __init__(self):
                super().__init__()
                # Must be named 'norm' and 'linear' to match from_modules expectations
                self.norm = nn.LayerNorm(hidden_size)
                self.linear = nn.Linear(hidden_size, hidden_size, bias=True)
                self.hidden_size = hidden_size

        class SimpleAttn(nn.Module):
            """Simple attention matching expected structure for from_modules."""
            def __init__(self):
                super().__init__()
                # Must have .qkv and .out_proj to match from_modules expectations
                qkv_dim = 3 * num_heads * head_dim
                self.qkv = nn.Linear(hidden_size, qkv_dim, bias=False)
                self.out_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
                self.num_heads = num_heads
                self.head_dim = head_dim
                self.scale = head_dim ** -0.5

        class SimpleSingleStreamBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.head_dim = head_dim

                # Adaptive norm
                self.norm = SimpleNorm()

                # Self-attention
                self.attn = SimpleAttn()

                # MLP
                mlp_hidden = hidden_size * 4
                self.mlp = nn.Sequential(
                    nn.Linear(hidden_size, mlp_hidden, bias=False),
                    nn.GELU(),
                    nn.Linear(mlp_hidden, hidden_size, bias=False)
                )

            def forward(self, x, vec, rope=None):
                B, S, H = x.shape

                # Adaptive norm
                x_norm = self.norm.norm(x)
                gate = self.norm.linear(vec)

                # Self-attention (manual computation to match structure)
                qkv = self.attn.qkv(x_norm).reshape(B, S, 3, self.num_heads, self.head_dim)
                q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3 x [B, num_heads, S, head_dim]

                attn_out = F.scaled_dot_product_attention(q, k, v, scale=self.attn.scale)
                attn_out = attn_out.transpose(1, 2).reshape(B, S, -1)
                attn_out = self.attn.out_proj(attn_out)

                # MLP + residual
                out = x + attn_out * gate.unsqueeze(1) + self.mlp(x_norm)
                return out

        return SimpleSingleStreamBlock()

    @staticmethod
    def _bench_input(n, d_model, n_heads, batch_sizes, seq_lengths, device='cpu', **kwargs):
        """Create input tensors."""
        batch_size = batch_sizes if isinstance(batch_sizes, int) else batch_sizes
        hidden_size = d_model if isinstance(d_model, int) else d_model
        seq_len = seq_lengths if isinstance(seq_lengths, int) else seq_lengths

        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
        vec = torch.randn(batch_size, hidden_size, device=device)
        return (x, vec, None)

    @classmethod
    def _bench_wide(cls, modules, strategy, **kwargs):
        """Create wide version."""
        if strategy == 'baseline':
            return None
        return cls.from_modules(modules, strategy=strategy)


__all__ = ['WideSingleStreamBlock']
