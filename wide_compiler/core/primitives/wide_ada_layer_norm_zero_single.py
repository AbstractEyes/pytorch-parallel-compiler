"""
WideAdaLayerNormZeroSingle - N parallel Adaptive LayerNorm with Zero initialization.

Adaptive normalization that modulates features based on conditioning vector.
Outputs both normalized features and modulation parameters (gate).

Used in Flux single-stream blocks for conditioning on timestep/text embeddings.

Expected speedup: 8-10x (combination of LayerNorm + Linear projections)

Strategies:
- 'fused': Fused normalization and projection (FASTEST)
- 'sequential': N separate operations (baseline)

Input/Output Format (v0.7.0):
- Input x:   [N, B, seq_len, hidden_size]  (N-first)
- Input vec: [N, B, emb_size]  (N-first, conditioning vector)
- Output: (x_norm, gate) where:
    - x_norm: [N, B, seq_len, hidden_size] (normalized, modulated features)
    - gate:   [N, B, hidden_size] (modulation parameters)

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class WideAdaLayerNormZeroSingle(nn.Module):
    """
    N parallel Adaptive LayerNorm with Zero initialization.

    Combines:
    1. LayerNorm on input features
    2. MLP projection of conditioning vector
    3. Modulation of normalized features
    """

    BENCHMARK_STRATEGIES = ['baseline', 'fused', 'sequential']

    def __init__(
        self,
        n: int,
        hidden_size: int,
        emb_size: Optional[int] = None,
        eps: float = 1e-6,
        strategy: str = 'fused',
    ):
        super().__init__()
        self.n = n
        self.hidden_size = hidden_size
        self.emb_size = emb_size if emb_size is not None else hidden_size
        self.eps = eps
        self._strategy = strategy

        # LayerNorm parameters: [N, hidden_size]
        self.norm_weight = nn.Parameter(torch.ones(n, hidden_size))
        self.norm_bias = nn.Parameter(torch.zeros(n, hidden_size))

        # MLP for conditioning: emb_size -> hidden_size (for gate)
        # Output: gate for attention modulation
        self.linear_weight = nn.Parameter(torch.empty(n, hidden_size, self.emb_size))
        self.linear_bias = nn.Parameter(torch.zeros(n, hidden_size))

        self._reset_parameters()

    @property
    def strategy(self) -> str:
        return self._strategy

    def _reset_parameters(self):
        """Initialize weights (zero-init for gate as in Flux)."""
        for i in range(self.n):
            nn.init.zeros_(self.linear_weight[i])  # Zero-init for stable training

    def forward(self, x: Tensor, vec: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with N-first format.

        Args:
            x: [N, B, seq_len, hidden_size] - input features
            vec: [N, B, emb_size] - conditioning vector (timestep/text embedding)

        Returns:
            x_norm: [N, B, seq_len, hidden_size] - normalized features
            gate: [N, B, hidden_size] - modulation parameters
        """
        if self._strategy == 'fused':
            return self._forward_fused(x, vec)
        else:
            return self._forward_sequential(x, vec)

    def _forward_fused(self, x: Tensor, vec: Tensor) -> Tuple[Tensor, Tensor]:
        """Fused adaptive normalization."""
        N, B, S, H = x.shape

        # LayerNorm: normalize over hidden dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Apply affine transformation: [N, B, S, H] * [N, 1, 1, H] + [N, 1, 1, H]
        weight = self.norm_weight.view(N, 1, 1, H)
        bias = self.norm_bias.view(N, 1, 1, H)
        x_norm = x_norm * weight + bias

        # Project conditioning vector to get gate
        # vec: [N, B, emb_size] @ [N, hidden, emb].T -> [N, B, hidden]
        gate = torch.einsum('nbe,nhe->nbh', vec, self.linear_weight)
        gate = gate + self.linear_bias.unsqueeze(1)  # [N, B, H]

        return x_norm, gate

    def _forward_sequential(self, x: Tensor, vec: Tensor) -> Tuple[Tensor, Tensor]:
        """Sequential adaptive normalization (baseline)."""
        N, B, S, H = x.shape
        x_norms = []
        gates = []

        for i in range(N):
            x_i = x[i]  # [B, S, H]
            vec_i = vec[i]  # [B, emb_size]

            # LayerNorm
            mean = x_i.mean(dim=-1, keepdim=True)
            var = x_i.var(dim=-1, keepdim=True, unbiased=False)
            x_norm = (x_i - mean) / torch.sqrt(var + self.eps)
            x_norm = x_norm * self.norm_weight[i] + self.norm_bias[i]

            # Gate from conditioning
            gate = F.linear(vec_i, self.linear_weight[i].T, self.linear_bias[i])

            x_norms.append(x_norm)
            gates.append(gate)

        return torch.stack(x_norms, dim=0), torch.stack(gates, dim=0)

    @classmethod
    def from_modules(cls, modules: List[nn.Module], strategy: str = 'fused') -> 'WideAdaLayerNormZeroSingle':
        """
        Create from N existing AdaLayerNormZeroSingle modules.

        Expects modules with .norm (LayerNorm) and .linear (Linear) attributes.
        """
        n = len(modules)
        t = modules[0]

        # Get dimensions
        hidden_size = t.norm.normalized_shape[0]
        emb_size = t.linear.in_features

        wide = cls(
            n=n,
            hidden_size=hidden_size,
            emb_size=emb_size,
            eps=t.norm.eps,
            strategy=strategy,
        )

        wide = wide.to(device=t.norm.weight.device, dtype=t.norm.weight.dtype)

        # Copy weights
        with torch.no_grad():
            for i, m in enumerate(modules):
                wide.norm_weight[i] = m.norm.weight
                wide.norm_bias[i] = m.norm.bias
                wide.linear_weight[i] = m.linear.weight
                wide.linear_bias[i] = m.linear.bias

        return wide

    def __repr__(self):
        return (f"WideAdaLayerNormZeroSingle({self.n}x[hidden={self.hidden_size}, "
                f"emb={self.emb_size}], strategy={self._strategy})")

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
                batch_sizes=[8],
                seq_lengths=[128],
                channels=[256],
                embedding_dims=[256],
            ),
            'full': SweepParams(
                n_values=[2, 4, 8, 16, 32, 64],
                batch_sizes=[4, 8, 16],
                seq_lengths=[64, 128, 256],
                channels=[128, 256, 512],
                embedding_dims=[128, 256, 512],
            ),
            'ci': SweepParams(
                n_values=[4, 8],
                batch_sizes=[8],
                seq_lengths=[64],
                channels=[128],
                embedding_dims=[128],
            ),
        }

    @classmethod
    def benchmark_job(cls, preset: str = 'full'):
        """Create benchmark job for WideAdaLayerNormZeroSingle."""
        cls._init_benchmark_sweeps()
        from ..benchmark.benchmark_schema import BenchmarkJob

        sweep = cls.BENCHMARK_SWEEPS.get(preset, cls.BENCHMARK_SWEEPS['full'])

        return BenchmarkJob(
            name=f'ada_layer_norm_zero_single_{preset}',
            primitive='ada_layer_norm_zero_single',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
            validate_fn=cls._bench_validate,
        )

    @staticmethod
    def _bench_model(channels=256, embedding_dims=256, **kwargs):
        """Create a single AdaLayerNormZeroSingle module."""
        class AdaLayerNormZeroSingle(nn.Module):
            def __init__(self, hidden_size, emb_size, eps=1e-6):
                super().__init__()
                self.norm = nn.LayerNorm(hidden_size, eps=eps)
                self.linear = nn.Linear(emb_size, hidden_size, bias=True)

            def forward(self, x, emb):
                # x: [B, S, H], emb: [B, E]
                x_norm = self.norm(x)
                gate = self.linear(emb)  # [B, H]
                return x_norm, gate

        return AdaLayerNormZeroSingle(channels, embedding_dims)

    @staticmethod
    def _bench_input(n: int, device: str, batch_sizes: int, seq_lengths: int,
                     channels: int, embedding_dims: int, **kwargs):
        """Create input for AdaLayerNormZeroSingle benchmark (returns tuple)."""
        x = torch.randn(batch_sizes, seq_lengths, channels, device=device)
        emb = torch.randn(batch_sizes, embedding_dims, device=device)
        return (x, emb)

    @classmethod
    def _bench_wide(cls, modules: List[nn.Module], strategy: str):
        """Create wide version for given strategy."""
        if strategy == 'baseline':
            return None
        return cls.from_modules(modules, strategy=strategy)

    @staticmethod
    def _bench_validate(wide_output, baseline_outputs, rtol=1e-4, atol=1e-4):
        """Custom validation for tuple outputs."""
        # wide_output: (x_norm [N, B, S, H], gate [N, B, H])
        # baseline_outputs: list of (x_norm [B, S, H], gate [B, H])

        wide_norm, wide_gate = wide_output
        baseline_norms = [out[0] for out in baseline_outputs]
        baseline_gates = [out[1] for out in baseline_outputs]

        stacked_norm = torch.stack(baseline_norms, dim=0)
        stacked_gate = torch.stack(baseline_gates, dim=0)

        # Validate norm output
        norm_match = torch.allclose(wide_norm, stacked_norm, rtol=rtol, atol=atol)
        if not norm_match:
            norm_diff = (wide_norm - stacked_norm).abs()
            raise ValueError(
                f"Norm output mismatch: max_diff={norm_diff.max():.2e}, "
                f"mean_diff={norm_diff.mean():.2e}"
            )

        # Validate gate output
        gate_match = torch.allclose(wide_gate, stacked_gate, rtol=rtol, atol=atol)
        if not gate_match:
            gate_diff = (wide_gate - stacked_gate).abs()
            raise ValueError(
                f"Gate output mismatch: max_diff={gate_diff.max():.2e}, "
                f"mean_diff={gate_diff.mean():.2e}"
            )


__all__ = ['WideAdaLayerNormZeroSingle']
