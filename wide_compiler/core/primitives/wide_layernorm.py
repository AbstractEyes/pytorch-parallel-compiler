"""
WideLayerNorm - N parallel LayerNorm operations fused into a single module.

LayerNorm operates on the last dimension independently per group.
Unlike BatchNorm, we can't simply concatenate channels - we need to
reshape, normalize each group's D dimensions, then apply per-group affine.

Expected speedup: Modest (1.2-2x) due to fused affine transform.

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Dict, Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class WideLayerNorm(nn.Module):
    """
    N parallel LayerNorm - operates on last dim per group.

    Input shape:  [B, N*D] or [B, T, N*D]
    Output shape: [B, N*D] or [B, T, N*D]

    Each group of D features is normalized independently.
    """

    def __init__(self, n: int, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.n = n
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Weight and bias per model
        self.weight = nn.Parameter(torch.ones(n * normalized_shape))
        self.bias = nn.Parameter(torch.zeros(n * normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        """
        Layer normalization with N-first format.

        Input:  [N, B, ..., D] N-first format
        Output: [N, B, ..., D]
        """
        D = self.normalized_shape
        N = self.n

        # Get per-model weight/bias: [N, D]
        weight = self.weight.view(N, D)
        bias = self.bias.view(N, D)

        # Expand weight/bias to match x dimensions
        # x: [N, B, ..., D], need weight: [N, 1, ..., 1, D]
        for _ in range(x.dim() - 2):
            weight = weight.unsqueeze(1)
            bias = bias.unsqueeze(1)

        # Manual layer norm: normalize over last dim, per-model affine
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * weight + bias

        return x

    @classmethod
    def from_modules(cls, modules: List[nn.LayerNorm]) -> 'WideLayerNorm':
        """Create from N existing LayerNorm modules."""
        n = len(modules)
        t = modules[0]
        norm_shape = t.normalized_shape[0] if isinstance(t.normalized_shape, tuple) else t.normalized_shape

        wide = cls(n, norm_shape, t.eps)

        # Copy to same device/dtype
        wide = wide.to(device=t.weight.device, dtype=t.weight.dtype)

        with torch.no_grad():
            for i, m in enumerate(modules):
                start = i * norm_shape
                end = start + norm_shape
                wide.weight[start:end] = m.weight
                wide.bias[start:end] = m.bias

        return wide

    def __repr__(self):
        return f"WideLayerNorm({self.n}x{self.normalized_shape})"

    # =========================================================================
    # BENCHMARK INTERFACE
    # =========================================================================

    BENCHMARK_STRATEGIES = ['baseline', 'wide']

    @classmethod
    def _get_sweep_params_class(cls):
        """Get SweepParams class with multiple import attempts."""
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
        """Get BenchmarkJob class with multiple import attempts."""
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
        """Initialize sweep configs (called once on first access)."""
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
                seq_lengths=[128],
            ),
            'full': SweepParams(
                n_values=[2, 4, 8, 16, 32, 64, 100],
                batch_sizes=[4, 8, 16, 32],
                d_model=[128, 256, 512, 768],
                seq_lengths=[64, 128, 256, 512],
            ),
            'ci': SweepParams(
                n_values=[4, 16],
                batch_sizes=[8],
                d_model=[256],
                seq_lengths=[128],
            ),
        }

    @classmethod
    def benchmark_job(cls, preset: str = 'full', **overrides) -> Any:
        """Get benchmark job for WideLayerNorm."""
        cls._init_benchmark_sweeps()

        BenchmarkJob = cls._get_benchmark_job_class()
        if BenchmarkJob is None:
            raise ImportError("Could not import BenchmarkJob from benchmark_schema")

        sweep = cls.BENCHMARK_SWEEPS.get(preset)
        if sweep is None:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(cls.BENCHMARK_SWEEPS.keys())}")

        if overrides:
            sweep = sweep.with_overrides(**overrides)

        return BenchmarkJob(
            name=f'layernorm_{preset}',
            primitive='layernorm',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
            # pack_fn/unpack_fn: use default N-first format [N, B, T, D]
        )

    @staticmethod
    def _bench_model(d_model: int, **_) -> nn.LayerNorm:
        """Create single LayerNorm for benchmarking."""
        return nn.LayerNorm(d_model)

    @staticmethod
    def _bench_input(n: int, batch_sizes: int, d_model: int, seq_lengths: int, device: str = 'cpu', **_) -> Tensor:
        """Create single input tensor [B, T, D]."""
        return torch.randn(batch_sizes, seq_lengths, d_model, device=device)

    @classmethod
    def _bench_wide(cls, modules: List[nn.LayerNorm], strategy: str) -> 'WideLayerNorm':
        """Create WideLayerNorm from modules."""
        return cls.from_modules(modules)


__all__ = ['WideLayerNorm']