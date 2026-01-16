"""
WideBatchNorm1d - N parallel BatchNorm1d operations fused into a single module.

BatchNorm1d fusion is straightforward - we just create a single BatchNorm1d
with N*num_features channels. This works because BatchNorm operates independently
per channel.

Expected speedup: Modest (1.2-2x) due to reduced kernel launch overhead.
BatchNorm is already memory-bound, so fusion mainly helps with launch overhead.

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Dict, Any

import torch
from torch import Tensor, nn


class WideBatchNorm1d(nn.Module):
    """
    N parallel BatchNorm1d as single BatchNorm1d.

    Input shape:  [B, N*C, L] or [B, N*C]
    Output shape: [B, N*C, L] or [B, N*C]

    BatchNorm operates independently per channel, so fusion is exact
    (no numerical difference from sequential execution).
    """

    def __init__(
        self,
        n: int,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.n = n
        self.num_features = num_features

        self.op = nn.BatchNorm1d(
            num_features=n * num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with N-first format.

        Input:  [N, B, C] or [N, B, C, L]
        Output: [N, B, C] or [N, B, C, L]
        """
        if x.dim() == 3:
            # [N, B, C] case
            N, B, C = x.shape
            x = x.permute(1, 0, 2).reshape(B, N * C)
            out = self.op(x)
            out = out.view(B, N, C).permute(1, 0, 2)
        else:
            # [N, B, C, L] case
            N, B, C, L = x.shape
            x = x.permute(1, 0, 2, 3).reshape(B, N * C, L)
            out = self.op(x)
            out = out.view(B, N, C, L).permute(1, 0, 2, 3)

        return out.contiguous()

    @classmethod
    def from_modules(cls, modules: List[nn.BatchNorm1d]) -> 'WideBatchNorm1d':
        """Create from N existing BatchNorm1d modules."""
        n = len(modules)
        t = modules[0]

        wide = cls(
            n=n,
            num_features=t.num_features,
            eps=t.eps,
            momentum=t.momentum,
            affine=t.affine,
            track_running_stats=t.track_running_stats,
        )

        # Copy to same device/dtype
        wide = wide.to(
            device=t.weight.device if t.weight is not None else 'cpu',
            dtype=t.weight.dtype if t.weight is not None else torch.float32,
        )

        with torch.no_grad():
            for i, m in enumerate(modules):
                start = i * t.num_features
                end = start + t.num_features
                if m.weight is not None:
                    wide.op.weight[start:end] = m.weight
                if m.bias is not None:
                    wide.op.bias[start:end] = m.bias
                if m.running_mean is not None:
                    wide.op.running_mean[start:end] = m.running_mean
                if m.running_var is not None:
                    wide.op.running_var[start:end] = m.running_var

        return wide

    def __repr__(self):
        return f"WideBatchNorm1d({self.n}x{self.num_features})"

    # =========================================================================
    # BENCHMARK INTERFACE
    # =========================================================================

    # Only two strategies: baseline (N separate) and wide (fused)
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
                channels=[64],
                seq_lengths=[256],
            ),
            'full': SweepParams(
                n_values=[2, 4, 8, 16, 32, 64, 100],
                batch_sizes=[4, 8, 16, 32],
                channels=[32, 64, 128, 256],
                seq_lengths=[64, 256, 1024],
            ),
            'ci': SweepParams(
                n_values=[4, 16],
                batch_sizes=[8],
                channels=[64],
                seq_lengths=[256],
            ),
        }

    @classmethod
    def benchmark_job(cls, preset: str = 'full', **overrides) -> Any:
        """Get benchmark job for WideBatchNorm1d."""
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
            name=f'batchnorm1d_{preset}',
            primitive='batchnorm1d',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
            # pack_fn/unpack_fn: use default N-first format [N, B, C, L]
        )

    @staticmethod
    def _bench_model(channels: int, **_) -> nn.BatchNorm1d:
        """Create single BatchNorm1d for benchmarking."""
        return nn.BatchNorm1d(channels)

    @staticmethod
    def _bench_input(n: int, batch_sizes: int, channels: int, seq_lengths: int, device: str = 'cpu', **_) -> Tensor:
        """Create single input tensor [B, C, L]."""
        return torch.randn(batch_sizes, channels, seq_lengths, device=device)

    @classmethod
    def _bench_wide(cls, modules: List[nn.BatchNorm1d], strategy: str) -> 'WideBatchNorm1d':
        """Create WideBatchNorm1d from modules."""
        return cls.from_modules(modules)


__all__ = ['WideBatchNorm1d']