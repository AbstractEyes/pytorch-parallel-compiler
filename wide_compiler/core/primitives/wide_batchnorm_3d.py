"""
WideBatchNorm3d - N parallel BatchNorm3d operations fused into a single module.

BatchNorm3d fusion works the same as BatchNorm2d - we create a single BatchNorm3d
with N*num_features channels. This works because BatchNorm operates independently
per channel.

Expected speedup: 30-40x based on BatchNorm2d results on A100.

Use cases: 3D medical imaging, video processing, volumetric CNNs.

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Dict, Any

import torch
from torch import nn, Tensor


class WideBatchNorm3d(nn.Module):
    """
    N parallel BatchNorm3d as single BatchNorm3d.

    Input shape:  [N, B, C, D, H, W]
    Output shape: [N, B, C, D, H, W]

    BatchNorm operates independently per channel, so fusion is exact.
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

        self.op = nn.BatchNorm3d(
            num_features=n * num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with N-first format.

        Input:  [N, B, C, D, H, W]
        Output: [N, B, C, D, H, W]
        """
        N, B, C, D, H, W = x.shape

        # Convert N-first to channel-packed: [N, B, C, D, H, W] -> [B, N*C, D, H, W]
        x = x.permute(1, 0, 2, 3, 4, 5).reshape(B, N * C, D, H, W)

        # Run batchnorm
        out = self.op(x)

        # Convert back to N-first: [B, N*C, D, H, W] -> [N, B, C, D, H, W]
        out = out.view(B, N, C, D, H, W).permute(1, 0, 2, 3, 4, 5)

        return out.contiguous()

    @classmethod
    def from_modules(cls, modules: List[nn.BatchNorm3d]) -> 'WideBatchNorm3d':
        """Create from N existing BatchNorm3d modules."""
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
        return f"WideBatchNorm3d({self.n}x{self.num_features})"

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
    def _init_sweeps(cls):
        """Lazy initialization of benchmark sweeps."""
        if cls._SWEEPS_INITIALIZED:
            return

        SweepParams = cls._get_sweep_params_class()
        if SweepParams is None:
            return

        cls.BENCHMARK_SWEEPS = {
            'quick': SweepParams(
                n_values=[4, 8, 16, 32],
                batch_sizes=[8],
                channels=[64],
                depths=[16],      # Depth dimension
                heights=[16],     # Height dimension
                widths=[16],      # Width dimension
            ),
            'full': SweepParams(
                n_values=[2, 4, 8, 16, 32, 64],
                batch_sizes=[8],
                channels=[32, 64, 128],
                depths=[8, 16],
                heights=[16, 32],
                widths=[16, 32],
            ),
            'ci': SweepParams(
                n_values=[4, 8],
                batch_sizes=[4],
                channels=[32],
                depths=[8],
                heights=[8],
                widths=[8],
            ),
        }
        cls._SWEEPS_INITIALIZED = True

    @classmethod
    def benchmark_job(cls, preset: str = 'full'):
        """
        Create benchmark job for WideBatchNorm3d.

        Args:
            preset: 'quick', 'full', or 'ci'

        Returns:
            BenchmarkJob for this primitive
        """
        cls._init_sweeps()

        BenchmarkJob = cls._get_benchmark_job_class()
        if BenchmarkJob is None:
            raise ImportError("Could not import BenchmarkJob")

        sweep = cls.BENCHMARK_SWEEPS.get(preset, cls.BENCHMARK_SWEEPS['full'])

        return BenchmarkJob(
            name=f'batchnorm3d_{preset}',
            primitive='batchnorm3d',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
        )

    @staticmethod
    def _bench_model(
        channels: int = 64,
        depths: int = 16,
        heights: int = 16,
        widths: int = 16,
        **kwargs
    ):
        """Create a single BatchNorm3d module."""
        return nn.BatchNorm3d(num_features=channels)

    @staticmethod
    def _bench_input(
        n: int,
        device: str,
        batch_sizes: int,
        channels: int = 64,
        depths: int = 16,
        heights: int = 16,
        widths: int = 16,
        **kwargs
    ):
        """Create input tensor [B, C, D, H, W]."""
        return torch.randn(
            batch_sizes, channels, depths, heights, widths,
            device=device
        )

    @classmethod
    def _bench_wide(cls, modules: List[nn.Module], strategy: str):
        """Create wide version for given strategy."""
        if strategy == 'baseline':
            return None
        return cls.from_modules(modules)


__all__ = ['WideBatchNorm3d']
