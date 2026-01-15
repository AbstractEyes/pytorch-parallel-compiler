"""
WideBatchNorm2d - N parallel BatchNorm2d operations fused into a single module.

BatchNorm2d fusion is straightforward - we just create a single BatchNorm2d
with N*num_features channels. This works because BatchNorm operates independently
per channel.

Expected speedup: Modest (1.2-2x) due to reduced kernel launch overhead.
BatchNorm is already memory-bound, so fusion mainly helps with launch overhead.

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Dict, Any

import torch
from torch import nn, Tensor


class WideBatchNorm2d(nn.Module):
    """
    N parallel BatchNorm2d as single BatchNorm2d.

    Input shape:  [B, N*C, H, W]
    Output shape: [B, N*C, H, W]

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

        self.op = nn.BatchNorm2d(
            num_features=n * num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with N-first format.

        Input:  [N, B, C, H, W]
        Output: [N, B, C, H, W]
        """
        N, B, C, H, W = x.shape

        # Convert N-first to channel-packed: [N, B, C, H, W] -> [B, N*C, H, W]
        x = x.permute(1, 0, 2, 3, 4).reshape(B, N * C, H, W)

        # Run batchnorm
        out = self.op(x)

        # Convert back to N-first: [B, N*C, H, W] -> [N, B, C, H, W]
        out = out.view(B, N, C, H, W).permute(1, 0, 2, 3, 4)

        return out.contiguous()

    @classmethod
    def from_modules(cls, modules: List[nn.BatchNorm2d]) -> 'WideBatchNorm2d':
        """Create from N existing BatchNorm2d modules."""
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
        return f"WideBatchNorm2d({self.n}x{self.num_features})"

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
                channels=[64],
                heights=[32],
                widths=[32],
            ),
            'full': SweepParams(
                n_values=[2, 4, 8, 16, 32, 64, 100],
                batch_sizes=[4, 8, 16, 32],
                channels=[32, 64, 128, 256],
                heights=[16, 32, 56],
                widths=[16, 32, 56],
            ),
            'ci': SweepParams(
                n_values=[4, 16],
                batch_sizes=[8],
                channels=[64],
                heights=[32],
                widths=[32],
            ),
        }

    @classmethod
    def benchmark_job(cls, preset: str = 'full', **overrides) -> Any:
        """Get benchmark job for WideBatchNorm2d."""
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
            name=f'batchnorm2d_{preset}',
            primitive='batchnorm2d',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
            pack_fn=cls._bench_pack,
            unpack_fn=cls._bench_unpack,
        )

    @staticmethod
    def _bench_model(channels: int, **_) -> nn.BatchNorm2d:
        """Create single BatchNorm2d for benchmarking."""
        return nn.BatchNorm2d(channels)

    @staticmethod
    def _bench_input(n: int, batch_sizes: int, channels: int, heights: int, widths: int, device: str = 'cpu', **_) -> Tensor:
        """Create single input tensor."""
        return torch.randn(batch_sizes, channels, heights, widths, device=device)

    @classmethod
    def _bench_wide(cls, modules: List[nn.BatchNorm2d], strategy: str) -> 'WideBatchNorm2d':
        """Create WideBatchNorm2d from modules."""
        return cls.from_modules(modules)

    @staticmethod
    def _bench_pack(inputs: List[Tensor]) -> Tensor:
        """Pack N inputs into wide format."""
        stacked = torch.stack(inputs, dim=1)  # [B, N, C, H, W]
        B, N, C, H, W = stacked.shape
        return stacked.view(B, N * C, H, W)

    @staticmethod
    def _bench_unpack(output: Tensor, n: int) -> List[Tensor]:
        """Unpack wide output to N outputs."""
        B, NC, H, W = output.shape
        C = NC // n
        reshaped = output.view(B, n, C, H, W)
        return [reshaped[:, i] for i in range(n)]


__all__ = ['WideBatchNorm2d']