"""
WideAdaptiveAvgPool2d - N parallel AdaptiveAvgPool2d operations.

Common in CNNs (ResNet, EfficientNet, etc.) for global pooling before classifier.
Expected speedup: 20-50x (trivial operation, high batching gain)

Strategies:
- 'batched': Single adaptive_avg_pool2d on channel-packed tensor (FASTEST)
- 'sequential': N separate adaptive_avg_pool2d calls (baseline)

Input/Output Format (v0.6.0):
- Input:  [N, B, C, H, W]           (N-first)
- Output: [N, B, C, H_out, W_out]   (N-first)

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Union, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class WideAdaptiveAvgPool2d(nn.Module):
    """
    N parallel AdaptiveAvgPool2d layers.

    Adaptive average pooling - outputs a fixed size regardless of input size.
    Used for global pooling in ResNets and many other CNN architectures.
    """

    BENCHMARK_STRATEGIES = ['baseline', 'batched', 'sequential']

    def __init__(
        self,
        n: int,
        output_size: Union[int, Tuple[int, int]],
        strategy: str = 'batched',
    ):
        super().__init__()
        self.n = n
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
        self._strategy = strategy
        self._use_batched = (strategy == 'batched')

    @property
    def strategy(self) -> str:
        return self._strategy

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with N-first format.

        Args:
            x: [N, B, C, H, W]

        Returns:
            [N, B, C, H_out, W_out]
        """
        if self._use_batched:
            return self._forward_batched(x)
        else:
            return self._forward_sequential(x)

    def _forward_batched(self, x: Tensor) -> Tensor:
        """Batched adaptive pooling (fastest)."""
        N, B, C, H, W = x.shape

        # Convert N-first to channel-packed: [N, B, C, H, W] -> [B, N*C, H, W]
        x_packed = x.permute(1, 0, 2, 3, 4).reshape(B, N * C, H, W)

        # Single adaptive pooling call
        out_packed = F.adaptive_avg_pool2d(x_packed, self.output_size)

        # Convert back to N-first: [B, N*C, H', W'] -> [N, B, C, H', W']
        B, NC, H_out, W_out = out_packed.shape
        out = out_packed.view(B, N, C, H_out, W_out).permute(1, 0, 2, 3, 4)

        return out.contiguous()

    def _forward_sequential(self, x: Tensor) -> Tensor:
        """Sequential adaptive pooling (baseline)."""
        N, B, C, H, W = x.shape

        outputs = []
        for i in range(N):
            out_i = F.adaptive_avg_pool2d(x[i], self.output_size)
            outputs.append(out_i)

        return torch.stack(outputs, dim=0)

    @classmethod
    def from_modules(cls, modules: List[nn.AdaptiveAvgPool2d], strategy: str = 'batched') -> 'WideAdaptiveAvgPool2d':
        """Create from N existing AdaptiveAvgPool2d modules."""
        n = len(modules)
        t = modules[0]

        # Verify all have same output_size
        for i, m in enumerate(modules):
            if m.output_size != t.output_size:
                raise ValueError(f"Module {i} has output_size={m.output_size}, expected {t.output_size}")

        return cls(n=n, output_size=t.output_size, strategy=strategy)

    def __repr__(self):
        return f"WideAdaptiveAvgPool2d({self.n}x[output_size={self.output_size}], strategy={self._strategy})"

    # =========================================================================
    # BENCHMARK INTERFACE
    # =========================================================================

    BENCHMARK_SWEEPS = {
        'quick': {'n_values': [4, 8, 16, 32]},
        'full': {'n_values': [2, 4, 8, 16, 32, 64]},
        'ci': {'n_values': [4, 8]},
    }

    @classmethod
    def benchmark_job(cls, preset: str = 'full'):
        """Create benchmark job for WideAdaptiveAvgPool2d."""
        from ..benchmark.benchmark_schema import BenchmarkJob, SweepParams

        sweep_config = cls.BENCHMARK_SWEEPS.get(preset, cls.BENCHMARK_SWEEPS['full'])
        sweep = SweepParams(
            n_values=sweep_config['n_values'],
            batch_sizes=[16],
            channels=[64, 256],
            heights=[56, 32],
            widths=[56, 32],
        )

        return BenchmarkJob(
            name=f'adaptive_avgpool2d_{preset}',
            primitive='adaptive_avgpool2d',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
        )

    @staticmethod
    def _bench_model(**kwargs):
        """
        Create a single AdaptiveAvgPool2d module.
        Uses fixed output_size=(1, 1) for global pooling (most common use case).
        """
        return nn.AdaptiveAvgPool2d(output_size=(1, 1))

    @staticmethod
    def _bench_input(n: int, device: str, batch_sizes: int, channels=64, heights=56, widths=56, **kwargs):
        """Create input tensor [B, C, H, W]."""
        return torch.randn(batch_sizes, channels, heights, widths, device=device)

    @classmethod
    def _bench_wide(cls, modules: List[nn.Module], strategy: str):
        """Create wide version for given strategy."""
        if strategy == 'baseline':
            return None
        return cls.from_modules(modules, strategy=strategy)


__all__ = ['WideAdaptiveAvgPool2d']
