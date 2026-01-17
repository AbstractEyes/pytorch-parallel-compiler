"""
WidePReLU - N parallel PReLU layers

Fuses N nn.PReLU modules into a single wide operation.
PReLU: f(x) = max(0, x) + a * min(0, x) where a is learnable.

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional


class WidePReLU(nn.Module):
    """
    N parallel PReLU layers.

    Input:  [N, B, C, ...]
    Output: [N, B, C, ...]

    PReLU applies: f(x) = max(0, x) + weight * min(0, x)
    where weight is a learnable parameter (one per channel or one shared).
    """

    BENCHMARK_STRATEGIES = ['baseline', 'wide']
    BENCHMARK_SWEEPS = {
        'quick': {'n_values': [4, 8, 16, 32]},
        'full': {'n_values': [2, 4, 8, 16, 32, 64]},
        'ci': {'n_values': [4, 8]},
    }

    def __init__(
        self,
        n: int,
        num_parameters: int = 1,
        init: float = 0.25,
    ):
        """
        Args:
            n: Number of parallel models
            num_parameters: Number of parameters (1 for shared, C for per-channel)
            init: Initial value for parameters
        """
        super().__init__()
        self.n = n
        self.num_parameters = num_parameters

        # Parameters: [N, num_parameters]
        self.weight = nn.Parameter(torch.empty(n, num_parameters).fill_(init))

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply PReLU with batched parameters.

        Args:
            x: [N, B, C, ...] or [N, B, ...]

        Returns:
            [N, B, C, ...] with PReLU applied
        """
        N, B = x.shape[0], x.shape[1]

        # Reshape weight for broadcasting
        if self.num_parameters == 1:
            # Shared: weight is [N, 1], broadcast to all dimensions
            # Need to reshape to [N, 1, 1, ...]
            weight = self.weight.view(N, *([1] * (x.ndim - 1)))
        else:
            # Per-channel: weight is [N, C], broadcast to [N, B, C, ...]
            # Reshape to [N, 1, C, 1, ...]
            weight = self.weight.view(N, 1, self.num_parameters, *([1] * (x.ndim - 3)))

        # PReLU formula: max(0, x) + weight * min(0, x)
        return torch.where(x > 0, x, x * weight)

    @classmethod
    def from_modules(
        cls,
        modules: List[nn.PReLU],
    ) -> 'WidePReLU':
        """
        Create WidePReLU from N PReLU modules.

        Args:
            modules: List of nn.PReLU modules

        Returns:
            WidePReLU instance with stacked parameters
        """
        n = len(modules)

        # Get num_parameters from first module
        num_parameters = modules[0].num_parameters

        # Verify all have same num_parameters
        for i, m in enumerate(modules):
            if m.num_parameters != num_parameters:
                raise ValueError(
                    f"All PReLU modules must have same num_parameters, "
                    f"got {m.num_parameters} at index {i}, expected {num_parameters}"
                )

        # Create wide module
        wide = cls(n=n, num_parameters=num_parameters)

        # Stack parameters: [N, num_parameters]
        with torch.no_grad():
            for i, m in enumerate(modules):
                wide.weight[i] = m.weight

        return wide

    def __repr__(self) -> str:
        return f"WidePReLU({self.n}x[num_parameters={self.num_parameters}])"

    # =========================================================================
    # Benchmark Interface
    # =========================================================================

    @classmethod
    def benchmark_job(cls, preset: str = 'full'):
        """Create benchmark job for WidePReLU."""
        from ..benchmark.benchmark_schema import BenchmarkJob, SweepParams

        sweep_config = cls.BENCHMARK_SWEEPS.get(preset, cls.BENCHMARK_SWEEPS['full'])
        sweep = SweepParams(
            n_values=sweep_config['n_values'],
            batch_sizes=[16],
            custom_params=[
                {'channels': 64, 'spatial': (32, 32)},   # 2D spatial
                {'channels': 128, 'spatial': (16, 16)},  # Smaller spatial
                {'channels': 256, 'spatial': None},      # Just channels (1D)
            ]
        )

        return BenchmarkJob(
            name=f'prelu_{preset}',
            primitive='prelu',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
        )

    @staticmethod
    def _bench_model(channels=64, spatial=(32, 32), **kwargs):
        """Create a single PReLU module (per-channel)."""
        return nn.PReLU(num_parameters=channels)

    @staticmethod
    def _bench_input(
        n: int,
        device: str,
        batch_sizes: int,
        channels=64,
        spatial=(32, 32),
        **kwargs
    ):
        """Create input: [B, C, ...] matching channels and spatial dims."""
        if spatial is None:
            # 1D case: [B, C]
            return torch.randn(batch_sizes, channels, device=device)
        else:
            # 2D case: [B, C, H, W]
            return torch.randn(batch_sizes, channels, *spatial, device=device)

    @classmethod
    def _bench_wide(cls, modules: List[nn.Module], strategy: str):
        """Create wide version for given strategy."""
        if strategy == 'baseline':
            return None
        return cls.from_modules(modules)


__all__ = ['WidePReLU']
