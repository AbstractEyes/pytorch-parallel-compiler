"""
WideDropout - N parallel Dropout layers with independent masks

Fuses N nn.Dropout modules into a single wide operation.
Uses independent dropout masks for each model to preserve ensemble semantics.

Strategies:
- 'independent': Each model gets its own dropout mask (true ensemble)
- 'shared': All models share the same mask (faster, less memory, breaks ensemble)

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional, Tuple


class WideDropout(nn.Module):
    """
    N parallel Dropout layers with independent masks.

    Input:  [N, B, ...] where ... is any number of dimensions
    Output: [N, B, ...]

    Each of the N models gets an independent dropout mask during training,
    preserving true ensemble semantics.
    """

    BENCHMARK_STRATEGIES = ['baseline', 'independent', 'shared']
    BENCHMARK_SWEEPS = {
        'quick': {'n_values': [4, 8, 16, 32]},
        'full': {'n_values': [2, 4, 8, 16, 32, 64]},
        'ci': {'n_values': [4, 8]},
    }

    def __init__(
        self,
        n: int,
        p: float = 0.5,
        inplace: bool = False,
        strategy: str = 'independent',
    ):
        super().__init__()
        self.n = n
        self.p = p
        self.inplace = inplace
        self.strategy = strategy

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply dropout with independent masks per model.

        Args:
            x: [N, B, ...] input tensor

        Returns:
            [N, B, ...] with dropout applied independently to each N
        """
        if not self.training or self.p == 0:
            return x

        if self.strategy == 'shared':
            # Shared mask across all N models (faster but breaks ensemble)
            # Apply dropout to the entire [N, B, ...] tensor at once
            return torch.nn.functional.dropout(x, self.p, True, self.inplace)
        else:
            # Independent masks (strategy == 'independent')
            # Apply dropout separately to each of the N models
            N = x.shape[0]
            results = []
            for i in range(N):
                dropped = torch.nn.functional.dropout(
                    x[i], self.p, True, self.inplace
                )
                results.append(dropped)
            return torch.stack(results, dim=0)

    @classmethod
    def from_modules(
        cls,
        modules: List[nn.Dropout],
        strategy: str = 'independent',
    ) -> 'WideDropout':
        """
        Create WideDropout from N Dropout modules.

        Args:
            modules: List of nn.Dropout modules
            strategy: 'independent' or 'shared'

        Returns:
            WideDropout instance
        """
        n = len(modules)

        # Verify all have same p
        p_values = [m.p for m in modules]
        if not all(p == p_values[0] for p in p_values):
            raise ValueError(f"All dropout modules must have same p, got {p_values}")

        # Verify all have same inplace
        inplace_values = [m.inplace for m in modules]
        if not all(inp == inplace_values[0] for inp in inplace_values):
            raise ValueError(
                f"All dropout modules must have same inplace, got {inplace_values}"
            )

        return cls(
            n=n,
            p=p_values[0],
            inplace=inplace_values[0],
            strategy=strategy,
        )

    def __repr__(self) -> str:
        return (
            f"WideDropout({self.n}x[p={self.p}, inplace={self.inplace}], "
            f"strategy={self.strategy})"
        )

    # =========================================================================
    # Benchmark Interface
    # =========================================================================

    @classmethod
    def benchmark_job(cls, preset: str = 'full'):
        """Create benchmark job for WideDropout."""
        from ..benchmark.benchmark_schema import BenchmarkJob, SweepParams

        sweep_config = cls.BENCHMARK_SWEEPS.get(preset, cls.BENCHMARK_SWEEPS['full'])
        sweep = SweepParams(
            n_values=sweep_config['n_values'],
            batch_sizes=[16],
            seq_lengths=[128, 256],
            channels=[64],
        )

        return BenchmarkJob(
            name=f'dropout_{preset}',
            primitive='dropout',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
        )

    @staticmethod
    def _bench_model(p=0.5, **kwargs):
        """Create a single Dropout module."""
        return nn.Dropout(p=p)

    @staticmethod
    def _bench_input(n: int, device: str, batch_sizes: int, seq_lengths=128, channels=64, **kwargs):
        """Create input: [B, T, C] for sequence case."""
        return torch.randn(batch_sizes, seq_lengths, channels, device=device)

    @classmethod
    def _bench_wide(cls, modules: List[nn.Module], strategy: str):
        """Create wide version for given strategy."""
        if strategy == 'baseline':
            return None  # Use sequential baseline
        return cls.from_modules(modules, strategy=strategy)


__all__ = ['WideDropout']
