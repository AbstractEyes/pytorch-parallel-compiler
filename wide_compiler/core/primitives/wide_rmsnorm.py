"""
WideRMSNorm - N parallel RMSNorm layers.

RMSNorm (Root Mean Square Layer Normalization) is a simplified normalization
technique that re-centers and re-scales activations using only the RMS statistic.
Faster than LayerNorm (no mean computation, no bias).

Common in modern LLMs (LLaMA, Mistral, etc.) and diffusion models (Flux).

Expected speedup: 12-15x (similar to LayerNorm, but simpler computation)

Strategies:
- 'batched': Batched normalization (FASTEST)
- 'sequential': N separate RMSNorm ops (baseline)

Input/Output Format (v0.7.0):
- Input:  [N, B, ..., D]  (N-first, any spatial dims)
- Output: [N, B, ..., D]  (N-first)

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class WideRMSNorm(nn.Module):
    """
    N parallel RMSNorm layers fused into a single module.

    RMSNorm: x_norm = x / RMS(x) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)
    """

    BENCHMARK_STRATEGIES = ['baseline', 'batched', 'sequential']

    def __init__(
        self,
        n: int,
        normalized_shape: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        strategy: str = 'batched',
    ):
        super().__init__()
        self.n = n
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self._strategy = strategy

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(n, normalized_shape))
        else:
            self.register_parameter('weight', None)

    @property
    def strategy(self) -> str:
        return self._strategy

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with N-first format.

        Input:  [N, B, ..., D]
        Output: [N, B, ..., D]
        """
        if self._strategy == 'batched':
            return self._forward_batched(x)
        else:
            return self._forward_sequential(x)

    def _forward_batched(self, x: Tensor) -> Tensor:
        """Batched RMSNorm across N models."""
        # x: [N, B, ..., D]
        # Compute RMS over last dimension for each [N, B, ...] position
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms

        if self.elementwise_affine:
            # Broadcast weight: [N, D] -> [N, 1, ..., 1, D]
            weight_shape = [self.n] + [1] * (x.ndim - 2) + [self.normalized_shape]
            weight = self.weight.view(weight_shape)
            x_norm = x_norm * weight

        return x_norm

    def _forward_sequential(self, x: Tensor) -> Tensor:
        """Sequential RMSNorm (baseline)."""
        N = x.shape[0]
        outputs = []

        for i in range(N):
            x_i = x[i]  # [B, ..., D]
            rms = torch.sqrt(torch.mean(x_i * x_i, dim=-1, keepdim=True) + self.eps)
            x_norm = x_i / rms

            if self.elementwise_affine:
                x_norm = x_norm * self.weight[i]

            outputs.append(x_norm)

        return torch.stack(outputs, dim=0)

    @classmethod
    def from_modules(cls, modules: List[nn.Module], strategy: str = 'batched') -> 'WideRMSNorm':
        """
        Create from N existing RMSNorm-like modules.

        Note: PyTorch doesn't have a built-in RMSNorm, so this expects
        custom modules with .weight and .eps attributes.
        """
        n = len(modules)
        t = modules[0]

        # Get normalized_shape from weight shape
        normalized_shape = t.weight.shape[0] if hasattr(t, 'weight') and t.weight is not None else None
        if normalized_shape is None:
            raise ValueError("Cannot determine normalized_shape from modules")

        wide = cls(
            n=n,
            normalized_shape=normalized_shape,
            eps=getattr(t, 'eps', 1e-6),
            elementwise_affine=(t.weight is not None),
            strategy=strategy,
        )

        wide = wide.to(device=t.weight.device, dtype=t.weight.dtype)

        # Copy weights
        if wide.elementwise_affine:
            with torch.no_grad():
                for i, m in enumerate(modules):
                    wide.weight[i] = m.weight

        return wide

    def __repr__(self):
        return (f"WideRMSNorm({self.n}x[normalized_shape={self.normalized_shape}, "
                f"eps={self.eps}], strategy={self._strategy})")

    # =========================================================================
    # BENCHMARK INTERFACE
    # =========================================================================

    BENCHMARK_STRATEGIES = ['baseline', 'batched', 'sequential']
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
            ),
            'full': SweepParams(
                n_values=[2, 4, 8, 16, 32, 64],
                batch_sizes=[4, 8, 16],
                seq_lengths=[64, 128, 256],
                channels=[128, 256, 512],
            ),
            'ci': SweepParams(
                n_values=[4, 8],
                batch_sizes=[8],
                seq_lengths=[64],
                channels=[128],
            ),
        }

    @classmethod
    def benchmark_job(cls, preset: str = 'full'):
        """Create benchmark job for WideRMSNorm."""
        cls._init_benchmark_sweeps()
        from ..benchmark.benchmark_schema import BenchmarkJob

        sweep = cls.BENCHMARK_SWEEPS.get(preset, cls.BENCHMARK_SWEEPS['full'])

        return BenchmarkJob(
            name=f'rmsnorm_{preset}',
            primitive='rmsnorm',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
        )

    @staticmethod
    def _bench_model(channels=256, **kwargs):
        """Create a single RMSNorm module."""
        class RMSNorm(nn.Module):
            def __init__(self, dim, eps=1e-6):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(dim))

            def forward(self, x):
                rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
                x_norm = x / rms
                return x_norm * self.weight

        return RMSNorm(channels)

    @staticmethod
    def _bench_input(n: int, device: str, batch_sizes: int, seq_lengths: int, channels: int, **kwargs):
        """Create input for RMSNorm benchmark."""
        return torch.randn(batch_sizes, seq_lengths, channels, device=device)

    @classmethod
    def _bench_wide(cls, modules: List[nn.Module], strategy: str):
        """Create wide version for given strategy."""
        if strategy == 'baseline':
            return None
        return cls.from_modules(modules, strategy=strategy)


__all__ = ['WideRMSNorm']
