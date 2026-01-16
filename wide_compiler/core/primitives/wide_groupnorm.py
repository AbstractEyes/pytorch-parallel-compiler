"""
WideGroupNorm - N parallel GroupNorm layers fused into a single module.

Key insight: GroupNorm normalizes within channel groups. For N models,
we can run a single GroupNorm with N*num_groups groups.

Strategies:
- 'fused': Single GroupNorm with N*num_groups (FASTEST)
- 'sequential': N separate GroupNorm calls (baseline)

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Dict, Any, Union
from enum import Enum

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class GroupNormStrategy(Enum):
    FUSED = 'fused'
    SEQUENTIAL = 'sequential'
    AUTO = 'auto'


class WideGroupNorm(nn.Module):
    """
    N parallel GroupNorm modules fused into single operation.

    Input shape:  [B, N*C, ...] (any spatial dims)
    Output shape: [B, N*C, ...]

    Strategies:
        'fused': Single GroupNorm with N*num_groups groups
        'sequential': N separate GroupNorm calls
    """

    def __init__(
        self,
        n: int,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        strategy: Union[str, GroupNormStrategy] = 'auto',
    ):
        super().__init__()

        self.n = n
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if isinstance(strategy, str):
            strategy = GroupNormStrategy(strategy)
        if strategy == GroupNormStrategy.AUTO:
            strategy = GroupNormStrategy.FUSED

        self._strategy = strategy
        self._use_fused = (strategy == GroupNormStrategy.FUSED)

        if self._use_fused:
            # Single GroupNorm with N*num_groups groups over N*num_channels
            self.norm = nn.GroupNorm(
                num_groups=n * num_groups,
                num_channels=n * num_channels,
                eps=eps,
                affine=affine,
            )
        else:
            # N separate GroupNorms
            self.norms = nn.ModuleList([
                nn.GroupNorm(
                    num_groups=num_groups,
                    num_channels=num_channels,
                    eps=eps,
                    affine=affine,
                )
                for _ in range(n)
            ])

    @property
    def strategy(self) -> GroupNormStrategy:
        return self._strategy

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with N-first format.

        Input:  [N, B, C, ...] any number of spatial dimensions
        Output: [N, B, C, ...]
        """
        N = x.shape[0]
        B = x.shape[1]
        C = x.shape[2]
        spatial = x.shape[3:]

        # Convert N-first to channel-packed
        # [N, B, C, ...] -> [B, N, C, ...] -> [B, N*C, ...]
        x = x.permute(1, 0, 2, *range(3, x.dim()))
        x = x.reshape(B, N * C, *spatial)

        # Run groupnorm
        if self._use_fused:
            out = self.norm(x)
        else:
            out = self._forward_sequential_internal(x)

        # Convert back: [B, N*C, ...] -> [N, B, C, ...]
        out = out.view(B, N, C, *spatial)
        out = out.permute(1, 0, 2, *range(3, out.dim()))

        return out.contiguous()

    def _forward_sequential_internal(self, x: Tensor) -> Tensor:
        """N separate GroupNorm calls on channel-packed input."""
        C = self.num_channels
        shape = x.shape
        B = shape[0]
        spatial = shape[2:]

        outputs = []
        for i in range(self.n):
            xi = x[:, i*C:(i+1)*C]  # [B, C, ...]
            out_i = self.norms[i](xi)
            outputs.append(out_i)

        return torch.cat(outputs, dim=1)

    @classmethod
    def from_modules(
        cls,
        modules: List[nn.GroupNorm],
        strategy: Union[str, GroupNormStrategy] = 'auto',
    ) -> 'WideGroupNorm':
        """Create from N existing GroupNorm modules."""
        n = len(modules)
        t = modules[0]

        wide = cls(
            n=n,
            num_groups=t.num_groups,
            num_channels=t.num_channels,
            eps=t.eps,
            affine=t.affine,
            strategy=strategy,
        )

        # Get device/dtype from weight or create dummy
        if t.affine:
            device = t.weight.device
            dtype = t.weight.dtype
        else:
            device = torch.device('cpu')
            dtype = torch.float32

        wide = wide.to(device=device, dtype=dtype)

        with torch.no_grad():
            if wide._use_fused and wide.affine:
                # Stack weights and biases
                weights = torch.cat([m.weight for m in modules], dim=0)
                biases = torch.cat([m.bias for m in modules], dim=0)
                wide.norm.weight.copy_(weights)
                wide.norm.bias.copy_(biases)
            elif not wide._use_fused:
                for i, m in enumerate(modules):
                    if wide.affine:
                        wide.norms[i].load_state_dict(m.state_dict())

        return wide

    def __repr__(self):
        return (
            f"WideGroupNorm({self.n}x[groups={self.num_groups}, "
            f"channels={self.num_channels}], strategy={self._strategy.value})"
        )

    # =========================================================================
    # BENCHMARK INTERFACE
    # =========================================================================

    BENCHMARK_STRATEGIES = ['baseline', 'fused', 'sequential']

    @classmethod
    def _get_sweep_params_class(cls):
        try:
            from ..benchmark.benchmark_schema import SweepParams
            return SweepParams
        except ImportError:
            pass
        try:
            from wide_compiler.core.benchmark.benchmark_schema import SweepParams
            return SweepParams
        except ImportError:
            return None

    @classmethod
    def _get_benchmark_job_class(cls):
        try:
            from ..benchmark.benchmark_schema import BenchmarkJob
            return BenchmarkJob
        except ImportError:
            pass
        try:
            from wide_compiler.core.benchmark.benchmark_schema import BenchmarkJob
            return BenchmarkJob
        except ImportError:
            return None

    BENCHMARK_SWEEPS: Dict[str, Any] = {}
    _SWEEPS_INITIALIZED = False

    @classmethod
    def _init_benchmark_sweeps(cls):
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
                n_values=[2, 4, 8, 16, 32, 64],
                batch_sizes=[4, 8, 16],
                channels=[32, 64, 128, 256],
                heights=[16, 32, 64],
                widths=[16, 32, 64],
            ),
            'ci': SweepParams(
                n_values=[4, 8],
                batch_sizes=[4],
                channels=[32],
                heights=[16],
                widths=[16],
            ),
        }

    @classmethod
    def benchmark_job(cls, preset: str = 'full', **overrides) -> Any:
        cls._init_benchmark_sweeps()

        BenchmarkJob = cls._get_benchmark_job_class()
        if BenchmarkJob is None:
            raise ImportError("Could not import BenchmarkJob")

        sweep = cls.BENCHMARK_SWEEPS.get(preset)
        if sweep is None:
            raise ValueError(f"Unknown preset '{preset}'")

        if overrides:
            sweep = sweep.with_overrides(**overrides)

        return BenchmarkJob(
            name=f'groupnorm_{preset}',
            primitive='groupnorm',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
            # pack_fn/unpack_fn: use default N-first format
        )

    @staticmethod
    def _bench_model(channels: int, **_) -> nn.GroupNorm:
        # 8 groups is common default
        num_groups = min(8, channels)
        return nn.GroupNorm(num_groups, channels)

    @staticmethod
    def _bench_input(n: int, batch_sizes: int, channels: int, heights: int,
                     widths: int, device: str = 'cpu', **_) -> Tensor:
        return torch.randn(batch_sizes, channels, heights, widths, device=device)

    @classmethod
    def _bench_wide(cls, modules: List[nn.GroupNorm], strategy: str) -> 'WideGroupNorm':
        strat_map = {
            'fused': GroupNormStrategy.FUSED,
            'sequential': GroupNormStrategy.SEQUENTIAL,
        }
        return cls.from_modules(modules, strategy=strat_map.get(strategy, GroupNormStrategy.FUSED))


__all__ = ['WideGroupNorm', 'GroupNormStrategy']