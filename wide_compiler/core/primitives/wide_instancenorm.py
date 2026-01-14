"""
WideInstanceNorm - N parallel InstanceNorm layers fused into a single module.

Key insight: InstanceNorm normalizes per-channel per-instance. For N models,
we run a single InstanceNorm over N*C channels.

Supports both 1D and 2D variants via WideInstanceNorm1d and WideInstanceNorm2d.

Strategies:
- 'fused': Single InstanceNorm over N*C channels (FASTEST)
- 'sequential': N separate InstanceNorm calls (baseline)

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Dict, Any, Union, Type
from enum import Enum

import torch
from torch import nn, Tensor


class InstanceNormStrategy(Enum):
    FUSED = 'fused'
    SEQUENTIAL = 'sequential'
    AUTO = 'auto'


class WideInstanceNorm1d(nn.Module):
    """
    N parallel InstanceNorm1d modules fused into single operation.

    Input shape:  [B, N*C, L]
    Output shape: [B, N*C, L]
    """

    def __init__(
            self,
            n: int,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = False,
            track_running_stats: bool = False,
            strategy: Union[str, InstanceNormStrategy] = 'auto',
    ):
        super().__init__()

        self.n = n
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if isinstance(strategy, str):
            strategy = InstanceNormStrategy(strategy)
        if strategy == InstanceNormStrategy.AUTO:
            strategy = InstanceNormStrategy.FUSED

        self._strategy = strategy
        self._use_fused = (strategy == InstanceNormStrategy.FUSED)

        if self._use_fused:
            self.norm = nn.InstanceNorm1d(
                num_features=n * num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
            )
        else:
            self.norms = nn.ModuleList([
                nn.InstanceNorm1d(
                    num_features=num_features,
                    eps=eps,
                    momentum=momentum,
                    affine=affine,
                    track_running_stats=track_running_stats,
                )
                for _ in range(n)
            ])

    @property
    def strategy(self) -> InstanceNormStrategy:
        return self._strategy

    def forward(self, x: Tensor) -> Tensor:
        if self._use_fused:
            return self.norm(x)
        return self._forward_sequential(x)

    def _forward_sequential(self, x: Tensor) -> Tensor:
        C = self.num_features
        outputs = []
        for i in range(self.n):
            xi = x[:, i * C:(i + 1) * C]
            out_i = self.norms[i](xi)
            outputs.append(out_i)
        return torch.cat(outputs, dim=1)

    @classmethod
    def from_modules(
            cls,
            modules: List[nn.InstanceNorm1d],
            strategy: Union[str, InstanceNormStrategy] = 'auto',
    ) -> 'WideInstanceNorm1d':
        n = len(modules)
        t = modules[0]

        wide = cls(
            n=n,
            num_features=t.num_features,
            eps=t.eps,
            momentum=t.momentum,
            affine=t.affine,
            track_running_stats=t.track_running_stats,
            strategy=strategy,
        )

        if t.affine:
            device = t.weight.device
            dtype = t.weight.dtype
            wide = wide.to(device=device, dtype=dtype)

            with torch.no_grad():
                if wide._use_fused:
                    weights = torch.cat([m.weight for m in modules], dim=0)
                    biases = torch.cat([m.bias for m in modules], dim=0)
                    wide.norm.weight.copy_(weights)
                    wide.norm.bias.copy_(biases)
                else:
                    for i, m in enumerate(modules):
                        wide.norms[i].load_state_dict(m.state_dict())

        return wide

    def __repr__(self):
        return (
            f"WideInstanceNorm1d({self.n}x[{self.num_features}], "
            f"strategy={self._strategy.value})"
        )


class WideInstanceNorm2d(nn.Module):
    """
    N parallel InstanceNorm2d modules fused into single operation.

    Input shape:  [B, N*C, H, W]
    Output shape: [B, N*C, H, W]
    """

    def __init__(
            self,
            n: int,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = False,
            track_running_stats: bool = False,
            strategy: Union[str, InstanceNormStrategy] = 'auto',
    ):
        super().__init__()

        self.n = n
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if isinstance(strategy, str):
            strategy = InstanceNormStrategy(strategy)
        if strategy == InstanceNormStrategy.AUTO:
            strategy = InstanceNormStrategy.FUSED

        self._strategy = strategy
        self._use_fused = (strategy == InstanceNormStrategy.FUSED)

        if self._use_fused:
            self.norm = nn.InstanceNorm2d(
                num_features=n * num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
            )
        else:
            self.norms = nn.ModuleList([
                nn.InstanceNorm2d(
                    num_features=num_features,
                    eps=eps,
                    momentum=momentum,
                    affine=affine,
                    track_running_stats=track_running_stats,
                )
                for _ in range(n)
            ])

    @property
    def strategy(self) -> InstanceNormStrategy:
        return self._strategy

    def forward(self, x: Tensor) -> Tensor:
        if self._use_fused:
            return self.norm(x)
        return self._forward_sequential(x)

    def _forward_sequential(self, x: Tensor) -> Tensor:
        C = self.num_features
        outputs = []
        for i in range(self.n):
            xi = x[:, i * C:(i + 1) * C]
            out_i = self.norms[i](xi)
            outputs.append(out_i)
        return torch.cat(outputs, dim=1)

    @classmethod
    def from_modules(
            cls,
            modules: List[nn.InstanceNorm2d],
            strategy: Union[str, InstanceNormStrategy] = 'auto',
    ) -> 'WideInstanceNorm2d':
        n = len(modules)
        t = modules[0]

        wide = cls(
            n=n,
            num_features=t.num_features,
            eps=t.eps,
            momentum=t.momentum,
            affine=t.affine,
            track_running_stats=t.track_running_stats,
            strategy=strategy,
        )

        if t.affine:
            device = t.weight.device
            dtype = t.weight.dtype
            wide = wide.to(device=device, dtype=dtype)

            with torch.no_grad():
                if wide._use_fused:
                    weights = torch.cat([m.weight for m in modules], dim=0)
                    biases = torch.cat([m.bias for m in modules], dim=0)
                    wide.norm.weight.copy_(weights)
                    wide.norm.bias.copy_(biases)
                else:
                    for i, m in enumerate(modules):
                        wide.norms[i].load_state_dict(m.state_dict())

        return wide

    def __repr__(self):
        return (
            f"WideInstanceNorm2d({self.n}x[{self.num_features}], "
            f"strategy={self._strategy.value})"
        )

    # =========================================================================
    # BENCHMARK INTERFACE (on 2D variant)
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
                channels=[32, 64, 128],
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
            name=f'instancenorm_{preset}',
            primitive='instancenorm',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
            pack_fn=cls._bench_pack,
            unpack_fn=cls._bench_unpack,
        )

    @staticmethod
    def _bench_model(channels: int, **_) -> nn.InstanceNorm2d:
        return nn.InstanceNorm2d(channels, affine=True)

    @staticmethod
    def _bench_input(n: int, batch_sizes: int, channels: int, heights: int,
                     widths: int, device: str = 'cpu', **_) -> Tensor:
        return torch.randn(batch_sizes, channels, heights, widths, device=device)

    @classmethod
    def _bench_wide(cls, modules: List[nn.InstanceNorm2d], strategy: str) -> 'WideInstanceNorm2d':
        strat_map = {
            'fused': InstanceNormStrategy.FUSED,
            'sequential': InstanceNormStrategy.SEQUENTIAL,
        }
        return cls.from_modules(modules, strategy=strat_map.get(strategy, InstanceNormStrategy.FUSED))

    @staticmethod
    def _bench_pack(inputs: List[Tensor]) -> Tensor:
        return torch.cat(inputs, dim=1)

    @staticmethod
    def _bench_unpack(output: Tensor, n: int) -> List[Tensor]:
        B, NC = output.shape[:2]
        C = NC // n
        return [output[:, i * C:(i + 1) * C] for i in range(n)]


__all__ = ['WideInstanceNorm1d', 'WideInstanceNorm2d', 'InstanceNormStrategy']