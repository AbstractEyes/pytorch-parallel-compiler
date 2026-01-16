"""
WideConv3d - N parallel Conv3d layers fused into a single grouped convolution.

Key insight: nn.Conv3d(groups=N) processes N independent channel groups
in one kernel launch.

Strategies:
- 'grouped': Single Conv3d with groups=N (FASTEST)
- 'sequential': N separate Conv3d calls (baseline)

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Dict, Any, Union, Tuple
from enum import Enum

import torch
from torch import nn, Tensor


class Conv3dStrategy(Enum):
    GROUPED = 'grouped'
    SEQUENTIAL = 'sequential'
    AUTO = 'auto'


class WideConv3d(nn.Module):
    """
    N parallel Conv3d modules fused into single grouped convolution.

    Input shape:  [B, N*C_in, D, H, W]
    Output shape: [B, N*C_out, D', H', W']

    Strategies:
        'grouped': Single Conv3d with groups=N (fastest for N >= 4)
        'sequential': N separate Conv3d calls
    """

    def __init__(
        self,
        n: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
        strategy: Union[str, Conv3dStrategy] = 'auto',
    ):
        super().__init__()

        self.n = n
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.has_bias = bias

        if isinstance(strategy, str):
            strategy = Conv3dStrategy(strategy)
        if strategy == Conv3dStrategy.AUTO:
            strategy = Conv3dStrategy.GROUPED

        self._strategy = strategy
        self._use_grouped = (strategy == Conv3dStrategy.GROUPED)

        if self._use_grouped:
            # Single grouped conv
            self.conv = nn.Conv3d(
                in_channels=n * in_channels,
                out_channels=n * out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=n,
                bias=bias,
            )
        else:
            # N separate convs
            self.convs = nn.ModuleList([
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=bias,
                )
                for _ in range(n)
            ])

    @property
    def strategy(self) -> Conv3dStrategy:
        return self._strategy

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with N-first format.

        Input:  [N, B, C_in, D, H, W]
        Output: [N, B, C_out, D', H', W']
        """
        N, B, C, D, H, W = x.shape

        # Convert N-first to channel-packed: [N, B, C, D, H, W] -> [B, N*C, D, H, W]
        x = x.permute(1, 0, 2, 3, 4, 5).reshape(B, N * C, D, H, W)

        # Run conv in channel-packed format
        if self._use_grouped:
            out = self.conv(x)
        else:
            out = self._forward_sequential_internal(x)

        # Convert back to N-first: [B, N*C_out, D', H', W'] -> [N, B, C_out, D', H', W']
        B, NC_out, D_out, H_out, W_out = out.shape
        C_out = NC_out // N
        out = out.view(B, N, C_out, D_out, H_out, W_out).permute(1, 0, 2, 3, 4, 5)

        return out.contiguous()

    def _forward_sequential_internal(self, x: Tensor) -> Tensor:
        """N separate Conv3d calls on channel-packed input."""
        B, NC, D, H, W = x.shape
        C_in = self.in_channels

        outputs = []
        for i in range(self.n):
            xi = x[:, i*C_in:(i+1)*C_in]  # [B, C_in, D, H, W]
            out_i = self.convs[i](xi)
            outputs.append(out_i)

        return torch.cat(outputs, dim=1)  # [B, N*C_out, D', H', W']

    @classmethod
    def from_modules(
        cls,
        modules: List[nn.Conv3d],
        strategy: Union[str, Conv3dStrategy] = 'auto',
    ) -> 'WideConv3d':
        """Create from N existing Conv3d modules."""
        n = len(modules)
        t = modules[0]

        wide = cls(
            n=n,
            in_channels=t.in_channels,
            out_channels=t.out_channels,
            kernel_size=t.kernel_size,
            stride=t.stride,
            padding=t.padding,
            dilation=t.dilation,
            bias=t.bias is not None,
            strategy=strategy,
        )

        device = t.weight.device
        dtype = t.weight.dtype
        wide = wide.to(device=device, dtype=dtype)

        with torch.no_grad():
            if wide._use_grouped:
                # Stack weights into grouped conv
                # weight: [out_ch, in_ch, kD, kH, kW] per module
                # grouped: [N*out_ch, in_ch, kD, kH, kW]
                weights = torch.cat([m.weight for m in modules], dim=0)
                wide.conv.weight.copy_(weights)
                if wide.has_bias:
                    biases = torch.cat([m.bias for m in modules], dim=0)
                    wide.conv.bias.copy_(biases)
            else:
                for i, m in enumerate(modules):
                    wide.convs[i].load_state_dict(m.state_dict())

        return wide

    def __repr__(self):
        return (
            f"WideConv3d({self.n}x[{self.in_channels}→{self.out_channels}, "
            f"k={self.kernel_size}], strategy={self._strategy.value})"
        )

    # =========================================================================
    # BENCHMARK INTERFACE
    # =========================================================================

    BENCHMARK_STRATEGIES = ['baseline', 'grouped', 'sequential']

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
                n_values=[4, 8, 16],
                batch_sizes=[4],
                channels=[32],
                kernel_sizes=[3],
                # Using heights/widths for D, H, W (all same for simplicity)
                heights=[16],
                widths=[16],
            ),
            'full': SweepParams(
                n_values=[2, 4, 8, 16, 32],
                batch_sizes=[2, 4, 8],
                channels=[16, 32, 64],
                kernel_sizes=[3],
                heights=[8, 16, 32],
                widths=[16, 32],
            ),
            'ci': SweepParams(
                n_values=[4, 8],
                batch_sizes=[2],
                channels=[16],
                kernel_sizes=[3],
                heights=[8],
                widths=[8],
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
            name=f'conv3d_{preset}',
            primitive='conv3d',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
            # pack_fn/unpack_fn: use default N-first format
        )

    @staticmethod
    def _bench_model(channels: int, kernel_sizes: int, **_) -> nn.Conv3d:
        padding = kernel_sizes // 2
        return nn.Conv3d(channels, channels, kernel_sizes, padding=padding)

    @staticmethod
    def _bench_input(n: int, batch_sizes: int, channels: int, heights: int,
                     widths: int, device: str = 'cpu', **_) -> Tensor:
        # D = heights for simplicity
        return torch.randn(batch_sizes, channels, heights, heights, widths, device=device)

    @classmethod
    def _bench_wide(cls, modules: List[nn.Conv3d], strategy: str) -> 'WideConv3d':
        strat_map = {
            'grouped': Conv3dStrategy.GROUPED,
            'sequential': Conv3dStrategy.SEQUENTIAL,
        }
        return cls.from_modules(modules, strategy=strat_map.get(strategy, Conv3dStrategy.GROUPED))


__all__ = ['WideConv3d', 'Conv3dStrategy']