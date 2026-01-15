"""
WideConv2d - N parallel Conv2d operations fused into a single module.

Numerical Accuracy Notes (from PyTorch docs):
- Batched operations are NOT guaranteed bitwise identical to sequential
- TF32 is enabled by default for convolutions on Ampere+ (10-bit mantissa)
- NCHW grouped: ~1e-6 relative error vs sequential
- NHWC grouped: 0.0 error (most accurate, but slower due to conversion)

Strategies:
- 'grouped': Grouped conv in NCHW format (FASTEST on A100, 2-4x speedup)
- 'channels_last': Grouped conv in NHWC format (most accurate, slower)
- 'sequential': N separate F.conv2d (exact, slowest)
- 'auto': Heuristic selection (prefers grouped NCHW)

Input/Output Format (v0.6.0):
- Input:  [N, B, C_in, H, W]   (N-first)
- Output: [N, B, C_out, H', W'] (N-first)

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Optional, Union, Tuple, Dict, Any
from enum import Enum

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import warnings


class ConvStrategy(Enum):
    """Strategy for executing N parallel convolutions.

    - AUTO: Heuristic selection (selects GROUPED on A100)
    - GROUPED: Single grouped conv in NCHW format (FASTEST: 2-4x speedup)
    - CHANNELS_LAST: Single grouped conv in NHWC format (most accurate: 0.0 error)
    - SEQUENTIAL: N separate F.conv2d (exact, slowest)
    """
    AUTO = 'auto'
    GROUPED = 'grouped'
    CHANNELS_LAST = 'channels_last'
    SEQUENTIAL = 'sequential'


# Tuned thresholds from A100 benchmarks (including ResNet18)
STRATEGY_THRESHOLDS = {
    'n_high': 16,             # N >= this: grouped always wins
    'n_crossover': 8,         # N >= this with B <= b_low: grouped wins
    'n_low': 3,               # N <= this: sequential (not enough parallelism)
    'b_low': 16,              # B <= this: grouped helps more
}

_WARNED_ABOUT_TUNING = False


def _is_datacenter_gpu() -> bool:
    """Check if running on datacenter GPU where grouped conv is optimized."""
    if not torch.cuda.is_available():
        return False
    name = torch.cuda.get_device_name().lower()
    datacenter = ['a100', 'h100', 'a10', 'v100', 'a30', 'a40', 'h200', 'b100', 'b200']
    return any(gpu in name for gpu in datacenter)


def select_strategy(n: int, b: int, c_in: int, c_out: int,
                    h: int = 56, w: int = 56) -> ConvStrategy:
    """
    Heuristic strategy selection based on A100 benchmarks.

    Key: GROUPED wins even at moderate N (8-16). Only use SEQUENTIAL
    for N <= 3 where kernel launch overhead dominates.
    """
    global _WARNED_ABOUT_TUNING
    if not _WARNED_ABOUT_TUNING:
        warnings.warn(
            "WideConv2d auto-selection thresholds may need tuning for your GPU. "
            "Run the test suite to verify optimal strategy for your hardware.",
            UserWarning
        )
        _WARNED_ABOUT_TUNING = True

    T = STRATEGY_THRESHOLDS

    # Very low N: not enough parallelism benefit
    if n <= T['n_low']:
        return ConvStrategy.SEQUENTIAL

    # Check if grouped provides benefit
    use_grouped = (n >= T['n_crossover'] and b <= T['b_low']) or n >= T['n_high']

    if use_grouped:
        return ConvStrategy.GROUPED

    return ConvStrategy.SEQUENTIAL


class WideConv2d(nn.Module):
    """
    N parallel Conv2d layers fused into a single module.

    Strategy is resolved at construction time and forward is delegated directly.
    This makes the module structurally immutable and torch.compile friendly.

    Strategies:
        'auto': Selects grouped NCHW (fastest on A100)
        'grouped': Single grouped conv in NCHW (FASTEST: 2-4x speedup)
        'channels_last': Single grouped conv in NHWC (most accurate: 0.0 error)
        'sequential': N separate F.conv2d (exact, slowest)
    """

    def __init__(
        self,
        n: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        strategy: Union[str, ConvStrategy] = 'auto',
    ):
        super().__init__()
        self.n = n
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.has_bias = bias

        # Parse and resolve strategy at construction time
        if isinstance(strategy, str):
            strategy = ConvStrategy(strategy)

        if strategy == ConvStrategy.AUTO:
            strategy = select_strategy(n, 8, in_channels, out_channels, 56, 56)

        self._strategy = strategy

        # Booleans for compile-friendly forward
        self._use_grouped = (strategy == ConvStrategy.GROUPED)
        self._use_channels_last = (strategy == ConvStrategy.CHANNELS_LAST)

        # Create grouped conv (used by all strategies for weight storage)
        self.grouped_conv = nn.Conv2d(
            in_channels=n * in_channels,
            out_channels=n * out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=n,
            bias=bias
        )

        # Cache for sequential strategy weight views
        self._weight_views: Optional[List[Tensor]] = None
        self._bias_views: Optional[List[Tensor]] = None

    @property
    def strategy(self) -> ConvStrategy:
        """Read-only strategy (immutable after construction)."""
        return self._strategy

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with N-first format.

        Input:  [N, B, C_in, H, W]
        Output: [N, B, C_out, H', W']
        """
        N, B, C, H, W = x.shape

        # Convert N-first to channel-packed: [N, B, C, H, W] -> [B, N*C, H, W]
        x_packed = x.permute(1, 0, 2, 3, 4).reshape(B, N * C, H, W)

        # Run conv in channel-packed format
        if self._use_channels_last:
            out = self._forward_channels_last_internal(x_packed)
        elif self._use_grouped:
            out = self.grouped_conv(x_packed)
        else:
            out = self._forward_sequential_internal(x_packed)

        # Convert back to N-first: [B, N*C_out, H', W'] -> [N, B, C_out, H', W']
        B, NC_out, H_out, W_out = out.shape
        C_out = NC_out // N
        out = out.view(B, N, C_out, H_out, W_out).permute(1, 0, 2, 3, 4)

        return out.contiguous()

    def _forward_channels_last_internal(self, x: Tensor) -> Tensor:
        """Execute grouped conv in NHWC format on channel-packed input."""
        x_nhwc = x.to(memory_format=torch.channels_last)
        w_nhwc = self.grouped_conv.weight.to(memory_format=torch.channels_last)

        out = F.conv2d(
            x_nhwc,
            w_nhwc,
            self.grouped_conv.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.n
        )

        return out.contiguous()

    def _forward_sequential_internal(self, x: Tensor) -> Tensor:
        """N separate convolutions on channel-packed input."""
        B = x.shape[0]
        H, W = x.shape[2], x.shape[3]

        # Reshape input: [B, N*C_in, H, W] -> [B, N, C_in, H, W]
        x_reshaped = x.view(B, self.n, self.in_channels, H, W)

        weight = self.grouped_conv.weight
        bias = self.grouped_conv.bias

        # Cast to input dtype if needed
        if weight.dtype != x.dtype:
            weight = weight.to(x.dtype)
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(x.dtype)

        outputs = []
        for i in range(self.n):
            xi = x_reshaped[:, i]
            wi = weight[i*self.out_channels:(i+1)*self.out_channels]
            bi = bias[i*self.out_channels:(i+1)*self.out_channels] if bias is not None else None
            out = F.conv2d(xi, wi, bi,
                          stride=self.stride, padding=self.padding,
                          dilation=self.dilation)
            outputs.append(out)

        stacked = torch.stack(outputs, dim=1)
        return stacked.view(B, -1, stacked.shape[3], stacked.shape[4])

    @classmethod
    def from_modules(
        cls,
        modules: List[nn.Conv2d],
        strategy: Union[str, ConvStrategy] = 'auto'
    ) -> 'WideConv2d':
        """Create WideConv2d from N existing Conv2d modules."""
        n = len(modules)
        t = modules[0]

        def to_tuple(x):
            return x if isinstance(x, tuple) else (x, x)

        k = to_tuple(t.kernel_size)
        s = to_tuple(t.stride)
        p = to_tuple(t.padding)
        d = to_tuple(t.dilation)

        wide = cls(
            n=n,
            in_channels=t.in_channels,
            out_channels=t.out_channels,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
            bias=t.bias is not None,
            strategy=strategy,
        )

        # Preserve device and dtype
        wide = wide.to(device=t.weight.device, dtype=t.weight.dtype)

        with torch.no_grad():
            for i, m in enumerate(modules):
                start = i * t.out_channels
                end = start + t.out_channels
                wide.grouped_conv.weight[start:end] = m.weight
                if m.bias is not None:
                    wide.grouped_conv.bias[start:end] = m.bias

        return wide

    def __repr__(self):
        return (
            f"WideConv2d({self.n}x[{self.in_channels}->{self.out_channels}], "
            f"k={self.kernel_size}, strategy={self._strategy.value})"
        )

    # =========================================================================
    # BENCHMARK INTERFACE
    # =========================================================================

    BENCHMARK_STRATEGIES = ['baseline', 'grouped', 'channels_last', 'sequential']

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
                kernel_sizes=[3],
                heights=[32],
                widths=[32],
            ),
            'full': SweepParams(
                n_values=[2, 4, 6, 8, 10, 12, 16, 20, 32, 64],
                batch_sizes=[4, 8, 16],
                channels=[32, 64, 128],
                kernel_sizes=[1, 3, 5],
                heights=[16, 32, 56],
                widths=[16, 32, 56],
            ),
            'ci': SweepParams(
                n_values=[4, 16],
                batch_sizes=[8],
                channels=[64],
                kernel_sizes=[3],
                heights=[32],
                widths=[32],
            ),
        }

    @classmethod
    def benchmark_job(cls, preset: str = 'full', **overrides) -> Any:
        """Get benchmark job for WideConv2d."""
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
            name=f'conv2d_{preset}',
            primitive='conv2d',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
            pack_fn=cls._bench_pack,
            unpack_fn=cls._bench_unpack,
        )

    @staticmethod
    def _bench_model(channels: int, kernel_sizes: int, **_) -> nn.Conv2d:
        """Create single Conv2d for benchmarking."""
        return nn.Conv2d(channels, channels, kernel_sizes, padding=kernel_sizes // 2)

    @staticmethod
    def _bench_input(n: int, batch_sizes: int, channels: int, heights: int, widths: int, device: str = 'cpu', **_) -> Tensor:
        """Create single input tensor for one model."""
        return torch.randn(batch_sizes, channels, heights, widths, device=device)

    @classmethod
    def _bench_wide(cls, modules: List[nn.Conv2d], strategy: str) -> 'WideConv2d':
        """Create WideConv2d with specific strategy."""
        strat_map = {
            'grouped': ConvStrategy.GROUPED,
            'channels_last': ConvStrategy.CHANNELS_LAST,
            'sequential': ConvStrategy.SEQUENTIAL,
        }
        strat = strat_map.get(strategy, ConvStrategy.GROUPED)
        return cls.from_modules(modules, strategy=strat)

    @staticmethod
    def _bench_pack(inputs: List[Tensor]) -> Tensor:
        """Pack N inputs into N-first format: [N, B, C, H, W]."""
        return torch.stack(inputs, dim=0)

    @staticmethod
    def _bench_unpack(output: Tensor, n: int) -> List[Tensor]:
        """Unpack N-first output to list of [B, C, H, W]."""
        return [output[i] for i in range(n)]


# Convenience function
def create_wide_conv2d(
    modules: List[nn.Conv2d],
    strategy: str = 'auto'
) -> WideConv2d:
    """Create WideConv2d from existing Conv2d modules."""
    return WideConv2d.from_modules(modules, strategy=strategy)


__all__ = [
    'WideConv2d',
    'ConvStrategy',
    'select_strategy',
    'create_wide_conv2d',
    'STRATEGY_THRESHOLDS',
]