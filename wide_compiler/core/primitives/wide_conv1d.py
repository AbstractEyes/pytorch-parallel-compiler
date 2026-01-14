"""
WideCompiler.core.primitives.wide_conv1d

N parallel Conv1d layers fused into single operation.

Strategy auto-selected based on N:
  - SEQUENTIAL (N < 10): Loop over individual convs
  - GROUPED (N >= 10): Single grouped convolution

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from __future__ import annotations

from enum import Enum, auto
from typing import List, Optional, Dict, Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class Conv1dStrategy(Enum):
    """Execution strategy for WideConv1d."""
    GROUPED = auto()     # Single grouped conv - best for large N
    SEQUENTIAL = auto()  # Loop over N convs - best for small N


# Threshold for strategy selection
_GROUPED_THRESHOLD = 10


class WideConv1d(nn.Module):
    """
    N parallel Conv1d layers as single fused operation.

    Input shape:  [B, N*C_in, L]
    Output shape: [B, N*C_out, L_out]

    Strategies:
        GROUPED: Single nn.Conv1d with groups=N
            - One kernel launch
            - Best when N >= 10

        SEQUENTIAL: Loop over N separate convolutions
            - N kernel launches but less memory reshaping
            - Best when N < 10

    Auto-selects strategy based on N threshold.
    """

    def __init__(
        self,
        n: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        strategy: Optional[Conv1dStrategy] = None,
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

        # Auto-select strategy
        if strategy is None:
            strategy = Conv1dStrategy.GROUPED if n >= _GROUPED_THRESHOLD else Conv1dStrategy.SEQUENTIAL
        self.strategy = strategy

        if strategy == Conv1dStrategy.GROUPED:
            self.op = nn.Conv1d(
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
            # SEQUENTIAL: store N separate weight tensors
            self.weight = nn.Parameter(
                torch.empty(n, out_channels, in_channels, kernel_size)
            )
            if bias:
                self.bias = nn.Parameter(torch.empty(n, out_channels))
            else:
                self.register_parameter('bias', None)

            self._init_sequential_weights()

    def _init_sequential_weights(self):
        """Initialize sequential weights to match PyTorch Conv1d defaults."""
        import math
        for i in range(self.n):
            # Match nn.Conv1d: kaiming_uniform with a=sqrt(5)
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in = self.in_channels * self.kernel_size
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if self.strategy == Conv1dStrategy.GROUPED:
            return self.op(x)
        else:
            return self._forward_sequential(x)

    def _forward_sequential(self, x: Tensor) -> Tensor:
        """Sequential forward: loop over N convolutions."""
        B, NC, L = x.shape
        C_in = self.in_channels
        C_out = self.out_channels

        # Split input: [B, N*C_in, L] -> [B, N, C_in, L]
        x = x.view(B, self.n, C_in, L)

        outputs = []
        for i in range(self.n):
            out = F.conv1d(
                x[:, i],
                self.weight[i],
                self.bias[i] if self.bias is not None else None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
            outputs.append(out)

        # Stack and reshape: [B, N, C_out, L_out] -> [B, N*C_out, L_out]
        stacked = torch.stack(outputs, dim=1)
        return stacked.view(B, self.n * C_out, -1)

    @classmethod
    def from_modules(
        cls,
        modules: List[nn.Conv1d],
        strategy: Optional[Conv1dStrategy] = None,
    ) -> 'WideConv1d':
        """Build WideConv1d from N existing Conv1d modules."""
        n = len(modules)
        t = modules[0]

        k = t.kernel_size[0] if isinstance(t.kernel_size, tuple) else t.kernel_size
        s = t.stride[0] if isinstance(t.stride, tuple) else t.stride
        p = t.padding[0] if isinstance(t.padding, tuple) else t.padding
        d = t.dilation[0] if isinstance(t.dilation, tuple) else t.dilation

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

        with torch.no_grad():
            if wide.strategy == Conv1dStrategy.GROUPED:
                for i, m in enumerate(modules):
                    start = i * t.out_channels
                    end = start + t.out_channels
                    wide.op.weight[start:end] = m.weight
                    if m.bias is not None:
                        wide.op.bias[start:end] = m.bias
            else:
                for i, m in enumerate(modules):
                    wide.weight[i] = m.weight
                    if m.bias is not None:
                        wide.bias[i] = m.bias

        return wide

    def __repr__(self):
        return (
            f"WideConv1d({self.n}x[{self.in_channels}->{self.out_channels}], "
            f"k={self.kernel_size}, strategy={self.strategy.name.lower()})"
        )

    # =========================================================================
    # BENCHMARK INTERFACE
    # =========================================================================

    BENCHMARK_STRATEGIES = ['baseline', 'grouped', 'sequential']

    @classmethod
    def _get_sweep_params_class(cls):
        """Get SweepParams class with multiple import attempts."""
        # Try relative import
        try:
            from ..benchmark.benchmark_schema import SweepParams
            return SweepParams
        except ImportError:
            pass

        # Try absolute import
        try:
            from wide_compiler.core.benchmark.benchmark_schema import SweepParams
            return SweepParams
        except ImportError:
            pass

        return None

    @classmethod
    def _get_benchmark_job_class(cls):
        """Get BenchmarkJob class with multiple import attempts."""
        # Try relative import
        try:
            from ..benchmark.benchmark_schema import BenchmarkJob
            return BenchmarkJob
        except ImportError:
            pass

        # Try absolute import
        try:
            from wide_compiler.core.benchmark.benchmark_schema import BenchmarkJob
            return BenchmarkJob
        except ImportError:
            pass

        return None

    # Sweep configurations - populated on first access
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
                seq_lengths=[256],
            ),
            'full': SweepParams(
                n_values=[2, 4, 6, 8, 10, 12, 16, 20, 32, 64],
                batch_sizes=[8],
                channels=[32, 64, 128],
                kernel_sizes=[1, 3, 5, 7],
                seq_lengths=[64, 256, 1024],
            ),
            'ci': SweepParams(
                n_values=[4, 16],
                batch_sizes=[8],
                channels=[64],
                kernel_sizes=[3],
                seq_lengths=[256],
            ),
        }

    @classmethod
    def benchmark_job(cls, preset: str = 'full', **overrides) -> Any:
        """
        Get benchmark job for WideConv1d.

        Args:
            preset: 'quick', 'full', or 'ci'
            **overrides: Override sweep params

        Returns:
            BenchmarkJob ready to run
        """
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
            name=f'conv1d_{preset}',
            primitive='conv1d',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
            pack_fn=cls._bench_pack,
            unpack_fn=cls._bench_unpack,
        )

    @staticmethod
    def _bench_model(channels: int, kernel_sizes: int, **_) -> nn.Conv1d:
        """Create single Conv1d for benchmarking."""
        return nn.Conv1d(channels, channels, kernel_sizes, padding=kernel_sizes // 2)

    @staticmethod
    def _bench_input(n: int, batch_sizes: int, channels: int, seq_lengths: int, device: str = 'cpu', **_) -> Tensor:
        """Create single input tensor (will be replicated for N models)."""
        return torch.randn(batch_sizes, channels, seq_lengths, device=device)

    @classmethod
    def _bench_wide(cls, modules: List[nn.Conv1d], strategy: str) -> 'WideConv1d':
        """Create WideConv1d with specific strategy."""
        strat = Conv1dStrategy.GROUPED if strategy == 'grouped' else Conv1dStrategy.SEQUENTIAL
        return cls.from_modules(modules, strategy=strat)

    @staticmethod
    def _bench_pack(inputs: List[Tensor]) -> Tensor:
        """Pack N inputs into wide format."""
        stacked = torch.stack(inputs, dim=1)  # [B, N, C, L]
        B, N, C, L = stacked.shape
        return stacked.view(B, N * C, L)

    @staticmethod
    def _bench_unpack(output: Tensor, n: int) -> List[Tensor]:
        """Unpack wide output to N outputs."""
        B, NC, L = output.shape
        C = NC // n
        reshaped = output.view(B, n, C, L)
        return [reshaped[:, i] for i in range(n)]


__all__ = ['WideConv1d', 'Conv1dStrategy']