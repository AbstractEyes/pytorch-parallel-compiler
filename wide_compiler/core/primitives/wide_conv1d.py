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
from typing import List, Optional
import warnings

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class Conv1dStrategy(Enum):
    """Execution strategy for WideConv1d."""
    GROUPED = auto()  # Single grouped conv - best for large N
    SEQUENTIAL = auto()  # Loop over N convs - best for small N


# Threshold for strategy selection (same as Conv2d)
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
            # Weight shape: [N, out_channels, in_channels, kernel_size]
            self.weight = nn.Parameter(
                torch.empty(n, out_channels, in_channels, kernel_size)
            )
            if bias:
                self.bias = nn.Parameter(torch.empty(n, out_channels))
            else:
                self.register_parameter('bias', None)

            self._init_sequential_weights()

    def _init_sequential_weights(self):
        """Initialize sequential weights with Kaiming uniform."""
        for i in range(self.n):
            nn.init.kaiming_uniform_(self.weight[i], a=5 ** 0.5)
            if self.bias is not None:
                fan_in = self.in_channels * self.kernel_size
                bound = 1 / (fan_in ** 0.5)
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
            # x[:, i]: [B, C_in, L]
            out = F.conv1d(
                x[:, i],
                self.weight[i],
                self.bias[i] if self.bias is not None else None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
            outputs.append(out)

        # Stack and reshape: [N, B, C_out, L_out] -> [B, N*C_out, L_out]
        stacked = torch.stack(outputs, dim=1)  # [B, N, C_out, L_out]
        return stacked.view(B, self.n * C_out, -1)

    @classmethod
    def from_modules(
            cls,
            modules: List[nn.Conv1d],
            strategy: Optional[Conv1dStrategy] = None,
    ) -> 'WideConv1d':
        """
        Build WideConv1d from N existing Conv1d modules.

        Args:
            modules: List of N Conv1d modules with identical architecture
            strategy: Force specific strategy, or None for auto-selection

        Returns:
            WideConv1d with weights copied from input modules
        """
        n = len(modules)
        t = modules[0]

        # Extract scalar values from potentially tuple attributes
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

        # Copy weights
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


__all__ = ['WideConv1d', 'Conv1dStrategy']

# =============================================================================
# TESTS
# =============================================================================

if __name__ == '__main__':
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"WideConv1d Tests | Device: {device}")
    print("=" * 60)

    # Test parameters
    B, C_in, C_out, L, K = 8, 32, 64, 128, 3

    for N in [4, 8, 16, 32]:
        print(f"\n--- N={N} ---")

        # Create N separate Conv1d
        convs = [nn.Conv1d(C_in, C_out, K, padding=1).to(device) for _ in range(N)]

        # Create wide versions with both strategies
        wide_grouped = WideConv1d.from_modules(convs, strategy=Conv1dStrategy.GROUPED).to(device)
        wide_seq = WideConv1d.from_modules(convs, strategy=Conv1dStrategy.SEQUENTIAL).to(device)
        wide_auto = WideConv1d.from_modules(convs).to(device)

        print(f"  Auto-selected: {wide_auto}")

        # Test inputs
        inputs = [torch.randn(B, C_in, L, device=device) for _ in range(N)]

        # Pack inputs
        packed = torch.cat([inp.unsqueeze(1) for inp in inputs], dim=1)
        packed = packed.view(B, N * C_in, L)

        # Reference: run separately
        with torch.no_grad():
            ref_outputs = [convs[i](inputs[i]) for i in range(N)]
            ref_stacked = torch.stack(ref_outputs, dim=1)
            ref_packed = ref_stacked.view(B, N * C_out, -1)

        # Test grouped
        with torch.no_grad():
            out_grouped = wide_grouped(packed)
        diff_grouped = (ref_packed - out_grouped).abs().max().item()

        # Test sequential
        with torch.no_grad():
            out_seq = wide_seq(packed)
        diff_seq = (ref_packed - out_seq).abs().max().item()

        print(f"  GROUPED diff:    {diff_grouped:.2e} {'OK' if diff_grouped < 1e-5 else 'FAIL'}")
        print(f"  SEQUENTIAL diff: {diff_seq:.2e} {'OK' if diff_seq < 1e-5 else 'FAIL'}")

    # Benchmark
    print("\n" + "=" * 60)
    print("Benchmark (N=20, B=16, C=64, L=256, K=3)")
    print("=" * 60)

    import time

    N, B, C_in, C_out, L, K = 20, 16, 64, 64, 256, 3
    convs = [nn.Conv1d(C_in, C_out, K, padding=1).to(device) for _ in range(N)]

    wide_grouped = WideConv1d.from_modules(convs, strategy=Conv1dStrategy.GROUPED).to(device)
    wide_seq = WideConv1d.from_modules(convs, strategy=Conv1dStrategy.SEQUENTIAL).to(device)

    inputs = [torch.randn(B, C_in, L, device=device) for _ in range(N)]
    packed = torch.cat([inp.unsqueeze(1) for inp in inputs], dim=1).view(B, N * C_in, L)


    def bench(fn, iters=100, warmup=20):
        for _ in range(warmup):
            fn()
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        if device == 'cuda':
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000


    with torch.no_grad():
        t_seq_baseline = bench(lambda: [convs[i](inputs[i]) for i in range(N)])
        t_grouped = bench(lambda: wide_grouped(packed))
        t_sequential = bench(lambda: wide_seq(packed))

    print(f"  N x Conv1d:      {t_seq_baseline:.3f}ms (baseline)")
    print(f"  GROUPED:         {t_grouped:.3f}ms ({t_seq_baseline / t_grouped:.2f}x)")
    print(f"  SEQUENTIAL:      {t_sequential:.3f}ms ({t_seq_baseline / t_sequential:.2f}x)")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)