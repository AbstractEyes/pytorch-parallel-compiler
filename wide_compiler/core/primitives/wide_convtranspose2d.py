"""
WideConvTranspose2d - N parallel ConvTranspose2d operations (deconvolution/upsampling).

Critical for GANs, VAEs, U-Nets, and any decoder/upsampling architecture.
Uses grouped transposed convolutions for efficient N-parallel execution.

Expected speedup: 8-15x (similar to Conv2d based on grouped conv optimizations)

Strategies:
- 'grouped': Grouped transposed conv in NCHW format (FASTEST)
- 'channels_last': Grouped transposed conv in NHWC format (most accurate)
- 'sequential': N separate F.conv_transpose2d (exact, slowest)

Input/Output Format (v0.6.0):
- Input:  [N, B, C_in, H, W]       (N-first)
- Output: [N, B, C_out, H', W']     (N-first, upsampled)

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Optional, Union, Tuple, Dict, Any
from enum import Enum

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class WideConvTranspose2d(nn.Module):
    """
    N parallel ConvTranspose2d layers fused into a single module.

    Transposed convolution (deconvolution) for upsampling.
    Used in decoder portions of GANs, VAEs, U-Nets, etc.
    """

    BENCHMARK_STRATEGIES = ['baseline', 'grouped', 'channels_last', 'sequential']

    def __init__(
        self,
        n: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        strategy: str = 'grouped',
    ):
        super().__init__()
        self.n = n
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.has_bias = bias
        self._strategy = strategy

        # Booleans for compile-friendly forward
        self._use_grouped = (strategy == 'grouped')
        self._use_channels_last = (strategy == 'channels_last')

        # Create grouped transposed conv
        self.grouped_conv = nn.ConvTranspose2d(
            in_channels=n * in_channels,
            out_channels=n * out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=n,
            bias=bias
        )

    @property
    def strategy(self) -> str:
        """Read-only strategy."""
        return self._strategy

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with N-first format.

        Input:  [N, B, C_in, H, W]
        Output: [N, B, C_out, H', W']  (upsampled)
        """
        N, B, C, H, W = x.shape

        # Convert N-first to channel-packed: [N, B, C, H, W] -> [B, N*C, H, W]
        x_packed = x.permute(1, 0, 2, 3, 4).reshape(B, N * C, H, W)

        # Run transposed conv in channel-packed format
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
        """Execute grouped transposed conv in NHWC format."""
        x_nhwc = x.to(memory_format=torch.channels_last)
        w_nhwc = self.grouped_conv.weight.to(memory_format=torch.channels_last)

        # ConvTranspose2d with NHWC
        out_nhwc = F.conv_transpose2d(
            x_nhwc, w_nhwc,
            bias=self.grouped_conv.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.n,
            dilation=self.dilation
        )

        return out_nhwc.contiguous(memory_format=torch.contiguous_format)

    def _forward_sequential_internal(self, x: Tensor) -> Tensor:
        """Execute N separate transposed convs (exact, slowest)."""
        B, NC, H, W = x.shape
        C = NC // self.n
        C_out = self.out_channels

        # Split: [B, N*C, H, W] -> N x [B, C, H, W]
        x_split = x.view(B, self.n, C, H, W)

        outputs = []
        for i in range(self.n):
            # Get weight and bias for this model
            start_in = i * C
            end_in = start_in + C
            start_out = i * C_out
            end_out = start_out + C_out

            w = self.grouped_conv.weight[start_out:end_out, start_in:end_in]
            b = self.grouped_conv.bias[start_out:end_out] if self.has_bias else None

            # Run conv_transpose2d
            out_i = F.conv_transpose2d(
                x_split[:, i],
                w, b,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                dilation=self.dilation
            )
            outputs.append(out_i)

        # Stack: N x [B, C_out, H', W'] -> [B, N*C_out, H', W']
        return torch.cat(outputs, dim=1)

    @classmethod
    def from_modules(cls, modules: List[nn.ConvTranspose2d], strategy: str = 'grouped') -> 'WideConvTranspose2d':
        """Create from N existing ConvTranspose2d modules."""
        n = len(modules)
        t = modules[0]

        # Verify all modules have same config
        for i, m in enumerate(modules):
            if m.in_channels != t.in_channels:
                raise ValueError(f"Module {i} has in_channels={m.in_channels}, expected {t.in_channels}")
            if m.out_channels != t.out_channels:
                raise ValueError(f"Module {i} has out_channels={m.out_channels}, expected {t.out_channels}")

        wide = cls(
            n=n,
            in_channels=t.in_channels,
            out_channels=t.out_channels,
            kernel_size=t.kernel_size,
            stride=t.stride,
            padding=t.padding,
            output_padding=t.output_padding,
            dilation=t.dilation,
            bias=(t.bias is not None),
            strategy=strategy,
        )

        # Copy to same device/dtype
        wide = wide.to(device=t.weight.device, dtype=t.weight.dtype)

        # Copy weights from N modules
        with torch.no_grad():
            for i, m in enumerate(modules):
                start_in = i * t.in_channels
                end_in = start_in + t.in_channels
                start_out = i * t.out_channels
                end_out = start_out + t.out_channels

                wide.grouped_conv.weight[start_out:end_out, start_in:end_in] = m.weight
                if m.bias is not None:
                    wide.grouped_conv.bias[start_out:end_out] = m.bias

        return wide

    def __repr__(self):
        return (f"WideConvTranspose2d({self.n}x[{self.in_channels}, {self.out_channels}, "
                f"kernel={self.kernel_size}, stride={self.stride}], strategy={self._strategy})")

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
        """Create benchmark job for WideConvTranspose2d."""
        from ..benchmark.benchmark_schema import BenchmarkJob, SweepParams

        sweep_config = cls.BENCHMARK_SWEEPS.get(preset, cls.BENCHMARK_SWEEPS['full'])
        sweep = SweepParams(
            n_values=sweep_config['n_values'],
            batch_sizes=[8],
            channels=[64],
            kernel_sizes=[4],  # Common for 2x upsampling
            heights=[16],
            widths=[16],
            strides=[2],  # 2x upsampling
            paddings=[1],
        )

        return BenchmarkJob(
            name=f'convtranspose2d_{preset}',
            primitive='convtranspose2d',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
        )

    @staticmethod
    def _bench_model(channels=64, kernel_sizes=4, strides=2, paddings=1, **kwargs):
        """Create a single ConvTranspose2d module."""
        return nn.ConvTranspose2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_sizes,
            stride=strides,
            padding=paddings,
        )

    @staticmethod
    def _bench_input(n: int, device: str, batch_sizes: int, channels=64, heights=16, widths=16, **kwargs):
        """Create input tensor [B, C, H, W]."""
        return torch.randn(batch_sizes, channels, heights, widths, device=device)

    @classmethod
    def _bench_wide(cls, modules: List[nn.Module], strategy: str):
        """Create wide version for given strategy."""
        if strategy == 'baseline':
            return None
        return cls.from_modules(modules, strategy=strategy)


__all__ = ['WideConvTranspose2d']
