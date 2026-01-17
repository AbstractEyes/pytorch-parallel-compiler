"""
WideConvTranspose1d - N parallel ConvTranspose1d operations (1D deconvolution/upsampling).

Critical for audio generation, speech synthesis, WaveNet-style models.
Uses grouped transposed convolutions for efficient N-parallel execution.

Expected speedup: 10-15x (similar to Conv1d based on grouped conv optimizations)

Strategies:
- 'grouped': Grouped transposed conv (FASTEST)
- 'sequential': N separate F.conv_transpose1d (exact, slowest)

Input/Output Format (v0.6.0):
- Input:  [N, B, C_in, L]       (N-first)
- Output: [N, B, C_out, L']     (N-first, upsampled)

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Optional, Union, Tuple, Dict, Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class WideConvTranspose1d(nn.Module):
    """
    N parallel ConvTranspose1d layers fused into a single module.

    Transposed convolution (deconvolution) for 1D upsampling.
    Used in audio/speech generation, WaveNet decoders, etc.
    """

    BENCHMARK_STRATEGIES = ['baseline', 'grouped', 'sequential']

    def __init__(
        self,
        n: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        strategy: str = 'grouped',
    ):
        super().__init__()
        self.n = n
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.has_bias = bias
        self._strategy = strategy

        # Booleans for compile-friendly forward
        self._use_grouped = (strategy == 'grouped')

        # Create grouped transposed conv
        self.grouped_conv = nn.ConvTranspose1d(
            in_channels=n * in_channels,
            out_channels=n * out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
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

        Input:  [N, B, C_in, L]
        Output: [N, B, C_out, L']  (upsampled)
        """
        N, B, C, L = x.shape

        # Convert N-first to channel-packed: [N, B, C, L] -> [B, N*C, L]
        x_packed = x.permute(1, 0, 2, 3).reshape(B, N * C, L)

        # Run transposed conv in channel-packed format
        if self._use_grouped:
            out = self.grouped_conv(x_packed)
        else:
            out = self._forward_sequential_internal(x_packed)

        # Convert back to N-first: [B, N*C_out, L'] -> [N, B, C_out, L']
        B, NC_out, L_out = out.shape
        C_out = NC_out // N
        out = out.view(B, N, C_out, L_out).permute(1, 0, 2, 3)

        return out.contiguous()

    def _forward_sequential_internal(self, x: Tensor) -> Tensor:
        """Execute N separate transposed convs (exact, slowest)."""
        B, NC, L = x.shape
        C = NC // self.n
        C_out = self.out_channels

        # Split: [B, N*C, L] -> N x [B, C, L]
        x_split = x.view(B, self.n, C, L)

        outputs = []
        for i in range(self.n):
            # Get weight and bias for this model
            start_in = i * C
            end_in = start_in + C
            start_out = i * C_out
            end_out = start_out + C_out

            w = self.grouped_conv.weight[start_out:end_out, start_in:end_in]
            b = self.grouped_conv.bias[start_out:end_out] if self.has_bias else None

            # Run conv_transpose1d
            out_i = F.conv_transpose1d(
                x_split[:, i],
                w, b,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                dilation=self.dilation
            )
            outputs.append(out_i)

        # Stack: N x [B, C_out, L'] -> [B, N*C_out, L']
        return torch.cat(outputs, dim=1)

    @classmethod
    def from_modules(cls, modules: List[nn.ConvTranspose1d], strategy: str = 'grouped') -> 'WideConvTranspose1d':
        """Create from N existing ConvTranspose1d modules."""
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
            kernel_size=t.kernel_size[0],
            stride=t.stride[0],
            padding=t.padding[0],
            output_padding=t.output_padding[0],
            dilation=t.dilation[0],
            bias=(t.bias is not None),
            strategy=strategy,
        )

        # Copy to same device/dtype
        wide = wide.to(device=t.weight.device, dtype=t.weight.dtype)

        # Copy weights from N modules
        # ConvTranspose1d weight format: [in_channels, out_channels/groups, kernel_size]
        # For grouped conv: [n*C_in, C_out, K]
        with torch.no_grad():
            for i, m in enumerate(modules):
                start_in = i * t.in_channels
                end_in = start_in + t.in_channels
                start_out = i * t.out_channels
                end_out = start_out + t.out_channels

                # Index by input channels (first dimension for ConvTranspose)
                wide.grouped_conv.weight[start_in:end_in] = m.weight
                if m.bias is not None:
                    wide.grouped_conv.bias[start_out:end_out] = m.bias

        return wide

    def __repr__(self):
        return (f"WideConvTranspose1d({self.n}x[{self.in_channels}, {self.out_channels}, "
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
        """Create benchmark job for WideConvTranspose1d."""
        from ..benchmark.benchmark_schema import BenchmarkJob, SweepParams

        sweep_config = cls.BENCHMARK_SWEEPS.get(preset, cls.BENCHMARK_SWEEPS['full'])
        sweep = SweepParams(
            n_values=sweep_config['n_values'],
            batch_sizes=[8],
            channels=[64],
            kernel_sizes=[4],  # Common for 2x upsampling
            seq_lengths=[256],
        )

        return BenchmarkJob(
            name=f'convtranspose1d_{preset}',
            primitive='convtranspose1d',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
        )

    @staticmethod
    def _bench_model(channels=64, kernel_sizes=4, **kwargs):
        """
        Create a single ConvTranspose1d module.
        Uses fixed stride=2 and padding=1 for 2x upsampling.
        """
        return nn.ConvTranspose1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_sizes,
            stride=2,  # 2x upsampling
            padding=1,
        )

    @staticmethod
    def _bench_input(n: int, device: str, batch_sizes: int, channels=64, seq_lengths=256, **kwargs):
        """Create input tensor [B, C, L]."""
        return torch.randn(batch_sizes, channels, seq_lengths, device=device)

    @classmethod
    def _bench_wide(cls, modules: List[nn.Module], strategy: str):
        """Create wide version for given strategy."""
        if strategy == 'baseline':
            return None
        return cls.from_modules(modules, strategy=strategy)


__all__ = ['WideConvTranspose1d']
