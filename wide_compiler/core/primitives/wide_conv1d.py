from typing import List

import torch
from torch import nn, Tensor


class WideConv1d(nn.Module):
    """N parallel Conv1d layers as single grouped Conv1d."""

    def __init__(self, n: int, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, padding: int = 0,
                 dilation: int = 1, bias: bool = True):
        super().__init__()
        self.n = n
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.op = nn.Conv1d(
            in_channels=n * in_channels,
            out_channels=n * out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=n,
            bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)

    @classmethod
    def from_modules(cls, modules: List[nn.Conv1d]) -> 'WideConv1d':
        n = len(modules)
        t = modules[0]

        k = t.kernel_size[0] if isinstance(t.kernel_size, tuple) else t.kernel_size
        s = t.stride[0] if isinstance(t.stride, tuple) else t.stride
        p = t.padding[0] if isinstance(t.padding, tuple) else t.padding
        d = t.dilation[0] if isinstance(t.dilation, tuple) else t.dilation

        wide = cls(n, t.in_channels, t.out_channels, k, s, p, d,
                   bias=t.bias is not None)

        with torch.no_grad():
            for i, m in enumerate(modules):
                start = i * t.out_channels
                end = start + t.out_channels
                wide.op.weight[start:end] = m.weight
                if m.bias is not None:
                    wide.op.bias[start:end] = m.bias

        return wide

    def __repr__(self):
        return f"WideConv1d({self.n}x[{self.in_channels}→{self.out_channels}])"
