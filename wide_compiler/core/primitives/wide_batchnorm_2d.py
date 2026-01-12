from typing import List

import torch
from torch import nn, Tensor


class WideBatchNorm2d(nn.Module):
    """N parallel BatchNorm2d as single BatchNorm2d."""

    def __init__(self, n: int, num_features: int, eps: float = 1e-5,
                 momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.n = n
        self.num_features = num_features

        # Single BN over N*C channels
        self.op = nn.BatchNorm2d(
            num_features=n * num_features,
            eps=eps,
            momentum=momentum,
            affine=affine
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, N*C, H, W]
        return self.op(x)

    @classmethod
    def from_modules(cls, modules: List[nn.BatchNorm2d]) -> 'WideBatchNorm2d':
        n = len(modules)
        t = modules[0]

        wide = cls(n, t.num_features, t.eps, t.momentum, t.affine)

        with torch.no_grad():
            for i, m in enumerate(modules):
                start = i * t.num_features
                end = start + t.num_features
                if m.weight is not None:
                    wide.op.weight[start:end] = m.weight
                if m.bias is not None:
                    wide.op.bias[start:end] = m.bias
                if m.running_mean is not None:
                    wide.op.running_mean[start:end] = m.running_mean
                if m.running_var is not None:
                    wide.op.running_var[start:end] = m.running_var

        return wide

    def __repr__(self):
        return f"WideBatchNorm2d({self.n}x{self.num_features})"