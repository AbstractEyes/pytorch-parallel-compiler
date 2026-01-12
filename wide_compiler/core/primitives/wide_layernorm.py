from typing import List

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class WideLayerNorm(nn.Module):
    """N parallel LayerNorm - operates on last dim per group."""

    def __init__(self, n: int, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.n = n
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Weight and bias per model
        self.weight = nn.Parameter(torch.ones(n * normalized_shape))
        self.bias = nn.Parameter(torch.zeros(n * normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, N*D] or [B, T, N*D]
        # Need to normalize each group of D independently

        if x.dim() == 2:
            B, ND = x.shape
            D = self.normalized_shape
            N = ND // D

            # Reshape to [B, N, D], normalize over D, reshape back
            x = x.view(B, N, D)
            x = F.layer_norm(x, (D,), eps=self.eps)
            x = x.view(B, ND)

            # Apply per-group affine
            x = x * self.weight + self.bias

        elif x.dim() == 3:
            B, T, ND = x.shape
            D = self.normalized_shape
            N = ND // D

            x = x.view(B, T, N, D)
            x = F.layer_norm(x, (D,), eps=self.eps)
            x = x.view(B, T, ND)
            x = x * self.weight + self.bias

        return x

    @classmethod
    def from_modules(cls, modules: List[nn.LayerNorm]) -> 'WideLayerNorm':
        n = len(modules)
        t = modules[0]
        norm_shape = t.normalized_shape[0] if isinstance(t.normalized_shape, tuple) else t.normalized_shape

        wide = cls(n, norm_shape, t.eps)

        with torch.no_grad():
            for i, m in enumerate(modules):
                start = i * norm_shape
                end = start + norm_shape
                wide.weight[start:end] = m.weight
                wide.bias[start:end] = m.bias

        return wide

    def __repr__(self):
        return f"WideLayerNorm({self.n}x{self.normalized_shape})"
