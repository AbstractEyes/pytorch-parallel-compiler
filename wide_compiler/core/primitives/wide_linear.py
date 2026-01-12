# =============================================================================
# WIDE OPS - Compile-friendly batched primitives
# =============================================================================
from typing import List

import torch
from torch import nn, Tensor


class WideLinear(nn.Module):
    """N parallel Linear layers as grouped Conv1d."""

    def __init__(self, n: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.n = n
        self.in_features = in_features
        self.out_features = out_features

        # Linear = Conv1d with kernel_size=1
        self.op = nn.Conv1d(
            in_channels=n * in_features,
            out_channels=n * out_features,
            kernel_size=1,
            groups=n,
            bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, N*in] or [B, N*in, T]
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, N*in, 1]
            out = self.op(x)
            return out.squeeze(-1)  # [B, N*out]
        return self.op(x)  # [B, N*out, T]

    @classmethod
    def from_modules(cls, modules: List[nn.Linear]) -> 'WideLinear':
        """Create from N existing Linear modules."""
        n = len(modules)
        template = modules[0]
        wide = cls(n, template.in_features, template.out_features,
                   bias=template.bias is not None)

        # Stack weights: [N, out, in] -> interleave into grouped conv
        with torch.no_grad():
            for i, m in enumerate(modules):
                # Conv1d weight: [N*out, in, 1], grouped
                # Group i uses indices [i*out : (i+1)*out]
                start = i * template.out_features
                end = start + template.out_features
                wide.op.weight[start:end, :, 0] = m.weight
                if m.bias is not None:
                    wide.op.bias[start:end] = m.bias

        return wide

    def __repr__(self):
        return f"WideLinear({self.n}x[{self.in_features}→{self.out_features}])"

