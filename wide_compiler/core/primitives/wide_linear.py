"""
WideLinear with multiple backend strategies.

Key findings from A100 benchmarks (sweep_boundaries.py):

1. Einsum wins 75.6% of cases overall
2. Einsum is MORE ACCURATE than grouped (6e-7 vs 3e-4 relative error)
3. Einsum speedups: 6-28x for N>=20, 2-8x for N=5-10
4. Einsum loses when:
   - N=2 (not enough parallelism)
   - Very large expansions (≥8x, like vocab projections)

Strategies:
- 'einsum': Batched matrix multiply via torch.einsum (RECOMMENDED)
- 'grouped': Grouped Conv1d (fallback for edge cases)
- 'sequential': N separate F.linear (baseline)
- 'auto': Heuristic selection
"""

from typing import List, Optional, Union
from enum import Enum

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class LinearStrategy(Enum):
    EINSUM = 'einsum'  # Batched matmul - best for most cases
    GROUPED = 'grouped'  # Grouped Conv1d - fallback
    SEQUENTIAL = 'sequential'  # N separate ops - baseline
    AUTO = 'auto'


# Thresholds from A100 benchmarks
LINEAR_THRESHOLDS = {
    'n_low': 2,  # N at or below this: sequential wins
    'n_einsum_wins': 5,  # N above this: einsum usually wins
    'expansion_max': 8.0,  # out/in ratio above this: sequential wins
    'vocab_size': 10000,  # out_features above this: likely vocab proj, sequential
}


def select_linear_strategy(n: int, b: int, in_features: int, out_features: int) -> LinearStrategy:
    """
    Heuristic for linear strategy selection based on A100 benchmarks.

    Key findings:
    1. Einsum wins 75.6% of cases
    2. Einsum wins almost always for N>=20 (6-28x speedup)
    3. Sequential wins for:
       - N=2 (not enough parallel work)
       - Large expansions ≥8x (memory bandwidth bound)
       - Vocab projections (very large output dim)

    Accuracy: Einsum has 6e-7 relative error vs 3e-4 for grouped
    """
    T = LINEAR_THRESHOLDS

    # Very low N: not enough parallel work
    if n <= T['n_low']:
        return LinearStrategy.SEQUENTIAL

    # Large expansion ratios (≥8x): memory bound, sequential wins
    expansion = out_features / in_features
    if expansion >= T['expansion_max']:
        return LinearStrategy.SEQUENTIAL

    # Vocab projections (very large output): sequential wins
    if out_features >= T['vocab_size']:
        return LinearStrategy.SEQUENTIAL

    # Default: einsum wins
    return LinearStrategy.EINSUM


class WideLinear(nn.Module):
    """
    N parallel Linear layers with selectable execution strategy.

    Default strategy is 'einsum' which provides:
    - 6-28x speedup for N>=20
    - 2-8x speedup for N=5-10  
    - Better numerical accuracy (6e-7 vs 3e-4 relative error)

    Strategies:
        'einsum': Batched matrix multiply (RECOMMENDED)
        'grouped': Fused via grouped Conv1d
        'sequential': N separate F.linear calls
        'auto': Select based on input shape and expansion ratio
    """

    def __init__(
            self,
            n: int,
            in_features: int,
            out_features: int,
            bias: bool = True,
            strategy: Union[str, LinearStrategy] = 'auto',
    ):
        super().__init__()
        self.n = n
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        if isinstance(strategy, str):
            strategy = LinearStrategy(strategy)
        self.strategy = strategy

        # Store weights in einsum-friendly format: [N, out, in]
        self.weight = nn.Parameter(torch.empty(n, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(n, out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize
        self._reset_parameters()

        # Lazy-created grouped conv for fallback
        self._grouped_conv: Optional[nn.Conv1d] = None

        # Stats
        self._call_count = 0

    def _reset_parameters(self):
        """Initialize parameters like nn.Linear."""
        import math
        # Kaiming uniform for each sub-weight
        for i in range(self.n):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in = self.in_features
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)

    def _get_grouped_conv(self) -> nn.Conv1d:
        """Lazy-create grouped conv for fallback strategy."""
        if self._grouped_conv is None:
            self._grouped_conv = nn.Conv1d(
                self.n * self.in_features,
                self.n * self.out_features,
                kernel_size=1,
                groups=self.n,
                bias=self.has_bias
            ).to(self.weight.device, self.weight.dtype)

            # Copy weights
            with torch.no_grad():
                for i in range(self.n):
                    start, end = i * self.out_features, (i + 1) * self.out_features
                    self._grouped_conv.weight[start:end, :, 0] = self.weight[i]
                    if self.bias is not None:
                        self._grouped_conv.bias[start:end] = self.bias[i]

        return self._grouped_conv

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: [B, N*in_features] or [B, N, in_features] or [N, B, in_features]

        Returns:
            [B, N*out_features] or matching input layout
        """
        strategy = self.strategy

        if strategy == LinearStrategy.AUTO:
            B = x.shape[0] if x.dim() == 2 else x.shape[1]
            strategy = select_linear_strategy(self.n, B, self.in_features, self.out_features)

        self._call_count += 1

        if strategy == LinearStrategy.EINSUM:
            return self._forward_einsum(x)
        elif strategy == LinearStrategy.GROUPED:
            return self._forward_grouped(x)
        elif strategy == LinearStrategy.SEQUENTIAL:
            return self._forward_sequential(x)
        else:
            return self._forward_einsum(x)

    def _forward_einsum(self, x: Tensor) -> Tensor:
        """
        Batched matrix multiply via einsum.

        Most efficient for N>=5, provides best numerical accuracy.
        """
        input_2d = x.dim() == 2

        if input_2d:
            # [B, N*in] -> [N, B, in]
            B = x.shape[0]
            x = x.view(B, self.n, self.in_features).permute(1, 0, 2)
        elif x.shape[0] == self.n:
            # Already [N, B, in]
            pass
        else:
            # [B, N, in] -> [N, B, in]
            x = x.permute(1, 0, 2)

        # Einsum: [N, out, in] @ [N, B, in] -> [N, B, out]
        out = torch.einsum('noi,nbi->nbo', self.weight, x)

        if self.bias is not None:
            out = out + self.bias.unsqueeze(1)  # [N, 1, out]

        if input_2d:
            # [N, B, out] -> [B, N*out]
            out = out.permute(1, 0, 2).reshape(B, -1)

        return out

    def _forward_grouped(self, x: Tensor) -> Tensor:
        """Grouped Conv1d approach."""
        grouped = self._get_grouped_conv()

        if x.dim() == 2:
            x = x.unsqueeze(-1)
            out = grouped(x)
            return out.squeeze(-1)
        return grouped(x)

    def _forward_sequential(self, x: Tensor) -> Tensor:
        """N separate linear operations."""
        input_2d = x.dim() == 2

        if input_2d:
            B = x.shape[0]
            x = x.view(B, self.n, self.in_features)

        outputs = []
        for i in range(self.n):
            xi = x[:, i] if x.dim() == 3 and x.shape[1] == self.n else x[i]
            bi = self.bias[i] if self.bias is not None else None
            out = F.linear(xi, self.weight[i], bi)
            outputs.append(out)

        out = torch.stack(outputs, dim=1)  # [B, N, out]

        if input_2d:
            out = out.view(out.shape[0], -1)  # [B, N*out]

        return out

    @classmethod
    def from_modules(
            cls,
            modules: List[nn.Linear],
            strategy: Union[str, LinearStrategy] = 'auto'
    ) -> 'WideLinear':
        """Create from N existing Linear modules."""
        n = len(modules)
        template = modules[0]
        wide = cls(
            n=n,
            in_features=template.in_features,
            out_features=template.out_features,
            bias=template.bias is not None,
            strategy=strategy,
        )

        with torch.no_grad():
            for i, m in enumerate(modules):
                wide.weight[i] = m.weight
                if m.bias is not None:
                    wide.bias[i] = m.bias

        return wide

    def set_strategy(self, strategy: Union[str, LinearStrategy]):
        """Change strategy at runtime."""
        if isinstance(strategy, str):
            strategy = LinearStrategy(strategy)
        self.strategy = strategy
        # Invalidate grouped conv cache
        self._grouped_conv = None

    def __repr__(self):
        return f"WideLinear({self.n}x[{self.in_features}→{self.out_features}], strategy={self.strategy.value})"


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    import time

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("=" * 60)
    print("WideLinear Strategy Comparison")
    print("=" * 60)

    N = 20
    B = 32
    in_f, out_f = 512, 256

    print(f"N={N}, B={B}, {in_f}→{out_f}")

    # Create modules
    linears = [nn.Linear(in_f, out_f).to(device).eval() for _ in range(N)]

    # Wide versions
    wide_einsum = WideLinear.from_modules(linears, strategy='einsum').to(device).eval()
    wide_grouped = WideLinear.from_modules(linears, strategy='grouped').to(device).eval()
    wide_seq = WideLinear.from_modules(linears, strategy='sequential').to(device).eval()
    wide_auto = WideLinear.from_modules(linears, strategy='auto').to(device).eval()

    # Input
    x_list = [torch.randn(B, in_f, device=device) for _ in range(N)]
    x_packed = torch.cat(x_list, dim=1)  # [B, N*in]

    # Warmup
    for _ in range(20):
        for lin in linears:
            lin(x_list[0])
        wide_einsum(x_packed)
        wide_grouped(x_packed)
        wide_seq(x_packed)
    if device == 'cuda':
        torch.cuda.synchronize()

    num_iters = 200

    # Benchmark
    methods = [
        ('N×Module', lambda: [linears[i](x_list[i]) for i in range(N)]),
        ('Wide einsum', lambda: wide_einsum(x_packed)),
        ('Wide grouped', lambda: wide_grouped(x_packed)),
        ('Wide sequential', lambda: wide_seq(x_packed)),
        ('Wide auto', lambda: wide_auto(x_packed)),
    ]

    print(f"\n{'Method':<20} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 45)

    results = {}
    for name, fn in methods:
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iters):
                fn()
        if device == 'cuda':
            torch.cuda.synchronize()
        results[name] = time.perf_counter() - t0

    baseline = results['N×Module']
    for name, t in results.items():
        speedup = baseline / t
        marker = "✓" if speedup > 1.05 else ""
        print(f"{name:<20} {t * 1000:<12.1f} {speedup:<10.2f}x {marker}")

    # Verify correctness
    print("\nCorrectness check:")
    with torch.no_grad():
        ref = torch.cat([linears[i](x_list[i]) for i in range(N)], dim=1)
        out_einsum = wide_einsum(x_packed)
        out_grouped = wide_grouped(x_packed)
        out_seq = wide_seq(x_packed)

        print(f"  Einsum vs ref:     {(ref - out_einsum).abs().max().item():.2e}")
        print(f"  Grouped vs ref:    {(ref - out_grouped).abs().max().item():.2e}")
        print(f"  Sequential vs ref: {(ref - out_seq).abs().max().item():.2e}")