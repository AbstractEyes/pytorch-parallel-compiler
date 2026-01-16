"""
WideLinear - N parallel Linear operations fused into a single module.

Key findings from A100 benchmarks:
1. Compiled einsum achieves 3-13x speedup vs N×Linear baseline
2. Einsum has ~1e-6 numerical error in fp32 (acceptable)
3. Low N + High B: baseline may be faster (N<8 with B>64)

Strategies:
- 'einsum': Batched matrix multiply via torch.einsum (RECOMMENDED)
- 'sequential': N separate F.linear (baseline, exact)
- 'auto': Heuristic selection (picks einsum for N>=8 or low B)

Input/Output Format (v0.6.0):
- Input:  [N, B, ..., in_features]  (N-first)
- Output: [N, B, ..., out_features] (N-first)

Speedup examples (compiled, A100):
- N=20, B=32:  3.0x
- N=50, B=16:  5.1x
- N=100, B=8:  12.8x

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Optional, Union, Dict, Any
from enum import Enum

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class LinearStrategy(Enum):
    EINSUM = 'einsum'         # Batched matmul - best for most cases
    SEQUENTIAL = 'sequential' # N separate ops - baseline, exact
    AUTO = 'auto'


# Thresholds from benchmarks
LINEAR_THRESHOLDS = {
    'n_min_einsum': 8,        # N below this with high B: sequential/baseline wins
    'b_max_for_low_n': 64,    # B above this with low N: baseline wins
    'expansion_max': 8.0,     # out/in ratio above this: sequential may win
    'vocab_size': 10000,      # out_features above this: sequential wins
}


def select_linear_strategy(n: int, b: int, in_features: int, out_features: int) -> LinearStrategy:
    """
    Heuristic for linear strategy selection.

    Einsum wins in most cases. Sequential only for edge cases:
    - Very small N with high B (not enough parallel work to offset overhead)
    - Very large output dimensions (vocab projections)
    """
    T = LINEAR_THRESHOLDS

    # Very small N with high B: overhead dominates
    if n < T['n_min_einsum'] and b > T['b_max_for_low_n']:
        return LinearStrategy.SEQUENTIAL

    # Vocab projections (very large output): sequential wins
    if out_features >= T['vocab_size']:
        return LinearStrategy.SEQUENTIAL

    # Default: einsum wins
    return LinearStrategy.EINSUM


class WideLinear(nn.Module):
    """
    N parallel Linear layers fused into a single module.

    Strategy is resolved at construction time and forward is delegated directly.
    This makes the module structurally immutable and torch.compile friendly.

    Strategies:
        'auto': Resolve to einsum or sequential based on N, B heuristics
        'einsum': Batched matrix multiply (RECOMMENDED - 3-13x speedup)
        'sequential': N separate F.linear calls (exact, slower)

    Note: For N<8 with B>64, baseline N×Linear may be faster.
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

        # Parse and resolve strategy at construction time
        if isinstance(strategy, str):
            strategy = LinearStrategy(strategy)

        if strategy == LinearStrategy.AUTO:
            strategy = select_linear_strategy(n, 32, in_features, out_features)

        self._strategy = strategy

        # Boolean for compile-friendly forward
        self._use_einsum = (strategy == LinearStrategy.EINSUM)

        # Store weights in einsum-friendly format: [N, out, in]
        self.weight = nn.Parameter(torch.empty(n, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(n, out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize
        self._reset_parameters()

    @property
    def strategy(self) -> LinearStrategy:
        """Read-only strategy (immutable after construction)."""
        return self._strategy

    def _reset_parameters(self):
        """Initialize parameters like nn.Linear."""
        import math
        for i in range(self.n):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in = self.in_features
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with N-first format.

        Input:  [N, B, ..., I] N-first with any batch dims
        Output: [N, B, ..., O]
        """
        if self._use_einsum:
            return self._forward_einsum(x)
        else:
            return self._forward_sequential(x)

    def _forward_einsum(self, x: Tensor) -> Tensor:
        """
        Batched matrix multiply via einsum.

        Input:  [N, B, ..., I] - N-first format
        Output: [N, B, ..., O]
        """
        N = self.n
        orig_shape = x.shape  # [N, B, ..., I]
        batch_shape = orig_shape[1:-1]  # [B, ...]

        # Flatten batch dims: [N, B, ..., I] -> [N, B*, I]
        x = x.reshape(N, -1, self.in_features)

        # Einsum: [N, out, in] @ [N, B*, in] -> [N, B*, out]
        out = torch.einsum('noi,nbi->nbo', self.weight, x)

        if self.bias is not None:
            out = out + self.bias.unsqueeze(1)  # [N, 1, out]

        # Restore batch dims: [N, B*, O] -> [N, B, ..., O]
        out = out.reshape(N, *batch_shape, self.out_features)

        return out

    def _forward_sequential(self, x: Tensor) -> Tensor:
        """
        N separate linear operations. Exact but slower.

        Input:  [N, B, ..., I] N-first format
        Output: [N, B, ..., O]
        """
        N = self.n
        orig_shape = x.shape  # [N, B, ..., I]
        batch_shape = orig_shape[1:-1]  # [B, ...]

        # Flatten batch dims: [N, B, ..., I] -> [N, B*, I]
        x = x.reshape(N, -1, self.in_features)

        outputs = []
        for i in range(N):
            xi = x[i]  # [B*, I]
            bi = self.bias[i] if self.bias is not None else None
            out = F.linear(xi, self.weight[i], bi)  # [B*, O]
            outputs.append(out)

        out = torch.stack(outputs, dim=0)  # [N, B*, O]

        # Restore batch dims: [N, B*, O] -> [N, B, ..., O]
        out = out.reshape(N, *batch_shape, self.out_features)

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

        # Preserve device and dtype from source modules
        wide = wide.to(device=template.weight.device, dtype=template.weight.dtype)

        with torch.no_grad():
            for i, m in enumerate(modules):
                wide.weight[i] = m.weight
                if m.bias is not None:
                    wide.bias[i] = m.bias

        return wide

    def __repr__(self):
        return (
            f"WideLinear({self.n}x[{self.in_features}->{self.out_features}], "
            f"strategy={self._strategy.value})"
        )

    # =========================================================================
    # BENCHMARK INTERFACE
    # =========================================================================

    BENCHMARK_STRATEGIES = ['baseline', 'einsum', 'sequential']

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
                batch_sizes=[16],
                d_model=[256],
            ),
            'full': SweepParams(
                n_values=[2, 4, 6, 8, 10, 12, 16, 20, 32, 64, 100],
                batch_sizes=[8, 16, 32, 64],
                d_model=[128, 256, 512, 768],
            ),
            'ci': SweepParams(
                n_values=[4, 16],
                batch_sizes=[16],
                d_model=[256],
            ),
        }

    @classmethod
    def benchmark_job(cls, preset: str = 'full', **overrides) -> Any:
        """Get benchmark job for WideLinear."""
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
            name=f'linear_{preset}',
            primitive='linear',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
            # pack_fn/unpack_fn/validate_fn: use defaults (N-first format)
        )

    @staticmethod
    def _bench_model(d_model: int, **_) -> nn.Linear:
        """Create single Linear for benchmarking."""
        return nn.Linear(d_model, d_model)

    @staticmethod
    def _bench_input(n: int, batch_sizes: int, d_model: int, device: str = 'cpu', **_) -> Tensor:
        """Create single input tensor [B, D]."""
        return torch.randn(batch_sizes, d_model, device=device)

    @classmethod
    def _bench_wide(cls, modules: List[nn.Linear], strategy: str) -> 'WideLinear':
        """Create WideLinear with specific strategy."""
        strat_map = {
            'einsum': LinearStrategy.EINSUM,
            'sequential': LinearStrategy.SEQUENTIAL,
        }
        strat = strat_map.get(strategy, LinearStrategy.EINSUM)
        return cls.from_modules(modules, strategy=strat)


__all__ = [
    'WideLinear',
    'LinearStrategy',
    'select_linear_strategy',
    'LINEAR_THRESHOLDS',
]