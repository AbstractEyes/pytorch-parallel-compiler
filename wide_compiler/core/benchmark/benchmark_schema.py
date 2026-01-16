"""
WideCompiler.core.benchmark.benchmark_schema

Data classes for benchmark configuration and results.
Includes compilation clause support for torch.compile optimization.

v0.6.0: N-first format support
- Default pack/unpack use torch.stack(dim=0) for [N, B, ...] format
- pack_fn/unpack_fn are optional - only needed for non-standard formats
- validate_fn optional - default compares output[i] vs baseline[i]

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from dataclasses import dataclass, field, replace
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from enum import Enum
import json
import torch
from torch import Tensor


class CompilationMode(Enum):
    """Compilation modes for benchmarking."""
    EAGER = 'eager'                       # No compilation
    DEFAULT = 'default'                   # torch.compile with default settings
    REDUCE_OVERHEAD = 'reduce-overhead'   # Optimized for low latency
    MAX_AUTOTUNE = 'max-autotune'         # Maximum optimization (slower compile)


def compilation_available() -> bool:
    """Check if torch.compile is available and functional."""
    try:
        major = int(torch.__version__.split('.')[0])
        if major < 2:
            return False
        # Quick sanity check
        @torch.compile(backend='inductor')
        def _test(x):
            return x + 1
        _test(torch.tensor(1.0))
        return True
    except Exception:
        return False


def get_compile_fn(mode: CompilationMode) -> Optional[Callable]:
    """Get compile function for given mode, or None for eager/unavailable."""
    if mode == CompilationMode.EAGER:
        return None
    if not compilation_available():
        return None

    if mode == CompilationMode.DEFAULT:
        return lambda m: torch.compile(m)
    elif mode == CompilationMode.REDUCE_OVERHEAD:
        return lambda m: torch.compile(m, mode='reduce-overhead')
    elif mode == CompilationMode.MAX_AUTOTUNE:
        return lambda m: torch.compile(m, mode='max-autotune')
    return None


# =============================================================================
# DEFAULT N-FIRST PACK/UNPACK/VALIDATE
# =============================================================================

def default_pack_fn(inputs: List[Tensor]) -> Tensor:
    """
    Default N-first packing: stack inputs along dim 0.

    Args:
        inputs: List of N tensors, each [B, ...]

    Returns:
        Stacked tensor [N, B, ...]
    """
    return torch.stack(inputs, dim=0)


def default_unpack_fn(output: Tensor, n: int) -> List[Tensor]:
    """
    Default N-first unpacking: index along dim 0.

    Args:
        output: Tensor [N, B, ...] or tuple where first element is [N, B, ...]
        n: Number of models

    Returns:
        List of N tensors, each [B, ...]
    """
    # Handle tuple outputs (e.g., GRU returns (output, h_n))
    if isinstance(output, tuple):
        output = output[0]
    return [output[i] for i in range(n)]


def default_validate_fn(
    wide_output: Tensor,
    baseline_outputs: List[Tensor],
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> Tuple[bool, str]:
    """
    Default N-first validation: compare output[i] vs baseline[i].

    Args:
        wide_output: [N, B, ...] or tuple with first element [N, B, ...]
        baseline_outputs: List of N tensors [B, ...]
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        (is_valid, message)
    """
    # Handle tuple outputs
    if isinstance(wide_output, tuple):
        wide_output = wide_output[0]

    n = len(baseline_outputs)

    # Check shape compatibility
    if wide_output.shape[0] != n:
        return False, f"N mismatch: wide has {wide_output.shape[0]}, expected {n}"

    # Compare each output
    max_diff = 0.0
    for i in range(n):
        wide_i = wide_output[i]
        base_i = baseline_outputs[i]

        if wide_i.shape != base_i.shape:
            return False, f"Shape mismatch at i={i}: {wide_i.shape} vs {base_i.shape}"

        diff = (wide_i - base_i).abs()
        max_diff = max(max_diff, diff.max().item())

        if not torch.allclose(wide_i, base_i, rtol=rtol, atol=atol):
            mean_diff = diff.mean().item()
            return False, f"Value mismatch at i={i}: max={max_diff:.6f}, mean={mean_diff:.6f}"

    return True, f"OK (max_diff={max_diff:.2e})"


@dataclass
class SweepParams:
    """
    Parameter ranges for benchmark sweep.

    Each primitive uses different subset of these.
    """
    # Common
    n_values: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    batch_sizes: List[int] = field(default_factory=lambda: [8])

    # Conv1d/Conv2d/Conv3d
    channels: List[int] = field(default_factory=list)
    kernel_sizes: List[int] = field(default_factory=list)

    # Sequence length (Conv1d, RNN, Attention)
    seq_lengths: List[int] = field(default_factory=list)

    # Spatial dims (Conv2d, Conv3d)
    heights: List[int] = field(default_factory=list)
    widths: List[int] = field(default_factory=list)

    # Linear/Embedding/Attention/RNN input
    d_model: List[int] = field(default_factory=list)

    # Attention heads
    n_heads: List[int] = field(default_factory=list)

    # RNN hidden size
    hidden_sizes: List[int] = field(default_factory=list)

    # Embedding specific
    vocab_sizes: List[int] = field(default_factory=list)
    embedding_dims: List[int] = field(default_factory=list)

    # Normalization
    num_groups: List[int] = field(default_factory=list)

    def with_overrides(self, **kwargs) -> 'SweepParams':
        """Create copy with overridden values."""
        return replace(self, **kwargs)


@dataclass
class BenchmarkJob:
    """
    Complete benchmark job specification.

    Contains sweep params and factory functions for creating
    models, inputs, and wide versions.

    v0.6.0 N-first format:
    - pack_fn/unpack_fn are OPTIONAL - defaults use torch.stack(dim=0)
    - validate_fn is OPTIONAL - default compares output[i] vs baseline[i]
    - Only override these for non-standard formats (e.g., tuple outputs)

    Compilation settings apply to BOTH Wide and baseline models
    for fair comparison.
    """
    name: str
    primitive: str
    strategies: List[str]
    sweep: SweepParams

    # Factory functions (required)
    model_factory: Callable[..., Any]  # (**params) -> nn.Module
    input_factory: Callable[..., Any]  # (n, device, **params) -> Tensor [B, ...]
    wide_factory: Callable[..., Any]   # (models, strategy) -> WideModule

    # Pack/unpack functions (optional - defaults to N-first)
    pack_fn: Optional[Callable[..., Any]] = None   # (inputs) -> packed [N, B, ...]
    unpack_fn: Optional[Callable[..., Any]] = None # (output, n) -> list of [B, ...]

    # Validation function (optional - defaults to N-first comparison)
    validate_fn: Optional[Callable[..., Any]] = None  # (wide_out, baseline_outs) -> (bool, str)

    # Validation tolerances (used by default validate_fn)
    validate_rtol: float = 1e-4
    validate_atol: float = 1e-5

    # Compilation settings
    compilation: CompilationMode = CompilationMode.EAGER
    compile_warmup_extra: int = 2  # Extra warmup iterations for compiled models

    # Validation settings
    validate: bool = True  # Run consistency check
    fail_on_invalid: bool = False  # Raise exception vs mark invalid

    def get_pack_fn(self) -> Callable:
        """Get pack function, using default if not specified."""
        return self.pack_fn if self.pack_fn is not None else default_pack_fn

    def get_unpack_fn(self) -> Callable:
        """Get unpack function, using default if not specified."""
        return self.unpack_fn if self.unpack_fn is not None else default_unpack_fn

    def get_validate_fn(self) -> Callable:
        """Get validation function, using default if not specified."""
        if self.validate_fn is not None:
            return self.validate_fn
        # Return default with configured tolerances
        return lambda w, b: default_validate_fn(w, b, self.validate_rtol, self.validate_atol)


@dataclass
class SingleResult:
    """Result from a single benchmark configuration."""
    n: int
    strategy: str
    params: Dict[str, Any]
    time_ms: float
    baseline_ms: float
    speedup: float
    compiled: bool = False  # Whether torch.compile was used
    valid: bool = True
    validation_msg: str = "OK"

    def to_dict(self) -> dict:
        return {
            'n': self.n,
            'strategy': self.strategy,
            'params': self.params,
            'time_ms': self.time_ms,
            'baseline_ms': self.baseline_ms,
            'speedup': self.speedup,
            'compiled': self.compiled,
            'valid': self.valid,
            'validation_msg': self.validation_msg,
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    name: str
    primitive: str
    device: str
    results: List[SingleResult]

    @property
    def valid_results(self) -> List[SingleResult]:
        """Get only valid results."""
        return [r for r in self.results if r.valid]

    @property
    def invalid_results(self) -> List[SingleResult]:
        """Get invalid results."""
        return [r for r in self.results if not r.valid]

    @property
    def crossover_n(self) -> Optional[int]:
        """Find N where wide becomes faster than baseline."""
        for r in sorted(self.valid_results, key=lambda x: x.n):
            if r.strategy != 'baseline' and r.speedup > 1.0:
                return r.n
        return None

    @property
    def best_speedup(self) -> float:
        """Maximum speedup achieved (valid results only)."""
        speedups = [r.speedup for r in self.valid_results if r.strategy != 'baseline']
        return max(speedups) if speedups else 1.0

    @property
    def best_result(self) -> Optional[SingleResult]:
        """Result with highest speedup (valid only)."""
        non_baseline = [r for r in self.valid_results if r.strategy != 'baseline']
        if not non_baseline:
            return None
        return max(non_baseline, key=lambda r: r.speedup)

    @property
    def strategy_wins(self) -> Dict[str, int]:
        """Count wins per strategy (excluding baseline, valid only)."""
        from collections import defaultdict

        # Group by (n, params)
        groups = defaultdict(list)
        for r in self.valid_results:
            if r.strategy != 'baseline':
                key = (r.n, tuple(sorted(r.params.items())))
                groups[key].append(r)

        # Find winner for each group
        wins = defaultdict(int)
        for group_results in groups.values():
            if group_results:
                best = min(group_results, key=lambda r: r.time_ms)
                wins[best.strategy] += 1

        return dict(wins)

    def top_results(self, n: int = 10) -> List[SingleResult]:
        """Get top N results by speedup (excluding baseline, valid only)."""
        non_baseline = [r for r in self.valid_results if r.strategy != 'baseline']
        return sorted(non_baseline, key=lambda r: -r.speedup)[:n]

    def summary(self) -> str:
        """Concise human-readable summary."""
        total = len(self.results)
        valid = len(self.valid_results)
        invalid = len(self.invalid_results)

        # Check if results were compiled
        compiled_results = [r for r in self.results if r.compiled]
        compiled_str = "compiled" if compiled_results else "eager"

        lines = [
            f"{'='*60}",
            f"  {self.name.upper()}",
            f"{'='*60}",
            f"  Device:       {self.device}",
            f"  Mode:         {compiled_str}",
            f"  Configs run:  {valid}" + (f" ({invalid} invalid)" if invalid else ""),
            f"  Crossover N:  {self.crossover_n or 'N/A'}",
            f"  Best speedup: {self.best_speedup:.2f}x",
        ]

        # Strategy wins
        wins = self.strategy_wins
        if wins:
            wins_str = ', '.join(f"{k}: {v}" for k, v in sorted(wins.items(), key=lambda x: -x[1]))
            lines.append(f"  Strategy wins: {wins_str}")

        # Best result details
        best = self.best_result
        if best:
            params_str = ', '.join(f"{k}={v}" for k, v in best.params.items())
            lines.extend([
                f"",
                f"  Best config:",
                f"    N={best.n}, strategy={best.strategy}",
                f"    {params_str}",
                f"    {best.baseline_ms:.3f}ms → {best.time_ms:.3f}ms ({best.speedup:.2f}x)",
            ])

        return '\n'.join(lines)

    def top_table(self, n: int = 10) -> str:
        """Format top N results as compact table."""
        top = self.top_results(n)
        if not top:
            return "No results."

        lines = [
            f"",
            f"  TOP {min(n, len(top))} RESULTS",
            f"  {'-'*56}",
            f"  {'N':>4} {'Strategy':<12} {'Time':>8} {'Baseline':>8} {'Speedup':>8}",
            f"  {'-'*56}",
        ]

        for r in top:
            lines.append(
                f"  {r.n:>4} {r.strategy:<12} {r.time_ms:>7.2f}ms {r.baseline_ms:>7.2f}ms {r.speedup:>7.2f}x"
            )

        if len(self.results) > n:
            lines.append(f"  ... ({len(self.results) - n} more results)")

        return '\n'.join(lines)

    def table(self, max_rows: int = 50) -> str:
        """Format all results as text table."""
        lines = [
            f"Benchmark: {self.name}",
            f"Device: {self.device}",
            f"Crossover N: {self.crossover_n}",
            f"Best speedup: {self.best_speedup:.2f}x",
            "",
            f"{'N':>4} {'Strategy':<12} {'Time(ms)':>10} {'Baseline':>10} {'Speedup':>8}",
            "-" * 50,
        ]

        for i, r in enumerate(self.results[:max_rows]):
            lines.append(
                f"{r.n:>4} {r.strategy:<12} {r.time_ms:>10.3f} {r.baseline_ms:>10.3f} {r.speedup:>7.2f}x"
            )

        if len(self.results) > max_rows:
            lines.append(f"... ({len(self.results) - max_rows} more rows)")

        return '\n'.join(lines)

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'primitive': self.primitive,
            'device': self.device,
            'crossover_n': self.crossover_n,
            'best_speedup': self.best_speedup,
            'strategy_wins': self.strategy_wins,
            'results': [r.to_dict() for r in self.results],
        }

    def save(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'BenchmarkResult':
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)

        results = [
            SingleResult(**r) for r in data['results']
        ]
        return cls(
            name=data['name'],
            primitive=data['primitive'],
            device=data['device'],
            results=results,
        )


__all__ = [
    'CompilationMode',
    'compilation_available',
    'get_compile_fn',
    'SweepParams',
    'BenchmarkJob',
    'SingleResult',
    'BenchmarkResult',
    # Default functions (for custom use)
    'default_pack_fn',
    'default_unpack_fn',
    'default_validate_fn',
]