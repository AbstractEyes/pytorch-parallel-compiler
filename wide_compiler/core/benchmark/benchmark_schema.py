"""
WideCompiler.core.benchmark.benchmark_schema

Data classes for benchmark configuration and results.

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from dataclasses import dataclass, field, replace
from typing import List, Dict, Any, Optional, Callable
import json


@dataclass
class SweepParams:
    """
    Parameter ranges for benchmark sweep.

    Each primitive uses different subset of these.
    """
    # Common
    n_values: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    batch_sizes: List[int] = field(default_factory=lambda: [8])

    # Conv1d/Conv2d
    channels: List[int] = field(default_factory=list)
    kernel_sizes: List[int] = field(default_factory=list)

    # Conv1d specific
    seq_lengths: List[int] = field(default_factory=list)

    # Conv2d specific
    heights: List[int] = field(default_factory=list)
    widths: List[int] = field(default_factory=list)

    # Linear/Embedding/Attention
    d_model: List[int] = field(default_factory=list)

    # Attention specific
    n_heads: List[int] = field(default_factory=list)

    # Embedding specific
    vocab_sizes: List[int] = field(default_factory=list)
    embedding_dims: List[int] = field(default_factory=list)

    def with_overrides(self, **kwargs) -> 'SweepParams':
        """Create copy with overridden values."""
        return replace(self, **kwargs)


@dataclass
class BenchmarkJob:
    """
    Complete benchmark job specification.

    Contains sweep params and factory functions for creating
    models, inputs, and wide versions.
    """
    name: str
    primitive: str
    strategies: List[str]
    sweep: SweepParams

    # Factory functions
    model_factory: Callable[..., Any]  # (**params) -> nn.Module
    input_factory: Callable[..., Any]  # (n, device, **params) -> Tensor
    wide_factory: Callable[..., Any]   # (models, strategy) -> WideModule
    pack_fn: Callable[..., Any]        # (inputs) -> packed
    unpack_fn: Callable[..., Any]      # (output, n) -> outputs


@dataclass
class SingleResult:
    """Result from a single benchmark configuration."""
    n: int
    strategy: str
    params: Dict[str, Any]
    time_ms: float
    baseline_ms: float
    speedup: float

    def to_dict(self) -> dict:
        return {
            'n': self.n,
            'strategy': self.strategy,
            'params': self.params,
            'time_ms': self.time_ms,
            'baseline_ms': self.baseline_ms,
            'speedup': self.speedup,
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    name: str
    primitive: str
    device: str
    results: List[SingleResult]

    @property
    def crossover_n(self) -> Optional[int]:
        """Find N where wide becomes faster than baseline."""
        for r in sorted(self.results, key=lambda x: x.n):
            if r.strategy != 'baseline' and r.speedup > 1.0:
                return r.n
        return None

    @property
    def best_speedup(self) -> float:
        """Maximum speedup achieved."""
        speedups = [r.speedup for r in self.results if r.strategy != 'baseline']
        return max(speedups) if speedups else 1.0

    @property
    def best_result(self) -> Optional[SingleResult]:
        """Result with highest speedup."""
        non_baseline = [r for r in self.results if r.strategy != 'baseline']
        if not non_baseline:
            return None
        return max(non_baseline, key=lambda r: r.speedup)

    @property
    def strategy_wins(self) -> Dict[str, int]:
        """Count wins per strategy (excluding baseline)."""
        from collections import defaultdict

        # Group by (n, params)
        groups = defaultdict(list)
        for r in self.results:
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
        """Get top N results by speedup (excluding baseline)."""
        non_baseline = [r for r in self.results if r.strategy != 'baseline']
        return sorted(non_baseline, key=lambda r: -r.speedup)[:n]

    def summary(self) -> str:
        """Concise human-readable summary."""
        lines = [
            f"{'='*60}",
            f"  {self.name.upper()}",
            f"{'='*60}",
            f"  Device:       {self.device}",
            f"  Configs run:  {len(self.results)}",
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
    'SweepParams',
    'BenchmarkJob',
    'SingleResult',
    'BenchmarkResult',
]