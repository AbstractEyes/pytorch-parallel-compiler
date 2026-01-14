"""
WideCompiler.core.benchmark.benchmark_schema

Minimal unified schema for benchmark system.

Classes:
    SweepParams     - Parameter ranges (lives in primitive files)
    BenchmarkJob    - Sweep + factories (runtime)
    SingleResult    - One (n, params) measurement
    BenchmarkResult - Complete output (serializable)

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Dict, List, Optional, Any, Callable, Iterator
from itertools import product
import json


# =============================================================================
# SWEEP PARAMS
# =============================================================================

@dataclass
class SweepParams:
    """
    Parameter ranges for benchmarking.

    Lives in each primitive file as BENCHMARK_SWEEPS dict.
    Use only the fields that apply to your primitive.
    """

    # Required
    n_values: List[int] = field(default_factory=lambda: [4, 8, 16, 32])

    # Optional - use what applies
    batch_sizes: List[int] = field(default_factory=lambda: [8])
    channels: List[int] = field(default_factory=list)
    features: List[int] = field(default_factory=list)
    seq_lengths: List[int] = field(default_factory=list)
    spatial_sizes: List[int] = field(default_factory=list)
    kernel_sizes: List[int] = field(default_factory=list)
    embed_dims: List[int] = field(default_factory=list)
    vocab_sizes: List[int] = field(default_factory=list)

    def param_grid(self) -> Iterator[Dict[str, Any]]:
        """
        Yield all parameter combinations (excluding n_values).

        Only includes non-empty fields.
        """
        fields = {}
        for name, val in asdict(self).items():
            if name == 'n_values':
                continue
            if isinstance(val, list) and len(val) > 0:
                fields[name] = val

        if not fields:
            yield {}
            return

        keys = list(fields.keys())
        for combo in product(*[fields[k] for k in keys]):
            yield dict(zip(keys, combo))

    def with_overrides(self, **kw) -> 'SweepParams':
        """Return copy with fields overridden."""
        return replace(self, **kw)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize (only non-empty fields)."""
        return {k: v for k, v in asdict(self).items()
                if isinstance(v, list) and len(v) > 0}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SweepParams':
        """Deserialize."""
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# =============================================================================
# BENCHMARK JOB
# =============================================================================

@dataclass
class BenchmarkJob:
    """
    Everything needed to run a benchmark.

    Created by primitive's benchmark_job() classmethod.
    Contains factories (not serializable).
    """

    # Identity
    name: str
    primitive: str
    strategies: List[str]
    sweep: SweepParams

    # Factories (primitive provides these)
    model_factory: Callable[..., Any]     # (**params) -> nn.Module
    input_factory: Callable[..., Any]     # (n, **params, device) -> Tensor
    wide_factory: Callable[..., Any]      # (modules, strategy) -> WideModule
    pack_fn: Callable[[List[Any]], Any]   # (inputs) -> packed
    unpack_fn: Callable[[Any, int], List] # (output, n) -> outputs

    # Settings
    device: str = 'cuda'
    warmup: int = 20
    iters: int = 100
    rtol: float = 1e-3
    atol: float = 1e-5


# =============================================================================
# RESULTS
# =============================================================================

@dataclass
class SingleResult:
    """Result for one (n, params) configuration."""

    n: int
    params: Dict[str, Any]

    # Timings (strategy -> milliseconds)
    timings: Dict[str, float] = field(default_factory=dict)

    # Speedups vs baseline
    speedups: Dict[str, float] = field(default_factory=dict)

    # Best
    best_strategy: str = ''
    best_speedup: float = 1.0

    # Correctness
    correct: bool = True
    max_error: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SingleResult':
        return cls(**d)


@dataclass
class BenchmarkResult:
    """
    Complete benchmark output.

    Fully serializable for save/load/compare.
    """

    # What was run
    name: str
    primitive: str
    strategies: List[str]
    sweep: SweepParams

    # All results
    results: List[SingleResult] = field(default_factory=list)

    # Summary
    crossover_n: Optional[int] = None
    recommended_threshold: Optional[int] = None
    strategy_wins: Dict[str, int] = field(default_factory=dict)
    best_speedup: float = 1.0
    best_config: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    device: str = ''
    started_at: str = ''
    completed_at: str = ''
    duration_s: float = 0.0
    errors: List[str] = field(default_factory=list)

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'primitive': self.primitive,
            'strategies': self.strategies,
            'sweep': self.sweep.to_dict(),
            'results': [r.to_dict() for r in self.results],
            'crossover_n': self.crossover_n,
            'recommended_threshold': self.recommended_threshold,
            'strategy_wins': self.strategy_wins,
            'best_speedup': self.best_speedup,
            'best_config': self.best_config,
            'device': self.device,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'duration_s': self.duration_s,
            'errors': self.errors,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BenchmarkResult':
        return cls(
            name=d['name'],
            primitive=d['primitive'],
            strategies=d['strategies'],
            sweep=SweepParams.from_dict(d['sweep']),
            results=[SingleResult.from_dict(r) for r in d.get('results', [])],
            crossover_n=d.get('crossover_n'),
            recommended_threshold=d.get('recommended_threshold'),
            strategy_wins=d.get('strategy_wins', {}),
            best_speedup=d.get('best_speedup', 1.0),
            best_config=d.get('best_config', {}),
            device=d.get('device', ''),
            started_at=d.get('started_at', ''),
            completed_at=d.get('completed_at', ''),
            duration_s=d.get('duration_s', 0.0),
            errors=d.get('errors', []),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> 'BenchmarkResult':
        return cls.from_dict(json.loads(s))

    def save(self, path: str):
        with open(path, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> 'BenchmarkResult':
        with open(path, 'r') as f:
            return cls.from_json(f.read())

    # --- Display ---

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"{'='*60}",
            f"Benchmark: {self.name}",
            f"Primitive: {self.primitive}",
            f"Device: {self.device}",
            f"Duration: {self.duration_s:.1f}s",
            f"{'='*60}",
            f"",
            f"Configurations tested: {len(self.results)}",
            f"Crossover N: {self.crossover_n}",
            f"Recommended threshold: {self.recommended_threshold}",
            f"Best speedup: {self.best_speedup:.2f}x",
            f"",
            f"Strategy wins:",
        ]
        for strat, wins in sorted(self.strategy_wins.items(), key=lambda x: -x[1]):
            lines.append(f"  {strat}: {wins}")

        if self.errors:
            lines.append(f"")
            lines.append(f"Errors: {len(self.errors)}")
            for e in self.errors[:3]:
                lines.append(f"  - {e}")

        return '\n'.join(lines)

    def table(self) -> str:
        """Results as ASCII table."""
        if not self.results:
            return "No results"

        strats = [s for s in self.strategies if s != 'baseline']
        header = f"{'N':>4} | " + " | ".join(f"{s:>10}" for s in strats) + " | Best"
        sep = "-" * len(header)

        lines = [header, sep]
        for r in self.results:
            speedup_strs = [f"{r.speedups.get(s, 0):.2f}x" for s in strats]
            lines.append(
                f"{r.n:>4} | " +
                " | ".join(f"{s:>10}" for s in speedup_strs) +
                f" | {r.best_strategy}"
            )

        return '\n'.join(lines)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SweepParams',
    'BenchmarkJob',
    'SingleResult',
    'BenchmarkResult',
]