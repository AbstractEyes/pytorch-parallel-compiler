"""
WideCompiler.core.benchmark.benchmark_registry

Registry for internal benchmarking scripts.
Each benchmark tests a specific component's performance characteristics.

Usage:
    from wide_compiler.core.benchmark import run_benchmark, list_benchmarks

    # Run specific benchmark
    run_benchmark('conv1d')

    # List available benchmarks
    print(list_benchmarks())

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from __future__ import annotations

from typing import Dict, Callable, List, Optional, Any
from dataclasses import dataclass


@dataclass
class BenchmarkInfo:
    """Metadata for a registered benchmark."""
    name: str
    description: str
    fn: Callable
    tags: List[str]


# Global registry
_BENCHMARKS: Dict[str, BenchmarkInfo] = {}


def register_benchmark(
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
) -> Callable:
    """
    Decorator to register a benchmark function.

    Args:
        name: Unique identifier for the benchmark
        description: Human-readable description
        tags: Categories (e.g., ['primitive', 'conv', 'strategy'])

    Example:
        @register_benchmark('conv1d', 'WideConv1d strategy comparison', ['primitive', 'conv'])
        def benchmark_conv1d(device='cuda', verbose=True):
            ...
    """

    def decorator(fn: Callable) -> Callable:
        _BENCHMARKS[name] = BenchmarkInfo(
            name=name,
            description=description or fn.__doc__ or "",
            fn=fn,
            tags=tags or [],
        )
        return fn

    return decorator


def list_benchmarks(tag: Optional[str] = None) -> List[str]:
    """
    List registered benchmark names.

    Args:
        tag: Filter by tag (e.g., 'primitive', 'traced')

    Returns:
        List of benchmark names
    """
    if tag is None:
        return list(_BENCHMARKS.keys())
    return [name for name, info in _BENCHMARKS.items() if tag in info.tags]


def get_benchmark_info(name: str) -> Optional[BenchmarkInfo]:
    """Get metadata for a benchmark."""
    return _BENCHMARKS.get(name)


def run_benchmark(
        name: str,
        device: str = 'cuda',
        verbose: bool = True,
        **kwargs,
) -> Any:
    """
    Run a registered benchmark.

    Args:
        name: Benchmark name
        device: 'cuda' or 'cpu'
        verbose: Print progress
        **kwargs: Additional args passed to benchmark function

    Returns:
        Benchmark results (format depends on benchmark)

    Raises:
        KeyError: If benchmark not found
    """
    if name not in _BENCHMARKS:
        available = ', '.join(_BENCHMARKS.keys())
        raise KeyError(f"Benchmark '{name}' not found. Available: {available}")

    info = _BENCHMARKS[name]
    if verbose:
        print(f"Running benchmark: {info.name}")
        print(f"Description: {info.description}")
        print("-" * 60)

    return info.fn(device=device, verbose=verbose, **kwargs)


def run_all_benchmarks(
        device: str = 'cuda',
        tag: Optional[str] = None,
        verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run all benchmarks (or filtered by tag).

    Args:
        device: 'cuda' or 'cpu'
        tag: Filter by tag
        verbose: Print progress

    Returns:
        Dict of {name: results}
    """
    names = list_benchmarks(tag)
    results = {}

    for name in names:
        try:
            results[name] = run_benchmark(name, device=device, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"FAILED: {name} - {e}")
            results[name] = {'error': str(e)}

    return results


def print_benchmark_summary():
    """Print summary of all registered benchmarks."""
    print("=" * 70)
    print("Registered Benchmarks")
    print("=" * 70)

    for name, info in sorted(_BENCHMARKS.items()):
        tags_str = ', '.join(info.tags) if info.tags else 'none'
        print(f"\n{name}")
        print(f"  Description: {info.description[:60]}...")
        print(f"  Tags: {tags_str}")

    print("\n" + "=" * 70)
    print(f"Total: {len(_BENCHMARKS)} benchmarks")


__all__ = [
    'register_benchmark',
    'list_benchmarks',
    'get_benchmark_info',
    'run_benchmark',
    'run_all_benchmarks',
    'print_benchmark_summary',
    'BenchmarkInfo',
]