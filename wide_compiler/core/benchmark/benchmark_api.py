"""
WideCompiler.core.benchmark.benchmark_api

Public benchmark API.

Usage:
    from wide_compiler.core.benchmark import benchmark, benchmark_custom

    # Run registered benchmark
    result = benchmark('conv1d')
    result = benchmark('conv1d', preset='quick')
    result = benchmark('conv1d', n_values=[8, 16, 32])

    # Run custom model benchmark
    result = benchmark_custom(MyModel, (32, 256), [8, 16, 32, 64])

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Union, Type, Tuple
import torch
import torch.nn as nn

from .benchmark_schema import BenchmarkJob, BenchmarkResult, SweepParams
from .benchmark_runner import run
from .benchmark_registry import get_primitive, list_primitives


def benchmark(
        name: str,
        preset: str = 'full',
        device: str = 'cuda',
        verbose: bool = True,
        **overrides,
) -> BenchmarkResult:
    """
    Run benchmark for a registered primitive.

    Args:
        name: Primitive name ('conv1d', 'conv2d', 'linear', etc.)
        preset: Sweep preset ('quick', 'full', 'ci')
        device: 'cuda' or 'cpu'
        verbose: Print progress
        **overrides: Override sweep params (e.g., n_values=[4,8])

    Returns:
        BenchmarkResult with all measurements

    Examples:
        result = benchmark('conv1d')
        result = benchmark('conv1d', preset='quick')
        result = benchmark('conv1d', n_values=[8, 16, 32])
        result = benchmark('linear', features=[256, 512])
    """
    # Get primitive class
    cls = get_primitive(name)

    # Get benchmark job from primitive
    job = cls.benchmark_job(preset, **overrides)
    job.device = device

    return run(job, verbose=verbose)


def benchmark_multi(
        names: List[str],
        preset: str = 'full',
        device: str = 'cuda',
        verbose: bool = True,
) -> Dict[str, BenchmarkResult]:
    """
    Run benchmarks for multiple primitives.

    Args:
        names: List of primitive names
        preset: Sweep preset
        device: 'cuda' or 'cpu'
        verbose: Print progress

    Returns:
        Dict mapping name -> result
    """
    results = {}
    for name in names:
        if verbose:
            print(f"\n{'#' * 60}")
            print(f"# {name}")
            print(f"{'#' * 60}\n")
        results[name] = benchmark(name, preset=preset, device=device, verbose=verbose)
    return results


def benchmark_all(
        preset: str = 'quick',
        device: str = 'cuda',
        verbose: bool = True,
) -> Dict[str, BenchmarkResult]:
    """
    Run benchmarks for all registered primitives.

    Args:
        preset: Sweep preset (default 'quick' for all)
        device: 'cuda' or 'cpu'
        verbose: Print progress

    Returns:
        Dict mapping name -> result
    """
    names = list_primitives()
    return benchmark_multi(names, preset=preset, device=device, verbose=verbose)


def benchmark_custom(
        model_class: Type[nn.Module],
        input_shape: Tuple[int, ...],
        n_values: List[int],
        name: str = 'custom',
        device: str = 'cuda',
        batch_size: int = 8,
        warmup: int = 20,
        iters: int = 100,
        verbose: bool = True,
        **model_kwargs,
) -> BenchmarkResult:
    """
    Benchmark an arbitrary model class.

    Uses TracedWideModel to compile the model.

    Args:
        model_class: nn.Module subclass
        input_shape: Shape for single model input (without batch)
        n_values: List of N values to test
        name: Name for this benchmark
        device: 'cuda' or 'cpu'
        batch_size: Batch size for inputs
        warmup: Warmup iterations
        iters: Benchmark iterations
        verbose: Print progress
        **model_kwargs: Arguments passed to model_class()

    Returns:
        BenchmarkResult

    Example:
        class Expert(nn.Module):
            def __init__(self, d=256):
                super().__init__()
                self.fc1 = nn.Linear(d, d*4)
                self.fc2 = nn.Linear(d*4, d)
            def forward(self, x):
                return self.fc2(F.gelu(self.fc1(x)))

        result = benchmark_custom(
            Expert,
            input_shape=(256,),  # Single input shape (without batch)
            n_values=[8, 16, 32, 64, 100],
            d=256,  # model kwarg
        )
    """
    from ..traced_wide import trace_and_build_wide
    from ..wide_model import pack_inputs, unpack_outputs

    # Build factories
    def model_factory(**_):
        return model_class(**model_kwargs)

    def input_factory(n, device, **_):
        return torch.randn(batch_size, *input_shape, device=device)

    def wide_factory(modules, strategy):
        # Use traced_wide to compile
        sample = torch.randn(1, *input_shape, device=device)
        return trace_and_build_wide(modules, sample)

    def pack_fn(inputs):
        return pack_inputs(inputs)

    def unpack_fn(output, n):
        return unpack_outputs(output, n)

    # Build job
    job = BenchmarkJob(
        name=name,
        primitive='custom',
        strategies=['baseline', 'wide'],
        sweep=SweepParams(n_values=n_values),
        model_factory=model_factory,
        input_factory=input_factory,
        wide_factory=wide_factory,
        pack_fn=pack_fn,
        unpack_fn=unpack_fn,
        device=device,
        warmup=warmup,
        iters=iters,
    )

    return run(job, verbose=verbose)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'benchmark',
    'benchmark_multi',
    'benchmark_all',
    'benchmark_custom',
]