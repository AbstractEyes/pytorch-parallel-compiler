"""
WideCompiler.core.benchmark.benchmark_api

High-level benchmark API.

Usage:
    from wide_compiler.core.benchmark import benchmark, benchmark_all

    result = benchmark('conv1d')
    result = benchmark('conv1d', preset='quick')
    results = benchmark_all()

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import Dict, List, Optional, Any
import torch
from torch import nn, Tensor

from .benchmark_registry import get_primitive, list_primitives, has_primitive
from .benchmark_schema import BenchmarkResult, BenchmarkJob
from .benchmark_runner import run


def benchmark(
    primitive: str,
    preset: str = 'full',
    device: str = 'cuda',
    verbose: bool = True,
    **overrides,
) -> BenchmarkResult:
    """
    Benchmark a registered primitive.

    Args:
        primitive: Name ('conv1d', 'conv2d', 'linear', etc.)
        preset: Sweep preset ('quick', 'full', 'ci')
        device: 'cuda' or 'cpu'
        verbose: Print progress
        **overrides: Override sweep params (e.g., n_values=[4,8,16])

    Returns:
        BenchmarkResult with all measurements

    Example:
        result = benchmark('conv1d')
        result = benchmark('conv1d', preset='quick')
        result = benchmark('conv1d', n_values=[8, 16, 32])
    """
    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        if verbose:
            print("CUDA not available, falling back to CPU")
        device = 'cpu'

    # Get primitive class
    primitive_cls = get_primitive(primitive)

    # Get benchmark job
    job = primitive_cls.benchmark_job(preset=preset, **overrides)

    # Run
    return run(job, device=device, verbose=verbose)


def benchmark_multi(
    primitives: List[str],
    preset: str = 'full',
    device: str = 'cuda',
    verbose: bool = True,
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark multiple primitives.

    Args:
        primitives: List of primitive names
        preset: Sweep preset
        device: 'cuda' or 'cpu'
        verbose: Print progress

    Returns:
        Dict mapping primitive name to BenchmarkResult
    """
    results = {}
    for name in primitives:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Benchmarking: {name}")
            print('='*60)
        results[name] = benchmark(name, preset=preset, device=device, verbose=verbose)
    return results


def benchmark_all(
    preset: str = 'full',
    device: str = 'cuda',
    verbose: bool = True,
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark all registered primitives.

    Args:
        preset: Sweep preset
        device: 'cuda' or 'cpu'
        verbose: Print progress

    Returns:
        Dict mapping primitive name to BenchmarkResult
    """
    available = list_primitives()
    if not available:
        if verbose:
            print("No primitives registered with benchmark interface.")
        return {}
    return benchmark_multi(available, preset=preset, device=device, verbose=verbose)


def benchmark_custom(
    model_class,
    input_shape,
    n_values: List[int],
    name: str = 'custom',
    device: str = 'cuda',
    batch_size: int = 8,
    warmup: int = 3,
    iterations: int = 10,
    verbose: bool = True,
    pack_fn: Optional[Any] = None,
    **model_kwargs,
) -> BenchmarkResult:
    """
    Benchmark an arbitrary model class.

    Compares:
    - baseline: N separate model forward passes
    - wide: TracedWideModel from N models

    Args:
        model_class: nn.Module subclass
        input_shape: Shape for single model input (without batch dim)
        n_values: List of N values to test
        name: Name for this benchmark
        device: 'cuda' or 'cpu'
        batch_size: Batch size
        warmup: Warmup iterations
        iterations: Timing iterations
        verbose: Print progress
        pack_fn: Optional custom pack function (List[Tensor] -> Tensor).
                 Defaults to channel concatenation [B, N*C, ...] for TracedWideModel.
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

        result = benchmark_custom(Expert, (256,), [8, 16, 32, 64], d=256)
    """
    from .benchmark_runner import time_fn
    from .benchmark_schema import SingleResult, BenchmarkResult

    try:
        from ..traced_wide import TracedWideModel
        from ..ensemble_util import pack_inputs as default_pack
    except ImportError:
        from wide_compiler.core.traced_wide import TracedWideModel
        from wide_compiler.core.ensemble_util import pack_inputs as default_pack

    # Use channel concatenation by default (for TracedWideModel compatibility)
    if pack_fn is None:
        pack_fn = default_pack

    if device == 'cuda' and not torch.cuda.is_available():
        if verbose:
            print("CUDA not available, falling back to CPU")
        device = 'cpu'

    results = []

    for n in n_values:
        if verbose:
            print(f"\nN={n}")

        # Create N models
        models = [model_class(**model_kwargs).to(device).eval() for _ in range(n)]

        # Create sample input
        sample = torch.randn(batch_size, *input_shape, device=device)

        # Create inputs
        inputs = [torch.randn_like(sample) for _ in range(n)]

        # Build wide model
        wide_model = TracedWideModel.from_models(models, sample).to(device).eval()
        packed = pack_fn(inputs)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                for m in models:
                    m(inputs[0])
                wide_model(packed)

        # Time baseline (sequential)
        def baseline_fn():
            for i, m in enumerate(models):
                m(inputs[i])

        # Time wide
        def wide_fn():
            wide_model(packed)

        baseline_ms = time_fn(baseline_fn, iterations=iterations, device=device)
        wide_ms = time_fn(wide_fn, iterations=iterations, device=device)
        speedup = baseline_ms / wide_ms if wide_ms > 0 else 0

        if verbose:
            print(f"  baseline: {baseline_ms:.3f}ms")
            print(f"  wide:     {wide_ms:.3f}ms")
            print(f"  speedup:  {speedup:.2f}x")

        results.append(SingleResult(
            n=n,
            strategy='wide',
            params={'batch_size': batch_size, 'input_shape': input_shape},
            time_ms=wide_ms,
            baseline_ms=baseline_ms,
            speedup=speedup,
        ))

    return BenchmarkResult(
        name=name,
        primitive='custom',
        device=device,
        results=results,
    )


__all__ = [
    'benchmark',
    'benchmark_multi',
    'benchmark_all',
    'benchmark_custom',
]