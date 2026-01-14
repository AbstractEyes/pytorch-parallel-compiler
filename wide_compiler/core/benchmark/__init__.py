"""
WideCompiler.core.benchmark

Internal benchmarking suite for WideCompiler components.

Benchmarks:
    - traced_wide: TracedWideModel forward pass variants
    - conv1d: WideConv1d strategy selection
    - conv2d: WideConv2d strategy selection
    - linear: WideLinear strategy selection

Usage:
    from wide_compiler.core.benchmark import run_benchmark, list_benchmarks

    # List available
    print(list_benchmarks())

    # Run one
    results = run_benchmark('conv1d')

    # Run all primitives
    results = run_benchmark_all(tag='primitive')

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from .benchmark_registry import (
    register_benchmark,
    list_benchmarks,
    get_benchmark_info,
    run_benchmark,
    run_all_benchmarks,
    print_benchmark_summary,
    BenchmarkInfo,
)

# Import benchmark modules to trigger registration
from . import wide_conv1d_benchmark

# from . import traced_wide_benchmark  # Add when moved
# from . import wide_conv2d_benchmark  # Add when created
# from . import wide_linear_benchmark  # Add when created

__all__ = [
    'register_benchmark',
    'list_benchmarks',
    'get_benchmark_info',
    'run_benchmark',
    'run_all_benchmarks',
    'print_benchmark_summary',
    'BenchmarkInfo',
]