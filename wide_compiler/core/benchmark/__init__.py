"""
WideCompiler.core.benchmark

Unified benchmark system.

API Usage:
    from wide_compiler.core.benchmark import benchmark, benchmark_custom

    # Run registered benchmark
    result = benchmark('conv1d')
    result = benchmark('conv1d', preset='quick')

    # Run custom model
    result = benchmark_custom(MyModel, (32, 256), [8, 16, 32])

Schema:
    SweepParams     - Parameter ranges (lives in primitive files)
    BenchmarkJob    - Sweep + factories (runtime)
    SingleResult    - One measurement
    BenchmarkResult - Full output (serializable)

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

# Schema
from .benchmark_schema import (
    SweepParams,
    BenchmarkJob,
    SingleResult,
    BenchmarkResult,
    CompilationMode,
    default_pack_fn,
    default_unpack_fn,
    default_validate_fn,
)

# Runner
from .benchmark_runner import run, run_single, time_fn

# Registry (import triggers auto-registration)
from .benchmark_registry import (
    register,
    register_primitive,
    get_primitive,
    list_primitives,
    has_primitive,
    get_import_errors,
    debug_registration,
)

# API (must come after registry)
from .benchmark_api import (
    benchmark,
    benchmark_multi,
    benchmark_all,
    benchmark_custom,
)

# Force registration on import
list_primitives()

__all__ = [
    # Schema
    'SweepParams',
    'BenchmarkJob',
    'SingleResult',
    'BenchmarkResult',
    'CompilationMode',
    'default_pack_fn',
    'default_unpack_fn',
    'default_validate_fn',

    # Runner
    'run',
    'run_single',
    'time_fn',

    # API
    'benchmark',
    'benchmark_multi',
    'benchmark_all',
    'benchmark_custom',

    # Registry
    'register',
    'register_primitive',
    'get_primitive',
    'list_primitives',
    'has_primitive',
    'get_import_errors',
    'debug_registration',
]