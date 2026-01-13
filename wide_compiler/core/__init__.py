"""
WideCompiler.core

Core components for Wide model compilation.

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

try:
    from .config import (
        WideConfig,
        get_default_config,
        set_default_config,
    )
except ImportError:
    from wide_compiler.core.config import (
        WideConfig,
        get_default_config,
        set_default_config,
    )

try:
    from .registry import (
        WideRegistry,
        get_registry,
        register,
        unregister,
        get_builder,
        build_wide,
        list_registered,
    )
except ImportError:
    from wide_compiler.core.registry import (
        WideRegistry,
        get_registry,
        register,
        unregister,
        get_builder,
        build_wide,
        list_registered,
    )

try:
    from .traced_wide import (
        TracedWideModel,
        TraceNode,
        WideStage,
        FunctionalOp,
        BinaryOp,
        analyze_trace,
        print_trace,
    )
except ImportError:
    from wide_compiler.core.traced_wide import (
        TracedWideModel,
        TraceNode,
        WideStage,
        FunctionalOp,
        BinaryOp,
        analyze_trace,
        print_trace,
    )

try:
    from .wide_model import (
        WideModel,
        TreeNode,
        traverse,
        get_leaves,
        make_wide,
        align_modules,
        pack_inputs,
        unpack_outputs,
    )
except ImportError:
    from wide_compiler.core.wide_model import (
        WideModel,
        TreeNode,
        traverse,
        get_leaves,
        make_wide,
        align_modules,
        pack_inputs,
        unpack_outputs,
    )

try:
    from .primitives import (
        WideLinear,
        WideConv2d,
        WideConv1d,
        WideBatchNorm2d,
        WideBatchNorm1d,
        WideLayerNorm,
        WideEmbedding,
    )
except ImportError:
    from wide_compiler.core.primitives import (
        WideLinear,
        WideConv2d,
        WideConv1d,
        WideBatchNorm2d,
        WideBatchNorm1d,
        WideLayerNorm,
        WideEmbedding,
    )

try:
    from .traced_benchmark import (
        benchmark_model,
        benchmark_models,
        profile_with_torch_profiler,
        BenchmarkResult,
        TimingResult,
        StageProfile,
        MemoryResult,
    )
except ImportError:
    from wide_compiler.core.traced_benchmark import (
        benchmark_model,
        benchmark_models,
        profile_with_torch_profiler,
        BenchmarkResult,
        TimingResult,
        StageProfile,
        MemoryResult,
    )

__all__ = [
    # Config
    'WideConfig',
    'get_default_config',
    'set_default_config',

    # Registry
    'WideRegistry',
    'get_registry',
    'register',
    'unregister',
    'get_builder',
    'build_wide',
    'list_registered',

    # Traced Wide
    'TracedWideModel',
    'TraceNode',
    'WideStage',
    'FunctionalOp',
    'BinaryOp',
    'analyze_trace',
    'print_trace',

    # Wide Model
    'WideModel',
    'TreeNode',
    'traverse',
    'get_leaves',
    'make_wide',
    'align_modules',
    'pack_inputs',
    'unpack_outputs',

    # Primitives
    'WideLinear',
    'WideConv2d',
    'WideConv1d',
    'WideBatchNorm2d',
    'WideBatchNorm1d',
    'WideLayerNorm',
    'WideEmbedding',

    # Benchmark
    'benchmark_model',
    'benchmark_models',
    'profile_with_torch_profiler',
    'BenchmarkResult',
    'TimingResult',
    'StageProfile',
    'MemoryResult',
]