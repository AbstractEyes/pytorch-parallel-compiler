"""
WideCompiler - Compile-friendly batched model execution.

Fuse N identical models into a single Wide model for massive speedups.

Main API:
    import wide_compiler

    # Simple usage
    models = [MyModel() for _ in range(100)]
    wide = wide_compiler.compile(models, sample_input)

    # With compilation
    wide = wide_compiler.compile(models, sample_input, compile_model=True)

    # From single model
    wide = wide_compiler.compile(MyModel(), sample_input, n=100)

    # Builder pattern
    wide = (wide_compiler.WideBuilder(models)
        .with_sample(sample_input)
        .compile()
        .build())

    # Pack/unpack
    packed = wide_compiler.pack(inputs)
    outputs = wide_compiler.unpack(wide_out, n=100)

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

# Main API
try:
    from .api import (
        compile,
        WideBuilder,
        pack,
        unpack,
    )
except ImportError:
    from wide_compiler.api import (
        compile,
        WideBuilder,
        pack,
        unpack,
    )

# Config
try:
    from .core.config import (
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

# Registry
try:
    from .core.registry import (
        register,
        unregister,
        get_registry,
        list_registered,
    )
except ImportError:
    from wide_compiler.core.registry import (
        register,
        unregister,
        get_registry,
        list_registered,
    )

# Core classes (for advanced usage)
try:
    from .core.traced_wide import (
        TracedWideModel,
        FunctionalOp,
        BinaryOp,
        print_trace,
    )
except ImportError:
    from wide_compiler.core.traced_wide import (
        TracedWideModel,
        FunctionalOp,
        BinaryOp,
        print_trace,
    )

try:
    from .core.wide_model import (
        WideModel,
        pack_inputs,
        unpack_outputs,
    )
except ImportError:
    from wide_compiler.core.wide_model import (
        WideModel,
        pack_inputs,
        unpack_outputs,
    )

# Primitives (for manual construction)
try:
    from .core.primitives import (
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

__all__ = [
    # Main API
    'compile',
    'WideBuilder',
    'pack',
    'unpack',

    # Config
    'WideConfig',
    'get_default_config',
    'set_default_config',

    # Registry
    'register',
    'unregister',
    'get_registry',
    'list_registered',

    # Core
    'TracedWideModel',
    'WideModel',
    'FunctionalOp',
    'BinaryOp',
    'print_trace',
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
]