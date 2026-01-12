"""
WideCompiler - Compile-friendly batched model execution.

Fuse N identical models into a single Wide model for massive speedups.

Usage:
    from wide_compiler import TracedWideModel, pack_inputs, unpack_outputs

    # Create N identical models
    models = [MyModel() for _ in range(100)]

    # Build wide model (requires sample input for tracing)
    sample = torch.randn(1, 64)
    wide_model = TracedWideModel.from_models(models, sample)

    # Pack inputs: list of [B, C, ...] -> [B, N*C, ...]
    inputs = [torch.randn(32, 64) for _ in range(100)]
    packed = pack_inputs(inputs)

    # Run wide model
    wide_out = wide_model(packed)

    # Unpack outputs: [B, N*C, ...] -> list of [B, C, ...]
    outputs = unpack_outputs(wide_out, n=100)

Copyright 2025 AbstractPhil
Apache License 2.0
"""

from .primitives import (
    # Wide primitives
    WideLinear,
    WideConv2d,
    WideConv1d,
    WideBatchNorm2d,
    WideBatchNorm1d,
    WideLayerNorm,
    WideEmbedding,
)

from .wide_model import (
    pack_inputs,
    unpack_outputs
)

from .traced_wide import (
    # Main API
    TracedWideModel,

    # Utilities
    print_trace,
    FunctionalOp,
    BinaryOp,
)

__all__ = [
    # Main API
    'TracedWideModel',
    'pack_inputs',
    'unpack_outputs',

    # Wide primitives (for manual construction)
    'WideLinear',
    'WideConv2d',
    'WideConv1d',
    'WideBatchNorm2d',
    'WideBatchNorm1d',
    'WideLayerNorm',
    'WideEmbedding',

    # Utilities
    'print_trace',
    'FunctionalOp',
    'BinaryOp',
]