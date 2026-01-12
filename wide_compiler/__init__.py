"""
WideCompiler - Compile-friendly batched model execution.

Fuse N identical models into a single Wide model for massive speedups.

Example:
    from wide_compiler import TracedWideModel, pack_inputs, unpack_outputs

    models = [MyModel() for _ in range(100)]
    wide = TracedWideModel.from_models(models, sample_input)

    packed = pack_inputs([x for x in inputs])
    out = wide(packed)
    outputs = unpack_outputs(out, n=100)

Copyright 2025 AbstractPhil
Apache License 2.0
"""

from .core import (
    # Main API
    TracedWideModel,
    pack_inputs,
    unpack_outputs,

    # Wide primitives
    WideLinear,
    WideConv2d,
    WideConv1d,
    WideBatchNorm2d,
    WideBatchNorm1d,
    WideLayerNorm,
    WideEmbedding,

    # Utilities
    print_trace,
    FunctionalOp,
    BinaryOp,
)

__all__ = [
    'TracedWideModel',
    'pack_inputs',
    'unpack_outputs',
    'WideLinear',
    'WideConv2d',
    'WideConv1d',
    'WideBatchNorm2d',
    'WideBatchNorm1d',
    'WideLayerNorm',
    'WideEmbedding',
    'print_trace',
    'FunctionalOp',
    'BinaryOp',
]