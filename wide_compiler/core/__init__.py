"""
WideCompiler.core

Core components for Wide model compilation.

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

# Traced model (main entry point)
from .traced_wide import (
    TracedWideModel,
    FunctionalOp,
    BinaryOp,
    print_trace,
    GetAttrOp,
    TraceNode,
    analyze_trace,
    print_trace,
    DEFAULT_TRACE_CONFIG,
    FunctionalOp,
    BinaryOp,
    GetAttrOp,
    SequentialPassthrough,
    TraceReport,
    WideStage,
)

# Pack/unpack utilities
from .ensemble_util import (
    pack_inputs,
    unpack_outputs,
    to_n_first,
    from_n_first,
    iter_n_first,
    stack_n_first,
    get_n_first_shape,
)

# Registry
from .registry import (
    WideRegistry,
    get_registry,
    register,
    unregister,
    get_builder,
    build_wide,
    list_registered,
)

# Config
from .config import (
    WideConfig,
    get_default_config,
    set_default_config,

)

# Primitives
from .primitives import (
    WideLinear,
    WideConv1d,
    WideConv2d,
    WideConv3d,
    WideConvTranspose1d,
    WideConvTranspose2d,
    WideBatchNorm1d,
    WideBatchNorm2d,
    WideBatchNorm3d,
    WideLayerNorm,
    WideGroupNorm,
    WideInstanceNorm1d,
    WideInstanceNorm2d,
    WideRMSNorm,
    WideAdaLayerNormZeroSingle,
    WideEmbedding,
    WideMLPEmbedder,
    WideAttention,
    WideMultiheadCrossAttention,
    WideGRU,
    WideLSTM,
    WideRNN,
    WidePReLU,
    WideDropout,
    WideAdaptiveAvgPool2d,
)

from .blocks import (
    WideDoubleStreamBlock,
    WideSingleStreamBlock,
    WideAttention,
    WideMLP,
    WideJointAttention,
)

__all__ = [
    # Traced model
    'TracedWideModel',
    'FunctionalOp',
    'BinaryOp',
    'print_trace',
    'TraceNode',
    'analyze_trace',
    'print_trace',
    'DEFAULT_TRACE_CONFIG',
    'GetAttrOp',
    'SequentialPassthrough',
    'TraceReport',
    'WideStage',

    # Pack/unpack
    'pack_inputs',
    'unpack_outputs',
    'to_n_first',
    'from_n_first',
    'iter_n_first',
    'stack_n_first',
    'get_n_first_shape',

    # Registry
    'WideRegistry',
    'get_registry',
    'register',
    'unregister',
    'get_builder',
    'build_wide',
    'list_registered',

    # Config
    'WideConfig',
    'get_default_config',
    'set_default_config',

    # Primitives
    'WideLinear',
    'WideConv1d',
    'WideConv2d',
    'WideConv3d',
    'WideConvTranspose1d',
    'WideConvTranspose2d',
    'WideBatchNorm1d',
    'WideBatchNorm2d',
    'WideBatchNorm3d',
    'WideLayerNorm',
    'WideGroupNorm',
    'WideInstanceNorm1d',
    'WideInstanceNorm2d',
    'WideRMSNorm',
    'WideAdaLayerNormZeroSingle',
    'WideEmbedding',
    'WideMLPEmbedder',
    'WideAttention',
    'WideMultiheadCrossAttention',
    'WideGRU',
    'WideLSTM',
    'WideRNN',
    'WidePReLU',
    'WideDropout',
    'WideAdaptiveAvgPool2d',

    # Blocks
    'WideDoubleStreamBlock',
    'WideSingleStreamBlock',
    'WideAttention',
    'WideMLP',
    'WideJointAttention',
]