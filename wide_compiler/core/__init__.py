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
)

# Pack/unpack utilities
from .ensemble_util import (
    pack_inputs,
    unpack_outputs,
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
    WideConv2d,
    WideConv1d,
    WideConvTranspose1d,
    WideConvTranspose2d,
    WideBatchNorm2d,
    WideBatchNorm1d,
    WideBatchNorm3d,
    WideLayerNorm,
    WideEmbedding,
    WideAdaptiveAvgPool2d,
    WideRNN,
    WideGRU,
    WideLSTM,
    WideDropout,
    WidePReLU,
    WideAttention,
    WideMultiheadCrossAttention,
    WideGroupNorm,
    WideConv3d,
    WideInstanceNorm1d,
    WideInstanceNorm2d,
)

__all__ = [
    # Traced model
    'TracedWideModel',
    'FunctionalOp',
    'BinaryOp',
    'print_trace',

    # Pack/unpack
    'pack_inputs',
    'unpack_outputs',

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
    'WideConv2d',
    'WideConv1d',
    'WideBatchNorm2d',
    'WideBatchNorm1d',
    'WideLayerNorm',
    'WideEmbedding',
    'WideAdaptiveAvgPool2d',
    'WideRNN',
    'WideGRU',
    'WideLSTM',
    'WideDropout',
    'WidePReLU',
    'WideAttention',
    'WideMultiheadCrossAttention',
    'WideGroupNorm',
    'WideConv3d',
    'WideConvTranspose1d',
    'WideConvTranspose2d',
    'WideInstanceNorm1d',
    'WideInstanceNorm2d',


]