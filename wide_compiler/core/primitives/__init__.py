"""
WideCompiler.core.primitives

Wide primitive modules for fusing N parallel operations.

Numerical Accuracy Notes:
-------------------------
PyTorch documents that batched operations (like grouped conv, einsum) are NOT
guaranteed to match sequential operations bitwise, even for identical math.
Additionally, TF32 is enabled by default for convolutions on Ampere+ GPUs.

Expected tolerances:
- NCHW grouped vs sequential: ~1e-6 relative error (IEEE fp32)
- NHWC grouped vs sequential: 0.0 error (most accurate)
- With TF32: ~1e-4 relative error
- Einsum vs sequential linear: ~1e-5 relative error

To disable TF32 globally at script start:
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

WideConv2d Strategies (A100 benchmarks):
- 'grouped': NCHW format (FASTEST: 2-4x speedup, ~1e-6 error)
- 'channels_last': NHWC format (most accurate: 0.0 error, but slower)
- 'sequential': N separate ops (exact, slowest)
- 'auto': Selects grouped NCHW

WideLinear Strategies:
- 'einsum': Batched matmul (6x+ speedup, ~1e-5 error)
- 'sequential': N separate ops (exact, slower)
- 'auto': Selects einsum for N>=3

Copyright 2025 AbstractPhil
MIT License
"""

from .wide_conv2d import (
    WideConv2d,
    ConvStrategy,
    select_strategy as select_conv_strategy,
    create_wide_conv2d,
    STRATEGY_THRESHOLDS as CONV_THRESHOLDS,
)

from .wide_linear import (
    WideLinear,
    LinearStrategy,
    select_linear_strategy,
    LINEAR_THRESHOLDS,
)

from .wide_conv1d import WideConv1d
from .wide_batchnorm_2d import WideBatchNorm2d
from .wide_batchnorm_1d import WideBatchNorm1d
from .wide_layernorm import WideLayerNorm
from .wide_embedding import WideEmbedding


__all__ = [
    # Conv2d with strategies
    'WideConv2d',
    'ConvStrategy',
    'select_conv_strategy',
    'create_wide_conv2d',
    'CONV_THRESHOLDS',

    # Linear with strategies
    'WideLinear',
    'LinearStrategy',
    'select_linear_strategy',
    'LINEAR_THRESHOLDS',

    # Other primitives
    'WideConv1d',
    'WideBatchNorm2d',
    'WideBatchNorm1d',
    'WideLayerNorm',
    'WideEmbedding',
]