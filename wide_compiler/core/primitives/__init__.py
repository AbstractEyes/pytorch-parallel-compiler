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
- Einsum vs sequential linear: ~1e-6 relative error (fp32)

To disable TF32 globally at script start:
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

# Conv layers
from .wide_conv1d import WideConv1d, Conv1dStrategy
from .wide_conv2d import (
    WideConv2d,
    ConvStrategy,
    select_strategy as select_conv_strategy,
    create_wide_conv2d,
    STRATEGY_THRESHOLDS as CONV_THRESHOLDS,
)
from .wide_conv3d import WideConv3d, Conv3dStrategy

# Linear
from .wide_linear import (
    WideLinear,
    LinearStrategy,
    select_linear_strategy,
    LINEAR_THRESHOLDS,
)

# Normalization layers
from .wide_batchnorm_1d import WideBatchNorm1d
from .wide_batchnorm_2d import WideBatchNorm2d
from .wide_layernorm import WideLayerNorm
from .wide_groupnorm import WideGroupNorm, GroupNormStrategy
from .wide_instancenorm import (
    WideInstanceNorm1d,
    WideInstanceNorm2d,
    InstanceNormStrategy,
)

# Embedding
from .wide_embedding import WideEmbedding, EmbeddingStrategy

# Attention
from .wide_attention import WideAttention, AttentionStrategy

# RNN layers
from .wide_gru import WideGRU, GRUStrategy
from .wide_lstm import WideLSTM, LSTMStrategy


__all__ = [
    # Conv layers
    'WideConv1d',
    'Conv1dStrategy',
    'WideConv2d',
    'ConvStrategy',
    'select_conv_strategy',
    'create_wide_conv2d',
    'CONV_THRESHOLDS',
    'WideConv3d',
    'Conv3dStrategy',

    # Linear
    'WideLinear',
    'LinearStrategy',
    'select_linear_strategy',
    'LINEAR_THRESHOLDS',

    # Normalization
    'WideBatchNorm1d',
    'WideBatchNorm2d',
    'WideLayerNorm',
    'WideGroupNorm',
    'GroupNormStrategy',
    'WideInstanceNorm1d',
    'WideInstanceNorm2d',
    'InstanceNormStrategy',

    # Embedding
    'WideEmbedding',
    'EmbeddingStrategy',

    # Attention
    'WideAttention',
    'AttentionStrategy',

    # RNN
    'WideGRU',
    'GRUStrategy',
    'WideLSTM',
    'LSTMStrategy',
]