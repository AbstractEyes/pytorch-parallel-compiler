"""
WideCompiler.core.blocks

Advanced composite blocks for Wide architectures.

This module contains higher-level building blocks that combine multiple
primitives into common architectural patterns (e.g., transformer blocks,
attention mechanisms with RoPE, etc.).

Unlike primitives/ which contains basic fused operations, blocks/ contains
composite structures meant for specific architectural use cases like Flux,
diffusion models, and other complex architectures.

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

# Attention blocks
from .wide_attention import WideAttention
from .wide_joint_attention import WideJointAttention

# Feed-forward blocks
from .wide_mlp import WideMLP

# Transformer blocks (Flux-style)
from .wide_double_stream_block import WideDoubleStreamBlock
from .wide_single_stream_block import WideSingleStreamBlock

__all__ = [
    # Attention blocks
    'WideAttention',
    'WideJointAttention',

    # Feed-forward blocks
    'WideMLP',

    # Transformer blocks
    'WideDoubleStreamBlock',
    'WideSingleStreamBlock',
]
