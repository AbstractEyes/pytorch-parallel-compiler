"""
Models module for wide model implementations.

Contains fused wide versions of popular architectures using WideCompiler primitives.
"""

from .wide_tiny_flux import (
    WideTinyFlux,
    WideTinyFluxConfig,
    WideDoubleStreamBlock,
    WideSingleStreamBlock,
    WideAdaLayerNormZero,
    WideRotaryEmbedding,
    WideMLPEmbedderTimestep,
    WideVectorIn,
)

__all__ = [
    'WideTinyFlux',
    'WideTinyFluxConfig',
    'WideDoubleStreamBlock',
    'WideSingleStreamBlock',
    'WideAdaLayerNormZero',
    'WideRotaryEmbedding',
    'WideMLPEmbedderTimestep',
    'WideVectorIn',
]
