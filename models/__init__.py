"""
Models module for wide model implementations.

Contains fused wide versions of popular architectures using WideCompiler primitives.
"""

from .wide_tiny_flux import WideTinyFlux

__all__ = ['WideTinyFlux']
