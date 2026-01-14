"""
WideCompiler.core.benchmark.benchmark_registry

Maps benchmark names to primitive classes.

Each primitive class must implement:
    - BENCHMARK_SWEEPS: Dict[str, SweepParams]
    - BENCHMARK_STRATEGIES: List[str]
    - benchmark_job(preset, **overrides) -> BenchmarkJob

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from __future__ import annotations

from typing import Dict, List, Type, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


# Registry of primitive name -> class
_REGISTRY: Dict[str, Type] = {}


def register(name: str):
    """
    Decorator to register a primitive class.

    Example:
        @register('conv1d')
        class WideConv1d(nn.Module):
            ...
    """
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def register_primitive(name: str, cls: Type):
    """Register a primitive class directly."""
    _REGISTRY[name] = cls


def get_primitive(name: str) -> Type:
    """
    Get primitive class by name.

    Raises:
        KeyError: If name not found
    """
    if name not in _REGISTRY:
        available = ', '.join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown primitive '{name}'. Available: {available}")
    return _REGISTRY[name]


def list_primitives() -> List[str]:
    """List all registered primitive names."""
    return sorted(_REGISTRY.keys())


def has_primitive(name: str) -> bool:
    """Check if primitive is registered."""
    return name in _REGISTRY


def get_all_primitives() -> Dict[str, Type]:
    """Get copy of full registry."""
    return dict(_REGISTRY)


# =============================================================================
# AUTO-REGISTRATION
# =============================================================================

def _auto_register():
    """
    Auto-register primitives that have benchmark interface.

    Called when primitives are imported.
    """
    try:
        from ..primitives import (
            WideLinear,
            WideConv1d,
            WideConv2d,
            WideBatchNorm1d,
            WideBatchNorm2d,
            WideLayerNorm,
            WideEmbedding,
        )

        # Only register if they have benchmark interface
        primitives = [
            ('linear', WideLinear),
            ('conv1d', WideConv1d),
            ('conv2d', WideConv2d),
            ('batchnorm1d', WideBatchNorm1d),
            ('batchnorm2d', WideBatchNorm2d),
            ('layernorm', WideLayerNorm),
            ('embedding', WideEmbedding),
        ]

        for name, cls in primitives:
            if hasattr(cls, 'benchmark_job'):
                _REGISTRY[name] = cls

    except ImportError:
        # Primitives not available yet
        pass


# Try to auto-register on import
_auto_register()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'register',
    'register_primitive',
    'get_primitive',
    'list_primitives',
    'has_primitive',
    'get_all_primitives',
]