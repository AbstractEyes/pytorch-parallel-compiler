"""
WideCompiler.core.registry

Registry for Wide primitives with decorator-based registration.

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from __future__ import annotations

from typing import Dict, Type, Callable, Optional, List, Any
import torch.nn as nn


# =============================================================================
# REGISTRY
# =============================================================================

class WideRegistry:
    """
    Registry mapping module types to their Wide equivalents.

    Usage:
        @registry.register('Linear')
        class WideLinear(nn.Module):
            @classmethod
            def from_modules(cls, modules: List[nn.Module]) -> 'WideLinear':
                ...
    """

    def __init__(self):
        self._builders: Dict[str, Callable[[List[nn.Module]], nn.Module]] = {}
        self._classes: Dict[str, Type[nn.Module]] = {}

    def register(self, module_type: str, wide_class: Optional[Type[nn.Module]] = None):
        """
        Register a Wide primitive.

        Can be used as decorator:
            @registry.register('Linear')
            class WideLinear: ...

        Or directly:
            registry.register('Linear', WideLinear)
        """
        def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
            if not hasattr(cls, 'from_modules'):
                raise TypeError(
                    f"{cls.__name__} must implement classmethod from_modules(modules: List[nn.Module])"
                )

            self._classes[module_type] = cls
            self._builders[module_type] = cls.from_modules
            return cls

        if wide_class is not None:
            # Direct registration
            decorator(wide_class)
            return wide_class

        # Decorator usage
        return decorator

    def unregister(self, module_type: str) -> None:
        """Remove a registration."""
        self._builders.pop(module_type, None)
        self._classes.pop(module_type, None)

    def get_builder(self, module_type: str) -> Optional[Callable[[List[nn.Module]], nn.Module]]:
        """Get builder function for a module type."""
        return self._builders.get(module_type)

    def get_class(self, module_type: str) -> Optional[Type[nn.Module]]:
        """Get Wide class for a module type."""
        return self._classes.get(module_type)

    def has(self, module_type: str) -> bool:
        """Check if module type is registered."""
        return module_type in self._builders

    def build(self, modules: List[nn.Module]) -> Optional[nn.Module]:
        """
        Build Wide version from N modules.

        Returns None if module type not registered.
        """
        if not modules:
            return None

        module_type = type(modules[0]).__name__
        builder = self.get_builder(module_type)

        if builder is None:
            return None

        return builder(modules)

    def list_registered(self) -> List[str]:
        """List all registered module types."""
        return list(self._builders.keys())

    def __contains__(self, module_type: str) -> bool:
        return self.has(module_type)

    def __repr__(self) -> str:
        types = ', '.join(sorted(self._builders.keys()))
        return f"WideRegistry([{types}])"


# =============================================================================
# GLOBAL REGISTRY
# =============================================================================

# Global registry instance
_global_registry = WideRegistry()


def get_registry() -> WideRegistry:
    """Get the global registry."""
    return _global_registry


def register(module_type: str, wide_class: Optional[Type[nn.Module]] = None):
    """
    Register a Wide primitive to the global registry.

    Usage as decorator:
        @register('MyModule')
        class WideMyModule(nn.Module):
            @classmethod
            def from_modules(cls, modules): ...

    Usage directly:
        register('MyModule', WideMyModule)
    """
    return _global_registry.register(module_type, wide_class)


def unregister(module_type: str) -> None:
    """Remove from global registry."""
    _global_registry.unregister(module_type)


def get_builder(module_type: str) -> Optional[Callable]:
    """Get builder from global registry."""
    return _global_registry.get_builder(module_type)


def build_wide(modules: List[nn.Module]) -> Optional[nn.Module]:
    """Build Wide module using global registry."""
    return _global_registry.build(modules)


def list_registered() -> List[str]:
    """List registered types in global registry."""
    return _global_registry.list_registered()


# =============================================================================
# AUTO-REGISTRATION
# =============================================================================

def auto_register_primitives() -> None:
    """
    Auto-register all built-in Wide primitives.
    Called on import.
    """
    try:
        try:
            from .primitives import (
                WideLinear,
                WideConv1d,
                WideConv2d,
                WideConv3d,
                WideBatchNorm1d,
                WideBatchNorm2d,
                WideLayerNorm,
                WideGroupNorm,
                WideInstanceNorm1d,
                WideInstanceNorm2d,
                WideEmbedding,
                WideAttention,
                WideGRU,
                WideLSTM,
            )
        except ImportError:
            from wide_compiler.core.primitives import (
                WideLinear,
                WideConv1d,
                WideConv2d,
                WideConv3d,
                WideBatchNorm1d,
                WideBatchNorm2d,
                WideLayerNorm,
                WideGroupNorm,
                WideInstanceNorm1d,
                WideInstanceNorm2d,
                WideEmbedding,
                WideAttention,
                WideGRU,
                WideLSTM,
            )

        # Linear
        _global_registry.register('Linear', WideLinear)
        # Convolutions
        _global_registry.register('Conv1d', WideConv1d)
        _global_registry.register('Conv2d', WideConv2d)
        _global_registry.register('Conv3d', WideConv3d)
        # Normalization
        _global_registry.register('BatchNorm1d', WideBatchNorm1d)
        _global_registry.register('BatchNorm2d', WideBatchNorm2d)
        _global_registry.register('LayerNorm', WideLayerNorm)
        _global_registry.register('GroupNorm', WideGroupNorm)
        _global_registry.register('InstanceNorm1d', WideInstanceNorm1d)
        _global_registry.register('InstanceNorm2d', WideInstanceNorm2d)
        # Embedding
        _global_registry.register('Embedding', WideEmbedding)
        # Attention
        _global_registry.register('MultiheadAttention', WideAttention)
        # RNNs
        _global_registry.register('GRU', WideGRU)
        _global_registry.register('LSTM', WideLSTM)

    except ImportError:
        # Primitives not available - skip
        pass


# Auto-register on module load
auto_register_primitives()


__all__ = [
    'WideRegistry',
    'get_registry',
    'register',
    'unregister',
    'get_builder',
    'build_wide',
    'list_registered',
    'auto_register_primitives',
]