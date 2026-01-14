"""
WideCompiler.core.benchmark.benchmark_registry

Auto-discovers and registers primitives that support benchmarking.

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import Dict, Type, Optional, List
import torch.nn as nn

# Registry of benchmark-capable primitives
_PRIMITIVE_REGISTRY: Dict[str, Type[nn.Module]] = {}

# Store import errors for debugging
_IMPORT_ERRORS: List[str] = []


def register(name: str):
    """Decorator to register a primitive for benchmarking."""
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        register_primitive(name, cls)
        return cls
    return decorator


def register_primitive(name: str, cls: Type[nn.Module]) -> None:
    """Register a primitive class for benchmarking."""
    if not hasattr(cls, 'benchmark_job'):
        raise ValueError(f"{cls.__name__} must have a benchmark_job classmethod")
    _PRIMITIVE_REGISTRY[name] = cls


def get_primitive(name: str) -> Optional[Type[nn.Module]]:
    """Get a primitive class by name."""
    _ensure_registered()
    return _PRIMITIVE_REGISTRY.get(name)


def has_primitive(name: str) -> bool:
    """Check if a primitive is registered."""
    _ensure_registered()
    return name in _PRIMITIVE_REGISTRY


def list_primitives() -> List[str]:
    """List all registered primitive names."""
    _ensure_registered()
    return sorted(_PRIMITIVE_REGISTRY.keys())


def get_all_primitives() -> Dict[str, Type[nn.Module]]:
    """Get all registered primitives."""
    _ensure_registered()
    return dict(_PRIMITIVE_REGISTRY)


def get_import_errors() -> List[str]:
    """Get any import errors that occurred during registration."""
    return list(_IMPORT_ERRORS)


def debug_registration() -> None:
    """Print debug info about registered primitives."""
    _ensure_registered()
    print(f"Registered primitives: {list(_PRIMITIVE_REGISTRY.keys())}")
    if _IMPORT_ERRORS:
        print(f"Import errors:")
        for err in _IMPORT_ERRORS:
            print(f"  - {err}")


_REGISTERED = False


def _ensure_registered():
    """Lazy registration on first access."""
    global _REGISTERED
    if not _REGISTERED:
        _auto_register()
        _REGISTERED = True


def _auto_register():
    """Auto-discover and register primitives with benchmark support."""
    global _PRIMITIVE_REGISTRY, _IMPORT_ERRORS

    primitives_module = False

    # Path 1: Relative import (when installed as package)
    try:
        from ..primitives import (
            WideConv1d,
            WideConv2d,
            WideConv3d,
            WideLinear,
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
        primitives_module = True
    except ImportError as e1:
        # Path 2: Absolute import
        try:
            from wide_compiler.core.primitives import (
                WideConv1d,
                WideConv2d,
                WideConv3d,
                WideLinear,
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
            primitives_module = True
        except ImportError as e2:
            # Store errors for debugging
            _IMPORT_ERRORS.append(f"Relative: {e1}")
            _IMPORT_ERRORS.append(f"Absolute: {e2}")
            return

    if not primitives_module:
        return

    # Map names to classes
    primitives = {
        'conv1d': WideConv1d,
        'conv2d': WideConv2d,
        'conv3d': WideConv3d,
        'linear': WideLinear,
        'batchnorm1d': WideBatchNorm1d,
        'batchnorm2d': WideBatchNorm2d,
        'layernorm': WideLayerNorm,
        'groupnorm': WideGroupNorm,
        'instancenorm1d': WideInstanceNorm1d,
        'instancenorm2d': WideInstanceNorm2d,
        'embedding': WideEmbedding,
        'attention': WideAttention,
        'gru': WideGRU,
        'lstm': WideLSTM,
    }

    # Register those with benchmark interface
    for name, cls in primitives.items():
        if hasattr(cls, 'benchmark_job'):
            # Initialize sweeps if needed
            if hasattr(cls, '_init_benchmark_sweeps'):
                try:
                    cls._init_benchmark_sweeps()
                except Exception as e:
                    _IMPORT_ERRORS.append(f"{name}._init_benchmark_sweeps: {e}")
                    continue

            # Check if BENCHMARK_SWEEPS is populated
            sweeps = getattr(cls, 'BENCHMARK_SWEEPS', None)
            if sweeps:  # Non-empty dict
                _PRIMITIVE_REGISTRY[name] = cls
            else:
                _IMPORT_ERRORS.append(f"{name}: BENCHMARK_SWEEPS empty after init")
        else:
            _IMPORT_ERRORS.append(f"{name}: no benchmark_job method")


__all__ = [
    'register',
    'register_primitive',
    'get_primitive',
    'has_primitive',
    'list_primitives',
    'get_all_primitives',
    'get_import_errors',
    'debug_registration',
]