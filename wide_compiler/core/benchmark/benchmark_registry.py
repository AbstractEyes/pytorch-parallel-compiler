"""
WideCompiler.core.benchmark.benchmark_registry

Registry for benchmarkable primitives.

Auto-discovers primitives with BENCHMARK_SWEEPS and benchmark_job().

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import Dict, Any, Optional, List

# Registry storage
_PRIMITIVE_REGISTRY: Dict[str, Any] = {}
_INITIALIZED = False
_IMPORT_ERRORS: List[str] = []


def _auto_register():
    """Auto-register primitives that have benchmark interface."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    # Try multiple import paths
    primitives_module = None

    # Path 1: Relative import (when installed as package)
    try:
        from ..primitives import (
            WideConv1d,
            WideConv2d,
            WideLinear,
            WideBatchNorm1d,
            WideBatchNorm2d,
            WideLayerNorm,
            WideEmbedding,
        )
        primitives_module = True
    except ImportError as e1:
        # Path 2: Absolute import
        try:
            from wide_compiler.core.primitives import (
                WideConv1d,
                WideConv2d,
                WideLinear,
                WideBatchNorm1d,
                WideBatchNorm2d,
                WideLayerNorm,
                WideEmbedding,
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
        'linear': WideLinear,
        'batchnorm1d': WideBatchNorm1d,
        'batchnorm2d': WideBatchNorm2d,
        'layernorm': WideLayerNorm,
        'embedding': WideEmbedding,
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
                _IMPORT_ERRORS.append(f"{name}: BENCHMARK_SWEEPS is empty")


def register(name: str, primitive_class: Any) -> None:
    """
    Manually register a primitive for benchmarking.

    Args:
        name: Short name ('conv1d', 'linear', etc.)
        primitive_class: Class with benchmark_job() and BENCHMARK_SWEEPS
    """
    _auto_register()
    if not hasattr(primitive_class, 'benchmark_job'):
        raise ValueError(f"{primitive_class} must have benchmark_job() method")
    _PRIMITIVE_REGISTRY[name] = primitive_class


def register_primitive(name: str, primitive_class: Any) -> None:
    """Alias for register()."""
    register(name, primitive_class)


def get_primitive(name: str) -> Any:
    """
    Get primitive class by name.

    Args:
        name: Primitive name ('conv1d', 'conv2d', etc.)

    Returns:
        Primitive class with benchmark interface

    Raises:
        KeyError if not found
    """
    _auto_register()
    if name not in _PRIMITIVE_REGISTRY:
        available = list(_PRIMITIVE_REGISTRY.keys())
        raise KeyError(f"Unknown primitive '{name}'. Available: {available}")
    return _PRIMITIVE_REGISTRY[name]


def list_primitives() -> List[str]:
    """List all registered primitive names."""
    _auto_register()
    return list(_PRIMITIVE_REGISTRY.keys())


def has_primitive(name: str) -> bool:
    """Check if primitive is registered."""
    _auto_register()
    return name in _PRIMITIVE_REGISTRY


def get_import_errors() -> List[str]:
    """Get any import errors encountered during registration."""
    _auto_register()
    return _IMPORT_ERRORS.copy()


def debug_registration() -> None:
    """Print debug info about registration."""
    _auto_register()
    print(f"Registered primitives: {list(_PRIMITIVE_REGISTRY.keys())}")
    if _IMPORT_ERRORS:
        print("Import errors:")
        for err in _IMPORT_ERRORS:
            print(f"  - {err}")


__all__ = [
    'register',
    'register_primitive',
    'get_primitive',
    'list_primitives',
    'has_primitive',
    'get_import_errors',
    'debug_registration',
]