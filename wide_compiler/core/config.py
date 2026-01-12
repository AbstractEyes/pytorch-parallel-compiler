"""
WideCompiler.core.config

Configuration for Wide model compilation.

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal


@dataclass
class WideConfig:
    """
    Configuration for Wide model compilation.

    Attributes:
        n: Number of models to fuse. If None, inferred from models list.

        compile: Whether to torch.compile the result.
        compile_mode: torch.compile mode ('default', 'reduce-overhead', 'max-autotune').
        compile_backend: torch.compile backend ('inductor', 'eager', etc).

        precision: Compute precision ('fp32', 'fp16', 'bf16').
        allow_tf32: Allow TF32 on Ampere+ GPUs.

        validate: Run correctness validation after build.
        validate_rtol: Relative tolerance for validation.
        validate_atol: Absolute tolerance for validation.

        fallback_passthrough: Wrap unknown ops as passthrough (vs raise).
        strict_architecture: Require identical architectures (vs best-effort).

        debug: Enable debug output.
        profile: Enable profiling hooks.
    """

    # Model count
    n: Optional[int] = None

    # Compilation
    compile: bool = False
    compile_mode: Literal['default', 'reduce-overhead', 'max-autotune'] = 'reduce-overhead'
    compile_backend: str = 'inductor'
    compile_options: Dict[str, Any] = field(default_factory=dict)

    # Precision
    precision: Literal['fp32', 'fp16', 'bf16'] = 'fp32'
    allow_tf32: bool = True

    # Validation
    validate: bool = False
    validate_rtol: float = 1e-3
    validate_atol: float = 1e-5

    # Behavior
    fallback_passthrough: bool = True
    strict_architecture: bool = True

    # Debug
    debug: bool = False
    profile: bool = False

    def __post_init__(self):
        """Validate config."""
        valid_modes = ('default', 'reduce-overhead', 'max-autotune')
        if self.compile_mode not in valid_modes:
            raise ValueError(f"compile_mode must be one of {valid_modes}")

        valid_precisions = ('fp32', 'fp16', 'bf16')
        if self.precision not in valid_precisions:
            raise ValueError(f"precision must be one of {valid_precisions}")

    @classmethod
    def default(cls) -> 'WideConfig':
        """Default config for general use."""
        return cls()

    @classmethod
    def fast(cls) -> 'WideConfig':
        """Config optimized for speed."""
        return cls(
            compile=True,
            compile_mode='reduce-overhead',
            allow_tf32=True,
            validate=False,
        )

    @classmethod
    def debug(cls) -> 'WideConfig':
        """Config for debugging."""
        return cls(
            compile=False,
            validate=True,
            debug=True,
            fallback_passthrough=False,
        )

    @classmethod
    def safe(cls) -> 'WideConfig':
        """Config with validation enabled."""
        return cls(
            compile=False,
            validate=True,
            strict_architecture=True,
        )


# Singleton default config
_default_config: Optional[WideConfig] = None


def get_default_config() -> WideConfig:
    """Get the default config."""
    global _default_config
    if _default_config is None:
        _default_config = WideConfig.default()
    return _default_config


def set_default_config(config: WideConfig) -> None:
    """Set the default config."""
    global _default_config
    _default_config = config


__all__ = [
    'WideConfig',
    'get_default_config',
    'set_default_config',
]