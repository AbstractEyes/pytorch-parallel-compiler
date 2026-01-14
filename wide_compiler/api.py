"""
WideCompiler.api

Main API entry point.

    import wide_compiler
    wide_model = wide_compiler.compile(models, sample_input)

Benchmark API:
    from wide_compiler.api import benchmark_primitive
    result = benchmark_primitive('conv1d')

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from __future__ import annotations

from typing import List, Optional, Union, overload
import torch
import torch.nn as nn
from torch import Tensor

try:
    from .core.config import WideConfig, get_default_config
    from .core.registry import get_registry
except ImportError:
    from wide_compiler.core.config import WideConfig, get_default_config
    from wide_compiler.core.registry import get_registry


# =============================================================================
# MAIN API
# =============================================================================

def compile(
    models: Union[nn.Module, List[nn.Module]],
    sample_input: Optional[Tensor] = None,
    *,
    n: Optional[int] = None,
    config: Optional[WideConfig] = None,
    # Convenience kwargs (override config)
    compile_model: Optional[bool] = None,
    compile_mode: Optional[str] = None,
    validate: Optional[bool] = None,
    debug: Optional[bool] = None,
) -> nn.Module:
    """
    Compile N models into a single Wide model.

    This is the main entry point for WideCompiler.

    Args:
        models: Either a single model (will create N copies) or list of N models.
        sample_input: Sample input tensor for FX tracing. Required for tracing.
        n: Number of model copies if models is a single model.
        config: WideConfig instance. Uses default if None.
        compile_model: Override config.compile
        compile_mode: Override config.compile_mode
        validate: Override config.validate
        debug: Override config.debug

    Returns:
        TracedWideModel ready for inference/training.

    Example:
        # From list of models
        models = [MyModel() for _ in range(100)]
        wide = wide_compiler.compile(models, sample_input)

        # From single model (creates N copies)
        wide = wide_compiler.compile(MyModel(), sample_input, n=100)

        # With config
        config = WideConfig.fast()
        wide = wide_compiler.compile(models, sample_input, config=config)

        # With overrides
        wide = wide_compiler.compile(models, sample_input, compile_model=True)
    """
    # Import here to avoid circular imports
    try:
        from .core.traced_wide import TracedWideModel
        from .core.ensemble_util import pack_inputs, unpack_outputs
    except ImportError:
        from wide_compiler.core.traced_wide import TracedWideModel
        from wide_compiler.core.ensemble_util import pack_inputs, unpack_outputs

    # Resolve config
    cfg = config or get_default_config()

    # Apply overrides
    if compile_model is not None:
        cfg = _with_override(cfg, 'compile', compile_model)
    if compile_mode is not None:
        cfg = _with_override(cfg, 'compile_mode', compile_mode)
    if validate is not None:
        cfg = _with_override(cfg, 'validate', validate)
    if debug is not None:
        cfg = _with_override(cfg, 'debug', debug)

    # Handle single model -> N copies
    if isinstance(models, nn.Module):
        if n is None:
            raise ValueError("n required when passing single model")

        # Create N copies with same architecture, different weights
        template = models
        models = [_clone_model(template) for _ in range(n)]

    if not models:
        raise ValueError("models list is empty")

    n_models = len(models)

    if cfg.debug:
        print(f"[WideCompiler] Compiling {n_models} models")
        print(f"[WideCompiler] Registry: {get_registry()}")

    # Require sample input for tracing
    if sample_input is None:
        raise ValueError(
            "sample_input required for FX tracing. "
            "Provide a sample input tensor with the expected shape."
        )

    # Build Wide model
    wide_model = TracedWideModel.from_models(models, sample_input)

    if cfg.debug:
        print(f"[WideCompiler] Built model:\n{wide_model.summary()}")

    # Validate correctness
    if cfg.validate:
        _validate_model(wide_model, models, sample_input, cfg)

    # Apply torch.compile if requested
    if cfg.compile:
        if cfg.debug:
            print(f"[WideCompiler] Compiling with mode={cfg.compile_mode}")

        wide_model = torch.compile(
            wide_model,
            mode=cfg.compile_mode,
            backend=cfg.compile_backend,
            **cfg.compile_options,
        )

    return wide_model


# =============================================================================
# BUILDER API (alternative fluent interface)
# =============================================================================

class WideBuilder:
    """
    Fluent builder for Wide models.

    Example:
        wide = (WideBuilder(models)
            .with_sample(sample_input)
            .with_config(WideConfig.fast())
            .validate()
            .compile()
            .build())
    """

    def __init__(self, models: Union[nn.Module, List[nn.Module]], n: Optional[int] = None):
        if isinstance(models, nn.Module):
            if n is None:
                raise ValueError("n required when passing single model")
            self._models = [_clone_model(models) for _ in range(n)]
        else:
            self._models = list(models)

        self._sample_input: Optional[Tensor] = None
        self._config = WideConfig.default()

    def with_sample(self, sample_input: Tensor) -> 'WideBuilder':
        """Set sample input for tracing."""
        self._sample_input = sample_input
        return self

    def with_config(self, config: WideConfig) -> 'WideBuilder':
        """Set configuration."""
        self._config = config
        return self

    def validate(self, enable: bool = True) -> 'WideBuilder':
        """Enable/disable validation."""
        self._config = _with_override(self._config, 'validate', enable)
        return self

    def compile(self, mode: str = 'reduce-overhead') -> 'WideBuilder':
        """Enable torch.compile."""
        self._config = _with_override(self._config, 'compile', True)
        self._config = _with_override(self._config, 'compile_mode', mode)
        return self

    def debug(self, enable: bool = True) -> 'WideBuilder':
        """Enable debug output."""
        self._config = _with_override(self._config, 'debug', enable)
        return self

    def register(self, module_type: str, wide_class: type) -> 'WideBuilder':
        """Register custom Wide primitive."""
        get_registry().register(module_type, wide_class)
        return self

    def build(self) -> nn.Module:
        """Build the Wide model."""
        try:
            from .core.traced_wide import TracedWideModel
        except ImportError:
            from wide_compiler.core.traced_wide import TracedWideModel

        if self._sample_input is None:
            raise ValueError("sample_input required - call .with_sample()")

        if self._config.debug:
            print(f"[WideBuilder] Building from {len(self._models)} models")

        wide_model = TracedWideModel.from_models(self._models, self._sample_input)

        if self._config.validate:
            _validate_model(wide_model, self._models, self._sample_input, self._config)

        if self._config.compile:
            wide_model = torch.compile(
                wide_model,
                mode=self._config.compile_mode,
                backend=self._config.compile_backend,
                **self._config.compile_options,
            )

        return wide_model


# =============================================================================
# UTILITIES
# =============================================================================

def _clone_model(model: nn.Module) -> nn.Module:
    """Create a deep copy of a model with new weights."""
    import copy
    cloned = copy.deepcopy(model)
    # Reinitialize parameters for diversity
    for m in cloned.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
    return cloned


def _with_override(config: WideConfig, field: str, value) -> WideConfig:
    """Create new config with field overridden."""
    import dataclasses
    return dataclasses.replace(config, **{field: value})


def _validate_model(
    wide_model: nn.Module,
    models: List[nn.Module],
    sample_input: Tensor,
    config: WideConfig,
) -> None:
    """Validate Wide model produces same outputs as individual models."""
    try:
        from .core.ensemble_util import pack_inputs, unpack_outputs
    except ImportError:
        from wide_compiler.core.ensemble_util import pack_inputs, unpack_outputs

    n = len(models)
    device = sample_input.device

    # Create test inputs
    inputs = [sample_input.clone() for _ in range(n)]
    packed = pack_inputs(inputs)

    # Run both
    wide_model.eval()
    for m in models:
        m.eval()

    with torch.no_grad():
        expected = [models[i](inputs[i]) for i in range(n)]
        wide_out = wide_model(packed)
        actual = unpack_outputs(wide_out, n)

    # Compare
    max_diff = max(
        (expected[i] - actual[i]).abs().max().item()
        for i in range(n)
    )

    if config.debug:
        print(f"[WideCompiler] Validation max_diff: {max_diff:.8f}")

    if max_diff > config.validate_rtol:
        raise ValueError(
            f"Validation failed: max_diff={max_diff:.8f} > rtol={config.validate_rtol}. "
            "Wide model output differs from individual models."
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def pack(inputs: List[Tensor]) -> Tensor:
    """Pack N inputs into wide format. Alias for pack_inputs."""
    try:
        from .core.ensemble_util import pack_inputs
    except ImportError:
        from wide_compiler.core.ensemble_util import pack_inputs
    return pack_inputs(inputs)


def unpack(output: Tensor, n: int) -> List[Tensor]:
    """Unpack wide output to N tensors. Alias for unpack_outputs."""
    try:
        from .core.ensemble_util import unpack_outputs
    except ImportError:
        from wide_compiler.core.ensemble_util import unpack_outputs
    return unpack_outputs(output, n)


# =============================================================================
# BENCHMARK API (convenience re-exports)
# =============================================================================

def benchmark_primitive(
    name: str,
    preset: str = 'full',
    device: str = 'cuda',
    verbose: bool = True,
    **overrides,
):
    """
    Benchmark a primitive's strategy selection.

    Args:
        name: Primitive name ('conv1d', 'conv2d', 'linear', etc.)
        preset: Sweep preset ('quick', 'full', 'ci')
        device: 'cuda' or 'cpu'
        verbose: Print progress
        **overrides: Override sweep params (e.g., n_values=[4,8])

    Returns:
        BenchmarkResult with all measurements

    Example:
        result = benchmark_primitive('conv1d')
        result = benchmark_primitive('conv1d', preset='quick')
        result = benchmark_primitive('conv1d', n_values=[8, 16, 32])
    """
    try:
        from .core.benchmark import benchmark
    except ImportError:
        from wide_compiler.core.benchmark import benchmark
    return benchmark(name, preset=preset, device=device, verbose=verbose, **overrides)


def benchmark_custom(
    model_class,
    input_shape,
    n_values,
    name: str = 'custom',
    device: str = 'cuda',
    **model_kwargs,
):
    """
    Benchmark an arbitrary model class.

    Args:
        model_class: nn.Module subclass
        input_shape: Shape for single model input (without batch)
        n_values: List of N values to test
        name: Name for this benchmark
        device: 'cuda' or 'cpu'
        **model_kwargs: Arguments passed to model_class()

    Returns:
        BenchmarkResult

    Example:
        class Expert(nn.Module):
            def __init__(self, d=256):
                super().__init__()
                self.fc1 = nn.Linear(d, d*4)
                self.fc2 = nn.Linear(d*4, d)
            def forward(self, x):
                return self.fc2(F.gelu(self.fc1(x)))

        result = benchmark_custom(Expert, (256,), [8, 16, 32, 64], d=256)
    """
    try:
        from .core.benchmark import benchmark_custom as _benchmark_custom
    except ImportError:
        from wide_compiler.core.benchmark import benchmark_custom as _benchmark_custom
    return _benchmark_custom(
        model_class, input_shape, n_values,
        name=name, device=device, **model_kwargs
    )


__all__ = [
    # Main API
    'compile',
    'WideBuilder',
    'pack',
    'unpack',

    # Benchmark API
    'benchmark_primitive',
    'benchmark_custom',
]