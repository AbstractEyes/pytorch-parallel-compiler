"""
WideCompiler.core.traced_wide

Tracing-based Wide model construction using torch.fx.
Trace forward pass → capture all ops → build Wide model.

Forward uses dict-based value lookup which compiles cleanly (0 graph breaks).
All string ops, isinstance checks, and per-stage allocations are traced through
by torch.compile without issue.

PACKING CONVENTION:
    TracedWideModel uses channel-packing: input [B, D] becomes [B, N*D].
    All Wide primitives (WideLinear, WideConv2d, etc.) expect this convention.

LIMITATIONS:
    Models with data-dependent reshapes (e.g., `x.view(B, H, W, C)`) will fail
    because the packed tensor has N*D elements, not D. Supported patterns:

    ✓ MLPs (Linear → activation → Linear)
    ✓ CNNs (Conv2d → BatchNorm → ReLU → Pool)
    ✓ Transformers WITHOUT explicit reshapes in forward()

    ✗ ViT with `x.view(B, num_patches, patch_dim)` - patch dim changes with N
    ✗ Any model using `.view()` or `.reshape()` with hardcoded dimensions

    For models with reshapes, use manual Wide construction with einsum (see demos).

CONTROL FLOW HANDLING:
    FX supports static control flow (branches that don't depend on tensor values).
    Use concrete_args to specialize on specific argument values:

        wide = TracedWideModel.from_models(
            models, sample,
            concrete_args={'output_attentions': False}
        )

    Use trace_config to auto-set common HuggingFace flags before tracing:

        wide = TracedWideModel.from_models(
            models, sample,
            trace_config={'use_cache': False, 'return_dict': False}
        )

PASSTHROUGH MODULES:
    Modules without registered Wide primitives fall back to SequentialPassthrough,
    which correctly handles N-first format but provides no fusion speedup.
    Enable verbose=True in from_models() to see warnings about these.

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
import warnings

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.fx as fx
import operator

try:
    from .ensemble_util import pack_inputs, unpack_outputs, iter_n_first, stack_n_first
    from .registry import get_registry
except ImportError:
    try:
        from wide_compiler.core.ensemble_util import pack_inputs, unpack_outputs, iter_n_first, stack_n_first
        from wide_compiler.core.registry import get_registry
    except ImportError:
        # Standalone mode - define minimal helpers
        def pack_inputs(inputs):
            stacked = torch.stack(inputs, dim=1)
            B, N = stacked.shape[:2]
            rest = stacked.shape[2:]
            if len(rest) == 0:
                return stacked.view(B, N)
            C = rest[0]
            spatial = rest[1:]
            return stacked.view(B, N * C, *spatial)

        def unpack_outputs(output, n):
            B = output.shape[0]
            NC = output.shape[1]
            spatial = output.shape[2:]
            C = NC // n
            reshaped = output.view(B, n, C, *spatial)
            return [reshaped[:, i] for i in range(n)]

        def iter_n_first(x):
            for i in range(x.shape[0]):
                yield x[i]

        def stack_n_first(tensors):
            return torch.stack(tensors, dim=0)

        def get_registry():
            return _MinimalRegistry()

        class _MinimalRegistry:
            def get_builder(self, name):
                return None


# =============================================================================
# DEFAULT TRACE CONFIG
# =============================================================================

# Common HuggingFace config flags that create static branches
# Setting these before tracing allows FX to trace through cleanly
DEFAULT_TRACE_CONFIG = {
    'output_attentions': False,
    'output_hidden_states': False,
    'use_cache': False,
    'return_dict': False,
    'torchscript': True,  # Some models have torchscript-friendly paths
}


def _prepare_for_trace(
    model: nn.Module,
    trace_config: Optional[Dict[str, Any]] = None,
    use_defaults: bool = True,
) -> None:
    """
    Prepare model for FX tracing by setting config flags.

    This mutates the model in-place to disable dynamic branches
    that would otherwise prevent tracing.

    Args:
        model: The model to prepare
        trace_config: Custom config overrides
        use_defaults: Whether to apply DEFAULT_TRACE_CONFIG first
    """
    # Build final config
    if use_defaults:
        cfg = {**DEFAULT_TRACE_CONFIG, **(trace_config or {})}
    else:
        cfg = trace_config or {}

    if not cfg:
        return

    # Try model.config (HuggingFace pattern)
    if hasattr(model, 'config'):
        for k, v in cfg.items():
            if hasattr(model.config, k):
                setattr(model.config, k, v)

    # Try direct attributes on model
    for k, v in cfg.items():
        if hasattr(model, k) and not callable(getattr(model, k)):
            try:
                setattr(model, k, v)
            except AttributeError:
                pass  # Some attributes are read-only

    # Force eval mode for consistent tracing
    model.eval()


# =============================================================================
# FX TRACE ANALYSIS
# =============================================================================

@dataclass
class TraceNode:
    """A node from fx trace."""
    order: int
    op: str           # 'call_module', 'call_function', 'call_method', 'placeholder', 'output'
    name: str         # Node name
    target: Any       # Module path or function
    args: Tuple
    kwargs: Dict


def analyze_trace(graph: fx.Graph) -> List[TraceNode]:
    """Extract ordered nodes from fx graph."""
    nodes = []
    for i, node in enumerate(graph.nodes):
        nodes.append(TraceNode(
            order=i,
            op=node.op,
            name=node.name,
            target=node.target,
            args=node.args,
            kwargs=dict(node.kwargs),
        ))
    return nodes


def print_trace(traced: fx.GraphModule) -> str:
    """Pretty print fx trace."""
    lines = ["FX Trace:", "=" * 60]

    for node in traced.graph.nodes:
        if node.op == 'placeholder':
            lines.append(f"  [input] {node.name}")
        elif node.op == 'call_module':
            lines.append(f"  [module] {node.target}")
        elif node.op == 'call_function':
            fn_name = getattr(node.target, '__name__', str(node.target))
            lines.append(f"  [func] {fn_name}")
        elif node.op == 'call_method':
            lines.append(f"  [method] .{node.target}()")
        elif node.op == 'get_attr':
            lines.append(f"  [attr] {node.target}")
        elif node.op == 'output':
            lines.append(f"  [output]")

    return "\n".join(lines)


# =============================================================================
# WIDE OP WRAPPERS
# =============================================================================

class FunctionalOp(nn.Module):
    """Wrapper for functional ops."""

    def __init__(self, fn: Callable, name: str = ""):
        super().__init__()
        self.fn = fn
        self.name = name or getattr(fn, '__name__', 'func')

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __repr__(self):
        return f"FunctionalOp({self.name})"


class BinaryOp(nn.Module):
    """Wrapper for binary ops like add, mul."""

    def __init__(self, op: str):
        super().__init__()
        self.op = op

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        if self.op == 'add':
            return a + b
        elif self.op == 'mul':
            return a * b
        elif self.op == 'sub':
            return a - b
        elif self.op == 'div':
            return a / b
        elif self.op == 'matmul':
            return a @ b
        else:
            raise ValueError(f"Unknown op: {self.op}")

    def __repr__(self):
        return f"BinaryOp({self.op})"


class GetAttrOp(nn.Module):
    """Wrapper for get_attr that returns a stored buffer/param."""

    def __init__(self, name: str):
        super().__init__()
        self.attr_name = name

    def forward(self, value: Tensor) -> Tensor:
        return value

    def __repr__(self):
        return f"GetAttrOp({self.attr_name})"


class SequentialPassthrough(nn.Module):
    """
    Fallback for unregistered modules - correct but unfused.

    When a module type has no registered Wide primitive, we fall back to
    sequential execution. This is slower than fused ops but maintains
    correctness with N-first format.

    Input:  [N, B, C, ...] N-first format
    Output: [N, B, C', ...] N-first format
    """

    def __init__(self, modules: List[nn.Module], name: str = ""):
        super().__init__()
        self.mods = nn.ModuleList(modules)
        self.n = len(modules)
        self.name = name
        self._module_type = type(modules[0]).__name__ if modules else "Unknown"

    def forward(self, x: Tensor) -> Tensor:
        """Execute N modules sequentially on N-first input."""
        outputs = [self.mods[i](xi) for i, xi in enumerate(iter_n_first(x))]
        return stack_n_first(outputs)

    def __repr__(self):
        return f"SequentialPassthrough({self._module_type}, n={self.n})"


# =============================================================================
# TRACE REPORT
# =============================================================================

@dataclass
class TraceReport:
    """Summary of tracing results."""
    n_models: int
    total_stages: int
    fused_stages: int
    passthrough_stages: int
    passthrough_modules: List[Tuple[str, str]]  # (node_name, module_type)
    attr_count: int
    concrete_args_used: Optional[Dict[str, Any]] = None
    trace_config_used: Optional[Dict[str, Any]] = None

    def has_passthroughs(self) -> bool:
        return self.passthrough_stages > 0

    def summary(self) -> str:
        lines = [
            f"TraceReport: {self.n_models} models",
            f"  Stages: {self.total_stages} ({self.fused_stages} fused, {self.passthrough_stages} passthrough)",
            f"  Attrs: {self.attr_count}",
        ]
        if self.concrete_args_used:
            lines.append(f"  concrete_args: {self.concrete_args_used}")
        if self.trace_config_used:
            lines.append(f"  trace_config: {self.trace_config_used}")
        if self.passthrough_modules:
            lines.append(f"  Passthroughs (no fusion):")
            for name, mtype in self.passthrough_modules:
                lines.append(f"    - {name}: {mtype}")
        return "\n".join(lines)

    def warnings(self) -> List[str]:
        """Generate warning messages for passthroughs."""
        if not self.passthrough_modules:
            return []

        msgs = []
        by_type: Dict[str, List[str]] = {}
        for name, mtype in self.passthrough_modules:
            by_type.setdefault(mtype, []).append(name)

        for mtype, names in by_type.items():
            if len(names) == 1:
                msgs.append(f"No Wide primitive for '{mtype}' at '{names[0]}' - using sequential fallback")
            else:
                msgs.append(f"No Wide primitive for '{mtype}' ({len(names)} instances) - using sequential fallback")

        return msgs


# =============================================================================
# TRACED WIDE MODEL
# =============================================================================

@dataclass
class WideStage:
    """A stage in the Wide model."""
    order: int
    name: str
    op_type: str      # 'module', 'function', 'method', 'getattr'
    target: str
    wide_op: nn.Module
    n: int
    num_inputs: int = 1
    is_passthrough: bool = False


def _get_nested_attr(obj: Any, attr_path: str) -> Any:
    """Get nested attribute like 'bn1.running_mean'."""
    parts = attr_path.split('.')
    for part in parts:
        obj = getattr(obj, part)
    return obj


def _infer_concat_dim(tensors: List[Tensor]) -> int:
    """Infer which dimension to concatenate on for wide attrs."""
    if not tensors:
        return 0
    shape = tensors[0].shape
    if len(shape) == 0:
        return 0
    elif len(shape) == 1:
        return 0
    elif len(shape) >= 2:
        return 0
    return 0


class TracedWideModel(nn.Module):
    """
    Wide model built from fx trace.

    Uses torch.fx to trace the model, then builds Wide ops
    for each node. Forward executes the graph respecting dataflow.

    Supports:
        - concrete_args: Specialize on specific argument values for static control flow
        - trace_config: Auto-set model config flags (HuggingFace patterns)
    """

    def __init__(self, n: int):
        super().__init__()
        self.n = n
        self.stages: nn.ModuleDict = nn.ModuleDict()
        self.stage_info: Dict[str, WideStage] = {}
        self._graph: Optional[fx.Graph] = None
        self._execution_order: List[str] = []
        self._node_args: Dict[str, Tuple] = {}
        self._node_kwargs: Dict[str, Dict] = {}
        self._input_name: str = 'x'
        self._output_name: str = ''
        self._attr_names: List[str] = []
        self._trace_report: Optional[TraceReport] = None

    @property
    def trace_report(self) -> Optional[TraceReport]:
        """Get trace report (available after from_models)."""
        return self._trace_report

    @classmethod
    def from_models(
        cls,
        models: List[nn.Module],
        sample_input: Tensor,
        verbose: bool = False,
        warn_passthroughs: bool = True,
        concrete_args: Optional[Dict[str, Any]] = None,
        trace_config: Optional[Dict[str, Any]] = None,
        use_default_trace_config: bool = True,
    ) -> 'TracedWideModel':
        """
        Build TracedWideModel from N models using fx tracing.

        Args:
            models: List of N identical models
            sample_input: Sample input for tracing (single model input shape)
            verbose: Print detailed trace info
            warn_passthroughs: Emit warnings for modules without Wide primitives
            concrete_args: Dict of argument names to concrete values for specializing
                          static control flow. Passed directly to fx.symbolic_trace().
                          Example: {'output_attentions': False, 'mask': None}
            trace_config: Dict of config attributes to set on model before tracing.
                         Useful for HuggingFace models with config-based branches.
                         Example: {'use_cache': False, 'return_dict': False}
            use_default_trace_config: Whether to apply DEFAULT_TRACE_CONFIG before
                                     trace_config. Set False to skip defaults.

        Returns:
            TracedWideModel with fused operations where possible

        Raises:
            RuntimeError: If FX tracing fails (usually due to dynamic control flow)

        Example:
            # Basic usage
            wide = TracedWideModel.from_models(models, sample)

            # With HuggingFace model
            wide = TracedWideModel.from_models(
                models, sample,
                trace_config={'use_cache': False, 'output_attentions': False},
                concrete_args={'attention_mask': None},
            )
        """
        n = len(models)
        wide_model = cls(n)

        template = models[0]

        # Prepare model for tracing (set config flags)
        effective_trace_config = {}
        if use_default_trace_config:
            effective_trace_config.update(DEFAULT_TRACE_CONFIG)
        if trace_config:
            effective_trace_config.update(trace_config)

        if effective_trace_config:
            _prepare_for_trace(template, effective_trace_config, use_defaults=False)
            # Also prepare other models for consistency
            for m in models[1:]:
                _prepare_for_trace(m, effective_trace_config, use_defaults=False)

        if verbose:
            if effective_trace_config:
                print(f"Applied trace_config: {effective_trace_config}")
            if concrete_args:
                print(f"Using concrete_args: {concrete_args}")

        # Trace with optional concrete_args
        try:
            traced = fx.symbolic_trace(template, concrete_args=concrete_args)
        except Exception as e:
            error_msg = str(e)
            hint = ""
            if "control flow" in error_msg.lower():
                hint = (
                    "\n\nHint: This model has data-dependent control flow. Try:\n"
                    "  1. Set trace_config to disable dynamic features:\n"
                    "     trace_config={'use_cache': False, 'output_attentions': False}\n"
                    "  2. Use concrete_args to specialize on specific values:\n"
                    "     concrete_args={'attention_mask': None}\n"
                    "  3. For HuggingFace models, check model.config for relevant flags."
                )
            raise RuntimeError(f"FX tracing failed: {e}{hint}")

        wide_model._graph = traced.graph

        if verbose:
            print(print_trace(traced))
            print()

        # Track passthroughs for reporting
        passthrough_modules: List[Tuple[str, str]] = []
        fused_count = 0
        passthrough_count = 0

        # First pass: collect get_attr nodes and register as buffers
        for node in traced.graph.nodes:
            if node.op == 'get_attr':
                attr_path = node.target

                attrs = []
                for m in models:
                    try:
                        attr = _get_nested_attr(m, attr_path)
                        if isinstance(attr, Tensor):
                            attrs.append(attr)
                        elif isinstance(attr, nn.Parameter):
                            attrs.append(attr.data)
                        else:
                            attrs = None
                            break
                    except AttributeError:
                        attrs = None
                        break

                if attrs is not None and len(attrs) == n:
                    concat_dim = _infer_concat_dim(attrs)
                    if attrs[0].dim() == 0:
                        wide_attr = torch.stack(attrs)
                    else:
                        wide_attr = torch.cat(attrs, dim=concat_dim)

                    safe_name = node.name.replace('.', '_')
                    wide_model.register_buffer(f'_attr_{safe_name}', wide_attr)
                    wide_model._attr_names.append(node.name)
                else:
                    try:
                        attr = _get_nested_attr(template, attr_path)
                        if isinstance(attr, Tensor):
                            safe_name = node.name.replace('.', '_')
                            wide_model.register_buffer(f'_attr_{safe_name}', attr.clone())
                            wide_model._attr_names.append(node.name)
                    except:
                        pass

        # Second pass: build stages
        for node in traced.graph.nodes:
            wide_op = None
            is_passthrough = False

            if node.op == 'placeholder':
                wide_model._input_name = node.name
                continue

            elif node.op == 'output':
                if node.args:
                    wide_model._output_name = node.args[0].name if hasattr(node.args[0], 'name') else str(node.args[0])
                continue

            elif node.op == 'get_attr':
                wide_model._execution_order.append(node.name)
                wide_model._node_args[node.name] = ()
                wide_model._node_kwargs[node.name] = {}
                continue

            elif node.op == 'call_module':
                target_path = node.target
                modules_list = [m.get_submodule(target_path) for m in models]
                module_type = type(modules_list[0]).__name__

                # Use registry to get builder
                registry = get_registry()
                builder = registry.get_builder(module_type)
                if builder is not None:
                    wide_op = builder(modules_list)
                    fused_count += 1
                    if verbose:
                        print(f"  ✓ Fused: {node.name} ({module_type}) → {type(wide_op).__name__}")
                else:
                    # Fallback to sequential passthrough
                    wide_op = SequentialPassthrough(modules_list, module_type)
                    is_passthrough = True
                    passthrough_count += 1
                    passthrough_modules.append((node.name, module_type))
                    if verbose:
                        print(f"  ⚠ Passthrough: {node.name} ({module_type})")

            elif node.op == 'call_function':
                fn = node.target
                fn_name = getattr(fn, '__name__', str(fn))

                if fn is operator.add or fn is torch.add:
                    wide_op = BinaryOp('add')
                elif fn is operator.mul or fn is torch.mul:
                    wide_op = BinaryOp('mul')
                elif fn is operator.sub:
                    wide_op = BinaryOp('sub')
                elif fn is operator.truediv:
                    wide_op = BinaryOp('div')
                elif fn is torch.matmul or fn is operator.matmul:
                    wide_op = BinaryOp('matmul')
                else:
                    wide_op = FunctionalOp(fn, fn_name)

            elif node.op == 'call_method':
                method_name = node.target
                def make_method_caller(method: str):
                    def caller(x, *args, **kwargs):
                        return getattr(x, method)(*args, **kwargs)
                    return caller
                wide_op = FunctionalOp(make_method_caller(method_name), f".{method_name}()")

            if wide_op is not None:
                arg_names = []
                for arg in node.args:
                    if hasattr(arg, 'name'):
                        arg_names.append(arg.name)
                    else:
                        arg_names.append(arg)

                kwarg_refs = {}
                for k, v in node.kwargs.items():
                    if hasattr(v, 'name'):
                        kwarg_refs[k] = v.name
                    else:
                        kwarg_refs[k] = v

                stage = WideStage(
                    order=len(wide_model._execution_order),
                    name=node.name,
                    op_type=node.op,
                    target=str(node.target),
                    wide_op=wide_op,
                    n=n,
                    num_inputs=len(arg_names),
                    is_passthrough=is_passthrough,
                )

                safe_name = node.name.replace('.', '_')
                wide_model.stages[safe_name] = wide_op
                wide_model.stage_info[node.name] = stage
                wide_model._execution_order.append(node.name)
                wide_model._node_args[node.name] = tuple(arg_names)
                wide_model._node_kwargs[node.name] = kwarg_refs

        # Build trace report
        wide_model._trace_report = TraceReport(
            n_models=n,
            total_stages=fused_count + passthrough_count,
            fused_stages=fused_count,
            passthrough_stages=passthrough_count,
            passthrough_modules=passthrough_modules,
            attr_count=len(wide_model._attr_names),
            concrete_args_used=concrete_args,
            trace_config_used=effective_trace_config if effective_trace_config else None,
        )

        # Emit warnings
        if warn_passthroughs and passthrough_modules:
            for msg in wide_model._trace_report.warnings():
                warnings.warn(msg, stacklevel=2)

        if verbose:
            print()
            print(wide_model._trace_report.summary())

        return wide_model

    def forward(self, x: Tensor) -> Tensor:
        """
        Execute graph with N-first internal format.

        Input:  [B, N*C, ...] channel-packed (C is first feature dim)
        Internal: [N, B, C, ...] N-first format
        Output: [B, N*C, ...] channel-packed
        """
        B = x.shape[0]
        nc = x.shape[1]
        spatial = x.shape[2:]

        c = nc // self.n
        x = x.view(B, self.n, c, *spatial)
        x = x.movedim(1, 0)

        values: Dict[str, Tensor] = {self._input_name: x}

        # Pre-populate get_attr values
        for attr_name in self._attr_names:
            safe_name = attr_name.replace('.', '_')
            buffer_name = f'_attr_{safe_name}'
            if hasattr(self, buffer_name):
                values[attr_name] = getattr(self, buffer_name)

        for node_name in self._execution_order:
            if node_name in self._attr_names:
                continue

            if node_name not in self.stage_info:
                continue

            stage = self.stage_info[node_name]
            safe_name = node_name.replace('.', '_')
            op = self.stages[safe_name]

            args = []
            for arg_name in self._node_args[node_name]:
                if isinstance(arg_name, str) and arg_name in values:
                    args.append(values[arg_name])
                else:
                    args.append(arg_name)

            kwargs = {}
            for k, v in self._node_kwargs.get(node_name, {}).items():
                if isinstance(v, str) and v in values:
                    kwargs[k] = values[v]
                else:
                    kwargs[k] = v

            if kwargs:
                values[node_name] = op(*args, **kwargs)
            elif len(args) == 1:
                values[node_name] = op(args[0])
            else:
                values[node_name] = op(*args)

        out = values[self._output_name]

        # Handle tuple outputs
        if isinstance(out, tuple):
            out = out[0]

        N, B = out.shape[0], out.shape[1]
        C = out.shape[2]
        spatial = out.shape[3:]

        out = out.movedim(0, 1)
        out = out.reshape(B, N * C, *spatial)

        return out

    def summary(self) -> str:
        """Print model summary."""
        lines = [
            f"TracedWideModel: {self.n} models",
            "=" * 60,
        ]

        if self._trace_report:
            lines.append(f"Stages: {self._trace_report.total_stages} "
                        f"({self._trace_report.fused_stages} fused, "
                        f"{self._trace_report.passthrough_stages} passthrough)")
            if self._trace_report.concrete_args_used:
                lines.append(f"concrete_args: {self._trace_report.concrete_args_used}")
            if self._trace_report.trace_config_used:
                lines.append(f"trace_config: {self._trace_report.trace_config_used}")
        else:
            lines.append(f"Stages: {len(self.stages)}")

        lines.append(f"Attrs: {len(self._attr_names)}")
        lines.append("")

        total_params = 0
        for node_name in self._execution_order:
            if node_name in self._attr_names:
                safe_name = node_name.replace('.', '_')
                buffer_name = f'_attr_{safe_name}'
                if hasattr(self, buffer_name):
                    buf = getattr(self, buffer_name)
                    lines.append(f"  [attr] {node_name}: {list(buf.shape)}")
                continue

            if node_name not in self.stage_info:
                continue

            stage = self.stage_info[node_name]
            safe_name = node_name.replace('.', '_')
            op = self.stages[safe_name]
            params = sum(p.numel() for p in op.parameters())
            total_params += params

            args_str = ", ".join(str(a) for a in self._node_args[node_name])

            marker = "⚠" if stage.is_passthrough else "✓"
            lines.append(
                f"  [{stage.order}] {marker} {node_name}({args_str}): {type(op).__name__} ({params:,} params)"
            )

        lines.append("")
        lines.append(f"Total: {total_params:,} params ({total_params // self.n:,} per model)")

        if self._trace_report and self._trace_report.has_passthroughs():
            lines.append("")
            lines.append("⚠ Passthrough modules (no fusion):")
            for name, mtype in self._trace_report.passthrough_modules:
                lines.append(f"    {name}: {mtype}")

        return "\n".join(lines)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Trace utilities
    'TraceNode',
    'analyze_trace',
    'print_trace',
    # Config
    'DEFAULT_TRACE_CONFIG',
    # Op wrappers
    'FunctionalOp',
    'BinaryOp',
    'GetAttrOp',
    'SequentialPassthrough',
    # Reporting
    'TraceReport',
    'WideStage',
    # Main class
    'TracedWideModel',
]


# =============================================================================
# MINIMAL SMOKE TEST
# =============================================================================

if __name__ == '__main__':
    import torch.nn.functional as F

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("TracedWideModel - Smoke Test")
    print(f"Device: {device}")
    print("=" * 50)

    # Simple MLP
    class MLP(nn.Module):
        def __init__(self, d=128):
            super().__init__()
            self.fc1 = nn.Linear(d, d*2)
            self.fc2 = nn.Linear(d*2, d)
        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))

    N, B, D = 10, 16, 128
    mlps = [MLP(D).to(device).eval() for _ in range(N)]
    sample = torch.randn(B, D, device=device)

    # Build with verbose output
    print(f"\nBuilding Wide model: N={N}")
    wide = TracedWideModel.from_models(mlps, sample, verbose=True).to(device).eval()

    # Verify
    inputs = [torch.randn(B, D, device=device) for _ in range(N)]
    packed = pack_inputs(inputs)

    with torch.inference_mode():
        ref = [mlps[i](inputs[i]) for i in range(N)]
        out = unpack_outputs(wide(packed), N)

    diff = max((ref[i] - out[i]).abs().max().item() for i in range(N))
    print(f"\n✓ Correctness: {diff:.2e}")

    # Quick timing
    import time
    with torch.inference_mode():
        for _ in range(10):
            _ = wide(packed)
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            _ = wide(packed)
        if device == 'cuda':
            torch.cuda.synchronize()
        t_wide = (time.perf_counter() - t0) / 50

        for _ in range(10):
            _ = [mlps[i](inputs[i]) for i in range(N)]
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            _ = [mlps[i](inputs[i]) for i in range(N)]
        if device == 'cuda':
            torch.cuda.synchronize()
        t_base = (time.perf_counter() - t0) / 50

    print(f"✓ Speedup: {t_base/t_wide:.1f}x ({t_base*1000:.2f} vs {t_wide*1000:.2f} ms)")

    print("\n" + wide.summary())

    # Test with static control flow
    print("\n" + "=" * 50)
    print("Testing static control flow with concrete_args...")

    class ConditionalMLP(nn.Module):
        def __init__(self, d=128):
            super().__init__()
            self.fc1 = nn.Linear(d, d*2)
            self.fc2 = nn.Linear(d*2, d)
            self.use_relu = True

        def forward(self, x, use_activation=True):
            x = self.fc1(x)
            if use_activation:  # Static control flow
                x = F.relu(x)
            return self.fc2(x)

    cond_mlps = [ConditionalMLP(D).to(device).eval() for _ in range(N)]

    # Trace with concrete_args to specialize on use_activation=True
    wide_cond = TracedWideModel.from_models(
        cond_mlps, sample,
        concrete_args={'use_activation': True},
        verbose=True,
    ).to(device).eval()

    print(f"✓ Conditional model traced successfully")

    print("\n" + "=" * 50)
    print("For full benchmarking: python -m wide_compiler.core.traced_benchmark")