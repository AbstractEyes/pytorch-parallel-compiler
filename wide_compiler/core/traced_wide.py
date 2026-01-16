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

ATTENTION HANDLING:
    WideAttention auto-detects input format and handles both conventions:
    - Direct call: [B, T, N*D] channel-packed → returns Tensor
    - MHA trace: [B, N*T, D] sequence-packed → auto-repacks, returns Tuple

    No wrapper needed - WideAttention handles repacking internally.

FUTURE OPTIMIZATION:
    Yield-tree execution (pre-compiled index-based plan with generator traversal)
    showed ~7% eager speedup in prototyping. The pattern:

        def _yield_exec(self, x):
            v = [None] * self._n_values
            v[0] = x
            for oi, arg_indices, const_args, out in self._plan:
                args = [v[i] if i >= 0 else const_args[j] for j, i in enumerate(arg_indices)]
                v[out] = self._ops[oi](*args)
                yield v[out]

        def forward(self, x):
            for out in self._yield_exec(x): pass
            return out

    Requires careful handling of constant args (e.g. flatten(1)) and method calls.
    Current dict-based approach is stable and compiles identically, so yield is
    deferred until stability across model zoo is achieved.

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.fx as fx
import operator

try:
    from .primitives import (
        WideLinear, WideConv1d, WideConv2d, WideConv3d,
        WideBatchNorm1d, WideBatchNorm2d, WideLayerNorm,
        WideGroupNorm, WideInstanceNorm1d, WideInstanceNorm2d,
        WideEmbedding, WideAttention, WideGRU, WideLSTM
    )

    from .ensemble_util import pack_inputs, unpack_outputs
except ImportError:
    from wide_compiler.core.primitives import (
        WideLinear, WideConv1d, WideConv2d, WideConv3d,
        WideBatchNorm1d, WideBatchNorm2d, WideLayerNorm,
        WideGroupNorm, WideInstanceNorm1d, WideInstanceNorm2d,
        WideEmbedding, WideAttention, WideGRU, WideLSTM
    )
    from wide_compiler.core.ensemble_util import pack_inputs, unpack_outputs


# =============================================================================
# WIDE BUILDERS
# =============================================================================

WIDE_BUILDERS = {
    # Linear
    'Linear': WideLinear.from_modules,
    # Convolutions
    'Conv1d': WideConv1d.from_modules,
    'Conv2d': WideConv2d.from_modules,
    'Conv3d': WideConv3d.from_modules,
    # Normalization
    'BatchNorm1d': WideBatchNorm1d.from_modules,
    'BatchNorm2d': WideBatchNorm2d.from_modules,
    'LayerNorm': WideLayerNorm.from_modules,
    'GroupNorm': WideGroupNorm.from_modules,
    'InstanceNorm1d': WideInstanceNorm1d.from_modules,
    'InstanceNorm2d': WideInstanceNorm2d.from_modules,
    # Embedding
    'Embedding': WideEmbedding.from_modules,
    # Attention
    'MultiheadAttention': WideAttention.from_modules,
    # RNNs
    'GRU': WideGRU.from_modules,
    'LSTM': WideLSTM.from_modules,
}


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
        # Value is passed in during forward
        return value

    def __repr__(self):
        return f"GetAttrOp({self.attr_name})"


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
    num_inputs: int = 1  # How many tensor inputs this op takes


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
        # Scalar - stack to create dim
        return 0
    elif len(shape) == 1:
        # 1D tensor (e.g., running_mean [C]) - concat on dim 0
        return 0
    elif len(shape) >= 2:
        # Higher dims - concat on channel dim (typically 0 for params/buffers)
        return 0
    return 0


class TracedWideModel(nn.Module):
    """
    Wide model built from fx trace.

    Uses torch.fx to trace the model, then builds Wide ops
    for each node. Forward executes the graph respecting dataflow.
    """

    def __init__(self, n: int):
        super().__init__()
        self.n = n
        self.stages: nn.ModuleDict = nn.ModuleDict()
        self.stage_info: Dict[str, WideStage] = {}
        self._graph: Optional[fx.Graph] = None
        self._execution_order: List[str] = []
        self._node_args: Dict[str, Tuple] = {}  # node_name -> arg node names
        self._node_kwargs: Dict[str, Dict] = {}  # node_name -> kwargs with node refs
        self._input_name: str = 'x'
        self._output_name: str = ''
        self._attr_names: List[str] = []  # Track get_attr nodes

    @classmethod
    def from_models(cls, models: List[nn.Module], sample_input: Tensor) -> 'TracedWideModel':
        """
        Build TracedWideModel from N models using fx tracing.

        Args:
            models: List of N identical models
            sample_input: Sample input for tracing (single model input shape)
        """
        n = len(models)
        wide_model = cls(n)

        # Trace template
        template = models[0]
        try:
            traced = fx.symbolic_trace(template)
        except Exception as e:
            raise RuntimeError(f"FX tracing failed: {e}. Model may have dynamic control flow.")

        wide_model._graph = traced.graph

        # First pass: collect get_attr nodes and register as buffers
        for node in traced.graph.nodes:
            if node.op == 'get_attr':
                attr_path = node.target

                # Get attribute from all N models
                attrs = []
                for m in models:
                    try:
                        attr = _get_nested_attr(m, attr_path)
                        if isinstance(attr, Tensor):
                            attrs.append(attr)
                        elif isinstance(attr, nn.Parameter):
                            attrs.append(attr.data)
                        else:
                            # Non-tensor attribute - store from template
                            attrs = None
                            break
                    except AttributeError:
                        attrs = None
                        break

                if attrs is not None and len(attrs) == n:
                    # Concatenate for wide model
                    concat_dim = _infer_concat_dim(attrs)
                    if attrs[0].dim() == 0:
                        # Scalars - stack them
                        wide_attr = torch.stack(attrs)
                    else:
                        wide_attr = torch.cat(attrs, dim=concat_dim)

                    # Register as buffer (non-trainable by default)
                    safe_name = node.name.replace('.', '_')
                    wide_model.register_buffer(f'_attr_{safe_name}', wide_attr)
                    wide_model._attr_names.append(node.name)
                else:
                    # Non-tensor or couldn't get from all models - use template
                    try:
                        attr = _get_nested_attr(template, attr_path)
                        if isinstance(attr, Tensor):
                            safe_name = node.name.replace('.', '_')
                            wide_model.register_buffer(f'_attr_{safe_name}', attr.clone())
                            wide_model._attr_names.append(node.name)
                    except:
                        pass  # Skip non-tensor attrs

        # Second pass: build stages
        for node in traced.graph.nodes:
            wide_op = None

            if node.op == 'placeholder':
                wide_model._input_name = node.name
                continue

            elif node.op == 'output':
                # Output args tell us which node is the final output
                if node.args:
                    wide_model._output_name = node.args[0].name if hasattr(node.args[0], 'name') else str(node.args[0])
                continue

            elif node.op == 'get_attr':
                # Already handled - just mark in execution order
                wide_model._execution_order.append(node.name)
                wide_model._node_args[node.name] = ()  # No inputs
                wide_model._node_kwargs[node.name] = {}
                continue

            elif node.op == 'call_module':
                target_path = node.target
                modules = [m.get_submodule(target_path) for m in models]
                module_type = type(modules[0]).__name__

                if module_type in WIDE_BUILDERS:
                    wide_op = WIDE_BUILDERS[module_type](modules)
                else:
                    wide_op = FunctionalOp(lambda x, m=modules[0]: m(x), f"Passthrough({module_type})")

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
                # Store arg names for graph execution
                arg_names = []
                for arg in node.args:
                    if hasattr(arg, 'name'):
                        arg_names.append(arg.name)
                    else:
                        arg_names.append(arg)  # Constant

                # Store kwargs with node references resolved
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
                )

                safe_name = node.name.replace('.', '_')
                wide_model.stages[safe_name] = wide_op
                wide_model.stage_info[node.name] = stage
                wide_model._execution_order.append(node.name)
                wide_model._node_args[node.name] = tuple(arg_names)
                wide_model._node_kwargs[node.name] = kwarg_refs

        return wide_model

    def forward(self, x: Tensor) -> Tensor:
        """
        Execute graph with N-first internal format.

        Input:  [B, N*C, ...] channel-packed (C is first feature dim)
        Internal: [N, B, C, ...] N-first format
        Output: [B, N*C, ...] channel-packed
        """
        # Unpack: [B, N*C, ...] -> [N, B, C, ...]
        # For images: [B, N*C, H, W] -> [N, B, C, H, W]
        # For sequences: [B, N*D, T] -> [N, B, D, T]
        # For 1D: [B, N*D] -> [N, B, D]

        B = x.shape[0]
        nc = x.shape[1]
        spatial = x.shape[2:]  # Could be (H, W), (T,), or ()

        c = nc // self.n
        x = x.view(B, self.n, c, *spatial)  # [B, N, C, ...]
        x = x.movedim(1, 0)                  # [N, B, C, ...]

        values: Dict[str, Tensor] = {self._input_name: x}

        # Pre-populate get_attr values
        for attr_name in self._attr_names:
            safe_name = attr_name.replace('.', '_')
            buffer_name = f'_attr_{safe_name}'
            if hasattr(self, buffer_name):
                values[attr_name] = getattr(self, buffer_name)

        for node_name in self._execution_order:
            # Skip get_attr nodes - already in values
            if node_name in self._attr_names:
                continue

            if node_name not in self.stage_info:
                continue

            stage = self.stage_info[node_name]
            safe_name = node_name.replace('.', '_')
            op = self.stages[safe_name]

            # Gather positional args
            args = []
            for arg_name in self._node_args[node_name]:
                if isinstance(arg_name, str) and arg_name in values:
                    args.append(values[arg_name])
                else:
                    args.append(arg_name)  # Constant

            # Gather kwargs
            kwargs = {}
            for k, v in self._node_kwargs.get(node_name, {}).items():
                if isinstance(v, str) and v in values:
                    kwargs[k] = values[v]
                else:
                    kwargs[k] = v

            # Execute
            if kwargs:
                values[node_name] = op(*args, **kwargs)
            elif len(args) == 1:
                values[node_name] = op(args[0])
            else:
                values[node_name] = op(*args)

        out = values[self._output_name]

        # Pack: [N, B, C, ...] -> [B, N*C, ...]
        # For images: [N, B, C, H, W] -> [B, N*C, H, W]
        # For sequences: [N, B, D, T] -> [B, N*D, T]
        # For 1D: [N, B, D] -> [B, N*D]

        # Handle tuple outputs (e.g., from attention)
        if isinstance(out, tuple):
            out = out[0]

        N, B = out.shape[0], out.shape[1]
        C = out.shape[2]
        spatial = out.shape[3:]  # Could be (H, W), (T,), or ()

        out = out.movedim(0, 1)                    # [B, N, C, ...]
        out = out.reshape(B, N * C, *spatial)      # [B, N*C, ...]

        return out

    def summary(self) -> str:
        """Print model summary."""
        lines = [
            f"TracedWideModel: {self.n} models",
            "=" * 60,
            f"Stages: {len(self.stages)}",
            f"Attrs: {len(self._attr_names)}",
            "",
        ]

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
            lines.append(
                f"  [{stage.order}] {node_name}({args_str}): {type(op).__name__} ({params:,} params)"
            )

        lines.append("")
        lines.append(f"Total: {total_params:,} params ({total_params // self.n:,} per model)")

        return "\n".join(lines)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'TraceNode',
    'analyze_trace',
    'print_trace',
    'FunctionalOp',
    'BinaryOp',
    'GetAttrOp',
    'WideStage',
    'TracedWideModel',
]



# =============================================================================
# MINIMAL SMOKE TEST
# =============================================================================

if __name__ == '__main__':
    """
    Minimal smoke test. For full benchmarking, use:
        python -m wide_compiler.core.traced_benchmark --model resnet18 --n 10
    
    Or programmatically:
        from wide_compiler.core import benchmark_model
        result = benchmark_model(MyModel, sample_input, n=10)
    """
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

    # Build
    print(f"\nBuilding Wide model: N={N}")
    wide = TracedWideModel.from_models(mlps, sample).to(device).eval()
    print(f"✓ {len(wide.stages)} stages")

    # Verify
    inputs = [torch.randn(B, D, device=device) for _ in range(N)]
    packed = pack_inputs(inputs)

    with torch.inference_mode():
        ref = [mlps[i](inputs[i]) for i in range(N)]
        out = unpack_outputs(wide(packed), N)

    diff = max((ref[i] - out[i]).abs().max().item() for i in range(N))
    print(f"✓ Correctness: {diff:.2e}")

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

    print("\n" + "=" * 50)
    print("For full benchmarking: python -m wide_compiler.core.traced_benchmark")