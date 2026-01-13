"""
WideCompiler.core.traced_wide

Tracing-based Wide model construction using torch.fx.
Trace forward pass → capture all ops → build Wide model.

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
        WideLinear, WideConv2d, WideConv1d,
        WideBatchNorm2d, WideBatchNorm1d, WideLayerNorm,
        WideEmbedding
    )

    from .wide_model import pack_inputs, unpack_outputs
except ImportError:
    from wide_compiler.core.primitives import (
        WideLinear, WideConv2d, WideConv1d,
        WideBatchNorm2d, WideBatchNorm1d, WideLayerNorm,
        WideEmbedding
    )
    from wide_compiler.core.wide_model import pack_inputs, unpack_outputs


# =============================================================================
# WIDE BUILDERS
# =============================================================================

WIDE_BUILDERS = {
    'Linear': WideLinear.from_modules,
    'Conv2d': WideConv2d.from_modules,
    'Conv1d': WideConv1d.from_modules,
    'BatchNorm2d': WideBatchNorm2d.from_modules,
    'BatchNorm1d': WideBatchNorm1d.from_modules,
    'LayerNorm': WideLayerNorm.from_modules,
    'Embedding': WideEmbedding.from_modules,
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
        """Execute graph respecting dataflow."""
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

        return values[self._output_name]

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
# TESTS - Profiling focused (this is a compiler library)
# =============================================================================

if __name__ == '__main__':
    import time
    import torch._dynamo
    from contextlib import contextmanager

    torch.manual_seed(42)
    torch._dynamo.config.cache_size_limit = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 70)
    print("TracedWideModel - Profiling & Bottleneck Analysis")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 70)

    def benchmark(fn, num_iters=100, warmup=20):
        """Benchmark function, return average time in seconds."""
        for _ in range(warmup):
            fn()
        if device == 'cuda':
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(num_iters):
            fn()
        if device == 'cuda':
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / num_iters

    # =========================================================================
    # TEST 1: Basic correctness
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Basic Correctness (MLP)")
    print("=" * 70)

    class MLP(nn.Module):
        def __init__(self, d=256):
            super().__init__()
            self.fc1 = nn.Linear(d, d*2)
            self.fc2 = nn.Linear(d*2, d)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x

    N, B, D = 50, 32, 256
    mlps = [MLP(D).to(device).eval() for _ in range(N)]
    sample = torch.randn(B, D, device=device)

    wide_mlp = TracedWideModel.from_models(mlps, sample).to(device).eval()

    inputs = [torch.randn(B, D, device=device) for _ in range(N)]
    packed = pack_inputs(inputs)

    with torch.inference_mode():
        ref_outs = [mlps[i](inputs[i]) for i in range(N)]
        wide_out = wide_mlp(packed)
        wide_outs = unpack_outputs(wide_out, N)

    max_diff = max((ref_outs[i] - wide_outs[i]).abs().max().item() for i in range(N))
    print(f"Max diff: {max_diff:.2e} {'✓' if max_diff < 1e-4 else '✗'}")
    print(wide_mlp.summary())

    # =========================================================================
    # TEST 2: Forward pass profiling (per-stage breakdown)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Per-Stage Time Breakdown")
    print("=" * 70)

    # Instrument forward to time each stage
    def profiled_forward(model, x):
        """Forward with per-stage timing."""
        values = {model._input_name: x}
        timings = {}

        # Pre-populate attrs
        for attr_name in model._attr_names:
            safe_name = attr_name.replace('.', '_')
            buffer_name = f'_attr_{safe_name}'
            if hasattr(model, buffer_name):
                values[attr_name] = getattr(model, buffer_name)

        for node_name in model._execution_order:
            if node_name in model._attr_names:
                continue
            if node_name not in model.stage_info:
                continue

            safe_name = node_name.replace('.', '_')
            op = model.stages[safe_name]

            # Gather args
            args = []
            for arg_name in model._node_args[node_name]:
                if isinstance(arg_name, str) and arg_name in values:
                    args.append(values[arg_name])
                else:
                    args.append(arg_name)

            # Gather kwargs
            kwargs = {}
            for k, v in model._node_kwargs.get(node_name, {}).items():
                if isinstance(v, str) and v in values:
                    kwargs[k] = values[v]
                else:
                    kwargs[k] = v

            # Time this stage
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            if kwargs:
                values[node_name] = op(*args, **kwargs)
            elif len(args) == 1:
                values[node_name] = op(args[0])
            else:
                values[node_name] = op(*args)

            if device == 'cuda':
                torch.cuda.synchronize()
            timings[node_name] = (time.perf_counter() - t0) * 1000  # ms

        return values[model._output_name], timings

    # Warmup
    with torch.inference_mode():
        for _ in range(20):
            _ = wide_mlp(packed)

    # Profile multiple runs and average
    all_timings = []
    with torch.inference_mode():
        for _ in range(10):
            _, timings = profiled_forward(wide_mlp, packed)
            all_timings.append(timings)

    # Average timings
    avg_timings = {}
    for key in all_timings[0].keys():
        avg_timings[key] = sum(t[key] for t in all_timings) / len(all_timings)

    print(f"\nConfig: N={N}, B={B}, D={D}")
    print(f"\n{'Stage':<20} {'Op Type':<25} {'Time (ms)':<12} {'%':<8}")
    print("-" * 70)

    total_time = sum(avg_timings.values())
    for node_name, t in sorted(avg_timings.items(), key=lambda x: -x[1]):
        safe_name = node_name.replace('.', '_')
        op = wide_mlp.stages.get(safe_name)
        op_type = type(op).__name__ if op else "?"
        pct = (t / total_time) * 100 if total_time > 0 else 0
        print(f"{node_name:<20} {op_type:<25} {t:<12.4f} {pct:<8.1f}%")

    print("-" * 70)
    print(f"{'TOTAL':<20} {'':<25} {total_time:<12.4f} {'100.0%':<8}")

    # =========================================================================
    # TEST 3: Eager vs Compiled performance
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Eager vs Compiled Performance")
    print("=" * 70)

    configs = [
        (20, 32, 256, "Small MLP"),
        (50, 16, 512, "Medium MLP"),
        (100, 8, 256, "High N MLP"),
    ]

    print(f"\n{'Config':<20} {'N×Model (ms)':<15} {'Wide Eager':<15} {'Wide Compiled':<15} {'Speedup'}")
    print("-" * 85)

    for N, B, D, desc in configs:
        torch._dynamo.reset()
        torch.manual_seed(42)

        mlps = [MLP(D).to(device).eval() for _ in range(N)]
        sample = torch.randn(B, D, device=device)
        wide = TracedWideModel.from_models(mlps, sample).to(device).eval()

        inputs = [torch.randn(B, D, device=device) for _ in range(N)]
        packed = pack_inputs(inputs)

        try:
            wide_compiled = torch.compile(wide, mode='default')
            # Warmup
            with torch.inference_mode():
                for _ in range(10):
                    _ = wide_compiled(packed)
            compile_ok = True
        except Exception as e:
            print(f"{desc:<20} COMPILE FAILED: {str(e)[:40]}")
            continue

        with torch.inference_mode():
            t_baseline = benchmark(lambda: [mlps[i](inputs[i]) for i in range(N)], num_iters=100)
            t_eager = benchmark(lambda: wide(packed), num_iters=100)
            t_compiled = benchmark(lambda: wide_compiled(packed), num_iters=100)

        speedup = t_baseline / t_compiled
        print(f"{desc:<20} {t_baseline*1000:<15.3f} {t_eager*1000:<15.3f} {t_compiled*1000:<15.3f} {speedup:<.2f}x")

    # =========================================================================
    # TEST 4: ResBlock profiling
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 4: ResBlock (Conv2d + BN + Residual)")
    print("=" * 70)

    class ResBlock(nn.Module):
        def __init__(self, channels=64):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            identity = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.relu(out + identity)

    N, B, C, H, W = 20, 8, 64, 32, 32
    resblocks = [ResBlock(C).to(device).eval() for _ in range(N)]
    sample = torch.randn(B, C, H, W, device=device)

    wide_res = TracedWideModel.from_models(resblocks, sample).to(device).eval()

    inputs = [torch.randn(B, C, H, W, device=device) for _ in range(N)]
    packed = pack_inputs(inputs)

    # Correctness
    with torch.inference_mode():
        ref_outs = [resblocks[i](inputs[i]) for i in range(N)]
        wide_out = wide_res(packed)
        wide_outs = unpack_outputs(wide_out, N)

    max_diff = max((ref_outs[i] - wide_outs[i]).abs().max().item() for i in range(N))
    print(f"\nCorrectness: max_diff = {max_diff:.2e} {'✓' if max_diff < 1e-3 else '✗'}")

    # Per-stage breakdown
    with torch.inference_mode():
        for _ in range(10):
            _ = wide_res(packed)

        all_timings = []
        for _ in range(5):
            _, timings = profiled_forward(wide_res, packed)
            all_timings.append(timings)

    avg_timings = {}
    for key in all_timings[0].keys():
        avg_timings[key] = sum(t[key] for t in all_timings) / len(all_timings)

    print(f"\nConfig: N={N}, B={B}, C={C}, H×W={H}×{W}")
    print(f"\n{'Stage':<20} {'Op Type':<25} {'Time (ms)':<12} {'%':<8}")
    print("-" * 70)

    total_time = sum(avg_timings.values())
    for node_name, t in sorted(avg_timings.items(), key=lambda x: -x[1]):
        safe_name = node_name.replace('.', '_')
        op = wide_res.stages.get(safe_name)
        op_type = type(op).__name__ if op else "?"
        pct = (t / total_time) * 100 if total_time > 0 else 0
        print(f"{node_name:<20} {op_type:<25} {t:<12.4f} {pct:<8.1f}%")

    print("-" * 70)
    print(f"{'TOTAL':<20} {'':<25} {total_time:<12.4f} {'100.0%':<8}")

    # Compiled comparison
    print("\nCompiled performance:")
    with torch.inference_mode():
        t_baseline = benchmark(lambda: [resblocks[i](inputs[i]) for i in range(N)], num_iters=50)
        t_eager = benchmark(lambda: wide_res(packed), num_iters=50)

    try:
        torch._dynamo.reset()
        wide_res_compiled = torch.compile(wide_res, mode='default')
        with torch.inference_mode():
            for _ in range(10):
                _ = wide_res_compiled(packed)
            t_compiled = benchmark(lambda: wide_res_compiled(packed), num_iters=50)
    except Exception as e:
        print(f"Compile failed: {e}")
        t_compiled = t_eager

    print(f"\n{'Method':<25} {'Time (ms)':<15} {'Speedup'}")
    print("-" * 55)
    print(f"{'N×ResBlock (baseline)':<25} {t_baseline*1000:<15.3f} {'1.00x'}")
    print(f"{'Wide (eager)':<25} {t_eager*1000:<15.3f} {t_baseline/t_eager:.2f}x")
    print(f"{'Wide (compiled)':<25} {t_compiled*1000:<15.3f} {t_baseline/t_compiled:.2f}x")

    # =========================================================================
    # TEST 5: torch.profiler detailed breakdown
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 5: torch.profiler Detailed Analysis")
    print("=" * 70)

    try:
        from torch.profiler import profile, ProfilerActivity, record_function

        N, B, D = 50, 32, 256
        mlps = [MLP(D).to(device).eval() for _ in range(N)]
        sample = torch.randn(B, D, device=device)
        wide = TracedWideModel.from_models(mlps, sample).to(device).eval()

        inputs = [torch.randn(B, D, device=device) for _ in range(N)]
        packed = pack_inputs(inputs)

        # Warmup
        with torch.inference_mode():
            for _ in range(20):
                _ = wide(packed)

        activities = [ProfilerActivity.CPU]
        if device == 'cuda':
            activities.append(ProfilerActivity.CUDA)

        print(f"\nProfiling Wide MLP: N={N}, B={B}, D={D}")
        print("-" * 70)

        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            with torch.inference_mode():
                for _ in range(10):
                    with record_function("wide_forward"):
                        _ = wide(packed)

        sort_key = "cuda_time_total" if device == 'cuda' else "cpu_time_total"
        print(prof.key_averages().table(sort_by=sort_key, row_limit=15))

    except Exception as e:
        print(f"torch.profiler failed: {e}")

    # =========================================================================
    # TEST 6: Memory profiling
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 6: Memory Analysis")
    print("=" * 70)

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        N, B, D = 50, 32, 512

        # Measure N×Model memory
        mlps = [MLP(D).to(device).eval() for _ in range(N)]
        mem_n_models = torch.cuda.max_memory_allocated() / 1024**2

        # Measure Wide memory
        torch.cuda.reset_peak_memory_stats()
        sample = torch.randn(B, D, device=device)
        wide = TracedWideModel.from_models(mlps, sample).to(device).eval()
        mem_wide = torch.cuda.max_memory_allocated() / 1024**2

        # Forward memory
        inputs = [torch.randn(B, D, device=device) for _ in range(N)]
        packed = pack_inputs(inputs)

        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            _ = wide(packed)
        mem_wide_fwd = torch.cuda.max_memory_allocated() / 1024**2

        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            _ = [mlps[i](inputs[i]) for i in range(N)]
        mem_baseline_fwd = torch.cuda.max_memory_allocated() / 1024**2

        print(f"\nConfig: N={N}, B={B}, D={D}")
        print(f"\n{'Metric':<35} {'Memory (MB)':<15}")
        print("-" * 55)
        print(f"{'N×Model parameters':<35} {mem_n_models:<15.1f}")
        print(f"{'Wide model construction':<35} {mem_wide:<15.1f}")
        print(f"{'Wide forward peak':<35} {mem_wide_fwd:<15.1f}")
        print(f"{'N×Model forward peak':<35} {mem_baseline_fwd:<15.1f}")

        savings = mem_baseline_fwd - mem_wide_fwd
        print(f"\n{'Memory saved (forward)':<35} {savings:<15.1f} ({savings/mem_baseline_fwd*100:.1f}%)")
    else:
        print("Memory profiling requires CUDA")

    # =========================================================================
    # TEST 7: Graph break analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 7: Graph Break Analysis")
    print("=" * 70)

    N, B, D = 20, 32, 256
    mlps = [MLP(D).to(device).eval() for _ in range(N)]
    sample = torch.randn(B, D, device=device)
    wide = TracedWideModel.from_models(mlps, sample).to(device).eval()

    packed = torch.randn(B, N * D, device=device)

    torch._dynamo.reset()

    try:
        explanation = torch._dynamo.explain(wide)(packed)
        print(f"\nGraph breaks: {explanation.graph_break_count}")
        print(f"Graphs compiled: {explanation.graph_count}")

        if explanation.graph_break_count > 0:
            print("\nBreak reasons:")
            for i, reason in enumerate(explanation.break_reasons[:5]):
                print(f"  {i+1}. {reason}")
        else:
            print("✓ No graph breaks - optimal compilation")

    except Exception as e:
        print(f"Graph analysis failed: {e}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("PROFILING SUMMARY")
    print("=" * 70)
    print("""
Bottleneck identification:

1. Per-stage timing shows which Wide ops are slowest
   - WideLinear/WideConv2d should dominate (compute bound)
   - FunctionalOp overhead should be <5%
   
2. Compiled vs Eager
   - Expect 1.5-3x speedup from torch.compile
   - No speedup = possible graph breaks
   
3. Memory analysis
   - Wide should use similar or less memory
   - Peak during forward is key metric

4. Graph breaks
   - 0 breaks = full graph fusion
   - Breaks = find and fix dynamic ops

Next steps if bottlenecked:
- FunctionalOp slow: Fuse into WideFunctional batch op
- Graph breaks: Replace dynamic control with static
- Memory high: Check intermediate tensor accumulation
- WideConv2d slow: Tune grouped vs channels_last
- WideLinear slow: Tune einsum threshold
""")