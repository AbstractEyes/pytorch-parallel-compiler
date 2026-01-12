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
        self._input_name: str = 'x'
        self._output_name: str = ''

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

        # Build stages from graph
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

            elif node.op == 'call_module':
                target_path = node.target
                modules = [m.get_submodule(target_path) for m in models]
                module_type = type(modules[0]).__name__

                if module_type in WIDE_BUILDERS:
                    wide_op = WIDE_BUILDERS[module_type](modules)
                else:
                    wide_op = FunctionalOp(modules[0], f"Passthrough({module_type})")

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

            elif node.op == 'get_attr':
                continue

            if wide_op is not None:
                # Store arg names for graph execution
                arg_names = []
                for arg in node.args:
                    if hasattr(arg, 'name'):
                        arg_names.append(arg.name)
                    else:
                        arg_names.append(arg)  # Constant

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

        return wide_model

    def forward(self, x: Tensor) -> Tensor:
        """Execute graph respecting dataflow."""
        values: Dict[str, Tensor] = {self._input_name: x}

        for node_name in self._execution_order:
            stage = self.stage_info[node_name]
            safe_name = node_name.replace('.', '_')
            op = self.stages[safe_name]

            # Gather inputs
            args = []
            for arg_name in self._node_args[node_name]:
                if isinstance(arg_name, str) and arg_name in values:
                    args.append(values[arg_name])
                else:
                    args.append(arg_name)  # Constant

            # Execute
            if len(args) == 1:
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
            "",
        ]

        total_params = 0
        for node_name in self._execution_order:
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


# Need operator for fx traced binary ops
import operator


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'TraceNode',
    'analyze_trace',
    'print_trace',
    'FunctionalOp',
    'BinaryOp',
    'WideStage',
    'TracedWideModel',
]


# =============================================================================
# TESTS
# =============================================================================

if __name__ == '__main__':
    import time
    torch.manual_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 60)
    print("TracedWideModel - FX Tracing-Based Wide Construction")
    print(f"Device: {device}")
    print("=" * 60)

    # =========================================================================
    # TEST 1: Basic FX tracing
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 1: FX Tracing (MLP with F.relu)")
    print("=" * 60)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, 64)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x

    mlp = MLP()
    traced = fx.symbolic_trace(mlp)
    print(print_trace(traced))
    try:
        print("\nGraph:")
        traced.graph.print_tabular()
    except ImportError:
        print("(Install 'tabulate' for detailed graph view)")

    # =========================================================================
    # TEST 2: TracedWideModel construction
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 2: TracedWideModel Construction")
    print("=" * 60)

    N = 50
    mlps = [MLP().to(device) for _ in range(N)]

    sample = torch.randn(4, 64, device=device)
    wide_mlp = TracedWideModel.from_models(mlps, sample).to(device)

    print(wide_mlp.summary())

    # =========================================================================
    # TEST 3: Correctness verification
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 3: Correctness Verification")
    print("=" * 60)

    B = 32
    inputs = [torch.randn(B, 64, device=device) for _ in range(N)]
    packed = pack_inputs(inputs)

    with torch.no_grad():
        separate_outs = [mlps[i](inputs[i]) for i in range(N)]
        wide_out = wide_mlp(packed)
        wide_outs = unpack_outputs(wide_out, N)

    max_diff = max((separate_outs[i] - wide_outs[i]).abs().max().item() for i in range(N))
    print(f"Max diff: {max_diff:.8f}")

    # =========================================================================
    # TEST 4: ResBlock with residual - full test
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 4: ResBlock with Residual (graph-based forward)")
    print("=" * 60)

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

    resblock = ResBlock()
    traced = fx.symbolic_trace(resblock)
    print(print_trace(traced))
    try:
        print("\nGraph:")
        traced.graph.print_tabular()
    except ImportError:
        print("(Install 'tabulate' for detailed graph view)")

    # Build wide resblock
    N = 20
    C = 64
    resblocks = [ResBlock().to(device) for _ in range(N)]
    for rb in resblocks:
        rb.eval()

    sample = torch.randn(4, C, 16, 16, device=device)
    wide_resblock = TracedWideModel.from_models(resblocks, sample).to(device)
    wide_resblock.eval()

    print("\n" + wide_resblock.summary())

    # Verify correctness
    print("\nVerifying correctness...")
    B = 8
    inputs = [torch.randn(B, C, 16, 16, device=device) for _ in range(N)]
    packed = pack_inputs(inputs)

    with torch.no_grad():
        separate_outs = [resblocks[i](inputs[i]) for i in range(N)]
        wide_out = wide_resblock(packed)
        wide_outs = unpack_outputs(wide_out, N)

    max_diff = max((separate_outs[i] - wide_outs[i]).abs().max().item() for i in range(N))
    print(f"Max diff: {max_diff:.8f}")

    if max_diff < 1e-4:
        print("✓ ResBlock with residual works correctly!")
    else:
        print("✗ ResBlock mismatch - need to debug")

    # =========================================================================
    # TEST 5: Benchmark
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 5: Benchmark (N=100)")
    print("=" * 60)

    N = 100
    mlps = [MLP().to(device) for _ in range(N)]
    wide_mlp = TracedWideModel.from_models(
        mlps,
        torch.randn(4, 64, device=device)
    ).to(device)

    inputs = [torch.randn(B, 64, device=device) for _ in range(N)]
    packed = pack_inputs(inputs)

    # Compile
    print("Compiling...")
    try:
        compiled_wide = torch.compile(wide_mlp, mode='reduce-overhead')
        compiled_single = torch.compile(mlps[0], mode='reduce-overhead')

        with torch.no_grad():
            for _ in range(5):
                compiled_wide(packed)
                compiled_single(inputs[0])

        compile_ok = True
    except Exception as e:
        print(f"Compile failed: {e}")
        compile_ok = False

    # Eager
    print("\n--- Eager Mode ---")

    if device == 'cuda':
        torch.cuda.synchronize()

    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(100):
            for i in range(N):
                mlps[i](inputs[i])
        if device == 'cuda':
            torch.cuda.synchronize()
        eager_seq = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(100):
            wide_mlp(packed)
        if device == 'cuda':
            torch.cuda.synchronize()
        eager_wide = time.perf_counter() - start

    print(f"Sequential ({N}x): {eager_seq*1000:.2f}ms")
    print(f"Wide:              {eager_wide*1000:.2f}ms")
    print(f"Speedup:           {eager_seq/eager_wide:.2f}x")

    if compile_ok:
        print("\n--- Compiled ---")

        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(100):
                compiled_single(inputs[0])
            if device == 'cuda':
                torch.cuda.synchronize()
            compiled_single_time = time.perf_counter() - start

            start = time.perf_counter()
            for _ in range(100):
                compiled_wide(packed)
            if device == 'cuda':
                torch.cuda.synchronize()
            compiled_wide_time = time.perf_counter() - start

        print(f"Single compiled x100:  {compiled_single_time*1000:.2f}ms")
        print(f"Wide compiled x100:    {compiled_wide_time*1000:.2f}ms")
        print(f"Effective speedup:     {(compiled_single_time * N)/compiled_wide_time:.2f}x")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)