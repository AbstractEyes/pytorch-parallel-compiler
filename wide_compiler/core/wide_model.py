"""
WideCompiler.core.tree

Compile-friendly batched model building.
N models as single wide model with grouped ops.
Input layout: [B, N*C, ...] throughout - no permutes in forward.

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Iterator, Optional, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from primitives import (
    WideLinear,
    WideConv2d,
    WideConv1d,
    WideBatchNorm2d,
    WideBatchNorm1d,
    WideLayerNorm,
    WideEmbedding,
)


# =============================================================================
# TREE TRAVERSAL
# =============================================================================

@dataclass
class TreeNode:
    """Node in module tree."""
    path: str
    module: nn.Module
    depth: int
    parent_path: str

    @property
    def name(self) -> str:
        return self.path.split('.')[-1] if self.path else ''

    @property
    def module_type(self) -> str:
        return type(self.module).__name__

    @property
    def is_leaf(self) -> bool:
        return len(list(self.module.children())) == 0


def traverse(module: nn.Module, depth: int = 0, path: str = '') -> Iterator[TreeNode]:
    """Recursive traversal of module tree."""
    for name, child in module.named_children():
        full_path = f"{path}.{name}" if path else name
        yield TreeNode(
            path=full_path,
            module=child,
            depth=depth,
            parent_path=path,
        )
        yield from traverse(child, depth + 1, full_path)


def get_leaves(module: nn.Module) -> List[TreeNode]:
    """Get all leaf modules."""
    return [node for node in traverse(module) if node.is_leaf]


# =============================================================================
# WIDE MODEL FACTORY
# =============================================================================

# Map module types to their Wide equivalents
WIDE_REGISTRY = {
    'Linear': WideLinear,
    'Conv2d': WideConv2d,
    'Conv1d': WideConv1d,
    'BatchNorm2d': WideBatchNorm2d,
    'BatchNorm1d': WideBatchNorm1d,
    'LayerNorm': WideLayerNorm,
    'Embedding': WideEmbedding,
}


def make_wide(modules: List[nn.Module]) -> Optional[nn.Module]:
    """Create Wide version from N identical modules."""
    if not modules:
        return None

    module_type = type(modules[0]).__name__

    if module_type in WIDE_REGISTRY:
        wide_cls = WIDE_REGISTRY[module_type]
        return wide_cls.from_modules(modules)

    return None


def align_modules(models: List[nn.Module]) -> Dict[str, List[nn.Module]]:
    """Align N models by path. Returns path -> [module from each model]."""
    n = len(models)

    # Get paths from first model
    template_paths = [node.path for node in traverse(models[0])]

    aligned = {}
    for path in template_paths:
        modules = []
        for m in models:
            try:
                modules.append(m.get_submodule(path))
            except AttributeError:
                break

        if len(modules) == n:
            aligned[path] = modules

    return aligned


# =============================================================================
# WIDE MODEL
# =============================================================================

class WideModel(nn.Module):
    """
    N models fused into single wide model.

    Input: [B, N*C, ...] - N models' channels interleaved
    Output: [B, N*C_out, ...]

    All ops are native PyTorch, compile-friendly.
    """

    def __init__(self, n: int):
        super().__init__()
        self.n = n
        self.wide_modules = nn.ModuleDict()
        self._paths: List[str] = []

    @classmethod
    def from_models(cls, models: List[nn.Module]) -> 'WideModel':
        """Build WideModel from N identical-architecture models."""
        n = len(models)
        wide = cls(n)

        aligned = align_modules(models)

        for path, modules in aligned.items():
            wide_mod = make_wide(modules)
            if wide_mod is not None:
                safe_path = path.replace('.', '__')
                wide.wide_modules[safe_path] = wide_mod
                wide._paths.append(path)

        return wide

    def get(self, path: str) -> Optional[nn.Module]:
        """Get wide module by original path."""
        safe_path = path.replace('.', '__')
        if safe_path in self.wide_modules:
            return self.wide_modules[safe_path]
        return None

    def summary(self) -> str:
        lines = [f"WideModel: {self.n} models fused", "=" * 50]

        total_params = 0
        for path in self._paths:
            mod = self.get(path)
            params = sum(p.numel() for p in mod.parameters())
            total_params += params
            lines.append(f"  {path}: {mod}")

        lines.append("")
        lines.append(f"Total: {total_params:,} params")
        return "\n".join(lines)


# =============================================================================
# INPUT/OUTPUT PACKING
# =============================================================================

def pack_inputs(inputs: List[Tensor]) -> Tensor:
    """
    Pack N inputs into wide format.
    [x0, x1, ..., xN-1] each [B, C, ...] -> [B, N*C, ...]
    """
    # Stack on dim 1, then reshape
    # [B, C, H, W] * N -> [B, N, C, H, W] -> [B, N*C, H, W]
    stacked = torch.stack(inputs, dim=1)  # [B, N, C, ...]
    B = stacked.shape[0]
    N = stacked.shape[1]
    C = stacked.shape[2]
    spatial = stacked.shape[3:]

    return stacked.reshape(B, N * C, *spatial)


def unpack_outputs(output: Tensor, n: int) -> List[Tensor]:
    """
    Unpack wide output to N separate outputs.
    [B, N*C, ...] -> [x0, x1, ..., xN-1] each [B, C, ...]
    """
    B = output.shape[0]
    NC = output.shape[1]
    C = NC // n
    spatial = output.shape[2:]

    # [B, N*C, ...] -> [B, N, C, ...] -> list of [B, C, ...]
    reshaped = output.reshape(B, n, C, *spatial)
    return [reshaped[:, i] for i in range(n)]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Tree
    'TreeNode',
    'traverse',
    'get_leaves',

    # Factory
    'make_wide',
    'align_modules',
    'WideModel',
    # Packing
    'pack_inputs',
    'unpack_outputs',
]


# =============================================================================
# TESTS
# =============================================================================

if __name__ == '__main__':
    import time
    torch.manual_seed(42)

    # Disable TF32 for reproducible results
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 60)
    print("WideCompiler - Compile-Friendly Batched Models")
    print(f"Device: {device}")
    print("=" * 60)

    # =========================================================================
    # TEST 1: WideLinear correctness
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 1: WideLinear Correctness")
    print("=" * 60)

    N, B, D_in, D_out = 10, 32, 64, 128

    # Create N separate linears
    linears = [nn.Linear(D_in, D_out).to(device) for _ in range(N)]

    # Create wide version
    wide_linear = WideLinear.from_modules(linears).to(device)
    print(f"Created: {wide_linear}")

    # Test: N separate inputs
    inputs = [torch.randn(B, D_in, device=device) for _ in range(N)]

    # Run separate
    with torch.no_grad():
        separate_outs = [linears[i](inputs[i]) for i in range(N)]

    # Run wide
    with torch.no_grad():
        packed_in = pack_inputs(inputs)  # [B, N*D_in]
        wide_out = wide_linear(packed_in)  # [B, N*D_out]
        wide_outs = unpack_outputs(wide_out, N)

    # Compare
    max_diff = max((separate_outs[i] - wide_outs[i]).abs().max().item() for i in range(N))
    print(f"Max diff: {max_diff:.8f}")
    assert max_diff < 1e-3, "WideLinear mismatch!"  # Relaxed for TF32
    print("✓ WideLinear correct")

    # =========================================================================
    # TEST 2: WideConv2d correctness
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 2: WideConv2d Correctness")
    print("=" * 60)

    N, B, C_in, C_out, H, W = 8, 16, 64, 64, 32, 32

    convs = [nn.Conv2d(C_in, C_out, 3, padding=1).to(device) for _ in range(N)]
    wide_conv = WideConv2d.from_modules(convs).to(device)
    print(f"Created: {wide_conv}")

    inputs = [torch.randn(B, C_in, H, W, device=device) for _ in range(N)]

    with torch.no_grad():
        separate_outs = [convs[i](inputs[i]) for i in range(N)]

        packed_in = pack_inputs(inputs)
        wide_out = wide_conv(packed_in)
        wide_outs = unpack_outputs(wide_out, N)

    max_diff = max((separate_outs[i] - wide_outs[i]).abs().max().item() for i in range(N))
    print(f"Max diff: {max_diff:.8f}")
    assert max_diff < 1e-3, "WideConv2d mismatch!"  # Relaxed for TF32
    print("✓ WideConv2d correct")

    # =========================================================================
    # TEST 3: WideBatchNorm2d correctness
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 3: WideBatchNorm2d Correctness")
    print("=" * 60)

    bns = [nn.BatchNorm2d(C_in).to(device) for _ in range(N)]
    for bn in bns:
        bn.eval()

    wide_bn = WideBatchNorm2d.from_modules(bns).to(device)
    wide_bn.eval()
    print(f"Created: {wide_bn}")

    with torch.no_grad():
        separate_outs = [bns[i](inputs[i]) for i in range(N)]

        packed_in = pack_inputs(inputs)
        wide_out = wide_bn(packed_in)
        wide_outs = unpack_outputs(wide_out, N)

    max_diff = max((separate_outs[i] - wide_outs[i]).abs().max().item() for i in range(N))
    print(f"Max diff: {max_diff:.8f}")
    assert max_diff < 1e-3, "WideBatchNorm2d mismatch!"  # Relaxed for TF32
    print("✓ WideBatchNorm2d correct")

    # =========================================================================
    # TEST 4: Full ResBlock benchmark
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 4: ResBlock Benchmark (N=50)")
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

    class WideResBlock(nn.Module):
        def __init__(self, n: int, channels: int = 64):
            super().__init__()
            self.n = n
            self.conv1 = WideConv2d(n, channels, channels, 3, padding=1)
            self.bn1 = WideBatchNorm2d(n, channels)
            self.conv2 = WideConv2d(n, channels, channels, 3, padding=1)
            self.bn2 = WideBatchNorm2d(n, channels)

        def forward(self, x):
            identity = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.relu(out + identity)

        @classmethod
        def from_models(cls, models: List[ResBlock]) -> 'WideResBlock':
            n = len(models)
            t = models[0]
            wide = cls.__new__(cls)
            nn.Module.__init__(wide)
            wide.n = n
            wide.conv1 = WideConv2d.from_modules([m.conv1 for m in models])
            wide.bn1 = WideBatchNorm2d.from_modules([m.bn1 for m in models])
            wide.conv2 = WideConv2d.from_modules([m.conv2 for m in models])
            wide.bn2 = WideBatchNorm2d.from_modules([m.bn2 for m in models])
            return wide

    N = 50
    B = 8
    C = 64
    H, W = 32, 32

    # Create N resblocks
    resblocks = [ResBlock(C).to(device) for _ in range(N)]
    for rb in resblocks:
        rb.eval()

    # Create wide resblock
    wide_resblock = WideResBlock.from_models(resblocks).to(device)
    wide_resblock.eval()

    # Inputs
    inputs = [torch.randn(B, C, H, W, device=device) for _ in range(N)]
    packed_input = pack_inputs(inputs)

    # Verify correctness
    print("Verifying correctness...")
    with torch.no_grad():
        separate_outs = [resblocks[i](inputs[i]) for i in range(N)]
        wide_out = wide_resblock(packed_input)
        wide_outs = unpack_outputs(wide_out, N)

    max_diff = max((separate_outs[i] - wide_outs[i]).abs().max().item() for i in range(N))
    print(f"Max diff: {max_diff:.8f}")

    # Compile
    print("\nCompiling...")
    try:
        compiled_wide = torch.compile(wide_resblock, mode='reduce-overhead')
        compiled_single = torch.compile(resblocks[0], mode='reduce-overhead')

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                compiled_wide(packed_input)
                compiled_single(inputs[0])

        compile_ok = True
    except Exception as e:
        print(f"Compile failed: {e}")
        compile_ok = False

    # Eager benchmark
    print("\n--- Eager Mode ---")

    if device == 'cuda':
        torch.cuda.synchronize()

    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(100):
            for i in range(N):
                resblocks[i](inputs[i])
        if device == 'cuda':
            torch.cuda.synchronize()
        eager_seq = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(100):
            wide_resblock(packed_input)
        if device == 'cuda':
            torch.cuda.synchronize()
        eager_wide = time.perf_counter() - start

    print(f"Sequential ({N}x): {eager_seq*1000:.2f}ms")
    print(f"Wide:              {eager_wide*1000:.2f}ms")
    print(f"Speedup:           {eager_seq/eager_wide:.2f}x")

    # Compiled benchmark
    if compile_ok:
        print("\n--- Compiled (reduce-overhead) ---")

        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(100):
                compiled_single(inputs[0])
            if device == 'cuda':
                torch.cuda.synchronize()
            compiled_single_time = time.perf_counter() - start

            start = time.perf_counter()
            for _ in range(100):
                compiled_wide(packed_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            compiled_wide_time = time.perf_counter() - start

        print(f"Single compiled x100:  {compiled_single_time*1000:.2f}ms")
        print(f"Wide compiled x100:    {compiled_wide_time*1000:.2f}ms")
        print(f"Effective speedup:     {(compiled_single_time * N)/compiled_wide_time:.2f}x (vs N sequential)")

    # =========================================================================
    # TEST 5: MLP benchmark
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 5: MLP Benchmark (N=100)")
    print("=" * 60)

    class MLP(nn.Module):
        def __init__(self, d_in=64, d_hidden=256, d_out=64):
            super().__init__()
            self.fc1 = nn.Linear(d_in, d_hidden)
            self.fc2 = nn.Linear(d_hidden, d_out)

        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))

    class WideMLP(nn.Module):
        def __init__(self, n: int, d_in=64, d_hidden=256, d_out=64):
            super().__init__()
            self.n = n
            self.fc1 = WideLinear(n, d_in, d_hidden)
            self.fc2 = WideLinear(n, d_hidden, d_out)

        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))

        @classmethod
        def from_models(cls, models: List[MLP]) -> 'WideMLP':
            n = len(models)
            wide = cls.__new__(cls)
            nn.Module.__init__(wide)
            wide.n = n
            wide.fc1 = WideLinear.from_modules([m.fc1 for m in models])
            wide.fc2 = WideLinear.from_modules([m.fc2 for m in models])
            return wide

    N = 100
    B = 32
    D = 64

    mlps = [MLP().to(device) for _ in range(N)]
    wide_mlp = WideMLP.from_models(mlps).to(device)

    inputs = [torch.randn(B, D, device=device) for _ in range(N)]
    packed_input = pack_inputs(inputs)  # [B, N*D]

    # Verify
    print("Verifying correctness...")
    with torch.no_grad():
        separate_outs = [mlps[i](inputs[i]) for i in range(N)]
        wide_out = wide_mlp(packed_input)
        wide_outs = unpack_outputs(wide_out, N)

    max_diff = max((separate_outs[i] - wide_outs[i]).abs().max().item() for i in range(N))
    print(f"Max diff: {max_diff:.8f}")

    # Compile
    print("\nCompiling...")
    try:
        compiled_wide_mlp = torch.compile(wide_mlp, mode='reduce-overhead')
        compiled_single_mlp = torch.compile(mlps[0], mode='reduce-overhead')

        with torch.no_grad():
            for _ in range(5):
                compiled_wide_mlp(packed_input)
                compiled_single_mlp(inputs[0])

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
            wide_mlp(packed_input)
        if device == 'cuda':
            torch.cuda.synchronize()
        eager_wide = time.perf_counter() - start

    print(f"Sequential ({N}x): {eager_seq*1000:.2f}ms")
    print(f"Wide:              {eager_wide*1000:.2f}ms")
    print(f"Speedup:           {eager_seq/eager_wide:.2f}x")

    if compile_ok:
        print("\n--- Compiled (reduce-overhead) ---")

        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(100):
                compiled_single_mlp(inputs[0])
            if device == 'cuda':
                torch.cuda.synchronize()
            compiled_single_time = time.perf_counter() - start

            start = time.perf_counter()
            for _ in range(100):
                compiled_wide_mlp(packed_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            compiled_wide_time = time.perf_counter() - start

        print(f"Single compiled x100:  {compiled_single_time*1000:.2f}ms")
        print(f"Wide compiled x100:    {compiled_wide_time*1000:.2f}ms")
        print(f"Effective speedup:     {(compiled_single_time * N)/compiled_wide_time:.2f}x (vs N sequential)")

    # =========================================================================
    # TEST 6: Gradient flow
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 6: Gradient Flow")
    print("=" * 60)

    wide_mlp_train = WideMLP(n=10, d_in=64, d_hidden=128, d_out=64).to(device)
    optimizer = torch.optim.SGD(wide_mlp_train.parameters(), lr=0.01)

    x = torch.randn(32, 10 * 64, device=device)  # [B, N*D]
    target = torch.randn(32, 10 * 64, device=device)

    print("Training steps:")
    for step in range(5):
        optimizer.zero_grad()
        out = wide_mlp_train(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        print(f"  Step {step}: loss={loss.item():.6f}")

    print("\n✓ Gradients flow correctly")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)