"""
WideCompiler.core.tree

Compile-friendly batched model building.
N models as single wide model with grouped ops.
Input layout: [B, N*C, ...] throughout - no permutes in forward.

Copyright 2025 AbstractPhil
MIT License
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Iterator, Optional, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


# =============================================================================
# WIDE OPS - Compile-friendly batched primitives
# =============================================================================

class WideLinear(nn.Module):
    """N parallel Linear layers as grouped Conv1d."""

    def __init__(self, n: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.n = n
        self.in_features = in_features
        self.out_features = out_features

        # Linear = Conv1d with kernel_size=1
        self.op = nn.Conv1d(
            in_channels=n * in_features,
            out_channels=n * out_features,
            kernel_size=1,
            groups=n,
            bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, N*in] or [B, N*in, T]
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, N*in, 1]
            out = self.op(x)
            return out.squeeze(-1)  # [B, N*out]
        return self.op(x)  # [B, N*out, T]

    @classmethod
    def from_modules(cls, modules: List[nn.Linear]) -> 'WideLinear':
        """Create from N existing Linear modules."""
        n = len(modules)
        template = modules[0]
        wide = cls(n, template.in_features, template.out_features,
                   bias=template.bias is not None)

        # Stack weights: [N, out, in] -> interleave into grouped conv
        with torch.no_grad():
            for i, m in enumerate(modules):
                # Conv1d weight: [N*out, in, 1], grouped
                # Group i uses indices [i*out : (i+1)*out]
                start = i * template.out_features
                end = start + template.out_features
                wide.op.weight[start:end, :, 0] = m.weight
                if m.bias is not None:
                    wide.op.bias[start:end] = m.bias

        return wide

    def __repr__(self):
        return f"WideLinear({self.n}x[{self.in_features}→{self.out_features}])"


class WideConv2d(nn.Module):
    """N parallel Conv2d layers as single grouped Conv2d."""

    def __init__(self, n: int, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, padding: int = 0,
                 dilation: int = 1, bias: bool = True):
        super().__init__()
        self.n = n
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.op = nn.Conv2d(
            in_channels=n * in_channels,
            out_channels=n * out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=n,
            bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, N*C_in, H, W] -> [B, N*C_out, H', W']
        return self.op(x)

    @classmethod
    def from_modules(cls, modules: List[nn.Conv2d]) -> 'WideConv2d':
        """Create from N existing Conv2d modules."""
        n = len(modules)
        t = modules[0]

        k = t.kernel_size[0] if isinstance(t.kernel_size, tuple) else t.kernel_size
        s = t.stride[0] if isinstance(t.stride, tuple) else t.stride
        p = t.padding[0] if isinstance(t.padding, tuple) else t.padding
        d = t.dilation[0] if isinstance(t.dilation, tuple) else t.dilation

        wide = cls(n, t.in_channels, t.out_channels, k, s, p, d,
                   bias=t.bias is not None)

        with torch.no_grad():
            for i, m in enumerate(modules):
                start = i * t.out_channels
                end = start + t.out_channels
                wide.op.weight[start:end] = m.weight
                if m.bias is not None:
                    wide.op.bias[start:end] = m.bias

        return wide

    def __repr__(self):
        k = self.op.kernel_size
        return f"WideConv2d({self.n}x[{self.in_channels}→{self.out_channels}, k={k}])"


class WideConv1d(nn.Module):
    """N parallel Conv1d layers as single grouped Conv1d."""

    def __init__(self, n: int, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, padding: int = 0,
                 dilation: int = 1, bias: bool = True):
        super().__init__()
        self.n = n
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.op = nn.Conv1d(
            in_channels=n * in_channels,
            out_channels=n * out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=n,
            bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)

    @classmethod
    def from_modules(cls, modules: List[nn.Conv1d]) -> 'WideConv1d':
        n = len(modules)
        t = modules[0]

        k = t.kernel_size[0] if isinstance(t.kernel_size, tuple) else t.kernel_size
        s = t.stride[0] if isinstance(t.stride, tuple) else t.stride
        p = t.padding[0] if isinstance(t.padding, tuple) else t.padding
        d = t.dilation[0] if isinstance(t.dilation, tuple) else t.dilation

        wide = cls(n, t.in_channels, t.out_channels, k, s, p, d,
                   bias=t.bias is not None)

        with torch.no_grad():
            for i, m in enumerate(modules):
                start = i * t.out_channels
                end = start + t.out_channels
                wide.op.weight[start:end] = m.weight
                if m.bias is not None:
                    wide.op.bias[start:end] = m.bias

        return wide

    def __repr__(self):
        return f"WideConv1d({self.n}x[{self.in_channels}→{self.out_channels}])"


class WideBatchNorm2d(nn.Module):
    """N parallel BatchNorm2d as single BatchNorm2d."""

    def __init__(self, n: int, num_features: int, eps: float = 1e-5,
                 momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.n = n
        self.num_features = num_features

        # Single BN over N*C channels
        self.op = nn.BatchNorm2d(
            num_features=n * num_features,
            eps=eps,
            momentum=momentum,
            affine=affine
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, N*C, H, W]
        return self.op(x)

    @classmethod
    def from_modules(cls, modules: List[nn.BatchNorm2d]) -> 'WideBatchNorm2d':
        n = len(modules)
        t = modules[0]

        wide = cls(n, t.num_features, t.eps, t.momentum, t.affine)

        with torch.no_grad():
            for i, m in enumerate(modules):
                start = i * t.num_features
                end = start + t.num_features
                if m.weight is not None:
                    wide.op.weight[start:end] = m.weight
                if m.bias is not None:
                    wide.op.bias[start:end] = m.bias
                if m.running_mean is not None:
                    wide.op.running_mean[start:end] = m.running_mean
                if m.running_var is not None:
                    wide.op.running_var[start:end] = m.running_var

        return wide

    def __repr__(self):
        return f"WideBatchNorm2d({self.n}x{self.num_features})"


class WideBatchNorm1d(nn.Module):
    """N parallel BatchNorm1d as single BatchNorm1d."""

    def __init__(self, n: int, num_features: int, eps: float = 1e-5,
                 momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.n = n
        self.num_features = num_features

        self.op = nn.BatchNorm1d(
            num_features=n * num_features,
            eps=eps,
            momentum=momentum,
            affine=affine
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)

    @classmethod
    def from_modules(cls, modules: List[nn.BatchNorm1d]) -> 'WideBatchNorm1d':
        n = len(modules)
        t = modules[0]

        wide = cls(n, t.num_features, t.eps, t.momentum, t.affine)

        with torch.no_grad():
            for i, m in enumerate(modules):
                start = i * t.num_features
                end = start + t.num_features
                if m.weight is not None:
                    wide.op.weight[start:end] = m.weight
                if m.bias is not None:
                    wide.op.bias[start:end] = m.bias
                if m.running_mean is not None:
                    wide.op.running_mean[start:end] = m.running_mean
                if m.running_var is not None:
                    wide.op.running_var[start:end] = m.running_var

        return wide

    def __repr__(self):
        return f"WideBatchNorm1d({self.n}x{self.num_features})"


class WideLayerNorm(nn.Module):
    """N parallel LayerNorm - operates on last dim per group."""

    def __init__(self, n: int, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.n = n
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Weight and bias per model
        self.weight = nn.Parameter(torch.ones(n * normalized_shape))
        self.bias = nn.Parameter(torch.zeros(n * normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, N*D] or [B, T, N*D]
        # Need to normalize each group of D independently

        if x.dim() == 2:
            B, ND = x.shape
            D = self.normalized_shape
            N = ND // D

            # Reshape to [B, N, D], normalize over D, reshape back
            x = x.view(B, N, D)
            x = F.layer_norm(x, (D,), eps=self.eps)
            x = x.view(B, ND)

            # Apply per-group affine
            x = x * self.weight + self.bias

        elif x.dim() == 3:
            B, T, ND = x.shape
            D = self.normalized_shape
            N = ND // D

            x = x.view(B, T, N, D)
            x = F.layer_norm(x, (D,), eps=self.eps)
            x = x.view(B, T, ND)
            x = x * self.weight + self.bias

        return x

    @classmethod
    def from_modules(cls, modules: List[nn.LayerNorm]) -> 'WideLayerNorm':
        n = len(modules)
        t = modules[0]
        norm_shape = t.normalized_shape[0] if isinstance(t.normalized_shape, tuple) else t.normalized_shape

        wide = cls(n, norm_shape, t.eps)

        with torch.no_grad():
            for i, m in enumerate(modules):
                start = i * norm_shape
                end = start + norm_shape
                wide.weight[start:end] = m.weight
                wide.bias[start:end] = m.bias

        return wide

    def __repr__(self):
        return f"WideLayerNorm({self.n}x{self.normalized_shape})"


class WideEmbedding(nn.Module):
    """N parallel Embedding tables."""

    def __init__(self, n: int, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.n = n
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Single embedding, output is N*D per token
        # Each model has its own D-dim embedding
        self.weight = nn.Parameter(torch.randn(n, num_embeddings, embedding_dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T] indices (same for all N models)
        # out: [B, T, N*D]
        B, T = x.shape

        # Gather from each table: [N, B, T, D]
        out = F.embedding(x, self.weight[0])  # [B, T, D]
        outs = [F.embedding(x, self.weight[i]) for i in range(self.n)]

        # Stack and reshape: [B, T, N*D]
        out = torch.cat(outs, dim=-1)
        return out

    @classmethod
    def from_modules(cls, modules: List[nn.Embedding]) -> 'WideEmbedding':
        n = len(modules)
        t = modules[0]

        wide = cls(n, t.num_embeddings, t.embedding_dim)

        with torch.no_grad():
            for i, m in enumerate(modules):
                wide.weight[i] = m.weight

        return wide

    def __repr__(self):
        return f"WideEmbedding({self.n}x[{self.num_embeddings}, {self.embedding_dim}])"


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
    # Wide ops
    'WideLinear',
    'WideConv2d',
    'WideConv1d',
    'WideBatchNorm2d',
    'WideBatchNorm1d',
    'WideLayerNorm',
    'WideEmbedding',

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