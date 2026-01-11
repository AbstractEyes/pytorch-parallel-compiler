"""
WideCompiler.core.stage_block
=============================

Atomic execution units for staged model structure.

SequentialBlock: Single module, standard execution
ParallelBlock: Multiple branches, parallel execution

The staged structure itself provides optimization through
organized execution. Compilation is additive.

Copyright 2025 AbstractPhil
MIT License
"""

from __future__ import annotations

from typing import List, Optional, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


# =============================================================================
# METADATA
# =============================================================================

@dataclass
class BlockMetadata:
    """Execution metadata for a block."""
    param_count: int
    type_name: str
    shape_info: str = ""

    def __repr__(self) -> str:
        if self.shape_info:
            return f"{self.type_name}({self.shape_info}, {self.param_count:,} params)"
        return f"{self.type_name}({self.param_count:,} params)"


# =============================================================================
# SEQUENTIAL BLOCK
# =============================================================================

class SequentialBlock(nn.Module):
    """
    Single module execution block.

    Wraps a single nn.Module for sequential execution.
    Provides structured representation for staged models.

    Attributes:
        module: The wrapped computation
        name: Block identifier
    """

    def __init__(self, module: nn.Module, name: str = ""):
        super().__init__()
        self.module = module
        self.name = name or type(module).__name__
        self._metadata: Optional[BlockMetadata] = None

    @property
    def is_parallel(self) -> bool:
        return False

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.module.parameters())

    @property
    def metadata(self) -> BlockMetadata:
        """Lazy-computed metadata."""
        if self._metadata is None:
            self._metadata = BlockMetadata(
                param_count=self.param_count,
                type_name=type(self.module).__name__,
                shape_info=self._get_shape_info(),
            )
        return self._metadata

    def _get_shape_info(self) -> str:
        """Extract shape info from module."""
        m = self.module
        if isinstance(m, nn.Linear):
            return f"{m.in_features}→{m.out_features}"
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return f"{m.in_channels}→{m.out_channels}"
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return f"{m.num_features}"
        if isinstance(m, nn.LayerNorm):
            return f"{list(m.normalized_shape)}"
        return ""

    def forward(self, x: Tensor) -> Tensor:
        """Standard forward execution."""
        return self.module(x)

    def __repr__(self) -> str:
        return f"SequentialBlock({self.name}, {self.metadata})"


# =============================================================================
# PARALLEL BLOCK
# =============================================================================

class ParallelBlock(nn.Module):
    """
    Multi-branch parallel execution block.

    Wraps multiple modules that execute in parallel on the same input.
    Outputs list of tensors - combination is caller's responsibility.

    Provides:
        - forward(): Standard parallel execution
        - wide_forward(): Protected from torch.compile capture

    Attributes:
        branches: The parallel modules
        name: Block identifier
    """

    def __init__(
            self,
            branches: List[nn.Module],
            name: str = "",
            validate: bool = True,
    ):
        super().__init__()

        if len(branches) < 2:
            raise ValueError("ParallelBlock requires at least 2 branches")

        self.branches = nn.ModuleList(branches)
        self.name = name or f"parallel_{len(branches)}"
        self._metadata: Optional[BlockMetadata] = None

        if validate:
            self._validate_branches()

    def _validate_branches(self) -> None:
        """Validate branches have similar structure."""
        signatures = [self._branch_signature(b) for b in self.branches]
        if len(set(signatures)) > 1:
            import warnings
            warnings.warn(
                f"ParallelBlock '{self.name}' has branches with different structures. "
                "This may be intentional, but could indicate a problem.",
                UserWarning
            )

    def _branch_signature(self, module: nn.Module) -> str:
        """Compute structural signature for branch comparison."""
        type_name = type(module).__name__
        param_count = sum(p.numel() for p in module.parameters())
        return f"{type_name}:{param_count}"

    @property
    def is_parallel(self) -> bool:
        return True

    @property
    def num_branches(self) -> int:
        return len(self.branches)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def metadata(self) -> BlockMetadata:
        """Lazy-computed metadata."""
        if self._metadata is None:
            branch_type = type(self.branches[0]).__name__
            self._metadata = BlockMetadata(
                param_count=self.param_count,
                type_name=f"Parallel[{branch_type}]",
                shape_info=f"×{self.num_branches}",
            )
        return self._metadata

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Standard parallel execution.

        Returns list of outputs from each branch.
        Caller handles combination.
        """
        return [branch(x) for branch in self.branches]

    @torch.compiler.disable
    def wide_forward(self, x: Tensor) -> List[Tensor]:
        """
        Protected parallel execution.

        Disabled from torch.compile capture to prevent
        CUDAGraphs buffer issues during training.

        Same semantics as forward().
        """
        return [branch(x) for branch in self.branches]

    def __repr__(self) -> str:
        return f"ParallelBlock({self.name}, {self.num_branches} branches, {self.metadata})"


# =============================================================================
# UTILITIES
# =============================================================================

def make_block(module: nn.Module, name: str = "") -> Union[SequentialBlock, ParallelBlock]:
    """
    Create appropriate block type from module.

    ModuleList/ModuleDict with 2+ children → ParallelBlock
    Everything else → SequentialBlock
    """
    if isinstance(module, nn.ModuleList) and len(module) >= 2:
        return ParallelBlock(list(module), name, validate=True)

    if isinstance(module, nn.ModuleDict) and len(module) >= 2:
        return ParallelBlock(list(module.values()), name, validate=True)

    return SequentialBlock(module, name)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'BlockMetadata',
    'SequentialBlock',
    'ParallelBlock',
    'make_block',
]

# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 50)
    print("stage_block.py tests")
    print("=" * 50)

    # Sequential
    print("\n--- SequentialBlock ---")
    seq = SequentialBlock(nn.Linear(64, 128), name="fc1")
    print(seq)
    print(f"  is_parallel: {seq.is_parallel}")
    print(f"  param_count: {seq.param_count:,}")

    x = torch.randn(4, 64)
    y = seq(x)
    print(f"  forward: {x.shape} → {y.shape}")

    # Parallel
    print("\n--- ParallelBlock ---")
    par = ParallelBlock(
        branches=[nn.Linear(64, 64) for _ in range(4)],
        name="towers"
    )
    print(par)
    print(f"  is_parallel: {par.is_parallel}")
    print(f"  num_branches: {par.num_branches}")
    print(f"  param_count: {par.param_count:,}")

    x = torch.randn(4, 64)
    outputs = par(x)
    print(f"  forward: {x.shape} → {len(outputs)} × {outputs[0].shape}")

    outputs_wide = par.wide_forward(x)
    print(f"  wide_forward: {x.shape} → {len(outputs_wide)} × {outputs_wide[0].shape}")

    # make_block utility
    print("\n--- make_block ---")
    mod_list = nn.ModuleList([nn.Linear(32, 32) for _ in range(3)])
    block = make_block(mod_list, "auto_parallel")
    print(f"  ModuleList → {type(block).__name__}")

    single = nn.Conv2d(3, 64, 3)
    block = make_block(single, "auto_seq")
    print(f"  Conv2d → {type(block).__name__}")

    print("\n" + "=" * 50)
    print("✓ stage_block.py complete")
    print("=" * 50)