"""
WideCompiler.core.stage_layer
=============================

Atomic compilation unit for staged model structure.

StageLayer is the fundamental abstraction:
- Executes blocks in sequence
- Everything inside compiles as one graph
- Graph breaks occur BETWEEN layers, not within
- Parallel blocks use wide_forward but stay in graph

This is the minimal, abstract form. Metrics and tracking
are added by wrapper layers when needed.

Copyright 2025 AbstractPhil
MIT License
"""

from __future__ import annotations

from typing import List, Union

import torch.nn as nn
from torch import Tensor

try:
    from .stage_block import Block, SequentialBlock, ParallelBlock
except ImportError:
    from stage_block import Block, SequentialBlock, ParallelBlock


# =============================================================================
# STAGE LAYER
# =============================================================================

class StageLayer(nn.Module):
    """
    Atomic compilation unit.

    Contains blocks that execute as a unified graph.
    Graph breaks occur between layers, not within.

    Assumes linear flow: single tensor in, single tensor out.
    Each block receives output of previous block.

    For DAG topologies (multiple inputs/outputs), wiring
    is handled at the model level.

    Attributes:
        blocks: Ordered computation blocks
        name: Layer identifier
    """

    def __init__(self, name: str = ""):
        super().__init__()
        self.blocks: nn.ModuleList = nn.ModuleList()
        self.name = name or "layer"
        self._parallel_indices: List[int] = []

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def has_parallel(self) -> bool:
        return len(self._parallel_indices) > 0

    @property
    def num_parallel(self) -> int:
        return len(self._parallel_indices)

    @property
    def parallel_indices(self) -> List[int]:
        return self._parallel_indices.copy()

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # =========================================================================
    # BUILDING
    # =========================================================================

    def add(self, block: Block) -> 'StageLayer':
        """Add block to layer."""
        idx = len(self.blocks)
        self.blocks.append(block)
        if block.is_parallel:
            self._parallel_indices.append(idx)
        return self

    def add_sequential(self, module: nn.Module, name: str = "") -> 'StageLayer':
        """Add sequential block."""
        return self.add(SequentialBlock(module, name))

    def add_parallel(
        self,
        branches: List[nn.Module],
        combine,
        name: str = "",
    ) -> 'StageLayer':
        """Add parallel block."""
        return self.add(ParallelBlock(branches, combine, name))

    # =========================================================================
    # EXECUTION
    # =========================================================================

    def forward(self, x: Tensor) -> Tensor:
        """
        Standard forward execution.

        Executes all blocks in sequence.
        Unified graph - no breaks within.
        """
        for block in self.blocks:
            x = block(x)
        return x

    def wide_forward(self, x: Tensor) -> Tensor:
        """
        Protected forward execution.

        Uses wide_forward on all blocks.
        Parallel blocks are protected from CUDAGraphs capture.
        Still unified graph - no breaks within layer.
        """
        for block in self.blocks:
            x = block.wide_forward(x)
        return x

    # =========================================================================
    # REPRESENTATION
    # =========================================================================

    def __repr__(self) -> str:
        parallel_str = f", {self.num_parallel} parallel" if self.has_parallel else ""
        return f"StageLayer({self.name}, {self.num_blocks} blocks{parallel_str})"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'StageLayer',
]


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    import torch

    print("=" * 50)
    print("stage_layer.py tests")
    print("=" * 50)

    # Build a layer
    print("\n--- Building StageLayer ---")
    layer = StageLayer("encoder")
    layer.add_sequential(nn.Linear(64, 128), "fc1")
    layer.add_sequential(nn.ReLU(), "act1")
    layer.add_parallel(
        branches=[nn.Linear(128, 128) for _ in range(4)],
        combine=lambda outs: sum(outs) / len(outs),
        name="towers"
    )
    layer.add_sequential(nn.Linear(128, 64), "fc2")

    print(layer)
    print(f"  num_blocks: {layer.num_blocks}")
    print(f"  has_parallel: {layer.has_parallel}")
    print(f"  parallel_indices: {layer.parallel_indices}")
    print(f"  param_count: {layer.param_count:,}")

    # Forward
    print("\n--- Forward ---")
    x = torch.randn(4, 64)
    y = layer(x)
    print(f"  forward: {x.shape} → {y.shape}")

    y_wide = layer.wide_forward(x)
    print(f"  wide_forward: {x.shape} → {y_wide.shape}")

    # Verify outputs match
    print(f"  outputs match: {torch.allclose(y, y_wide, atol=1e-6)}")

    # Pure sequential layer
    print("\n--- Sequential Only Layer ---")
    seq_layer = StageLayer("mlp")
    seq_layer.add_sequential(nn.Linear(32, 64))
    seq_layer.add_sequential(nn.ReLU())
    seq_layer.add_sequential(nn.Linear(64, 32))

    print(seq_layer)
    print(f"  has_parallel: {seq_layer.has_parallel}")

    x = torch.randn(4, 32)
    y = seq_layer(x)
    print(f"  forward: {x.shape} → {y.shape}")

    print("\n" + "=" * 50)
    print("✓ stage_layer.py complete")
    print("✓ Atomic compilation unit")
    print("✓ Minimal, abstract, reusable")
    print("=" * 50)