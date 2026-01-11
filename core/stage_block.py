"""
WideCompiler.core.stage_block
=============================

Atomic execution units with input/output cardinality.

Blocks are defined by their data flow signature:
    (input_count → output_count)

SequentialBlock: (1 → 1) Single module passthrough
ParallelBlock:   (N → 1) Parallel branches with combine
    - Broadcast: 1 input → all branches → combine
    - Routed: N inputs → N branches → combine

The cardinality abstraction enables both homogeneous parallel
(same input broadcast) and heterogeneous parallel (different inputs routed).

Copyright 2025 AbstractPhil
MIT License
"""

from __future__ import annotations

from typing import List, Optional, Union, Callable, Tuple
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
    input_count: int
    output_count: int
    shape_info: str = ""

    def __repr__(self) -> str:
        sig = f"({self.input_count}→{self.output_count})"
        if self.shape_info:
            return f"{self.type_name}{sig}({self.shape_info}, {self.param_count:,} params)"
        return f"{self.type_name}{sig}({self.param_count:,} params)"


# =============================================================================
# BASE BLOCK
# =============================================================================

class StageBlock(nn.Module):
    """
    Base class for staged computation units.

    Defines cardinality contract:
        input_count: Expected number of inputs (-1 = variadic)
        output_count: Number of outputs produced (-1 = variadic)

    Subclasses implement specific data flow patterns.
    """

    @property
    def input_count(self) -> int:
        """Number of expected inputs. -1 = variadic."""
        raise NotImplementedError

    @property
    def output_count(self) -> int:
        """Number of produced outputs. -1 = variadic."""
        raise NotImplementedError

    @property
    def is_parallel(self) -> bool:
        """Whether block contains parallel execution."""
        return False

    def forward(self, *inputs: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
        raise NotImplementedError

    def wide_forward(self, *inputs: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
        raise NotImplementedError


# =============================================================================
# SEQUENTIAL BLOCK (1 → 1)
# =============================================================================

class SequentialBlock(StageBlock):
    """
    Single module execution block. (1 → 1)

    Wraps a single nn.Module for sequential execution.
    One input, one output.

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
    def input_count(self) -> int:
        return 1

    @property
    def output_count(self) -> int:
        return 1

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
                input_count=self.input_count,
                output_count=self.output_count,
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

    def forward(self, *inputs: Tensor) -> Tensor:
        """Standard forward execution. (1 → 1)"""
        if len(inputs) != 1:
            raise ValueError(f"SequentialBlock expects 1 input, got {len(inputs)}")
        return self.module(inputs[0])

    def wide_forward(self, *inputs: Tensor) -> Tensor:
        """Passthrough - sequential doesn't need protection."""
        return self.forward(*inputs)

    def __repr__(self) -> str:
        return f"SequentialBlock({self.name}, {self.metadata})"


# =============================================================================
# PARALLEL BLOCK (N → 1)
# =============================================================================

class ParallelBlock(StageBlock):
    """
    Multi-branch parallel execution block. (N → 1)

    Supports two input modes:
        Broadcast (1 input): Same input sent to all branches
        Routed (N inputs): Each input sent to corresponding branch

    Mode is auto-detected from input count at forward time.

    Attributes:
        branches: The parallel modules
        combine: Combination callable (extracted from user)
        name: Block identifier
    """

    def __init__(
        self,
        branches: List[nn.Module],
        combine: Callable[[List[Tensor]], Tensor] = None,
        name: str = "",
        validate: bool = True,
    ):
        super().__init__()

        if len(branches) < 2:
            raise ValueError("ParallelBlock requires at least 2 branches")

        self.branches = nn.ModuleList(branches)
        self.combine = combine or self._default_combine
        self.name = name or f"parallel_{len(branches)}"
        self._metadata: Optional[BlockMetadata] = None

        if validate:
            self._validate_branches()

    @staticmethod
    def _default_combine(outputs: List[Tensor]) -> Tensor:
        """Default: mean of outputs."""
        return torch.stack(outputs, dim=0).mean(dim=0)

    def _validate_branches(self) -> None:
        """Validate branches have similar structure (warning only)."""
        signatures = [self._branch_signature(b) for b in self.branches]
        if len(set(signatures)) > 1:
            import warnings
            warnings.warn(
                f"ParallelBlock '{self.name}' has branches with different structures. "
                "This may be intentional (e.g., CLIP), but could indicate a problem.",
                UserWarning
            )

    def _branch_signature(self, module: nn.Module) -> str:
        """Compute structural signature for branch comparison."""
        type_name = type(module).__name__
        param_count = sum(p.numel() for p in module.parameters())
        return f"{type_name}:{param_count}"

    @property
    def input_count(self) -> int:
        """Accepts 1 (broadcast) or N (routed) inputs."""
        return -1  # Variadic: 1 or len(branches)

    @property
    def output_count(self) -> int:
        return 1

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
            combine_name = getattr(self.combine, '__name__', 'custom')
            self._metadata = BlockMetadata(
                param_count=self.param_count,
                type_name=f"Parallel[{branch_type}]",
                input_count=self.input_count,
                output_count=self.output_count,
                shape_info=f"×{self.num_branches}→{combine_name}",
            )
        return self._metadata

    def _execute_branches(self, *inputs: Tensor) -> List[Tensor]:
        """Execute branches with broadcast or routed inputs."""
        n_inputs = len(inputs)
        n_branches = self.num_branches

        if n_inputs == 1:
            # Broadcast: same input to all branches
            x = inputs[0]
            return [branch(x) for branch in self.branches]

        elif n_inputs == n_branches:
            # Routed: zip inputs to branches
            return [branch(inp) for branch, inp in zip(self.branches, inputs)]

        else:
            raise ValueError(
                f"ParallelBlock expects 1 input (broadcast) or {n_branches} inputs (routed), "
                f"got {n_inputs}"
            )

    def forward(self, *inputs: Tensor) -> Tensor:
        """
        Parallel execution with combination. (N → 1)

        Broadcast mode: forward(x) - same x to all branches
        Routed mode: forward(x1, x2, ..., xN) - each to corresponding branch
        """
        outputs = self._execute_branches(*inputs)
        return self.combine(outputs)

    @torch.compiler.disable
    def wide_forward(self, *inputs: Tensor) -> Tensor:
        """
        Protected parallel execution.

        Disabled from torch.compile capture to prevent
        CUDAGraphs buffer issues during training.
        """
        outputs = self._execute_branches(*inputs)
        return self.combine(outputs)

    def __repr__(self) -> str:
        return f"ParallelBlock({self.name}, {self.num_branches} branches, {self.metadata})"


# =============================================================================
# TYPE ALIAS
# =============================================================================

Block = Union[SequentialBlock, ParallelBlock]


# =============================================================================
# UTILITIES
# =============================================================================

def make_block(
    module: nn.Module,
    name: str = "",
    combine: Callable[[List[Tensor]], Tensor] = None,
) -> Block:
    """
    Create appropriate block type from module.

    ModuleList/ModuleDict with 2+ children → ParallelBlock
    Everything else → SequentialBlock
    """
    if isinstance(module, nn.ModuleList) and len(module) >= 2:
        return ParallelBlock(list(module), combine, name, validate=True)

    if isinstance(module, nn.ModuleDict) and len(module) >= 2:
        return ParallelBlock(list(module.values()), combine, name, validate=True)

    return SequentialBlock(module, name)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base
    'StageBlock',
    'BlockMetadata',

    # Blocks
    'SequentialBlock',
    'ParallelBlock',
    'Block',

    # Utilities
    'make_block',
]


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("stage_block.py tests - Cardinality Paradigm")
    print("=" * 60)

    # Sequential (1 → 1)
    print("\n--- SequentialBlock (1 → 1) ---")
    seq = SequentialBlock(nn.Linear(64, 128), name="fc1")
    print(seq)
    print(f"  input_count: {seq.input_count}")
    print(f"  output_count: {seq.output_count}")

    x = torch.randn(4, 64)
    y = seq(x)
    print(f"  forward: {x.shape} → {y.shape}")

    # Parallel - Broadcast mode (1 → 1 internally)
    print("\n--- ParallelBlock Broadcast (1 → 1) ---")
    par = ParallelBlock(
        branches=[nn.Linear(64, 64) for _ in range(4)],
        combine=lambda outs: sum(outs) / len(outs),
        name="towers"
    )
    print(par)
    print(f"  input_count: {par.input_count} (variadic)")
    print(f"  output_count: {par.output_count}")

    x = torch.randn(4, 64)
    y = par(x)  # Broadcast: one input
    print(f"  forward(x): {x.shape} → {y.shape}")

    # Parallel - Routed mode (N → 1)
    print("\n--- ParallelBlock Routed (N → 1) ---")
    # Simulating CLIP-style: different encoders, different inputs
    text_encoder = nn.Linear(128, 64)
    vision_encoder = nn.Linear(256, 64)

    clip_block = ParallelBlock(
        branches=[text_encoder, vision_encoder],
        combine=lambda outs: outs[0] @ outs[1].T,  # Similarity
        name="clip"
    )
    print(clip_block)

    text = torch.randn(4, 128)
    image = torch.randn(8, 256)
    similarity = clip_block(text, image)  # Routed: two inputs
    print(f"  forward(text, image): {text.shape}, {image.shape} → {similarity.shape}")

    # Wide forward
    print("\n--- wide_forward ---")
    y_wide = par.wide_forward(x)
    print(f"  ParallelBlock: {x.shape} → {y_wide.shape}")

    sim_wide = clip_block.wide_forward(text, image)
    print(f"  CLIP block: {text.shape}, {image.shape} → {sim_wide.shape}")

    # Cardinality check
    print("\n--- Cardinality Summary ---")
    blocks = [seq, par, clip_block]
    for block in blocks:
        print(f"  {block.name}: ({block.input_count} → {block.output_count})")

    print("\n" + "=" * 60)
    print("✓ stage_block.py complete")
    print("✓ Cardinality-based abstraction")
    print("✓ Broadcast (1 → N → 1) and Routed (N → N → 1) modes")
    print("=" * 60)