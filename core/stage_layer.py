"""
WideCompiler.core.stage_layer
=============================

Compilation boundary for staged models.

StageLayer wraps a section of the model.
Marks graph boundary - breaks occur between layers, not within.

Pure passthrough. No execution assumptions.
User's forward defines behavior.

Copyright 2025 AbstractPhil
MIT License
"""

from __future__ import annotations

from typing import Any, List

import torch
import torch.nn as nn

try:
    from .stage_block import StageBlock
except ImportError:
    from stage_block import StageBlock


class StageLayer(nn.Module):
    """
    Compilation boundary.

    Wraps a section of user's model.
    Everything inside compiles as one unit.
    Graph breaks occur between layers.

    Pure passthrough - preserves user's forward.
    wide_forward routes to block.wide_forward.

    Attributes:
        module: The wrapped section (StageBlock or user module)
        blocks: Contained blocks (for analysis)
        name: Layer identifier
    """

    def __init__(
        self,
        module: nn.Module,
        name: str = "",
    ):
        super().__init__()
        self.module = module
        self.name = name or "layer"
        self._blocks: List[StageBlock] = []

        # Collect StageBlocks for analysis
        self._collect_blocks(module)

    def _collect_blocks(self, module: nn.Module) -> None:
        """Find all StageBlocks in module tree."""
        if isinstance(module, StageBlock):
            self._blocks.append(module)
        for child in module.children():
            self._collect_blocks(child)

    @property
    def blocks(self) -> List[StageBlock]:
        return self._blocks.copy()

    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    @property
    def has_parallel(self) -> bool:
        return any(b.is_parallel for b in self._blocks)

    @property
    def parallel_blocks(self) -> List[StageBlock]:
        return [b for b in self._blocks if b.is_parallel]

    @property
    def num_parallel(self) -> int:
        return len(self.parallel_blocks)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.module.parameters())

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Pure passthrough to wrapped module."""
        return self.module(*args, **kwargs)

    @torch.compiler.disable
    def wide_forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Protected passthrough.

        Disabled from torch.compile capture.
        Use when layer contains parallel regions.
        """
        return self.module(*args, **kwargs)

    def __repr__(self) -> str:
        p_str = f", {self.num_parallel} parallel" if self.has_parallel else ""
        return f"StageLayer({self.name}, {self.num_blocks} blocks{p_str})"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ['StageLayer']


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 50)
    print("stage_layer.py - Compilation Boundary")
    print("=" * 50)

    # Simple layer
    print("\n--- Simple Layer ---")

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, 64)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    layer = StageLayer(Encoder(), name="encoder")
    print(layer)
    print(f"  has_parallel: {layer.has_parallel}")

    x = torch.randn(4, 64)
    y = layer(x)
    print(f"  forward: {x.shape} → {y.shape}")

    # Layer with wrapped blocks
    print("\n--- Layer with StageBlocks ---")

    class MixedSection(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = StageBlock(nn.Linear(64, 128), is_parallel=False, name="stem")

            towers = nn.ModuleList([nn.Linear(128, 128) for _ in range(4)])
            self.towers = StageBlock(towers, is_parallel=True, name="towers")

            self.head = StageBlock(nn.Linear(128, 64), is_parallel=False, name="head")

        def forward(self, x):
            x = torch.relu(self.stem(x))
            # User's parallel execution + combination
            outputs = [t(x) for t in self.towers.module]
            x = sum(outputs) / len(outputs)
            return self.head(x)

    layer2 = StageLayer(MixedSection(), name="mixed")
    print(layer2)
    print(f"  blocks: {[b.name for b in layer2.blocks]}")
    print(f"  parallel_blocks: {[b.name for b in layer2.parallel_blocks]}")

    x = torch.randn(4, 64)
    y = layer2(x)
    print(f"  forward: {x.shape} → {y.shape}")

    y_wide = layer2.wide_forward(x)
    print(f"  wide_forward: {y_wide.shape}")

    # Multi-input layer
    print("\n--- Multi-Input Layer ---")

    class CLIPSection(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_enc = nn.Linear(128, 64)
            self.vision_enc = nn.Linear(256, 64)

        def forward(self, text, image):
            t = self.text_enc(text)
            v = self.vision_enc(image)
            return t @ v.T

    clip_layer = StageLayer(CLIPSection(), name="clip")
    print(clip_layer)

    text = torch.randn(4, 128)
    image = torch.randn(8, 256)
    sim = clip_layer(text, image)
    print(f"  forward(text, image): → {sim.shape}")

    print("\n" + "=" * 50)
    print("✓ Pure passthrough")
    print("✓ Compilation boundary")
    print("✓ Collects blocks for analysis")
    print("=" * 50)