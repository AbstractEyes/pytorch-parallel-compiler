"""
WideCompiler.core.stage_block
=============================

Structural wrappers for compilation organization.

StageBlock wraps a module region. Pure passthrough.
Adds wide_forward capability for parallel protection.

No execution assumptions. No wiring logic.
User's forward defines behavior. We just wrap.

Copyright 2025 AbstractPhil
MIT License
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class StageBlock(nn.Module):
    """
    Wraps a module region for staged compilation.

    Pure passthrough - preserves user's forward behavior.
    Adds wide_forward for parallel region protection.

    Attributes:
        module: The wrapped computation
        is_parallel: Whether this region executes in parallel
        name: Block identifier
    """

    def __init__(
        self,
        module: nn.Module,
        is_parallel: bool = False,
        name: str = "",
    ):
        super().__init__()
        self.module = module
        self.is_parallel = is_parallel
        self.name = name or type(module).__name__

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
        Same behavior as forward.
        """
        return self.module(*args, **kwargs)

    def __repr__(self) -> str:
        p = "parallel" if self.is_parallel else "sequential"
        return f"StageBlock({self.name}, {p}, {self.param_count:,} params)"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ['StageBlock']


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 50)
    print("stage_block.py - Pure Passthrough")
    print("=" * 50)

    # Sequential block
    print("\n--- Sequential Block ---")
    seq = StageBlock(nn.Linear(64, 128), is_parallel=False, name="fc1")
    print(seq)

    x = torch.randn(4, 64)
    y = seq(x)
    print(f"  forward: {x.shape} → {y.shape}")

    y_wide = seq.wide_forward(x)
    print(f"  wide_forward: {x.shape} → {y_wide.shape}")

    # Parallel block (wraps user's parallel structure)
    print("\n--- Parallel Block ---")

    class UserTowers(nn.Module):
        def __init__(self):
            super().__init__()
            self.towers = nn.ModuleList([nn.Linear(64, 64) for _ in range(4)])

        def forward(self, x):
            outputs = [t(x) for t in self.towers]
            return sum(outputs) / len(outputs)

    par = StageBlock(UserTowers(), is_parallel=True, name="towers")
    print(par)

    x = torch.randn(4, 64)
    y = par(x)
    print(f"  forward: {x.shape} → {y.shape}")

    y_wide = par.wide_forward(x)
    print(f"  wide_forward: {x.shape} → {y_wide.shape}")

    # CLIP-style (wraps user's multi-input structure)
    print("\n--- Multi-Input Block (CLIP-style) ---")

    class UserCLIP(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_enc = nn.Linear(128, 64)
            self.vision_enc = nn.Linear(256, 64)

        def forward(self, text, image):
            t = self.text_enc(text)
            v = self.vision_enc(image)
            return t @ v.T

    clip = StageBlock(UserCLIP(), is_parallel=True, name="clip")
    print(clip)

    text = torch.randn(4, 128)
    image = torch.randn(8, 256)
    sim = clip(text, image)
    print(f"  forward(text, image): {text.shape}, {image.shape} → {sim.shape}")

    sim_wide = clip.wide_forward(text, image)
    print(f"  wide_forward: {sim_wide.shape}")

    print("\n" + "=" * 50)
    print("✓ Pure passthrough")
    print("✓ No execution assumptions")
    print("✓ Wraps anything")
    print("=" * 50)