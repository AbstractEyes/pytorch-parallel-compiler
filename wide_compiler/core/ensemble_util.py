"""
WideCompiler.core.ensemble_util

Packing utilities for N-model ensembles.

Three formats:
    1. List[Tensor]     - N separate [B, C, ...] tensors (user API)
    2. Channel-packed   - [B, N*C, ...] (TracedWideModel I/O)
    3. N-first          - [N, B, C, ...] (internal Wide ops)

Conversions:
    pack_inputs     : List[Tensor] → [B, N*C, ...]
    unpack_outputs  : [B, N*C, ...] → List[Tensor]

    to_n_first      : [B, N*C, ...] → [N, B, C, ...]
    from_n_first    : [N, B, C, ...] → [B, N*C, ...]

    iter_n_first    : [N, B, C, ...] → Iterator of [B, C, ...]
    stack_n_first   : List of [B, C, ...] → [N, B, C, ...]

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from __future__ import annotations

from typing import List, Iterator
import torch
from torch import Tensor


# =============================================================================
# LIST ↔ CHANNEL-PACKED (User API)
# =============================================================================

def pack_inputs(inputs: List[Tensor]) -> Tensor:
    """
    Pack N input tensors into channel-packed format.

    List of N × [B, C, ...] → [B, N*C, ...]
    """
    if not inputs:
        raise ValueError("inputs list is empty")

    stacked = torch.stack(inputs, dim=1)  # [B, N, C, ...]
    B, N = stacked.shape[:2]
    rest = stacked.shape[2:]

    if len(rest) == 0:
        return stacked.view(B, N)

    C = rest[0]
    spatial = rest[1:]
    return stacked.view(B, N * C, *spatial)


def unpack_outputs(output: Tensor, n: int) -> List[Tensor]:
    """
    Unpack channel-packed tensor to N separate outputs.

    [B, N*C, ...] → List of N × [B, C, ...]
    """
    B = output.shape[0]
    NC = output.shape[1]
    spatial = output.shape[2:]

    if NC % n != 0:
        raise ValueError(f"Cannot split dim 1 ({NC}) into {n} equal parts")

    C = NC // n
    reshaped = output.view(B, n, C, *spatial)  # [B, N, C, ...]
    return [reshaped[:, i] for i in range(n)]


# =============================================================================
# CHANNEL-PACKED ↔ N-FIRST (TracedWideModel boundaries)
# =============================================================================

def to_n_first(x: Tensor, n: int) -> Tensor:
    """
    Convert channel-packed to N-first format.

    [B, N*C, ...] → [N, B, C, ...]
    """
    B, NC = x.shape[0], x.shape[1]
    spatial = x.shape[2:]

    if NC % n != 0:
        raise ValueError(f"Cannot split dim 1 ({NC}) into {n} parts")

    C = NC // n
    x = x.view(B, n, C, *spatial)  # [B, N, C, ...]
    return x.movedim(1, 0)          # [N, B, C, ...]


def from_n_first(x: Tensor) -> Tensor:
    """
    Convert N-first to channel-packed format.

    [N, B, C, ...] → [B, N*C, ...]
    """
    N, B, C = x.shape[0], x.shape[1], x.shape[2]
    spatial = x.shape[3:]

    x = x.movedim(0, 1)              # [B, N, C, ...]
    return x.reshape(B, N * C, *spatial)


# =============================================================================
# N-FIRST ↔ ITERATION (SequentialPassthrough)
# =============================================================================

def iter_n_first(x: Tensor) -> Iterator[Tensor]:
    """
    Iterate over N dimension of N-first tensor.

    [N, B, C, ...] → yields N × [B, C, ...]

    Usage:
        for i, xi in enumerate(iter_n_first(x)):
            out_i = module[i](xi)
    """
    for i in range(x.shape[0]):
        yield x[i]


def stack_n_first(tensors: List[Tensor]) -> Tensor:
    """
    Stack list of tensors into N-first format.

    List of N × [B, C, ...] → [N, B, C, ...]
    """
    return torch.stack(tensors, dim=0)


# =============================================================================
# CONVENIENCE
# =============================================================================

def get_n_first_shape(x: Tensor) -> tuple:
    """Extract (N, B, C, *spatial) from N-first tensor."""
    return x.shape[0], x.shape[1], x.shape[2], x.shape[3:]


__all__ = [
    # List ↔ Channel-packed
    'pack_inputs',
    'unpack_outputs',
    # Channel-packed ↔ N-first
    'to_n_first',
    'from_n_first',
    # N-first iteration
    'iter_n_first',
    'stack_n_first',
    # Utility
    'get_n_first_shape',
]