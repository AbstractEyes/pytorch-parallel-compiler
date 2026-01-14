"""
WideCompiler.core.ensemble_util

Packing utilities for N-model ensembles.

    pack_inputs:   List[Tensor] -> Tensor  (N inputs -> wide input)
    unpack_outputs: Tensor -> List[Tensor] (wide output -> N outputs)

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from __future__ import annotations

from typing import List
import torch
from torch import Tensor


def pack_inputs(inputs: List[Tensor]) -> Tensor:
    """
    Pack N input tensors into wide format.

    Stacks along dim=1, then flattens dims 1-2.

    Args:
        inputs: List of N tensors, each [B, *feature_dims]

    Returns:
        Packed tensor [B, N*F, *spatial_dims] where F is first feature dim

    Example:
        # N=10 inputs of shape [B, C, H, W]
        inputs = [torch.randn(8, 64, 32, 32) for _ in range(10)]
        packed = pack_inputs(inputs)  # [8, 640, 32, 32]

        # N=10 inputs of shape [B, D]
        inputs = [torch.randn(8, 256) for _ in range(10)]
        packed = pack_inputs(inputs)  # [8, 2560]
    """
    if not inputs:
        raise ValueError("inputs list is empty")

    # Stack: [B, *F] -> [B, N, *F]
    stacked = torch.stack(inputs, dim=1)

    # Get shape
    B = stacked.shape[0]
    N = stacked.shape[1]
    rest = stacked.shape[2:]

    if len(rest) == 0:
        # Scalar per sample (unlikely but handle it)
        return stacked.view(B, N)

    # Flatten first two dims after batch: [B, N, F, ...] -> [B, N*F, ...]
    F = rest[0]
    spatial = rest[1:]

    return stacked.view(B, N * F, *spatial)


def unpack_outputs(output: Tensor, n: int) -> List[Tensor]:
    """
    Unpack wide output tensor to N separate outputs.

    Inverse of pack_inputs.

    Args:
        output: Packed tensor [B, N*F, *spatial_dims]
        n: Number of outputs to unpack

    Returns:
        List of N tensors, each [B, F, *spatial_dims]

    Example:
        # Unpack [8, 640, 32, 32] with N=10
        outputs = unpack_outputs(wide_out, n=10)  # 10x [8, 64, 32, 32]

        # Unpack [8, 2560] with N=10
        outputs = unpack_outputs(wide_out, n=10)  # 10x [8, 256]
    """
    B = output.shape[0]
    rest = output.shape[1:]

    if len(rest) == 0:
        raise ValueError("output must have at least 2 dimensions")

    NF = rest[0]
    spatial = rest[1:]

    if NF % n != 0:
        raise ValueError(f"Cannot split dim 1 ({NF}) into {n} equal parts")

    F = NF // n

    # Reshape: [B, N*F, ...] -> [B, N, F, ...]
    reshaped = output.view(B, n, F, *spatial)

    # Unbind along N dimension
    return [reshaped[:, i] for i in range(n)]


__all__ = ['pack_inputs', 'unpack_outputs']