"""
WideCompiler.main
=================

Public API: compile(), analyze(), summary()

Copyright 2025 AbstractPhil
MIT License
"""

from __future__ import annotations

from typing import List, Optional
import warnings

import torch
import torch.nn as nn

from .util import (
    RegionType,
    RegionInfo,
    AnalysisResult,
    ModuleClassifier,
    SignatureComputer,
    PatternDetector,
    iter_modules,
    protect_module,
)


# =============================================================================
# ANALYZER
# =============================================================================

class Analyzer:
    """Analyzes PyTorch model structure."""

    def __init__(self, min_parallel_size: int = 2):
        self.min_parallel_size = min_parallel_size

    def analyze(self, model: nn.Module) -> AnalysisResult:
        """Analyze model structure."""
        regions: List[RegionInfo] = []
        parallel_indices: List[int] = []
        max_depth = 0

        for path, module in iter_modules(model):
            depth = path.count('.')
            max_depth = max(max_depth, depth)

            parallel = PatternDetector.detect(
                module, path, self.min_parallel_size
            )

            if parallel is not None:
                parallel.depth = depth
                regions.append(parallel)
                parallel_indices.append(len(regions) - 1)

            elif ModuleClassifier.is_leaf(module):
                regions.append(RegionInfo(
                    region_type=RegionType.SEQUENTIAL,
                    path=path,
                    module=module,
                    signature=SignatureComputer.compute(module),
                    depth=depth,
                ))

        return AnalysisResult(
            regions=regions,
            parallel_indices=parallel_indices,
            total_params=sum(p.numel() for p in model.parameters()),
            max_depth=max_depth,
        )


# =============================================================================
# COMPILER
# =============================================================================

class Compiler:
    """Compiles PyTorch models with parallel region protection."""

    def __init__(self, min_parallel_size: int = 2):
        self.analyzer = Analyzer(min_parallel_size)

    def compile(
        self,
        model: nn.Module,
        mode: str = 'default',
        **kwargs
    ) -> nn.Module:
        """Compile model with specified mode."""
        analysis = self.analyzer.analyze(model)

        inference = False
        if mode == 'inference':
            mode = 'reduce-overhead'
            inference = True

        print(f"WideCompiler: {analysis.num_regions} regions, {analysis.num_parallel} parallel")

        if not analysis.has_parallel:
            print(f"  Mode: {mode}")
            return torch.compile(model, mode=mode, **kwargs)

        if mode == 'hybrid':
            return self._compile_hybrid(model, analysis, **kwargs)

        if mode == 'reduce-overhead' and not inference:
            warnings.warn(
                f"reduce-overhead with {analysis.num_parallel} parallel region(s) "
                "may crash on backward. Use mode='hybrid' for training.",
                RuntimeWarning
            )

        print(f"  Mode: {mode}")
        return torch.compile(model, mode=mode, **kwargs)

    def _compile_hybrid(
        self,
        model: nn.Module,
        analysis: AnalysisResult,
        **kwargs
    ) -> nn.Module:
        """Hybrid compilation: protect parallel, compile rest."""
        print(f"  Mode: hybrid")

        for idx in analysis.parallel_indices:
            region = analysis.regions[idx]
            protect_module(region.module)
            print(f"    Protected: {region.path}")

        return torch.compile(model, mode='reduce-overhead', **kwargs)


# =============================================================================
# PUBLIC API
# =============================================================================

_analyzer = Analyzer()
_compiler = Compiler()


def analyze(model: nn.Module, min_parallel_size: int = 2) -> AnalysisResult:
    """Analyze model structure."""
    return Analyzer(min_parallel_size).analyze(model)


def summary(model: nn.Module) -> str:
    """Get analysis summary string."""
    analysis = _analyzer.analyze(model)

    lines = [
        "WideCompiler Analysis",
        "=" * 40,
        f"Regions: {analysis.num_regions}",
        f"Parallel: {analysis.num_parallel}",
        f"Parameters: {analysis.total_params:,}",
        "",
    ]

    if analysis.num_parallel > 0:
        lines.append("Parallel regions:")
        for idx in analysis.parallel_indices:
            region = analysis.regions[idx]
            lines.append(f"  {region.path} ({region.num_children} children)")

    return "\n".join(lines)


def compile(
    model: nn.Module,
    mode: str = 'default',
    **kwargs
) -> nn.Module:
    """
    Compile any PyTorch model.

    Args:
        model: Any nn.Module
        mode: 'default', 'hybrid', 'reduce-overhead', 'inference'
        **kwargs: Additional args for torch.compile

    Returns:
        Compiled model
    """
    return _compiler.compile(model, mode, **kwargs)


__all__ = ['Analyzer', 'Compiler', 'analyze', 'summary', 'compile']