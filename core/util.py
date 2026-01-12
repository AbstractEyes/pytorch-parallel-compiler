"""
WideCompiler.util
=================

Foundation utilities for type-agnostic compilation.

Provides:
    - Module classification
    - Signature computation for structural matching
    - Pattern detection for parallel regions
    - Path navigation utilities

Copyright 2025 AbstractPhil
MIT License
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple, Type, Iterator, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import hashlib

import torch
import torch.nn as nn
from torch import Tensor


# =============================================================================
# ENUMS
# =============================================================================

class RegionType(Enum):
    """Execution region classification."""
    SEQUENTIAL = auto()  # Single path execution
    PARALLEL = auto()    # Multiple path execution


class ModuleCategory(Enum):
    """High-level module classification."""
    LINEAR = auto()
    CONV = auto()
    NORM = auto()
    ACTIVATION = auto()
    ATTENTION = auto()
    EMBEDDING = auto()
    DROPOUT = auto()
    POOLING = auto()
    CONTAINER = auto()
    RECURRENT = auto()
    IDENTITY = auto()
    CUSTOM = auto()
    UNKNOWN = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ModuleSignature:
    """Structural signature for similarity comparison."""
    category: ModuleCategory
    type_name: str
    param_count: int
    shape_info: str = ""
    children_hash: str = ""

    @property
    def key(self) -> str:
        """Signature key for comparison."""
        parts = [self.type_name, self.shape_info, str(self.param_count)]
        if self.children_hash:
            parts.append(self.children_hash[:8])
        return ":".join(filter(None, parts))

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ModuleSignature):
            return self.key == other.key
        return False


@dataclass
class RegionInfo:
    """Detected execution region."""
    region_type: RegionType
    path: str
    module: nn.Module
    signature: ModuleSignature
    children_paths: List[str] = field(default_factory=list)
    depth: int = 0

    @property
    def is_parallel(self) -> bool:
        return self.region_type == RegionType.PARALLEL

    @property
    def num_children(self) -> int:
        return len(self.children_paths)


@dataclass
class AnalysisResult:
    """Model analysis result."""
    regions: List[RegionInfo] = field(default_factory=list)
    parallel_indices: List[int] = field(default_factory=list)
    total_params: int = 0
    max_depth: int = 0

    @property
    def has_parallel(self) -> bool:
        return len(self.parallel_indices) > 0

    @property
    def num_parallel(self) -> int:
        return len(self.parallel_indices)

    @property
    def num_regions(self) -> int:
        return len(self.regions)


# =============================================================================
# MODULE CLASSIFIER
# =============================================================================

class ModuleClassifier:
    """Classifies PyTorch modules by category."""

    LINEAR_TYPES: Tuple[Type, ...] = (nn.Linear, nn.Bilinear)

    CONV_TYPES: Tuple[Type, ...] = (
        nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
    )

    NORM_TYPES: Tuple[Type, ...] = (
        nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    )

    ACTIVATION_TYPES: Tuple[Type, ...] = (
        nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.ELU, nn.SELU,
        nn.GELU, nn.SiLU, nn.Mish, nn.Softplus,
        nn.Tanh, nn.Sigmoid, nn.Softmax, nn.LogSoftmax,
    )

    POOLING_TYPES: Tuple[Type, ...] = (
        nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
        nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
        nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d,
        nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d,
    )

    DROPOUT_TYPES: Tuple[Type, ...] = (
        nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
    )

    CONTAINER_TYPES: Tuple[Type, ...] = (
        nn.Sequential, nn.ModuleList, nn.ModuleDict,
    )

    RECURRENT_TYPES: Tuple[Type, ...] = (nn.RNN, nn.LSTM, nn.GRU)

    ATTENTION_TYPES: Tuple[Type, ...] = (nn.MultiheadAttention,)

    EMBEDDING_TYPES: Tuple[Type, ...] = (nn.Embedding, nn.EmbeddingBag)

    @classmethod
    def classify(cls, module: nn.Module) -> ModuleCategory:
        """Classify module into category."""
        if isinstance(module, cls.LINEAR_TYPES):
            return ModuleCategory.LINEAR
        if isinstance(module, cls.CONV_TYPES):
            return ModuleCategory.CONV
        if isinstance(module, cls.NORM_TYPES):
            return ModuleCategory.NORM
        if isinstance(module, cls.ACTIVATION_TYPES):
            return ModuleCategory.ACTIVATION
        if isinstance(module, cls.POOLING_TYPES):
            return ModuleCategory.POOLING
        if isinstance(module, cls.DROPOUT_TYPES):
            return ModuleCategory.DROPOUT
        if isinstance(module, cls.CONTAINER_TYPES):
            return ModuleCategory.CONTAINER
        if isinstance(module, cls.RECURRENT_TYPES):
            return ModuleCategory.RECURRENT
        if isinstance(module, cls.ATTENTION_TYPES):
            return ModuleCategory.ATTENTION
        if isinstance(module, cls.EMBEDDING_TYPES):
            return ModuleCategory.EMBEDDING
        if isinstance(module, nn.Identity):
            return ModuleCategory.IDENTITY
        if len(list(module.children())) > 0:
            return ModuleCategory.CUSTOM
        return ModuleCategory.UNKNOWN

    @classmethod
    def is_leaf(cls, module: nn.Module) -> bool:
        """Check if module is a leaf (no children to traverse)."""
        leaf_categories = {
            ModuleCategory.LINEAR, ModuleCategory.CONV,
            ModuleCategory.NORM, ModuleCategory.ACTIVATION,
            ModuleCategory.POOLING, ModuleCategory.DROPOUT,
            ModuleCategory.RECURRENT, ModuleCategory.ATTENTION,
            ModuleCategory.EMBEDDING, ModuleCategory.IDENTITY,
        }
        category = cls.classify(module)
        if category in leaf_categories:
            return True
        if category == ModuleCategory.CONTAINER:
            return False
        return len(list(module.children())) == 0


# =============================================================================
# SIGNATURE COMPUTATION
# =============================================================================

class SignatureComputer:
    """Computes structural signatures for modules."""

    @classmethod
    def compute(cls, module: nn.Module) -> ModuleSignature:
        """Compute signature for module."""
        category = ModuleClassifier.classify(module)
        type_name = type(module).__name__
        param_count = sum(p.numel() for p in module.parameters(recurse=False))
        shape_info = cls._get_shape_info(module)
        children_hash = ""

        if category in (ModuleCategory.CONTAINER, ModuleCategory.CUSTOM):
            children_hash = cls._hash_children(module)

        return ModuleSignature(
            category=category,
            type_name=type_name,
            param_count=param_count,
            shape_info=shape_info,
            children_hash=children_hash,
        )

    @classmethod
    def _get_shape_info(cls, module: nn.Module) -> str:
        """Extract shape information."""
        if isinstance(module, nn.Linear):
            return f"{module.in_features}x{module.out_features}"
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return f"{module.in_channels}x{module.out_channels}"
        if isinstance(module, nn.LayerNorm):
            return str(list(module.normalized_shape))
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return str(module.num_features)
        if isinstance(module, nn.Embedding):
            return f"{module.num_embeddings}x{module.embedding_dim}"
        return ""

    @classmethod
    def _hash_children(cls, module: nn.Module) -> str:
        """Hash children structure."""
        child_sigs = []
        for name, child in module.named_children():
            sig = cls.compute(child)
            child_sigs.append(f"{name}:{sig.key}")
        combined = "|".join(child_sigs)
        return hashlib.md5(combined.encode()).hexdigest()

    @classmethod
    def are_similar(cls, modules: List[nn.Module]) -> bool:
        """Check if modules have similar structure."""
        if len(modules) < 2:
            return False
        signatures = [cls.compute(m) for m in modules]
        first = signatures[0]
        return all(sig == first for sig in signatures[1:])


# =============================================================================
# PATTERN DETECTION
# =============================================================================

class PatternDetector:
    """Detects parallel patterns in module structures."""

    @classmethod
    def detect_module_list(
        cls,
        module: nn.ModuleList,
        path: str,
        min_size: int = 2,
    ) -> Optional[RegionInfo]:
        """Detect parallel pattern in ModuleList."""
        if len(module) < min_size:
            return None

        modules = list(module)
        if not SignatureComputer.are_similar(modules):
            return None

        return RegionInfo(
            region_type=RegionType.PARALLEL,
            path=path,
            module=module,
            signature=SignatureComputer.compute(module),
            children_paths=[f"{path}.{i}" for i in range(len(module))],
        )

    @classmethod
    def detect_module_dict(
        cls,
        module: nn.ModuleDict,
        path: str,
        min_size: int = 2,
    ) -> Optional[RegionInfo]:
        """Detect parallel pattern in ModuleDict."""
        if len(module) < min_size:
            return None

        modules = list(module.values())
        if not SignatureComputer.are_similar(modules):
            return None

        return RegionInfo(
            region_type=RegionType.PARALLEL,
            path=path,
            module=module,
            signature=SignatureComputer.compute(module),
            children_paths=[f"{path}.{k}" for k in module.keys()],
        )

    @classmethod
    def detect_indexed_pattern(
        cls,
        module: nn.Module,
        path: str,
        min_size: int = 2,
    ) -> Optional[RegionInfo]:
        """
        Detect indexed naming pattern (name_0, name_1, ...).

        This is a heuristic - works with common conventions.
        """
        groups: Dict[str, List[Tuple[str, nn.Module]]] = defaultdict(list)

        for name, child in module.named_children():
            parts = name.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                groups[parts[0]].append((name, child))

        for prefix, members in groups.items():
            if len(members) >= min_size:
                modules = [m for _, m in members]
                if SignatureComputer.are_similar(modules):
                    return RegionInfo(
                        region_type=RegionType.PARALLEL,
                        path=f"{path}.[{prefix}_*]",
                        module=module,
                        signature=ModuleSignature(
                            category=ModuleCategory.CUSTOM,
                            type_name=f"IndexedGroup",
                            param_count=sum(p.numel() for m in modules for p in m.parameters()),
                            shape_info=f"{len(members)}",
                        ),
                        children_paths=[f"{path}.{name}" for name, _ in members],
                    )

        return None

    @classmethod
    def detect(
        cls,
        module: nn.Module,
        path: str,
        min_size: int = 2,
    ) -> Optional[RegionInfo]:
        """Run all detection patterns."""
        if isinstance(module, nn.ModuleList):
            result = cls.detect_module_list(module, path, min_size)
            if result:
                return result

        if isinstance(module, nn.ModuleDict):
            result = cls.detect_module_dict(module, path, min_size)
            if result:
                return result

        return cls.detect_indexed_pattern(module, path, min_size)


# =============================================================================
# UTILITIES
# =============================================================================

def get_module_by_path(root: nn.Module, path: str) -> Optional[nn.Module]:
    """Get module by dotted path."""
    parts = path.split('.')
    module = root

    for part in parts:
        if part == 'root' or part.startswith('['):
            continue

        if part.isdigit():
            idx = int(part)
            if isinstance(module, (nn.ModuleList, nn.Sequential)) and idx < len(module):
                module = module[idx]
            else:
                return None
        elif hasattr(module, part):
            module = getattr(module, part)
        elif isinstance(module, nn.ModuleDict) and part in module:
            module = module[part]
        else:
            return None

    return module


def iter_modules(
    module: nn.Module,
    prefix: str = "root",
) -> Iterator[Tuple[str, nn.Module]]:
    """Iterate modules with paths."""
    yield prefix, module
    for name, child in module.named_children():
        yield from iter_modules(child, f"{prefix}.{name}")


def protect_module(module: nn.Module) -> None:
    """
    Protect module's forward from torch.compile.

    Wraps forward with @torch.compiler.disable.
    Mutates module in place.
    """
    original = module.forward

    @torch.compiler.disable
    def _protected(*args, **kwargs):
        return original(*args, **kwargs)

    module.forward = _protected


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'RegionType',
    'ModuleCategory',
    'ModuleSignature',
    'RegionInfo',
    'AnalysisResult',
    'ModuleClassifier',
    'SignatureComputer',
    'PatternDetector',
    'get_module_by_path',
    'iter_modules',
    'protect_module',
]