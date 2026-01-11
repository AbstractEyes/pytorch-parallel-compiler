"""
WideCompiler.core
=================

Structural components for compilation organization.

Hierarchy (bottom to top):
    Block  - Single computational unit
    Layer  - Collection of blocks
    Model  - Full model wrapper

Types:
    compiled_* - Sequential execution
    parallel_* - Parallel execution (wide_forward eligible)

#Copyright 2025 AbstractPhil
#MIT License
#"""
#
#from .compiled_block import CompiledBlock
#from .compiled_layer import CompiledLayer
#from .compiled_model import CompiledModel
#
#from .stage_block import ParallelBlock
#from .layer import ParallelLayer
#from .parallel_model import ParallelModel
#
#__all__ = [
#    'CompiledBlock',
#    'CompiledLayer',
#    'CompiledModel',
#    'ParallelBlock',
#    'ParallelLayer',
#    'ParallelModel',
#]