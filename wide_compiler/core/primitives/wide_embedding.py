"""
WideEmbedding - N parallel Embedding tables fused into a single module.

Each of N models has its own embedding table. Given input indices,
we gather from all N tables and concatenate along the embedding dimension.

Strategies:
- 'indexed': Batched advanced indexing (FASTEST - single op)
- 'gather': torch.gather based (alternative batched approach)
- 'sequential': N separate F.embedding calls (baseline, exact)

Expected speedup: 2-5x with indexed strategy.

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Dict, Any, Union
from enum import Enum

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class EmbeddingStrategy(Enum):
    INDEXED = 'indexed'       # Batched advanced indexing - fastest
    GATHER = 'gather'         # torch.gather based
    SEQUENTIAL = 'sequential' # N separate ops - baseline
    AUTO = 'auto'


class WideEmbedding(nn.Module):
    """
    N parallel Embedding tables.

    Input shape:  [B, T] indices (same indices for all N models)
    Output shape: [B, T, N*D]

    Each model has its own D-dim embedding table.

    Strategies:
        'indexed': Batched advanced indexing - weight[:, indices]
            - Single operation, fastest for most cases
        'gather': torch.gather based batched lookup
            - Alternative batched approach
        'sequential': N separate F.embedding calls
            - Baseline, exact but slower
    """

    def __init__(
        self,
        n: int,
        num_embeddings: int,
        embedding_dim: int,
        strategy: Union[str, EmbeddingStrategy] = 'auto',
    ):
        super().__init__()
        self.n = n
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Parse strategy
        if isinstance(strategy, str):
            strategy = EmbeddingStrategy(strategy)
        if strategy == EmbeddingStrategy.AUTO:
            strategy = EmbeddingStrategy.INDEXED  # Default to fastest

        self._strategy = strategy
        self._use_indexed = (strategy == EmbeddingStrategy.INDEXED)
        self._use_gather = (strategy == EmbeddingStrategy.GATHER)

        # Weight: [N, V, D] - N tables, each with V embeddings of dim D
        self.weight = nn.Parameter(torch.empty(n, num_embeddings, embedding_dim))
        self._reset_parameters()

    @property
    def strategy(self) -> EmbeddingStrategy:
        return self._strategy

    def _reset_parameters(self):
        """Initialize like nn.Embedding."""
        nn.init.normal_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        if self._use_indexed:
            return self._forward_indexed(x)
        elif self._use_gather:
            return self._forward_gather(x)
        else:
            return self._forward_sequential(x)

    def _forward_indexed(self, x: Tensor) -> Tensor:
        """
        Batched advanced indexing - fastest approach.

        Input:  [B, T] indices
        Output: [N, B, T, D] N-first format
        """
        # x: [B, T], weight: [N, V, D]
        # weight[:, x] -> [N, B, T, D]
        return self.weight[:, x]  # [N, B, T, D]

    def _forward_gather(self, x: Tensor) -> Tensor:
        """
        torch.gather based batched lookup.

        Input:  [B, T] indices
        Output: [N, B, T, D] N-first format
        """
        B, T = x.shape
        N, V, D = self.weight.shape

        # Expand indices for gather: [N, B*T, D]
        x_flat = x.view(-1)  # [B*T]
        x_exp = x_flat.unsqueeze(0).unsqueeze(-1)  # [1, B*T, 1]
        x_exp = x_exp.expand(N, -1, D)  # [N, B*T, D]

        # Gather along vocab dimension
        out = torch.gather(self.weight, 1, x_exp)  # [N, B*T, D]
        return out.view(N, B, T, D)  # [N, B, T, D]

    def _forward_sequential(self, x: Tensor) -> Tensor:
        """
        N separate embedding lookups - baseline.

        Input:  [B, T] indices
        Output: [N, B, T, D] N-first format
        """
        outs = [F.embedding(x, self.weight[i]) for i in range(self.n)]
        return torch.stack(outs, dim=0)  # [N, B, T, D]

    @classmethod
    def from_modules(
        cls,
        modules: List[nn.Embedding],
        strategy: Union[str, EmbeddingStrategy] = 'auto',
    ) -> 'WideEmbedding':
        """Create from N existing Embedding modules."""
        n = len(modules)
        t = modules[0]

        wide = cls(
            n=n,
            num_embeddings=t.num_embeddings,
            embedding_dim=t.embedding_dim,
            strategy=strategy,
        )

        # Copy to same device/dtype
        wide = wide.to(device=t.weight.device, dtype=t.weight.dtype)

        with torch.no_grad():
            for i, m in enumerate(modules):
                wide.weight[i] = m.weight

        return wide

    def __repr__(self):
        return (
            f"WideEmbedding({self.n}x[{self.num_embeddings}, {self.embedding_dim}], "
            f"strategy={self._strategy.value})"
        )

    # =========================================================================
    # BENCHMARK INTERFACE
    # =========================================================================

    BENCHMARK_STRATEGIES = ['baseline', 'indexed', 'gather', 'sequential']

    @classmethod
    def _get_sweep_params_class(cls):
        """Get SweepParams class with multiple import attempts."""
        try:
            from ..benchmark.benchmark_schema import SweepParams
            return SweepParams
        except ImportError:
            pass
        try:
            from wide_compiler.core.benchmark.benchmark_schema import SweepParams
            return SweepParams
        except ImportError:
            pass
        return None

    @classmethod
    def _get_benchmark_job_class(cls):
        """Get BenchmarkJob class with multiple import attempts."""
        try:
            from ..benchmark.benchmark_schema import BenchmarkJob
            return BenchmarkJob
        except ImportError:
            pass
        try:
            from wide_compiler.core.benchmark.benchmark_schema import BenchmarkJob
            return BenchmarkJob
        except ImportError:
            pass
        return None

    BENCHMARK_SWEEPS: Dict[str, Any] = {}
    _SWEEPS_INITIALIZED = False

    @classmethod
    def _init_benchmark_sweeps(cls):
        """Initialize sweep configs (called once on first access)."""
        if cls._SWEEPS_INITIALIZED:
            return
        cls._SWEEPS_INITIALIZED = True

        SweepParams = cls._get_sweep_params_class()
        if SweepParams is None:
            return

        cls.BENCHMARK_SWEEPS = {
            'quick': SweepParams(
                n_values=[4, 8, 16, 32],
                batch_sizes=[8],
                vocab_sizes=[1000],
                embedding_dims=[256],
                seq_lengths=[128],
            ),
            'full': SweepParams(
                n_values=[2, 4, 8, 16, 32, 64],
                batch_sizes=[4, 8, 16, 32],
                vocab_sizes=[1000, 10000, 50000],
                embedding_dims=[128, 256, 512],
                seq_lengths=[64, 128, 256],
            ),
            'ci': SweepParams(
                n_values=[4, 16],
                batch_sizes=[8],
                vocab_sizes=[1000],
                embedding_dims=[256],
                seq_lengths=[128],
            ),
        }

    @classmethod
    def benchmark_job(cls, preset: str = 'full', **overrides) -> Any:
        """Get benchmark job for WideEmbedding."""
        cls._init_benchmark_sweeps()

        BenchmarkJob = cls._get_benchmark_job_class()
        if BenchmarkJob is None:
            raise ImportError("Could not import BenchmarkJob from benchmark_schema")

        sweep = cls.BENCHMARK_SWEEPS.get(preset)
        if sweep is None:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(cls.BENCHMARK_SWEEPS.keys())}")

        if overrides:
            sweep = sweep.with_overrides(**overrides)

        return BenchmarkJob(
            name=f'embedding_{preset}',
            primitive='embedding',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
            pack_fn=cls._bench_pack,
            unpack_fn=cls._bench_unpack,
        )

    @staticmethod
    def _bench_model(vocab_sizes: int, embedding_dims: int, **_) -> nn.Embedding:
        """Create single Embedding for benchmarking."""
        return nn.Embedding(vocab_sizes, embedding_dims)

    @staticmethod
    def _bench_input(n: int, batch_sizes: int, vocab_sizes: int, seq_lengths: int, device: str = 'cpu', **_) -> Tensor:
        """Create single input tensor (indices)."""
        return torch.randint(0, vocab_sizes, (batch_sizes, seq_lengths), device=device)

    @classmethod
    def _bench_wide(cls, modules: List[nn.Embedding], strategy: str) -> 'WideEmbedding':
        """Create WideEmbedding with specific strategy."""
        strat_map = {
            'indexed': EmbeddingStrategy.INDEXED,
            'gather': EmbeddingStrategy.GATHER,
            'sequential': EmbeddingStrategy.SEQUENTIAL,
        }
        strat = strat_map.get(strategy, EmbeddingStrategy.INDEXED)
        return cls.from_modules(modules, strategy=strat)

    @staticmethod
    def _bench_pack(inputs: List[Tensor]) -> Tensor:
        """Pack N inputs - for embedding, all inputs are the same indices."""
        return inputs[0]

    @staticmethod
    def _bench_unpack(output: Tensor, n: int) -> List[Tensor]:
        """Unpack wide output to N outputs."""
        B, T, ND = output.shape
        D = ND // n
        return [output[..., i*D:(i+1)*D] for i in range(n)]


__all__ = ['WideEmbedding', 'EmbeddingStrategy']