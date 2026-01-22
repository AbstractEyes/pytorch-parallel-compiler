"""
WideMLPEmbedder - N parallel MLP-based embedding projectors.

Projects conditioning vectors (e.g., timestep, text embeddings) through
an MLP to produce modulation parameters for adaptive normalization.

Common in diffusion models (Flux, SD3) for timestep/class conditioning.

Expected speedup: 10-12x (similar to Linear, mostly compute bound)

Strategies:
- 'fused': Fused einsum for all projections (FASTEST)
- 'sequential': N separate MLPs (baseline)

Input/Output Format (v0.7.0):
- Input:  [N, B, in_features]  (N-first)
- Output: [N, B, hidden_features] (N-first, after activation and projection)

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Optional, Callable

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class WideMLPEmbedder(nn.Module):
    """
    N parallel MLP embedders for conditioning.

    Typical architecture:
        input -> Linear -> SiLU -> Linear -> output
    """

    BENCHMARK_STRATEGIES = ['baseline', 'fused', 'sequential']

    def __init__(
        self,
        n: int,
        in_features: int,
        hidden_features: int,
        activation: Optional[Callable] = None,
        strategy: str = 'fused',
    ):
        super().__init__()
        self.n = n
        self.in_features = in_features
        self.hidden_features = hidden_features
        self._strategy = strategy
        self._activation = activation if activation is not None else F.silu

        # First projection: [N, hidden, in]
        self.fc1_weight = nn.Parameter(torch.empty(n, hidden_features, in_features))
        self.fc1_bias = nn.Parameter(torch.empty(n, hidden_features))

        # Second projection: [N, in, hidden] - projects back to in_features
        self.fc2_weight = nn.Parameter(torch.empty(n, in_features, hidden_features))
        self.fc2_bias = nn.Parameter(torch.empty(n, in_features))

        self._reset_parameters()

    @property
    def strategy(self) -> str:
        return self._strategy

    def _reset_parameters(self):
        """Initialize weights."""
        for i in range(self.n):
            nn.init.xavier_uniform_(self.fc1_weight[i])
            nn.init.zeros_(self.fc1_bias[i])
            nn.init.xavier_uniform_(self.fc2_weight[i])
            nn.init.zeros_(self.fc2_bias[i])

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with N-first format.

        Input:  [N, B, in_features]
        Output: [N, B, hidden_features]
        """
        if self._strategy == 'fused':
            return self._forward_fused(x)
        else:
            return self._forward_sequential(x)

    def _forward_fused(self, x: Tensor) -> Tensor:
        """Fused MLP via einsum."""
        N, B, D_in = x.shape

        # First projection: [N, B, D_in] @ [N, H, D_in].T -> [N, B, H]
        h = torch.einsum('nbi,nhi->nbh', x, self.fc1_weight)
        h = h + self.fc1_bias.unsqueeze(1)  # [N, B, H] + [N, 1, H]

        # Activation
        h = self._activation(h)

        # Second projection: [N, B, H] @ [N, H, H].T -> [N, B, H]
        out = torch.einsum('nbh,noh->nbo', h, self.fc2_weight)
        out = out + self.fc2_bias.unsqueeze(1)

        return out

    def _forward_sequential(self, x: Tensor) -> Tensor:
        """Sequential MLP (baseline)."""
        N = x.shape[0]
        outputs = []

        for i in range(N):
            x_i = x[i]  # [B, D_in]

            # First projection (fc1_weight already in [out, in] format)
            h = F.linear(x_i, self.fc1_weight[i], self.fc1_bias[i])
            h = self._activation(h)

            # Second projection (fc2_weight already in [out, in] format)
            out = F.linear(h, self.fc2_weight[i], self.fc2_bias[i])

            outputs.append(out)

        return torch.stack(outputs, dim=0)

    @classmethod
    def from_modules(cls, modules: List[nn.Module], strategy: str = 'fused') -> 'WideMLPEmbedder':
        """
        Create from N existing MLP embedder modules.

        Expects modules with .fc1, .fc2 (nn.Linear) and activation.
        """
        n = len(modules)
        t = modules[0]

        # Determine dimensions from first module
        in_features = t.fc1.in_features
        hidden_features = t.fc1.out_features

        wide = cls(
            n=n,
            in_features=in_features,
            hidden_features=hidden_features,
            activation=getattr(t, 'activation', F.silu),
            strategy=strategy,
        )

        wide = wide.to(device=t.fc1.weight.device, dtype=t.fc1.weight.dtype)

        # Copy weights
        with torch.no_grad():
            for i, m in enumerate(modules):
                wide.fc1_weight[i] = m.fc1.weight
                wide.fc1_bias[i] = m.fc1.bias
                wide.fc2_weight[i] = m.fc2.weight
                wide.fc2_bias[i] = m.fc2.bias

        return wide

    def __repr__(self):
        return (f"WideMLPEmbedder({self.n}x[{self.in_features}->{self.hidden_features}], "
                f"strategy={self._strategy})")

    # =========================================================================
    # BENCHMARK INTERFACE
    # =========================================================================

    BENCHMARK_STRATEGIES = ['baseline', 'fused', 'sequential']
    BENCHMARK_SWEEPS = {}
    _SWEEPS_INITIALIZED = False

    @classmethod
    def _init_benchmark_sweeps(cls):
        """Initialize sweep configs (called once)."""
        if cls._SWEEPS_INITIALIZED:
            return
        cls._SWEEPS_INITIALIZED = True

        try:
            from ..benchmark.benchmark_schema import SweepParams
        except ImportError:
            return

        cls.BENCHMARK_SWEEPS = {
            'quick': SweepParams(
                n_values=[4, 8, 16, 32],
                batch_sizes=[8],
                embedding_dims=[128, 256],
            ),
            'full': SweepParams(
                n_values=[2, 4, 8, 16, 32, 64],
                batch_sizes=[4, 8, 16],
                embedding_dims=[128, 256, 512, 768],
            ),
            'ci': SweepParams(
                n_values=[4, 8],
                batch_sizes=[8],
                embedding_dims=[128],
            ),
        }

    @classmethod
    def benchmark_job(cls, preset: str = 'full'):
        """Create benchmark job for WideMLPEmbedder."""
        cls._init_benchmark_sweeps()
        from ..benchmark.benchmark_schema import BenchmarkJob

        sweep = cls.BENCHMARK_SWEEPS.get(preset, cls.BENCHMARK_SWEEPS['full'])

        return BenchmarkJob(
            name=f'mlp_embedder_{preset}',
            primitive='mlp_embedder',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
        )

    @staticmethod
    def _bench_model(embedding_dims=256, **kwargs):
        """Create a single MLPEmbedder module."""
        class MLPEmbedder(nn.Module):
            def __init__(self, in_dim, hidden_dim=None):
                super().__init__()
                if hidden_dim is None:
                    hidden_dim = in_dim * 4
                self.fc1 = nn.Linear(in_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, in_dim)
                self.activation = F.silu

            def forward(self, x):
                h = self.fc1(x)
                h = self.activation(h)
                return self.fc2(h)

        return MLPEmbedder(embedding_dims)

    @staticmethod
    def _bench_input(n: int, device: str, batch_sizes: int, embedding_dims: int, **kwargs):
        """Create input for MLPEmbedder benchmark."""
        return torch.randn(batch_sizes, embedding_dims, device=device)

    @classmethod
    def _bench_wide(cls, modules: List[nn.Module], strategy: str):
        """Create wide version for given strategy."""
        if strategy == 'baseline':
            return None
        return cls.from_modules(modules, strategy=strategy)


__all__ = ['WideMLPEmbedder']
