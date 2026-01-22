"""
WideMLP - N parallel MLP blocks with overridable activation.

Standard feed-forward network used in transformer architectures.
Supports various activation functions (GELU, SiLU, ReLU, etc.).

Expected speedup: 8-12x (dominated by linear layers)

Strategies:
- 'fused': Fused linear projections via einsum (FASTEST)
- 'sequential': N separate MLP blocks (baseline)

Input/Output Format (v0.7.0):
- Input:  [N, B, seq_len, hidden_size]  (N-first)
- Output: [N, B, seq_len, hidden_size]  (N-first)

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Optional, Callable

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class WideMLP(nn.Module):
    """
    N parallel MLP blocks with overridable activation.

    Architecture:
        x -> Linear(hidden_size, intermediate_size)
          -> Activation
          -> Linear(intermediate_size, hidden_size)
    """

    def __init__(
        self,
        n: int,
        hidden_size: int,
        intermediate_size: int,
        activation: str = 'gelu',
        bias: bool = True,
        strategy: str = 'fused',
    ):
        super().__init__()
        self.n = n
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_name = activation
        self._strategy = strategy
        self.has_bias = bias

        # First linear: hidden_size -> intermediate_size
        self.fc1_weight = nn.Parameter(torch.empty(n, intermediate_size, hidden_size))
        if bias:
            self.fc1_bias = nn.Parameter(torch.zeros(n, intermediate_size))
        else:
            self.register_parameter('fc1_bias', None)

        # Second linear: intermediate_size -> hidden_size
        self.fc2_weight = nn.Parameter(torch.empty(n, hidden_size, intermediate_size))
        if bias:
            self.fc2_bias = nn.Parameter(torch.zeros(n, hidden_size))
        else:
            self.register_parameter('fc2_bias', None)

        # Activation function
        self._activation = self._get_activation(activation)

        self._reset_parameters()

    @property
    def strategy(self) -> str:
        return self._strategy

    def _get_activation(self, name: str) -> Callable:
        """Get activation function by name."""
        activations = {
            'gelu': F.gelu,
            'relu': F.relu,
            'silu': F.silu,
            'swish': F.silu,  # SiLU is also called Swish
            'tanh': torch.tanh,
            'gelu_tanh': lambda x: F.gelu(x, approximate='tanh'),
        }
        if name not in activations:
            raise ValueError(f"Unsupported activation: {name}. Supported: {list(activations.keys())}")
        return activations[name]

    def _reset_parameters(self):
        """Initialize weights."""
        for i in range(self.n):
            nn.init.xavier_uniform_(self.fc1_weight[i])
            nn.init.xavier_uniform_(self.fc2_weight[i])

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with N-first format.

        Args:
            x: [N, B, seq_len, hidden_size]

        Returns:
            [N, B, seq_len, hidden_size]
        """
        if self._strategy == 'fused':
            return self._forward_fused(x)
        else:
            return self._forward_sequential(x)

    def _forward_fused(self, x: Tensor) -> Tensor:
        """Fused MLP via batched operations."""
        N, B, S, H = x.shape

        # First linear: [N, B, S, H] @ [N, intermediate, H].T -> [N, B, S, intermediate]
        h = torch.einsum('nbsh,nih->nbsi', x, self.fc1_weight)
        if self.fc1_bias is not None:
            h = h + self.fc1_bias.view(N, 1, 1, -1)

        # Activation
        h = self._activation(h)

        # Second linear: [N, B, S, intermediate] @ [N, H, intermediate].T -> [N, B, S, H]
        out = torch.einsum('nbsi,nhi->nbsh', h, self.fc2_weight)
        if self.fc2_bias is not None:
            out = out + self.fc2_bias.view(N, 1, 1, -1)

        return out

    def _forward_sequential(self, x: Tensor) -> Tensor:
        """Sequential MLP (baseline)."""
        N, B, S, H = x.shape
        outputs = []

        for i in range(N):
            x_i = x[i]  # [B, S, H]

            # First linear (weights already in [out, in] format)
            h_i = F.linear(x_i, self.fc1_weight[i], self.fc1_bias[i] if self.fc1_bias is not None else None)

            # Activation
            h_i = self._activation(h_i)

            # Second linear (weights already in [out, in] format)
            out_i = F.linear(h_i, self.fc2_weight[i], self.fc2_bias[i] if self.fc2_bias is not None else None)

            outputs.append(out_i)

        return torch.stack(outputs, dim=0)

    @classmethod
    def from_modules(cls, modules: List[nn.Module], strategy: str = 'fused') -> 'WideMLP':
        """
        Create from N existing MLP modules.

        Expects modules with:
        - .fc1 or .linear1 (Linear): first linear layer
        - .fc2 or .linear2 (Linear): second linear layer
        - .activation or .act_fn (Callable): activation function

        Also supports sequential containers with [Linear, Activation, Linear] pattern.
        """
        n = len(modules)
        t = modules[0]

        # Detect module structure
        if isinstance(t, nn.Sequential):
            # Sequential container: [Linear, Activation, Linear]
            fc1 = t[0]
            fc2 = t[-1]
            # Try to detect activation from middle layer
            activation = 'gelu'  # Default
            if len(t) > 2:
                act_layer = t[1]
                if isinstance(act_layer, nn.GELU):
                    activation = 'gelu'
                elif isinstance(act_layer, nn.ReLU):
                    activation = 'relu'
                elif isinstance(act_layer, nn.SiLU):
                    activation = 'silu'
        else:
            # Module with attributes
            fc1 = getattr(t, 'fc1', getattr(t, 'linear1', None))
            fc2 = getattr(t, 'fc2', getattr(t, 'linear2', None))

            if fc1 is None or fc2 is None:
                raise ValueError("Could not find fc1/fc2 or linear1/linear2 in module")

            # Try to detect activation
            activation = 'gelu'  # Default
            act_fn = getattr(t, 'activation', getattr(t, 'act_fn', None))
            if act_fn is not None:
                if isinstance(act_fn, str):
                    activation = act_fn
                elif isinstance(act_fn, nn.GELU):
                    activation = 'gelu'
                elif isinstance(act_fn, nn.ReLU):
                    activation = 'relu'
                elif isinstance(act_fn, nn.SiLU):
                    activation = 'silu'

        hidden_size = fc1.in_features
        intermediate_size = fc1.out_features

        wide = cls(
            n=n,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            bias=(fc1.bias is not None),
            strategy=strategy,
        )

        wide = wide.to(device=fc1.weight.device, dtype=fc1.weight.dtype)

        # Copy weights
        with torch.no_grad():
            for i, m in enumerate(modules):
                if isinstance(m, nn.Sequential):
                    m_fc1 = m[0]
                    m_fc2 = m[-1]
                else:
                    m_fc1 = getattr(m, 'fc1', getattr(m, 'linear1', None))
                    m_fc2 = getattr(m, 'fc2', getattr(m, 'linear2', None))

                wide.fc1_weight[i] = m_fc1.weight
                wide.fc2_weight[i] = m_fc2.weight

                if m_fc1.bias is not None:
                    wide.fc1_bias[i] = m_fc1.bias
                if m_fc2.bias is not None:
                    wide.fc2_bias[i] = m_fc2.bias

        return wide

    def __repr__(self):
        return (f"WideMLP({self.n}x[hidden={self.hidden_size}, "
                f"intermediate={self.intermediate_size}, activation={self.activation_name}], "
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
                seq_lengths=[128],
                channels=[256],
            ),
            'full': SweepParams(
                n_values=[2, 4, 8, 16, 32, 64],
                batch_sizes=[4, 8, 16],
                seq_lengths=[64, 128, 256],
                channels=[128, 256, 512, 768],
            ),
            'ci': SweepParams(
                n_values=[4, 8],
                batch_sizes=[8],
                seq_lengths=[64],
                channels=[128],
            ),
        }

    @classmethod
    def benchmark_job(cls, preset: str = 'full'):
        """Create benchmark job for WideMLP."""
        cls._init_benchmark_sweeps()
        from ..benchmark.benchmark_schema import BenchmarkJob

        sweep = cls.BENCHMARK_SWEEPS.get(preset, cls.BENCHMARK_SWEEPS['full'])

        return BenchmarkJob(
            name=f'mlp_block_{preset}',
            primitive='mlp_block',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
        )

    @staticmethod
    def _bench_model(channels=256, **kwargs):
        """Create a single MLP module."""
        class MLP(nn.Module):
            def __init__(self, hidden_size, intermediate_size=None):
                super().__init__()
                if intermediate_size is None:
                    intermediate_size = hidden_size * 4
                self.fc1 = nn.Linear(hidden_size, intermediate_size)
                self.fc2 = nn.Linear(intermediate_size, hidden_size)
                self.activation = F.gelu

            def forward(self, x):
                h = self.fc1(x)
                h = self.activation(h)
                return self.fc2(h)

        return MLP(channels)

    @staticmethod
    def _bench_input(n: int, device: str, batch_sizes: int, seq_lengths: int, channels: int, **kwargs):
        """Create input for MLP benchmark."""
        return torch.randn(batch_sizes, seq_lengths, channels, device=device)

    @classmethod
    def _bench_wide(cls, modules: List[nn.Module], strategy: str):
        """Create wide version for given strategy."""
        if strategy == 'baseline':
            return None
        return cls.from_modules(modules, strategy=strategy)


__all__ = ['WideMLP']
