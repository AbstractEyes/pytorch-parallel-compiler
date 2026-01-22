"""
WideRNN - N parallel vanilla RNN layers with fused input projections.

Requires additional optimizations and a custom kernel fusion for multi-layer RNNs.

Simpler than GRU/LSTM - single tanh activation, no gates.
Expected speedup: 2-4x (similar to GRU/LSTM, crossover at N=16)

Strategies:
- 'fused': Pre-compute input projections via einsum (FASTEST)
- 'sequential': N separate RNN calls (baseline)

Input/Output Format (v0.6.0):
- Input:  [N, B, T, input_size]  (N-first)
- Output: [N, B, T, hidden_size] (N-first)
- h_n:    [N, B, hidden_size]    (N-first)

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Dict, Any, Union, Optional, Tuple
from enum import Enum

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class RNNStrategy(Enum):
    FUSED = 'fused'
    SEQUENTIAL = 'sequential'
    AUTO = 'auto'


class WideRNN(nn.Module):
    """
    N parallel RNN modules with fused input projections.

    Simple RNN: h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)

    Input shape:  [N, B, T, input_size]
    Output shape: [N, B, T, hidden_size]
    Hidden shape: [N, B, hidden_size]
    """

    BENCHMARK_STRATEGIES = ['baseline', 'fused', 'sequential']

    def __init__(
        self,
        n: int,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = 'tanh',
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        strategy: Union[str, RNNStrategy] = 'auto',
    ):
        super().__init__()

        if num_layers != 1:
            raise NotImplementedError("WideRNN only supports num_layers=1")
        if bidirectional:
            raise NotImplementedError("WideRNN only supports unidirectional")
        if nonlinearity not in ['tanh', 'relu']:
            raise ValueError(f"nonlinearity must be 'tanh' or 'relu', got {nonlinearity}")

        self.n = n
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.has_bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        if isinstance(strategy, str):
            strategy = RNNStrategy(strategy)
        if strategy == RNNStrategy.AUTO:
            strategy = RNNStrategy.FUSED

        self._strategy = strategy
        self._use_fused = (strategy == RNNStrategy.FUSED)

        # Activation function
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu

        H = hidden_size
        I = input_size

        # Weights: [N, H, I] for input, [N, H, H] for hidden
        self.weight_ih = nn.Parameter(torch.empty(n, H, I))
        self.weight_hh = nn.Parameter(torch.empty(n, H, H))

        if bias:
            self.bias_ih = nn.Parameter(torch.empty(n, H))
            self.bias_hh = nn.Parameter(torch.empty(n, H))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self._reset_parameters()

    @property
    def strategy(self) -> RNNStrategy:
        return self._strategy

    def _reset_parameters(self):
        """Initialize parameters like nn.RNN."""
        import math
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in [self.weight_ih, self.weight_hh]:
            nn.init.uniform_(weight, -stdv, stdv)
        if self.has_bias:
            nn.init.uniform_(self.bias_ih, -stdv, stdv)
            nn.init.uniform_(self.bias_hh, -stdv, stdv)

    def forward(
        self,
        x: Tensor,
        h_0: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with N-first format.

        Args:
            x: [N, B, T, input_size]
            h_0: Optional [N, B, hidden_size] initial hidden state

        Returns:
            output: [N, B, T, hidden_size]
            h_n: [N, B, hidden_size] final hidden state
        """
        if self._use_fused:
            return self._forward_fused(x, h_0)
        else:
            return self._forward_sequential(x, h_0)

    def _forward_fused(
        self,
        x: Tensor,
        h_0: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Fused RNN: pre-compute all input projections.

        h_t = activation(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
        """
        N, B, T, I = x.shape
        H = self.hidden_size

        # Initialize hidden state
        if h_0 is None:
            h_t = torch.zeros(N, B, H, device=x.device, dtype=x.dtype)
        else:
            h_t = h_0

        # Pre-compute ALL input projections: [N, B, T, I] @ [N, H, I].T -> [N, B, T, H]
        x_projected = torch.einsum('nbti,nhi->nbth', x, self.weight_ih)
        if self.bias_ih is not None:
            x_projected = x_projected + self.bias_ih.view(N, 1, 1, H)

        # Recurrence over timesteps
        outputs = []
        for t in range(T):
            # x_t already projected: [N, B, H]
            x_t_proj = x_projected[:, :, t, :]

            # Hidden projection: [N, B, H] @ [N, H, H].T -> [N, B, H]
            h_proj = torch.einsum('nbh,noh->nbo', h_t, self.weight_hh)
            if self.bias_hh is not None:
                h_proj = h_proj + self.bias_hh.view(N, 1, H)

            # New hidden state: h_t = activation(x_proj + h_proj)
            h_t = self._activation(x_t_proj + h_proj)

            outputs.append(h_t)

        # Stack outputs: T x [N, B, H] -> [N, B, T, H]
        output = torch.stack(outputs, dim=2)

        return output, h_t

    def _forward_sequential(
        self,
        x: Tensor,
        h_0: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Sequential RNN (baseline)."""
        N, B, T, I = x.shape
        H = self.hidden_size

        outputs = []
        h_finals = []

        for i in range(N):
            # Initialize hidden for this model
            if h_0 is None:
                h_t = torch.zeros(B, H, device=x.device, dtype=x.dtype)
            else:
                h_t = h_0[i]

            model_outputs = []
            for t in range(T):
                x_t = x[i, :, t, :]  # [B, I]

                # x projection
                x_proj = F.linear(x_t, self.weight_ih[i], self.bias_ih[i] if self.bias_ih is not None else None)

                # h projection
                h_proj = F.linear(h_t, self.weight_hh[i], self.bias_hh[i] if self.bias_hh is not None else None)

                # New hidden
                h_t = self._activation(x_proj + h_proj)

                model_outputs.append(h_t)

            outputs.append(torch.stack(model_outputs, dim=1))  # [B, T, H]
            h_finals.append(h_t)

        # Stack: N x [B, T, H] -> [N, B, T, H]
        output = torch.stack(outputs, dim=0)
        h_n = torch.stack(h_finals, dim=0)

        return output, h_n

    @classmethod
    def from_modules(cls, modules: List[nn.RNN], strategy: str = 'fused') -> 'WideRNN':
        """Create from N existing RNN modules."""
        n = len(modules)
        t = modules[0]

        # Map PyTorch's mode ('RNN_TANH', 'RNN_RELU') to our format ('tanh', 'relu')
        mode_map = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}
        nonlinearity = mode_map.get(t.mode, t.mode.lower().replace('rnn_', ''))

        wide = cls(
            n=n,
            input_size=t.input_size,
            hidden_size=t.hidden_size,
            num_layers=t.num_layers,
            nonlinearity=nonlinearity,
            bias=t.bias,
            batch_first=t.batch_first,
            dropout=t.dropout,
            bidirectional=t.bidirectional,
            strategy=strategy,
        )

        wide = wide.to(device=t.weight_ih_l0.device, dtype=t.weight_ih_l0.dtype)

        # Copy weights
        # PyTorch RNN weights are [hidden_size, input_size] and [hidden_size, hidden_size]
        # For einsum 'nbti,nhi->nbth' we need weight[n, h, i] = [N, hidden, input]
        # For F.linear we also need [out, in] = [hidden, input]
        # So NO transpose needed!
        with torch.no_grad():
            for i, m in enumerate(modules):
                wide.weight_ih[i] = m.weight_ih_l0  # Keep as [H, I]
                wide.weight_hh[i] = m.weight_hh_l0  # Keep as [H, H]
                if m.bias:
                    wide.bias_ih[i] = m.bias_ih_l0
                    wide.bias_hh[i] = m.bias_hh_l0

        return wide

    def __repr__(self):
        return (f"WideRNN({self.n}x[{self.input_size}, {self.hidden_size}], "
                f"nonlinearity={self.nonlinearity}, strategy={self._strategy.value})")

    # =========================================================================
    # BENCHMARK INTERFACE
    # =========================================================================

    BENCHMARK_SWEEPS = {
        'quick': {'n_values': [8, 16, 32]},  # RNN needs higher N
        'full': {'n_values': [4, 8, 16, 32, 64]},
        'ci': {'n_values': [8, 16]},
    }

    @classmethod
    def benchmark_job(cls, preset: str = 'full'):
        """Create benchmark job for WideRNN."""
        from ..benchmark.benchmark_schema import BenchmarkJob, SweepParams

        sweep_config = cls.BENCHMARK_SWEEPS.get(preset, cls.BENCHMARK_SWEEPS['full'])
        sweep = SweepParams(
            n_values=sweep_config['n_values'],
            batch_sizes=[8],
            seq_lengths=[32],
            d_model=[64],
            hidden_sizes=[128],
        )

        return BenchmarkJob(
            name=f'rnn_{preset}',
            primitive='rnn',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
            validate_fn=cls._bench_validate,  # Custom: handles tuple output
            validate_rtol=1e-3,  # Relaxed for RNN accumulation
        )

    @staticmethod
    def _bench_model(d_model=64, hidden_sizes=128, **kwargs):
        """Create a single RNN module."""
        return nn.RNN(input_size=d_model, hidden_size=hidden_sizes, batch_first=True)

    @staticmethod
    def _bench_input(n: int, device: str, batch_sizes: int, seq_lengths: int, d_model: int, **kwargs):
        """Create input tensor [B, T, D]."""
        return torch.randn(batch_sizes, seq_lengths, d_model, device=device)

    @classmethod
    def _bench_wide(cls, modules: List[nn.Module], strategy: str):
        """Create wide version for given strategy."""
        if strategy == 'baseline':
            return None
        return cls.from_modules(modules, strategy=strategy)

    @staticmethod
    def _bench_validate(
        wide_output: Tuple[Tensor, Tensor],
        baseline_outputs: List[Tuple[Tensor, Tensor]],
        rtol: float = 1e-3,
        atol: float = 1e-3,
    ) -> Tuple[bool, str]:
        """
        Validate wide output matches stacked baseline outputs.

        Note: RNNs accumulate numerical differences over timesteps,
        so we use relaxed tolerances (1e-3 vs 1e-5 for feedforward).

        Args:
            wide_output: (output, h_n) tuple from WideRNN
                output: [N, B, T, H]
                h_n: [N, B, H]
            baseline_outputs: List of N x (output, h_n) tuples from individual RNNs
                each output: [B, T, H]
                each h_n: [B, H]
        """
        # Extract outputs and hidden states
        wide_out, wide_h = wide_output

        # Baseline outputs are tuples (output, h_n)
        baseline_outs = [out[0] for out in baseline_outputs]
        baseline_hs = [out[1] for out in baseline_outputs]

        n = len(baseline_outputs)

        # Stack baseline outputs: N x [B, T, H] -> [N, B, T, H]
        stacked_out = torch.stack(baseline_outs, dim=0)

        # Stack baseline hidden states: N x [num_layers, B, H] -> [N, num_layers, B, H]
        # For single-layer RNN (num_layers=1), squeeze to [N, B, H]
        stacked_h = torch.stack(baseline_hs, dim=0)
        if stacked_h.shape[1] == 1:  # num_layers dimension
            stacked_h = stacked_h.squeeze(1)  # [N, 1, B, H] -> [N, B, H]

        # Validate shapes
        if wide_out.shape != stacked_out.shape:
            return False, f"Output shape mismatch: {wide_out.shape} vs {stacked_out.shape}"
        if wide_h.shape != stacked_h.shape:
            return False, f"Hidden state shape mismatch: {wide_h.shape} vs {stacked_h.shape}"

        # Validate output values
        out_diff = (wide_out - stacked_out).abs()
        out_max_diff = out_diff.max().item()
        out_mean_diff = out_diff.mean().item()

        out_close = torch.allclose(wide_out, stacked_out, rtol=rtol, atol=atol)

        # Validate hidden state values
        h_diff = (wide_h - stacked_h).abs()
        h_max_diff = h_diff.max().item()
        h_mean_diff = h_diff.mean().item()

        h_close = torch.allclose(wide_h, stacked_h, rtol=rtol, atol=atol)

        if not out_close:
            return False, f"Output mismatch: max={out_max_diff:.6f}, mean={out_mean_diff:.6f}"
        if not h_close:
            return False, f"Hidden state mismatch: max={h_max_diff:.6f}, mean={h_mean_diff:.6f}"

        return True, "OK"


__all__ = ['WideRNN', 'RNNStrategy']
