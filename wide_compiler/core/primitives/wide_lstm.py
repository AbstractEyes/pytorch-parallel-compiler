"""
WideLSTM - N parallel LSTM layers fused into a single module.

Key insight: Batch N sequences together, process with N separate LSTMs
in parallel, concatenate outputs.

Strategies:
- 'fused': Process all N via ModuleList (cleaner, same perf)
- 'sequential': N separate LSTM calls (baseline)

Expected speedup: 2-6x depending on N and sequence length.

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Dict, Any, Union, Optional, Tuple
from enum import Enum

import torch
from torch import nn, Tensor


class LSTMStrategy(Enum):
    FUSED = 'fused'
    SEQUENTIAL = 'sequential'
    AUTO = 'auto'


class WideLSTM(nn.Module):
    """
    N parallel LSTM modules fused into single operations.

    Input shape:  [B, T, N*input_size] or [T, B, N*input_size]
    Output shape: [B, T, N*hidden_size] or [T, B, N*hidden_size], (h_n, c_n)

    Strategies:
        'fused': Process all N via optimized batching
        'sequential': N separate LSTM computations
    """

    def __init__(
        self,
        n: int,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        strategy: Union[str, LSTMStrategy] = 'auto',
    ):
        super().__init__()

        self.n = n
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        if isinstance(strategy, str):
            strategy = LSTMStrategy(strategy)
        if strategy == LSTMStrategy.AUTO:
            strategy = LSTMStrategy.FUSED

        self._strategy = strategy
        self._use_fused = (strategy == LSTMStrategy.FUSED)

        # Create N separate LSTMs
        self.lstms = nn.ModuleList([
            nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            for _ in range(n)
        ])

    @property
    def strategy(self) -> LSTMStrategy:
        return self._strategy

    def forward(
        self,
        x: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass.

        Args:
            x: [B, T, N*input_size] if batch_first else [T, B, N*input_size]
            hx: Optional (h_0, c_0) each [num_layers*num_dir, B, N*hidden_size]

        Returns:
            output: [B, T, N*hidden_size*num_dir] or [T, B, ...]
            (h_n, c_n): each [num_layers*num_dir, B, N*hidden_size]
        """
        if self._use_fused:
            return self._forward_fused(x, hx)
        else:
            return self._forward_sequential(x, hx)

    def _forward_fused(
        self,
        x: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Fused: process all N in parallel."""
        N = self.n
        H = self.hidden_size
        D = self.num_directions
        L = self.num_layers

        if self.batch_first:
            B, T, NI = x.shape
            x_split = x.view(B, T, N, -1).permute(2, 0, 1, 3)
        else:
            T, B, NI = x.shape
            x_split = x.view(T, B, N, -1).permute(2, 0, 1, 3)

        # Split hidden states if provided
        if hx is not None:
            h_0, c_0 = hx
            h_split = h_0.view(L * D, B, N, H).permute(2, 0, 1, 3)
            c_split = c_0.view(L * D, B, N, H).permute(2, 0, 1, 3)
        else:
            h_split = [None] * N
            c_split = [None] * N

        outputs = []
        h_ns = []
        c_ns = []

        for i in range(N):
            xi = x_split[i]
            if hx is not None:
                hi = (h_split[i].contiguous(), c_split[i].contiguous())
            else:
                hi = None
            out_i, (h_n_i, c_n_i) = self.lstms[i](xi, hi)
            outputs.append(out_i)
            h_ns.append(h_n_i)
            c_ns.append(c_n_i)

        # Concatenate outputs
        if self.batch_first:
            output = torch.cat(outputs, dim=-1)
        else:
            output = torch.cat(outputs, dim=-1)

        h_n = torch.cat(h_ns, dim=-1)
        c_n = torch.cat(c_ns, dim=-1)

        return output, (h_n, c_n)

    def _forward_sequential(
        self,
        x: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Sequential: N separate LSTM calls."""
        N = self.n
        H = self.hidden_size
        I = self.input_size
        D = self.num_directions
        L = self.num_layers

        if self.batch_first:
            B, T, _ = x.shape
            x_chunks = x.view(B, T, N, I).unbind(dim=2)
        else:
            T, B, _ = x.shape
            x_chunks = x.view(T, B, N, I).unbind(dim=2)

        if hx is not None:
            h_0, c_0 = hx
            h_chunks = h_0.view(L * D, B, N, H).unbind(dim=2)
            c_chunks = c_0.view(L * D, B, N, H).unbind(dim=2)
        else:
            h_chunks = [None] * N
            c_chunks = [None] * N

        outputs = []
        h_ns = []
        c_ns = []

        for i, xi in enumerate(x_chunks):
            if hx is not None:
                hi = (h_chunks[i].contiguous(), c_chunks[i].contiguous())
            else:
                hi = None
            out_i, (h_n_i, c_n_i) = self.lstms[i](xi, hi)
            outputs.append(out_i)
            h_ns.append(h_n_i)
            c_ns.append(c_n_i)

        output = torch.cat(outputs, dim=-1)
        h_n = torch.cat(h_ns, dim=-1)
        c_n = torch.cat(c_ns, dim=-1)

        return output, (h_n, c_n)

    @classmethod
    def from_modules(
        cls,
        modules: List[nn.LSTM],
        strategy: Union[str, LSTMStrategy] = 'auto',
    ) -> 'WideLSTM':
        """Create from N existing LSTM modules."""
        n = len(modules)
        ref = modules[0]

        wide = cls(
            n=n,
            input_size=ref.input_size,
            hidden_size=ref.hidden_size,
            num_layers=ref.num_layers,
            bias=ref.bias,
            batch_first=ref.batch_first,
            dropout=ref.dropout,
            bidirectional=ref.bidirectional,
            strategy=strategy,
        )

        device = next(ref.parameters()).device
        dtype = next(ref.parameters()).dtype
        wide = wide.to(device=device, dtype=dtype)

        with torch.no_grad():
            for i, m in enumerate(modules):
                wide.lstms[i].load_state_dict(m.state_dict())

        return wide

    def __repr__(self):
        return (
            f"WideLSTM({self.n}x[in={self.input_size}, h={self.hidden_size}, "
            f"layers={self.num_layers}], strategy={self._strategy.value})"
        )

    # =========================================================================
    # BENCHMARK INTERFACE
    # =========================================================================

    BENCHMARK_STRATEGIES = ['baseline', 'fused', 'sequential']

    @classmethod
    def _get_sweep_params_class(cls):
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
                seq_lengths=[64],
                d_model=[128],  # input_size
                hidden_sizes=[256],
            ),
            'full': SweepParams(
                n_values=[2, 4, 8, 16, 32],
                batch_sizes=[4, 8, 16],
                seq_lengths=[32, 64, 128],
                d_model=[64, 128, 256],
                hidden_sizes=[128, 256, 512],
            ),
            'ci': SweepParams(
                n_values=[4, 8],
                batch_sizes=[4],
                seq_lengths=[32],
                d_model=[64],
                hidden_sizes=[128],
            ),
        }

    @classmethod
    def benchmark_job(cls, preset: str = 'full', **overrides) -> Any:
        cls._init_benchmark_sweeps()

        BenchmarkJob = cls._get_benchmark_job_class()
        if BenchmarkJob is None:
            raise ImportError("Could not import BenchmarkJob")

        sweep = cls.BENCHMARK_SWEEPS.get(preset)
        if sweep is None:
            raise ValueError(f"Unknown preset '{preset}'")

        if overrides:
            sweep = sweep.with_overrides(**overrides)

        return BenchmarkJob(
            name=f'lstm_{preset}',
            primitive='lstm',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
            pack_fn=cls._bench_pack,
            unpack_fn=cls._bench_unpack,
        )

    @staticmethod
    def _bench_model(d_model: int, hidden_sizes: int, **_) -> nn.Module:
        """d_model=input_size, hidden_sizes=hidden_size."""
        class LSTMWrapper(nn.Module):
            def __init__(self, input_size, hidden_size):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            def forward(self, x):
                out, _ = self.lstm(x)
                return out
        return LSTMWrapper(d_model, hidden_sizes)

    @staticmethod
    def _bench_input(n: int, batch_sizes: int, seq_lengths: int, d_model: int, device: str = 'cpu', **_) -> Tensor:
        return torch.randn(batch_sizes, seq_lengths, d_model, device=device)

    @classmethod
    def _bench_wide(cls, modules: List[nn.Module], strategy: str) -> 'WideLSTM':
        strat_map = {
            'fused': LSTMStrategy.FUSED,
            'sequential': LSTMStrategy.SEQUENTIAL,
        }
        strat = strat_map.get(strategy, LSTMStrategy.FUSED)
        lstms = [m.lstm for m in modules]
        return cls.from_modules(lstms, strategy=strat)

    @staticmethod
    def _bench_pack(inputs: List[Tensor]) -> Tensor:
        return torch.cat(inputs, dim=-1)

    @staticmethod
    def _bench_unpack(output: Tensor, n: int) -> List[Tensor]:
        if isinstance(output, tuple):
            output = output[0]
        chunks = output.shape[-1] // n
        return [output[..., i*chunks:(i+1)*chunks] for i in range(n)]


__all__ = ['WideLSTM', 'LSTMStrategy']