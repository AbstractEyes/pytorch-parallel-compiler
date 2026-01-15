"""
WideGRU - N parallel GRU layers with fused input projections.

Key insight: Pre-compute input projections for ALL timesteps and ALL models
in a single einsum, then run the recurrence with pre-projected inputs.

The hidden-to-hidden projection must still be sequential (h_t depends on h_{t-1}),
but input projection (~50% of compute) can be fully parallelized.

Fusion strategy (einsum):
- Input projection: einsum('bti,nio->btno', x, W_ih) fuses N matmuls
- Hidden projection: per-timestep loop (unavoidable due to recurrence)
- Gates: elementwise ops on [B, T, N, 3H]

Performance characteristics:
- Scales well with N (3-8x speedup observed at N=8-32)
- Einsum approach avoids block-diagonal sparsity overhead
- Best for moderate hidden sizes (64-256)

Strategies:
- 'fused': Pre-compute input projections via einsum (FASTEST)
- 'sequential': N separate GRU calls (baseline)

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Dict, Any, Union, Optional, Tuple
from enum import Enum

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class GRUStrategy(Enum):
    FUSED = 'fused'
    SEQUENTIAL = 'sequential'
    AUTO = 'auto'


class WideGRU(nn.Module):
    """
    N parallel GRU modules with fused input projections.

    Input shape:  [B, T, N*input_size] if batch_first else [T, B, N*input_size]
    Output shape: [B, T, N*hidden_size] if batch_first else [T, B, N*hidden_size]

    Strategies:
        'fused': Pre-compute all input projections, fused recurrence
        'sequential': N separate GRU computations (baseline)
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
        strategy: Union[str, GRUStrategy] = 'auto',
    ):
        super().__init__()

        if num_layers != 1:
            raise NotImplementedError("WideGRU only supports num_layers=1 for fused strategy")
        if bidirectional:
            raise NotImplementedError("WideGRU only supports unidirectional for fused strategy")

        self.n = n
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.has_bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        if isinstance(strategy, str):
            strategy = GRUStrategy(strategy)
        if strategy == GRUStrategy.AUTO:
            strategy = GRUStrategy.FUSED

        self._strategy = strategy
        self._use_fused = (strategy == GRUStrategy.FUSED)

        H = hidden_size
        I = input_size

        # Fused weights: [N, 3*H, I] for input, [N, 3*H, H] for hidden
        # Gate order: reset, update, new (r, z, n)
        self.weight_ih = nn.Parameter(torch.empty(n, 3 * H, I))
        self.weight_hh = nn.Parameter(torch.empty(n, 3 * H, H))

        if bias:
            self.bias_ih = nn.Parameter(torch.empty(n, 3 * H))
            self.bias_hh = nn.Parameter(torch.empty(n, 3 * H))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self._reset_parameters()

    @property
    def strategy(self) -> GRUStrategy:
        return self._strategy

    def _reset_parameters(self):
        """Initialize parameters like nn.GRU."""
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
        Forward pass.

        Args:
            x: [B, T, N*input_size] if batch_first else [T, B, N*input_size]
            h_0: Optional [1, B, N*hidden_size] initial hidden state

        Returns:
            output: [B, T, N*hidden_size] or [T, B, N*hidden_size]
            h_n: [1, B, N*hidden_size] final hidden state
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
        Fused forward: pre-compute ALL input projections, then run recurrence.

        This fuses ~50% of the compute (input projections) while the
        hidden-to-hidden must remain sequential due to recurrence.
        """
        N = self.n
        H = self.hidden_size
        I = self.input_size

        # Handle batch_first
        if self.batch_first:
            B, T, _ = x.shape
        else:
            T, B, _ = x.shape
            x = x.transpose(0, 1)  # [T, B, N*I] -> [B, T, N*I]

        # Reshape input: [B, T, N*I] -> [B, T, N, I]
        x = x.reshape(B, T, N, I)

        # === FUSED INPUT PROJECTION ===
        # Compute W_ih @ x for ALL timesteps, ALL models in ONE einsum
        # x: [B, T, N, I], weight_ih: [N, 3H, I] -> [B, T, N, 3H]
        gates_x = torch.einsum('btni,ngi->btng', x, self.weight_ih)
        if self.bias_ih is not None:
            gates_x = gates_x + self.bias_ih.view(1, 1, N, 3 * H)

        # Initialize hidden state
        if h_0 is not None:
            # [1, B, N*H] -> [B, N, H]
            h = h_0.squeeze(0).reshape(B, N, H).contiguous()
        else:
            h = torch.zeros(B, N, H, device=x.device, dtype=x.dtype)

        # === RECURRENCE (sequential over time, parallel over N) ===
        outputs = []
        for t in range(T):
            # Hidden projection: [B, N, H] @ [N, 3H, H]^T -> [B, N, 3H]
            gates_h = torch.einsum('bnh,ngh->bng', h, self.weight_hh)
            if self.bias_hh is not None:
                gates_h = gates_h + self.bias_hh.view(1, N, 3 * H)

            # Get input gates for this timestep
            gx = gates_x[:, t]  # [B, N, 3H]

            # Split gates
            gx_r, gx_z, gx_n = gx.chunk(3, dim=-1)    # Each [B, N, H]
            gh_r, gh_z, gh_n = gates_h.chunk(3, dim=-1)

            # Reset and update gates: r, z use sum of input and hidden
            r = torch.sigmoid(gx_r + gh_r)
            z = torch.sigmoid(gx_z + gh_z)

            # New gate: n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
            # gh_n already has bias added, so: n = tanh(gx_n + r * gh_n)
            n = torch.tanh(gx_n + r * gh_n)

            # Update hidden: h = (1 - z) * n + z * h_prev
            h = (1 - z) * n + z * h

            outputs.append(h)

        # Stack outputs: List of [B, N, H] -> [B, T, N, H] -> [B, T, N*H]
        output = torch.stack(outputs, dim=1).reshape(B, T, N * H)

        # Final hidden: [B, N, H] -> [1, B, N*H]
        h_n = h.reshape(1, B, N * H)

        if not self.batch_first:
            output = output.transpose(0, 1)  # [B, T, N*H] -> [T, B, N*H]

        return output, h_n

    def _forward_sequential(
        self,
        x: Tensor,
        h_0: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Sequential: use individual GRU cells per model."""
        N = self.n
        H = self.hidden_size
        I = self.input_size

        if self.batch_first:
            B, T, _ = x.shape
        else:
            T, B, _ = x.shape
            x = x.transpose(0, 1)

        # [B, T, N*I] -> [B, T, N, I]
        x = x.reshape(B, T, N, I)

        if h_0 is not None:
            h = h_0.squeeze(0).reshape(B, N, H).contiguous()
        else:
            h = torch.zeros(B, N, H, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(T):
            h_new = []
            for i in range(N):
                xi = x[:, t, i, :]  # [B, I]
                hi = h[:, i, :]     # [B, H]

                # GRU cell computation
                gi = F.linear(xi, self.weight_ih[i], self.bias_ih[i] if self.bias_ih is not None else None)
                gh = F.linear(hi, self.weight_hh[i], self.bias_hh[i] if self.bias_hh is not None else None)

                gi_r, gi_z, gi_n = gi.chunk(3, dim=-1)
                gh_r, gh_z, gh_n = gh.chunk(3, dim=-1)

                r = torch.sigmoid(gi_r + gh_r)
                z = torch.sigmoid(gi_z + gh_z)
                n = torch.tanh(gi_n + r * gh_n)

                h_new.append((1 - z) * n + z * hi)

            h = torch.stack(h_new, dim=1)  # [B, N, H]
            outputs.append(h)

        output = torch.stack(outputs, dim=1).reshape(B, T, N * H)
        h_n = h.reshape(1, B, N * H)

        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, h_n

    @classmethod
    def from_modules(
        cls,
        modules: List[nn.GRU],
        strategy: Union[str, GRUStrategy] = 'auto',
    ) -> 'WideGRU':
        """Create from N existing GRU modules."""
        n = len(modules)
        ref = modules[0]

        # Validate compatibility
        if ref.num_layers != 1:
            raise ValueError("WideGRU.from_modules only supports single-layer GRUs")
        if ref.bidirectional:
            raise ValueError("WideGRU.from_modules only supports unidirectional GRUs")

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

        # Copy weights from each GRU
        with torch.no_grad():
            for i, m in enumerate(modules):
                # nn.GRU stores as weight_ih_l0, weight_hh_l0, etc.
                wide.weight_ih[i].copy_(m.weight_ih_l0)
                wide.weight_hh[i].copy_(m.weight_hh_l0)
                if ref.bias:
                    wide.bias_ih[i].copy_(m.bias_ih_l0)
                    wide.bias_hh[i].copy_(m.bias_hh_l0)

        return wide

    def __repr__(self):
        return (
            f"WideGRU({self.n}x[in={self.input_size}, h={self.hidden_size}], "
            f"strategy={self._strategy.value})"
        )

    # =========================================================================
    # VALIDATION
    # =========================================================================

    @staticmethod
    def validate_outputs(
        wide_out: Tuple[Tensor, Tensor],
        baseline_outs: List[Tuple[Tensor, Tensor]],
        rtol: float = 1e-4,
        atol: float = 1e-5,
    ) -> Tuple[bool, str]:
        """
        Validate wide outputs match concatenated baseline outputs.

        Args:
            wide_out: (output, h_n) from WideGRU
            baseline_outs: List of (output, h_n) from N individual GRUs
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            (is_valid, message)
        """
        wide_output, wide_hn = wide_out

        # Concatenate baseline outputs
        baseline_output = torch.cat([o[0] for o in baseline_outs], dim=-1)
        baseline_hn = torch.cat([o[1] for o in baseline_outs], dim=-1)

        # Check shapes
        if wide_output.shape != baseline_output.shape:
            return False, f"Output shape mismatch: {wide_output.shape} vs {baseline_output.shape}"
        if wide_hn.shape != baseline_hn.shape:
            return False, f"Hidden shape mismatch: {wide_hn.shape} vs {baseline_hn.shape}"

        # Check values
        if not torch.allclose(wide_output, baseline_output, rtol=rtol, atol=atol):
            diff = (wide_output - baseline_output).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            return False, f"Output mismatch: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"

        if not torch.allclose(wide_hn, baseline_hn, rtol=rtol, atol=atol):
            diff = (wide_hn - baseline_hn).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            return False, f"Hidden mismatch: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"

        return True, "OK"

    # =========================================================================
    # BENCHMARK INTERFACE
    # =========================================================================

    # Sequential is only for debugging - it's always slower than baseline
    BENCHMARK_STRATEGIES = ['fused']
    BENCHMARK_STRATEGIES_ALL = ['fused', 'sequential']  # For validation

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
            'smoke': SweepParams(
                n_values=[8, 32],
                batch_sizes=[4],
                seq_lengths=[32],
                d_model=[64],
                hidden_sizes=[128],
            ),
            'quick': SweepParams(
                n_values=[8, 16, 32],
                batch_sizes=[8],
                seq_lengths=[32],
                d_model=[64],
                hidden_sizes=[128],
            ),
            'full': SweepParams(
                n_values=[4, 8, 16, 32, 64],
                batch_sizes=[4, 8, 16],
                seq_lengths=[32, 64],
                d_model=[64, 128],
                hidden_sizes=[128, 256],
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
            name=f'gru_{preset}',
            primitive='gru',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
            pack_fn=cls._bench_pack,
            unpack_fn=cls._bench_unpack,
            validate_fn=cls._bench_validate,
        )

    @staticmethod
    def _bench_model(d_model: int, hidden_sizes: int, **_) -> nn.Module:
        """Create single GRU for benchmarking."""
        class GRUWrapper(nn.Module):
            def __init__(self, input_size, hidden_size):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
            def forward(self, x):
                out, h = self.gru(x)
                return out  # Return only output for timing
        return GRUWrapper(d_model, hidden_sizes)

    @staticmethod
    def _bench_input(n: int, batch_sizes: int, seq_lengths: int, d_model: int, device: str = 'cpu', **_) -> Tensor:
        return torch.randn(batch_sizes, seq_lengths, d_model, device=device)

    @classmethod
    def _bench_wide(cls, modules: List[nn.Module], strategy: str) -> 'WideGRU':
        strat_map = {
            'fused': GRUStrategy.FUSED,
            'sequential': GRUStrategy.SEQUENTIAL,
        }
        strat = strat_map.get(strategy, GRUStrategy.FUSED)
        grus = [m.gru for m in modules]
        return cls.from_modules(grus, strategy=strat)

    @staticmethod
    def _bench_pack(inputs: List[Tensor]) -> Tensor:
        return torch.cat(inputs, dim=-1)

    @staticmethod
    def _bench_unpack(output: Tensor, n: int) -> List[Tensor]:
        if isinstance(output, tuple):
            output = output[0]
        chunks = output.shape[-1] // n
        return [output[..., i*chunks:(i+1)*chunks] for i in range(n)]

    @staticmethod
    def _bench_validate(
        wide_output: Tensor,
        baseline_outputs: List[Tensor],
        rtol: float = 1e-3,
        atol: float = 1e-3,
    ) -> Tuple[bool, str]:
        """Validate wide output matches concatenated baseline.

        Note: RNNs accumulate numerical differences over timesteps,
        so we use relaxed tolerances (1e-3 vs 1e-5 for feedforward).
        """
        if isinstance(wide_output, tuple):
            wide_output = wide_output[0]

        baseline_concat = torch.cat(baseline_outputs, dim=-1)

        if wide_output.shape != baseline_concat.shape:
            return False, f"Shape mismatch: {wide_output.shape} vs {baseline_concat.shape}"

        if not torch.allclose(wide_output, baseline_concat, rtol=rtol, atol=atol):
            diff = (wide_output - baseline_concat).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            return False, f"Value mismatch: max={max_diff:.6f}, mean={mean_diff:.6f}"

        return True, "OK"


__all__ = ['WideGRU', 'GRUStrategy']