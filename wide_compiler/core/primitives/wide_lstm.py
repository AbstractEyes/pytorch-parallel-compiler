"""
WideLSTM - N parallel LSTM layers fused into a single module.

Fusion strategy: Block-diagonal weight matrices allow N separate LSTMs
to be computed with a single large matmul instead of N small ones.

For N LSTMs with input [B, T, N*I]:
- Input projection:  [B, T, N*I] @ [N*I, N*4H] -> [B, T, N*4H]
- Hidden projection: [B, N*H] @ [N*H, N*4H] -> [B, N*4H]
- Gates: elementwise ops on [B, N*4H] - trivially parallel

Performance characteristics (A100, compiled):
- N=4:  **3.3x speedup** - sweet spot, kernel launch savings dominate
- N=8:  0.5x (2x SLOWER) - block-diagonal overhead exceeds benefits
- N=16: 1.0x breakeven
- N=32: 0.8x slower
- N=64: 0.6x slower

Recommendation: Use WideLSTM ONLY for N <= 4. For larger N, use:
- WideGRU (better scaling via einsum, 5-8x at N=8-32)
- Multiple WideLSTM(N=4) groups
- Sequential baseline

Limitations:
- Block-diagonal weights contain O(N²) zeros, inefficient at N > 4
- Currently supports: num_layers=1, bidirectional=False, batch_first=True

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Dict, Any, Union, Optional, Tuple
from enum import Enum

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class LSTMStrategy(Enum):
    FUSED = 'fused'
    SEQUENTIAL = 'sequential'
    AUTO = 'auto'


class WideLSTM(nn.Module):
    """
    N parallel LSTM modules fused via block-diagonal weight matrices.

    Input shape:  [B, T, N*input_size] (batch_first=True)
    Output shape: [B, T, N*hidden_size], (h_n, c_n)

    Weight layout:
        weight_ih: [N*4*H, N*I] block-diagonal
        weight_hh: [N*4*H, N*H] block-diagonal
        bias: [N*4*H]
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

        if num_layers != 1:
            raise NotImplementedError("WideLSTM currently only supports num_layers=1")
        if bidirectional:
            raise NotImplementedError("WideLSTM currently only supports bidirectional=False")
        if not batch_first:
            raise NotImplementedError("WideLSTM currently only supports batch_first=True")

        self.n = n
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        if isinstance(strategy, str):
            strategy = LSTMStrategy(strategy)
        if strategy == LSTMStrategy.AUTO:
            strategy = LSTMStrategy.FUSED

        self._strategy = strategy

        # Fused weights: block-diagonal structure
        # weight_ih: [N*4H, N*I] with N blocks of [4H, I] on diagonal
        # weight_hh: [N*4H, N*H] with N blocks of [4H, H] on diagonal
        I, H, N = input_size, hidden_size, n

        self.weight_ih = nn.Parameter(torch.empty(N * 4 * H, N * I))
        self.weight_hh = nn.Parameter(torch.empty(N * 4 * H, N * H))

        if bias:
            self.bias_ih = nn.Parameter(torch.empty(N * 4 * H))
            self.bias_hh = nn.Parameter(torch.empty(N * 4 * H))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self._init_weights()

    def _init_weights(self):
        """Initialize with block-diagonal structure (zeros off-diagonal)."""
        I, H, N = self.input_size, self.hidden_size, self.n

        # Initialize to zeros (ensures block-diagonal structure)
        nn.init.zeros_(self.weight_ih)
        nn.init.zeros_(self.weight_hh)

        # Initialize each block with standard LSTM init
        stdv = 1.0 / (H ** 0.5)
        for i in range(N):
            # weight_ih block [i]: rows [i*4H : (i+1)*4H], cols [i*I : (i+1)*I]
            self.weight_ih.data[i*4*H:(i+1)*4*H, i*I:(i+1)*I].uniform_(-stdv, stdv)
            # weight_hh block [i]: rows [i*4H : (i+1)*4H], cols [i*H : (i+1)*H]
            self.weight_hh.data[i*4*H:(i+1)*4*H, i*H:(i+1)*H].uniform_(-stdv, stdv)

        if self.bias:
            nn.init.zeros_(self.bias_ih)
            nn.init.zeros_(self.bias_hh)

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
            x: [B, T, N*input_size]
            hx: Optional (h_0, c_0) each [1, B, N*hidden_size]

        Returns:
            output: [B, T, N*hidden_size]
            (h_n, c_n): each [1, B, N*hidden_size]
        """
        if self._strategy == LSTMStrategy.SEQUENTIAL:
            return self._forward_sequential(x, hx)
        else:
            return self._forward_fused(x, hx)

    def _forward_fused(
        self,
        x: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Fused forward: block-diagonal matmuls + sequential timesteps."""
        B, T, _ = x.shape
        N, H = self.n, self.hidden_size

        # Initialize hidden states
        if hx is not None:
            h_t = hx[0].squeeze(0)  # [B, N*H]
            c_t = hx[1].squeeze(0)  # [B, N*H]
        else:
            h_t = x.new_zeros(B, N * H)
            c_t = x.new_zeros(B, N * H)

        # Precompute input projections for all timesteps: [B, T, N*4H]
        # This is the key optimization - one big matmul instead of T*N small ones
        gates_x = F.linear(x, self.weight_ih, self.bias_ih)  # [B, T, N*4H]

        outputs = []

        for t in range(T):
            # Hidden projection: [B, N*H] @ [N*H, N*4H].T -> [B, N*4H]
            gates_h = F.linear(h_t, self.weight_hh, self.bias_hh)

            # Combined gates: [B, N*4H]
            gates = gates_x[:, t, :] + gates_h

            # Reshape to [B, N, 4H] then split into 4 gates of [B, N, H]
            # Layout after matmul: [i₀,f₀,g₀,o₀, i₁,f₁,g₁,o₁, ...]
            gates = gates.view(B, N, 4 * H)
            i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=-1)  # each [B, N, H]

            # Flatten back to [B, N*H]
            i_gate = i_gate.reshape(B, N * H)
            f_gate = f_gate.reshape(B, N * H)
            g_gate = g_gate.reshape(B, N * H)
            o_gate = o_gate.reshape(B, N * H)

            # Apply activations
            i_t = torch.sigmoid(i_gate)
            f_t = torch.sigmoid(f_gate)
            g_t = torch.tanh(g_gate)
            o_t = torch.sigmoid(o_gate)

            # Cell state update
            c_t = f_t * c_t + i_t * g_t

            # Hidden state
            h_t = o_t * torch.tanh(c_t)

            outputs.append(h_t)

        # Stack outputs: [B, T, N*H]
        output = torch.stack(outputs, dim=1)

        # Return with layer dimension: [1, B, N*H]
        return output, (h_t.unsqueeze(0), c_t.unsqueeze(0))

    def _forward_sequential(
        self,
        x: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Sequential forward: process each of N LSTMs separately (for validation)."""
        B, T, _ = x.shape
        N, I, H = self.n, self.input_size, self.hidden_size

        # Split input: [B, T, N*I] -> list of [B, T, I]
        x_chunks = x.view(B, T, N, I).unbind(dim=2)

        # Split hidden states if provided
        if hx is not None:
            h_0, c_0 = hx
            h_0 = h_0.squeeze(0)  # [B, N*H]
            c_0 = c_0.squeeze(0)
            h_chunks = h_0.view(B, N, H).unbind(dim=1)
            c_chunks = c_0.view(B, N, H).unbind(dim=1)
        else:
            h_chunks = [None] * N
            c_chunks = [None] * N

        outputs = []
        h_ns = []
        c_ns = []

        for i in range(N):
            # Extract block weights for this LSTM
            w_ih = self.weight_ih[i*4*H:(i+1)*4*H, i*I:(i+1)*I]  # [4H, I]
            w_hh = self.weight_hh[i*4*H:(i+1)*4*H, i*H:(i+1)*H]  # [4H, H]
            b_ih = self.bias_ih[i*4*H:(i+1)*4*H] if self.bias else None
            b_hh = self.bias_hh[i*4*H:(i+1)*4*H] if self.bias else None

            xi = x_chunks[i]  # [B, T, I]

            if h_chunks[i] is not None:
                h_t = h_chunks[i]  # [B, H]
                c_t = c_chunks[i]  # [B, H]
            else:
                h_t = xi.new_zeros(B, H)
                c_t = xi.new_zeros(B, H)

            out_seq = []
            for t in range(T):
                gates = F.linear(xi[:, t], w_ih, b_ih) + F.linear(h_t, w_hh, b_hh)
                i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=-1)

                i_t = torch.sigmoid(i_gate)
                f_t = torch.sigmoid(f_gate)
                g_t = torch.tanh(g_gate)
                o_t = torch.sigmoid(o_gate)

                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * torch.tanh(c_t)
                out_seq.append(h_t)

            outputs.append(torch.stack(out_seq, dim=1))  # [B, T, H]
            h_ns.append(h_t)
            c_ns.append(c_t)

        output = torch.cat(outputs, dim=-1)  # [B, T, N*H]
        h_n = torch.cat(h_ns, dim=-1).unsqueeze(0)  # [1, B, N*H]
        c_n = torch.cat(c_ns, dim=-1).unsqueeze(0)

        return output, (h_n, c_n)

    @classmethod
    def from_modules(
        cls,
        modules: List[nn.LSTM],
        strategy: Union[str, LSTMStrategy] = 'auto',
    ) -> 'WideLSTM':
        """Create WideLSTM from N existing nn.LSTM modules."""
        n = len(modules)
        ref = modules[0]

        # Validate compatibility
        for m in modules:
            assert m.input_size == ref.input_size
            assert m.hidden_size == ref.hidden_size
            assert m.num_layers == 1, "Only single-layer LSTM supported"
            assert not m.bidirectional, "Bidirectional not supported"
            assert m.batch_first == ref.batch_first

        wide = cls(
            n=n,
            input_size=ref.input_size,
            hidden_size=ref.hidden_size,
            num_layers=1,
            bias=ref.bias,
            batch_first=True,
            dropout=ref.dropout,
            bidirectional=False,
            strategy=strategy,
        )

        device = next(ref.parameters()).device
        dtype = next(ref.parameters()).dtype
        wide = wide.to(device=device, dtype=dtype)

        # Copy weights into block-diagonal structure
        I, H = ref.input_size, ref.hidden_size

        with torch.no_grad():
            # Reset to zeros
            wide.weight_ih.zero_()
            wide.weight_hh.zero_()
            if wide.bias:
                wide.bias_ih.zero_()
                wide.bias_hh.zero_()

            for i, m in enumerate(modules):
                # weight_ih: [4H, I] -> block at [i*4H:(i+1)*4H, i*I:(i+1)*I]
                wide.weight_ih[i*4*H:(i+1)*4*H, i*I:(i+1)*I] = m.weight_ih_l0
                # weight_hh: [4H, H] -> block at [i*4H:(i+1)*4*H, i*H:(i+1)*H]
                wide.weight_hh[i*4*H:(i+1)*4*H, i*H:(i+1)*H] = m.weight_hh_l0

                if wide.bias:
                    wide.bias_ih[i*4*H:(i+1)*4*H] = m.bias_ih_l0
                    wide.bias_hh[i*4*H:(i+1)*4*H] = m.bias_hh_l0

        return wide

    def __repr__(self):
        return (
            f"WideLSTM(n={self.n}, in={self.input_size}, h={self.hidden_size}, "
            f"strategy={self._strategy.value})"
        )

    # =========================================================================
    # BENCHMARK INTERFACE
    # =========================================================================

    BENCHMARK_STRATEGIES = ['fused']
    BENCHMARK_STRATEGIES_ALL = ['fused', 'sequential']

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
            name=f'lstm_{preset}',
            primitive='lstm',
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
        """Create single LSTM for benchmarking."""
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

    @staticmethod
    def _bench_validate(
        wide_output: Tensor,
        baseline_outputs: List[Tensor],
        rtol: float = 1e-3,
        atol: float = 1e-3,
    ) -> Tuple[bool, str]:
        """Validate wide output matches concatenated baseline."""
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


__all__ = ['WideLSTM', 'LSTMStrategy']


# =============================================================================
# TESTS
# =============================================================================

def _test_basic_forward():
    """Test basic forward pass shapes."""
    N, B, T, I, H = 4, 2, 10, 32, 64

    wide = WideLSTM(n=N, input_size=I, hidden_size=H)
    x = torch.randn(B, T, N * I)

    out, (h_n, c_n) = wide(x)

    assert out.shape == (B, T, N * H), f"Output shape: {out.shape}"
    assert h_n.shape == (1, B, N * H), f"h_n shape: {h_n.shape}"
    assert c_n.shape == (1, B, N * H), f"c_n shape: {c_n.shape}"
    print("✓ basic_forward")


def _test_from_modules():
    """Test from_modules factory preserves weights."""
    N, I, H = 4, 32, 64

    lstms = [nn.LSTM(I, H, batch_first=True) for _ in range(N)]
    wide = WideLSTM.from_modules(lstms)

    assert wide.n == N
    assert wide.input_size == I
    assert wide.hidden_size == H

    # Verify weights were copied to correct blocks
    for i, lstm in enumerate(lstms):
        w_ih_block = wide.weight_ih[i*4*H:(i+1)*4*H, i*I:(i+1)*I]
        w_hh_block = wide.weight_hh[i*4*H:(i+1)*4*H, i*H:(i+1)*H]

        assert torch.equal(w_ih_block, lstm.weight_ih_l0), f"weight_ih mismatch at {i}"
        assert torch.equal(w_hh_block, lstm.weight_hh_l0), f"weight_hh mismatch at {i}"

    print("✓ from_modules")


def _test_single_lstm_cell():
    """Test N=1 case matches nn.LSTM exactly - validates our manual cell impl."""
    B, T, I, H = 2, 8, 32, 64

    lstm = nn.LSTM(I, H, batch_first=True)
    x = torch.randn(B, T, I)

    with torch.no_grad():
        expected, _ = lstm(x)

    # Create WideLSTM with N=1 from this LSTM
    wide = WideLSTM.from_modules([lstm], strategy='fused')

    with torch.no_grad():
        actual, _ = wide(x)

    if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-5):
        diff = (actual - expected).abs()
        print(f"  N=1 test failed!")
        print(f"  Max diff: {diff.max().item():.6f}")
        print(f"  Mean diff: {diff.mean().item():.6f}")

        # Check per-timestep
        for t in range(min(T, 4)):
            diff_t = (actual[:, t, :] - expected[:, t, :]).abs().max().item()
            print(f"  t={t}: max diff = {diff_t:.6f}")

        raise AssertionError("N=1 mismatch - manual cell implementation is wrong")

    print("✓ single_lstm_cell")


def _test_single_timestep():
    """Test single timestep to isolate gate computation issues."""
    N, B, I, H = 4, 2, 32, 64
    T = 1  # Single timestep

    lstms = [nn.LSTM(I, H, batch_first=True) for _ in range(N)]
    inputs = [torch.randn(B, T, I) for _ in range(N)]

    # Baseline
    baseline_outputs = []
    with torch.no_grad():
        for lstm, inp in zip(lstms, inputs):
            out, _ = lstm(inp)
            baseline_outputs.append(out)
    baseline_concat = torch.cat(baseline_outputs, dim=-1)

    # Fused
    wide = WideLSTM.from_modules(lstms, strategy='fused')
    packed = torch.cat(inputs, dim=-1)
    with torch.no_grad():
        wide_out, _ = wide(packed)

    if not torch.allclose(wide_out, baseline_concat, rtol=1e-4, atol=1e-4):
        print(f"  Single timestep failed!")
        print(f"  Max diff: {(wide_out - baseline_concat).abs().max().item():.6f}")

        # Check each LSTM separately
        for i in range(N):
            wide_i = wide_out[:, :, i*H:(i+1)*H]
            base_i = baseline_outputs[i]
            diff_i = (wide_i - base_i).abs().max().item()
            print(f"  LSTM {i}: max diff = {diff_i:.6f}")

        raise AssertionError("Single timestep mismatch")

    print("✓ single_timestep")


def _test_correctness():
    """Test fused output matches N separate LSTMs."""
    N, B, T, I, H = 4, 2, 16, 32, 64

    lstms = [nn.LSTM(I, H, batch_first=True) for _ in range(N)]
    inputs = [torch.randn(B, T, I) for _ in range(N)]

    # Baseline: run each LSTM separately
    baseline_outputs = []
    with torch.no_grad():
        for lstm, inp in zip(lstms, inputs):
            out, _ = lstm(inp)
            baseline_outputs.append(out)
    baseline_concat = torch.cat(baseline_outputs, dim=-1)

    # Fused
    wide = WideLSTM.from_modules(lstms, strategy='fused')
    packed = torch.cat(inputs, dim=-1)
    with torch.no_grad():
        wide_out, _ = wide(packed)

    assert wide_out.shape == baseline_concat.shape, \
        f"Shape: {wide_out.shape} vs {baseline_concat.shape}"

    if not torch.allclose(wide_out, baseline_concat, rtol=1e-4, atol=1e-4):
        diff = (wide_out - baseline_concat).abs()
        print(f"  Max diff: {diff.max().item():.6f}")
        print(f"  Mean diff: {diff.mean().item():.6f}")

        # Debug: check per-LSTM outputs
        for i in range(N):
            wide_i = wide_out[:, :, i*H:(i+1)*H]
            base_i = baseline_outputs[i]
            diff_i = (wide_i - base_i).abs().max().item()
            print(f"  LSTM {i} max diff: {diff_i:.6f}")

        # Debug: check first timestep only
        print("\n  First timestep analysis:")
        for i in range(N):
            wide_t0_i = wide_out[:, 0, i*H:(i+1)*H]
            base_t0_i = baseline_outputs[i][:, 0, :]
            diff_t0 = (wide_t0_i - base_t0_i).abs().max().item()
            print(f"  LSTM {i} t=0 diff: {diff_t0:.6f}")

        raise AssertionError("Value mismatch")

    print("✓ correctness")


def _test_correctness_with_hidden():
    """Test correctness with initial hidden state."""
    N, B, T, I, H = 4, 2, 16, 32, 64

    lstms = [nn.LSTM(I, H, batch_first=True) for _ in range(N)]
    inputs = [torch.randn(B, T, I) for _ in range(N)]
    h_0s = [torch.randn(1, B, H) for _ in range(N)]
    c_0s = [torch.randn(1, B, H) for _ in range(N)]

    # Baseline
    baseline_outputs = []
    with torch.no_grad():
        for lstm, inp, h0, c0 in zip(lstms, inputs, h_0s, c_0s):
            out, _ = lstm(inp, (h0, c0))
            baseline_outputs.append(out)
    baseline_concat = torch.cat(baseline_outputs, dim=-1)

    # Fused
    wide = WideLSTM.from_modules(lstms, strategy='fused')
    packed = torch.cat(inputs, dim=-1)
    h_0_packed = torch.cat(h_0s, dim=-1)
    c_0_packed = torch.cat(c_0s, dim=-1)

    with torch.no_grad():
        wide_out, _ = wide(packed, (h_0_packed, c_0_packed))

    assert torch.allclose(wide_out, baseline_concat, rtol=1e-4, atol=1e-4), \
        f"Max diff: {(wide_out - baseline_concat).abs().max().item()}"

    print("✓ correctness_with_hidden")


def _test_strategy_equivalence():
    """Test fused and sequential strategies produce same output."""
    N, B, T, I, H = 4, 2, 16, 32, 64

    lstms = [nn.LSTM(I, H, batch_first=True) for _ in range(N)]

    wide_fused = WideLSTM.from_modules(lstms, strategy='fused')
    wide_seq = WideLSTM.from_modules(lstms, strategy='sequential')

    x = torch.cat([torch.randn(B, T, I) for _ in range(N)], dim=-1)

    with torch.no_grad():
        out_fused, (h_fused, c_fused) = wide_fused(x)
        out_seq, (h_seq, c_seq) = wide_seq(x)

    assert torch.allclose(out_fused, out_seq, rtol=1e-5, atol=1e-5), \
        f"Output diff: {(out_fused - out_seq).abs().max().item()}"
    assert torch.allclose(h_fused, h_seq, rtol=1e-5, atol=1e-5)
    assert torch.allclose(c_fused, c_seq, rtol=1e-5, atol=1e-5)

    print("✓ strategy_equivalence")


def _test_gradient_flow():
    """Test gradients flow correctly."""
    N, B, T, I, H = 4, 2, 16, 32, 64

    lstms = [nn.LSTM(I, H, batch_first=True) for _ in range(N)]
    wide = WideLSTM.from_modules(lstms, strategy='fused')

    x = torch.randn(B, T, N * I, requires_grad=True)
    out, _ = wide(x)

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert wide.weight_ih.grad is not None
    assert wide.weight_hh.grad is not None

    print("✓ gradient_flow")


def _test_block_diagonal_structure():
    """Verify weight matrices are truly block-diagonal."""
    N, I, H = 4, 32, 64

    lstms = [nn.LSTM(I, H, batch_first=True) for _ in range(N)]
    wide = WideLSTM.from_modules(lstms)

    # Check off-diagonal blocks are zero
    for i in range(N):
        for j in range(N):
            if i != j:
                # weight_ih block [i,j] should be zero
                block_ih = wide.weight_ih[i*4*H:(i+1)*4*H, j*I:(j+1)*I]
                assert block_ih.abs().max() == 0, f"Non-zero off-diagonal at weight_ih[{i},{j}]"

                # weight_hh block [i,j] should be zero
                block_hh = wide.weight_hh[i*4*H:(i+1)*4*H, j*H:(j+1)*H]
                assert block_hh.abs().max() == 0, f"Non-zero off-diagonal at weight_hh[{i},{j}]"

    print("✓ block_diagonal_structure")


def _test_compilation():
    """Test torch.compile compatibility."""
    N, B, T, I, H = 4, 2, 16, 32, 64

    lstms = [nn.LSTM(I, H, batch_first=True) for _ in range(N)]
    wide = WideLSTM.from_modules(lstms, strategy='fused')

    x = torch.randn(B, T, N * I)

    with torch.no_grad():
        out_eager, _ = wide(x)

    wide_compiled = torch.compile(wide, mode='reduce-overhead')

    with torch.no_grad():
        for _ in range(3):
            wide_compiled(x)
        out_compiled, _ = wide_compiled(x)

    assert torch.allclose(out_eager, out_compiled, rtol=1e-4, atol=1e-4), \
        f"Compiled diff: {(out_eager - out_compiled).abs().max().item()}"

    print("✓ compilation")


def _test_compilation_cuda():
    """Test torch.compile on CUDA."""
    if not torch.cuda.is_available():
        print("⊘ compilation_cuda (no CUDA)")
        return

    N, B, T, I, H = 8, 4, 32, 64, 128
    device = 'cuda'

    lstms = [nn.LSTM(I, H, batch_first=True).to(device) for _ in range(N)]
    wide = WideLSTM.from_modules(lstms, strategy='fused')

    x = torch.randn(B, T, N * I, device=device)

    with torch.no_grad():
        out_eager, _ = wide(x)

    wide_compiled = torch.compile(wide, mode='reduce-overhead')

    with torch.no_grad():
        for _ in range(3):
            wide_compiled(x)
        out_compiled, _ = wide_compiled(x)

    assert torch.allclose(out_eager, out_compiled, rtol=1e-4, atol=1e-4)

    print("✓ compilation_cuda")


def run_tests():
    """Run all tests."""
    print("=" * 50)
    print("WideLSTM Tests")
    print("=" * 50)

    _test_basic_forward()
    _test_from_modules()
    _test_block_diagonal_structure()
    _test_single_lstm_cell()
    _test_single_timestep()
    _test_correctness()
    _test_correctness_with_hidden()
    _test_strategy_equivalence()
    _test_gradient_flow()
    _test_compilation()
    _test_compilation_cuda()

    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == '__main__':
    run_tests()