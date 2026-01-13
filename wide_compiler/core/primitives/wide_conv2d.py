"""
WideConv2d with multiple backend strategies.

Strategies:
- 'grouped': Single grouped conv (best for high N, low B)
- 'sequential': N separate convs (best for low N, high B)
- 'batch_parallel': Process all N in parallel via batching tricks
- 'auto': Heuristic selection based on N, B, C

Each strategy has different performance characteristics depending on:
- N: Number of models
- B: Batch size per model  
- C: Channel count
- H, W: Spatial dimensions
"""

from typing import List, Literal, Optional, Union, Tuple
from enum import Enum

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class ConvStrategy(Enum):
    GROUPED = 'grouped'           # Single grouped conv
    SEQUENTIAL = 'sequential'     # N separate F.conv2d
    CHUNK_BATCH = 'chunk_batch'   # Stack on batch dim, chunk outputs
    HYBRID = 'hybrid'             # Mixed: grouped for some, sequential for others
    CHANNELS_LAST = 'channels_last'  # Grouped with channels_last memory format
    AUTO = 'auto'


# Tuned thresholds from A100 benchmarks (sweep_boundaries.py)
# WARNING: These thresholds are optimized for A100. Other GPUs may behave differently.
# Run sweep_boundaries.py on your hardware to find optimal thresholds.
STRATEGY_THRESHOLDS = {
    'n_high': 32,             # N above this: grouped wins on A100
    'n_crossover': 16,        # N above this with low B: grouped helps on A100
    'n_low': 4,               # N at or below this: sequential
    'b_low': 8,               # B at or below this: grouped helps on A100
    'c_in_grouped_min': 2,    # C_in below this: grouped loses
    'c_in_grouped_max': 4,    # C_in above this: grouped starts losing
}

_WARNED_ABOUT_OPTIMIZATION = False

def _warn_optimization():
    """Warn once that thresholds are A100-tuned."""
    global _WARNED_ABOUT_OPTIMIZATION
    if not _WARNED_ABOUT_OPTIMIZATION:
        import warnings
        warnings.warn(
            "WideConv2d auto-selection is currently optimized for A100 GPUs. "
            "Performance on other GPUs may vary. Run sweep_boundaries.py to tune for your hardware.",
            UserWarning
        )
        _WARNED_ABOUT_OPTIMIZATION = True


def select_strategy(n: int, b: int, c_in: int, c_out: int,
                    h: int = 56, w: int = 56) -> ConvStrategy:
    """
    Heuristic strategy selection based on A100 benchmarks.

    WARNING: Optimized for A100. Other GPUs may need different thresholds.

    Key findings from A100 sweep_boundaries.py:

    1. N vs B effect (fixed N*B=128):
       - N=2, B=64:  Seq wins (0.87x)
       - N=16, B=8:  Tie (1.04x)
       - N=32, B=4:  Grp wins (1.82x)
       - N=128, B=1: Grp wins (6.81x)

    2. Channel effect (N=10, B=16):
       - C_in=1:   Seq wins (0.62x)
       - C_in=2-4: Grp wins (1.15-1.18x)
       - C_in>=8:  Seq wins or tie
    """
    _warn_optimization()

    T = STRATEGY_THRESHOLDS

    # Very high N: grouped wins on A100
    if n >= T['n_high']:
        return ConvStrategy.GROUPED

    # Very low N: sequential (not enough parallelism)
    if n <= T['n_low']:
        return ConvStrategy.SEQUENTIAL

    # Channel-based decision
    if c_in < T['c_in_grouped_min']:
        return ConvStrategy.SEQUENTIAL

    if c_in > T['c_in_grouped_max'] and b > T['b_low']:
        return ConvStrategy.SEQUENTIAL

    # Low batch size with decent N: grouped helps on A100
    if b <= T['b_low'] and n >= T['n_crossover']:
        return ConvStrategy.GROUPED

    # Default: sequential is safest
    return ConvStrategy.SEQUENTIAL


class WideConv2d(nn.Module):
    """
    N parallel Conv2d layers with selectable execution strategy.

    Strategies:
        'grouped': Fused grouped convolution (1 kernel launch)
        'sequential': N separate convolutions (N kernel launches)
        'batch_parallel': Experimental batched execution
        'auto': Automatically select based on input shape

    The 'auto' strategy selects at forward() time based on actual batch size.
    Other strategies are fixed at construction.

    Example:
        #>>> convs = [nn.Conv2d(64, 128, 3) for _ in range(10)]
        #>>> wide = WideConv2d.from_modules(convs, strategy='auto')
        #>>> x = pack_inputs([torch.randn(8, 64, 56, 56) for _ in range(10)])
        #>>> y = wide(x)  # Auto-selects best strategy
    """

    def __init__(
        self,
        n: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        strategy: Union[str, ConvStrategy] = 'auto',
    ):
        super().__init__()
        self.n = n
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.has_bias = bias

        # Parse strategy
        if isinstance(strategy, str):
            strategy = ConvStrategy(strategy)
        self.strategy = strategy

        # Always create grouped conv (used by grouped and auto strategies)
        self.grouped_conv = nn.Conv2d(
            in_channels=n * in_channels,
            out_channels=n * out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=n,
            bias=bias
        )

        # For sequential strategy, we use views into grouped_conv weights
        # This avoids duplicating parameters
        self._weight_views: Optional[List[Tensor]] = None
        self._bias_views: Optional[List[Tensor]] = None

        # Stats for auto strategy
        self._call_count = 0
        self._strategy_counts = {s: 0 for s in ConvStrategy}

    def _get_weight_views(self) -> List[Tensor]:
        """Get per-model weight views (lazy creation)."""
        if self._weight_views is None:
            self._weight_views = [
                self.grouped_conv.weight[i*self.out_channels:(i+1)*self.out_channels]
                for i in range(self.n)
            ]
        return self._weight_views

    def _get_bias_views(self) -> Optional[List[Tensor]]:
        """Get per-model bias views (lazy creation)."""
        if self.grouped_conv.bias is None:
            return None
        if self._bias_views is None:
            self._bias_views = [
                self.grouped_conv.bias[i*self.out_channels:(i+1)*self.out_channels]
                for i in range(self.n)
            ]
        return self._bias_views

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with strategy selection.

        Args:
            x: Input tensor [B, N*C_in, H, W] for grouped/auto
               or List[Tensor] for sequential

        Returns:
            Output tensor [B, N*C_out, H', W']
        """
        strategy = self.strategy

        # Auto strategy: select based on input shape
        if strategy == ConvStrategy.AUTO:
            B = x.shape[0]
            H, W = x.shape[2], x.shape[3]
            strategy = select_strategy(self.n, B, self.in_channels, self.out_channels, H, W)
            self._strategy_counts[strategy] += 1

        self._call_count += 1

        if strategy == ConvStrategy.GROUPED:
            return self._forward_grouped(x)
        elif strategy == ConvStrategy.SEQUENTIAL:
            return self._forward_sequential(x)
        elif strategy == ConvStrategy.CHUNK_BATCH:
            return self._forward_chunk_batch(x)
        elif strategy == ConvStrategy.CHANNELS_LAST:
            return self._forward_channels_last(x)
        elif strategy == ConvStrategy.HYBRID:
            return self._forward_hybrid(x)
        else:
            # Fallback
            return self._forward_grouped(x)

    def _forward_grouped(self, x: Tensor) -> Tensor:
        """Single grouped convolution."""
        return self.grouped_conv(x)

    def _forward_sequential(self, x: Tensor) -> Tensor:
        """N separate convolutions, concatenated."""
        B = x.shape[0]
        H, W = x.shape[2], x.shape[3]

        # Reshape input: [B, N*C_in, H, W] -> [B, N, C_in, H, W]
        x_reshaped = x.view(B, self.n, self.in_channels, H, W)

        weights = self._get_weight_views()
        biases = self._get_bias_views()

        outputs = []
        for i in range(self.n):
            xi = x_reshaped[:, i]  # [B, C_in, H, W]
            bi = biases[i] if biases else None
            out = F.conv2d(xi, weights[i], bi,
                          stride=self.stride, padding=self.padding,
                          dilation=self.dilation)
            outputs.append(out)

        # Stack and reshape: [N, B, C_out, H', W'] -> [B, N*C_out, H', W']
        stacked = torch.stack(outputs, dim=1)  # [B, N, C_out, H', W']
        return stacked.view(B, -1, stacked.shape[3], stacked.shape[4])

    def _forward_chunk_batch(self, x: Tensor) -> Tensor:
        """
        Reshape to batch dimension, process, reshape back.

        Input: [B, N*C_in, H, W]
        -> Reshape to: [B*N, C_in, H, W]
        -> Run N separate convs (each on B samples)
        -> Reshape back: [B, N*C_out, H', W']

        This has same kernel count as sequential but different memory access pattern.
        May benefit from better cache locality on some architectures.
        """
        B = x.shape[0]
        H, W = x.shape[2], x.shape[3]

        # Reshape: [B, N*C_in, H, W] -> [B, N, C_in, H, W] -> [N, B, C_in, H, W]
        x_reshaped = x.view(B, self.n, self.in_channels, H, W).permute(1, 0, 2, 3, 4)

        weights = self._get_weight_views()
        biases = self._get_bias_views()

        # Process each model's batch
        outputs = []
        for i in range(self.n):
            xi = x_reshaped[i]  # [B, C_in, H, W] - contiguous batch for model i
            bi = biases[i] if biases else None
            out = F.conv2d(xi, weights[i], bi,
                          stride=self.stride, padding=self.padding,
                          dilation=self.dilation)
            outputs.append(out)

        # Stack: [N, B, C_out, H', W'] -> [B, N, C_out, H', W'] -> [B, N*C_out, H', W']
        stacked = torch.stack(outputs, dim=0).permute(1, 0, 2, 3, 4)
        return stacked.reshape(B, -1, stacked.shape[3], stacked.shape[4])

    def _forward_channels_last(self, x: Tensor) -> Tensor:
        """
        Grouped conv with channels_last memory format.

        Channels-last (NHWC) can be faster for some GPU architectures
        by avoiding layout conversions that cuDNN sometimes does internally.
        """
        # Convert to channels_last
        x_cl = x.to(memory_format=torch.channels_last)

        # Ensure grouped_conv is also channels_last
        if not self.grouped_conv.weight.is_contiguous(memory_format=torch.channels_last):
            self.grouped_conv = self.grouped_conv.to(memory_format=torch.channels_last)

        out = self.grouped_conv(x_cl)

        # Convert back to contiguous if needed
        return out.contiguous()

    def _forward_hybrid(self, x: Tensor) -> Tensor:
        """
        Hybrid strategy: split N into groups, grouped conv within each.

        For very large N, this can balance between:
        - Too many kernel launches (sequential)
        - Single kernel with poor occupancy (fully grouped)

        Splits N into chunks of size ~8-16 and runs grouped conv per chunk.
        """
        B = x.shape[0]
        H, W = x.shape[2], x.shape[3]

        # Choose chunk size based on N
        if self.n <= 16:
            # Not worth splitting, just use grouped
            return self._forward_grouped(x)

        chunk_size = min(16, self.n // 2)  # 2-16 models per chunk
        num_chunks = (self.n + chunk_size - 1) // chunk_size

        # Reshape input
        x_reshaped = x.view(B, self.n, self.in_channels, H, W)

        outputs = []
        for c in range(num_chunks):
            start_idx = c * chunk_size
            end_idx = min(start_idx + chunk_size, self.n)
            actual_chunk = end_idx - start_idx

            # Get chunk of inputs
            xi = x_reshaped[:, start_idx:end_idx]  # [B, chunk, C_in, H, W]
            xi = xi.reshape(B, actual_chunk * self.in_channels, H, W)

            # Get chunk of weights
            w_start = start_idx * self.out_channels
            w_end = end_idx * self.out_channels
            w_chunk = self.grouped_conv.weight[w_start:w_end]
            b_chunk = self.grouped_conv.bias[w_start:w_end] if self.grouped_conv.bias is not None else None

            # Run grouped conv for this chunk
            out = F.conv2d(xi, w_chunk, b_chunk,
                          stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=actual_chunk)
            outputs.append(out)

        # Concatenate chunks
        return torch.cat(outputs, dim=1)

    @classmethod
    def from_modules(
        cls,
        modules: List[nn.Conv2d],
        strategy: Union[str, ConvStrategy] = 'auto'
    ) -> 'WideConv2d':
        """
        Create WideConv2d from N existing Conv2d modules.

        Args:
            modules: List of N Conv2d modules with same architecture
            strategy: Execution strategy ('auto', 'grouped', 'sequential')

        Returns:
            WideConv2d instance with fused weights
        """
        n = len(modules)
        t = modules[0]  # Template

        # Extract config (handle tuple vs int)
        def to_tuple(x):
            return x if isinstance(x, tuple) else (x, x)

        k = to_tuple(t.kernel_size)
        s = to_tuple(t.stride)
        p = to_tuple(t.padding)
        d = to_tuple(t.dilation)

        wide = cls(
            n=n,
            in_channels=t.in_channels,
            out_channels=t.out_channels,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
            bias=t.bias is not None,
            strategy=strategy,
        )

        # Copy weights
        with torch.no_grad():
            for i, m in enumerate(modules):
                start = i * t.out_channels
                end = start + t.out_channels
                wide.grouped_conv.weight[start:end] = m.weight
                if m.bias is not None:
                    wide.grouped_conv.bias[start:end] = m.bias

        return wide

    def get_stats(self) -> dict:
        """Get strategy selection statistics."""
        return {
            'total_calls': self._call_count,
            'strategy_counts': dict(self._strategy_counts),
            'current_strategy': self.strategy.value,
        }

    def set_strategy(self, strategy: Union[str, ConvStrategy]):
        """Change strategy at runtime."""
        if isinstance(strategy, str):
            strategy = ConvStrategy(strategy)
        self.strategy = strategy
        # Clear cached views if switching to sequential
        if strategy == ConvStrategy.SEQUENTIAL:
            self._weight_views = None
            self._bias_views = None

    def __repr__(self):
        return (
            f"WideConv2d({self.n}x[{self.in_channels}→{self.out_channels}], "
            f"k={self.kernel_size}, strategy={self.strategy.value})"
        )


# =============================================================================
# Convenience functions
# =============================================================================

def create_wide_conv2d(
    n: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    strategy: str = 'auto',
    **kwargs
) -> WideConv2d:
    """Factory function for creating WideConv2d."""
    return WideConv2d(n, in_channels, out_channels, kernel_size, strategy=strategy, **kwargs)


# =============================================================================
# Tests
# =============================================================================

if __name__ == '__main__':
    import time

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("=" * 70)
    print("WideConv2d Strategy Comparison")
    print("=" * 70)

    def benchmark(name, fn, num_iters=100, warmup=20):
        for _ in range(warmup):
            fn()
        if device == 'cuda':
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(num_iters):
            fn()
        if device == 'cuda':
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    # Test configs
    configs = [
        # (N, B, description)
        (50, 4, "High N, Low B (Wide should win)"),
        (10, 32, "Medium N, Medium B"),
        (4, 128, "Low N, High B (Sequential may win)"),
        (100, 2, "Very High N, Very Low B"),
    ]

    C_in, C_out = 64, 64
    H, W = 56, 56
    K = 3

    print(f"\nC={C_in}→{C_out}, K={K}, H×W={H}×{W}")

    for N, B, desc in configs:
        print(f"\n{'='*70}")
        print(f"Config: N={N}, B={B} - {desc}")
        print("=" * 70)

        # Create models
        convs = [nn.Conv2d(C_in, C_out, K, padding=1).to(device).eval() for _ in range(N)]

        # Create wide versions with different strategies
        wide_grouped = WideConv2d.from_modules(convs, strategy='grouped').to(device).eval()
        wide_sequential = WideConv2d.from_modules(convs, strategy='sequential').to(device).eval()
        wide_auto = WideConv2d.from_modules(convs, strategy='auto').to(device).eval()

        # Inputs
        x_list = [torch.randn(B, C_in, H, W, device=device) for _ in range(N)]

        # Pack for wide: [B, N*C_in, H, W]
        # Stack on dim 1 then reshape - keeps channels from each model contiguous
        x_packed = torch.stack(x_list, dim=1).view(B, N*C_in, H, W)

        # Benchmark
        results = {}

        # Sequential baseline (N separate module calls)
        def run_sequential_baseline():
            return [convs[i](x_list[i]) for i in range(N)]
        results['N×Module calls'] = benchmark('baseline', run_sequential_baseline)

        # Wide grouped
        def run_grouped():
            return wide_grouped(x_packed)
        results['Wide (grouped)'] = benchmark('grouped', run_grouped)

        # Wide sequential
        def run_wide_seq():
            return wide_sequential(x_packed)
        results['Wide (sequential)'] = benchmark('sequential', run_wide_seq)

        # Wide chunk_batch
        wide_chunk = WideConv2d.from_modules(convs, strategy='chunk_batch').to(device).eval()
        def run_chunk():
            return wide_chunk(x_packed)
        results['Wide (chunk_batch)'] = benchmark('chunk_batch', run_chunk)

        # Wide channels_last
        wide_cl = WideConv2d.from_modules(convs, strategy='channels_last').to(device).eval()
        x_packed_cl = x_packed.clone()
        def run_cl():
            return wide_cl(x_packed_cl)
        results['Wide (channels_last)'] = benchmark('channels_last', run_cl)

        # Wide hybrid (only for large N)
        if N >= 16:
            wide_hybrid = WideConv2d.from_modules(convs, strategy='hybrid').to(device).eval()
            def run_hybrid():
                return wide_hybrid(x_packed)
            results['Wide (hybrid)'] = benchmark('hybrid', run_hybrid)

        # Wide auto
        def run_auto():
            return wide_auto(x_packed)
        results['Wide (auto)'] = benchmark('auto', run_auto)

        # Print results
        baseline = results['N×Module calls']
        print(f"\n{'Method':<25} {'Time (ms)':<12} {'Speedup':<10}")
        print("-" * 50)
        for name, t in results.items():
            speedup = baseline / t
            marker = "✓" if speedup > 1.05 else ""
            print(f"{name:<25} {t*1000:<12.1f} {speedup:<10.2f}x {marker}")

        # Find best
        best_name = min(results, key=results.get)
        print(f"\n→ Best: {best_name}")

        # Show auto strategy selection
        auto_strategy = select_strategy(N, B, C_in, C_out, H, W)
        print(f"→ Auto selected: {auto_strategy.value}")

        # Verify correctness
        with torch.no_grad():
            # Reference: run each conv separately, stack properly
            ref_outs = [convs[i](x_list[i]) for i in range(N)]
            ref = torch.stack(ref_outs, dim=1).view(B, N*C_out, ref_outs[0].shape[2], ref_outs[0].shape[3])

            grouped_out = wide_grouped(x_packed)
            seq_out = wide_sequential(x_packed)
            chunk_out = wide_chunk(x_packed)
            cl_out = wide_cl(x_packed_cl)

            print(f"\nCorrectness:")
            print(f"  grouped:      {(ref - grouped_out).abs().max().item():.2e}")
            print(f"  sequential:   {(ref - seq_out).abs().max().item():.2e}")
            print(f"  chunk_batch:  {(ref - chunk_out).abs().max().item():.2e}")
            print(f"  channels_last:{(ref - cl_out).abs().max().item():.2e}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)