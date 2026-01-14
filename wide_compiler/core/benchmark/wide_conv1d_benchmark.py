"""
WideConv1d Comprehensive Benchmark

Tests GROUPED vs SEQUENTIAL strategies across:
- N values: 2, 4, 6, 8, 10, 12, 16, 20, 32, 64, 100
- Kernel sizes: 1, 3, 5, 7, 11
- Channel sizes: 32, 64, 128, 256
- Sequence lengths: 64, 256, 1024, 4096

Goal: Find optimal crossover threshold for strategy selection.

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

import torch
import torch.nn as nn
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import sys

# Import from package or local
try:
    from wide_conv1d import WideConv1d, Conv1dStrategy
except ImportError:
    from wide_compiler.core.primitives.wide_conv1d import WideConv1d, Conv1dStrategy


@dataclass
class BenchResult:
    n: int
    kernel_size: int
    channels: int
    seq_len: int
    batch: int
    baseline_ms: float
    grouped_ms: float
    sequential_ms: float
    grouped_speedup: float
    sequential_speedup: float
    best_strategy: str


def benchmark_fn(fn, iters: int = 100, warmup: int = 20, device: str = 'cuda') -> float:
    """Benchmark a function, return avg time in ms."""
    for _ in range(warmup):
        fn()
    if device == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device == 'cuda':
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) / iters * 1000


def run_single_benchmark(
        N: int,
        C: int,
        L: int,
        K: int,
        B: int = 8,
        iters: int = 100,
        device: str = 'cuda',
) -> BenchResult:
    """Run benchmark for a single configuration."""

    # Create N Conv1d modules
    convs = [nn.Conv1d(C, C, K, padding=K // 2).to(device) for _ in range(N)]
    for conv in convs:
        conv.eval()

    # Create wide versions
    wide_grouped = WideConv1d.from_modules(convs, strategy=Conv1dStrategy.GROUPED).to(device).eval()
    wide_sequential = WideConv1d.from_modules(convs, strategy=Conv1dStrategy.SEQUENTIAL).to(device).eval()

    # Create inputs
    inputs = [torch.randn(B, C, L, device=device) for _ in range(N)]
    packed = torch.cat([inp.unsqueeze(1) for inp in inputs], dim=1).view(B, N * C, L)

    # Verify correctness first
    with torch.no_grad():
        ref = torch.cat([convs[i](inputs[i]).unsqueeze(1) for i in range(N)], dim=1).view(B, N * C, -1)
        out_g = wide_grouped(packed)
        out_s = wide_sequential(packed)

        diff_g = (ref - out_g).abs().max().item()
        diff_s = (ref - out_s).abs().max().item()

        if diff_g > 1e-4 or diff_s > 1e-4:
            print(f"  WARNING: Correctness issue! GROUPED={diff_g:.2e}, SEQ={diff_s:.2e}")

    # Benchmark
    with torch.no_grad():
        t_baseline = benchmark_fn(lambda: [convs[i](inputs[i]) for i in range(N)], iters, device=device)
        t_grouped = benchmark_fn(lambda: wide_grouped(packed), iters, device=device)
        t_sequential = benchmark_fn(lambda: wide_sequential(packed), iters, device=device)

    grouped_speedup = t_baseline / t_grouped
    sequential_speedup = t_baseline / t_sequential

    best = "GROUPED" if t_grouped <= t_sequential else "SEQUENTIAL"

    return BenchResult(
        n=N,
        kernel_size=K,
        channels=C,
        seq_len=L,
        batch=B,
        baseline_ms=t_baseline,
        grouped_ms=t_grouped,
        sequential_ms=t_sequential,
        grouped_speedup=grouped_speedup,
        sequential_speedup=sequential_speedup,
        best_strategy=best,
    )


def print_results_table(results: List[BenchResult], title: str):
    """Print results as formatted table."""
    print(f"\n{'=' * 90}")
    print(f"{title}")
    print(f"{'=' * 90}")
    print(
        f"{'N':>4} {'K':>3} {'C':>4} {'L':>5} {'Baseline':>10} {'GROUPED':>10} {'SEQ':>10} {'G-Speed':>8} {'S-Speed':>8} {'Best':>10}")
    print("-" * 90)

    for r in results:
        print(f"{r.n:>4} {r.kernel_size:>3} {r.channels:>4} {r.seq_len:>5} "
              f"{r.baseline_ms:>9.3f}ms {r.grouped_ms:>9.3f}ms {r.sequential_ms:>9.3f}ms "
              f"{r.grouped_speedup:>7.2f}x {r.sequential_speedup:>7.2f}x {r.best_strategy:>10}")


def find_crossover(results: List[BenchResult]) -> Dict[Tuple[int, int, int], int]:
    """Find N crossover point for each (K, C, L) configuration."""
    crossovers = {}

    # Group by (K, C, L)
    configs = {}
    for r in results:
        key = (r.kernel_size, r.channels, r.seq_len)
        if key not in configs:
            configs[key] = []
        configs[key].append(r)

    for key, rs in configs.items():
        rs_sorted = sorted(rs, key=lambda x: x.n)
        crossover_n = None
        for r in rs_sorted:
            if r.best_strategy == "GROUPED":
                crossover_n = r.n
                break
        crossovers[key] = crossover_n

    return crossovers


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"WideConv1d Comprehensive Benchmark")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 90)

    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    all_results = []

    # ==========================================================================
    # TEST 1: Vary N with fixed K=3, C=64, L=256 (typical audio/time series)
    # ==========================================================================
    print("\n[TEST 1] Varying N (K=3, C=64, L=256, B=8)")
    results_n = []
    for N in [2, 4, 6, 8, 10, 12, 16, 20, 32, 64]:
        print(f"  N={N}...", end=" ", flush=True)
        r = run_single_benchmark(N=N, C=64, L=256, K=3, B=8, iters=100, device=device)
        results_n.append(r)
        print(f"G={r.grouped_speedup:.2f}x S={r.sequential_speedup:.2f}x -> {r.best_strategy}")

    print_results_table(results_n, "N Sweep (K=3, C=64, L=256)")
    all_results.extend(results_n)

    # ==========================================================================
    # TEST 2: Vary kernel size with fixed N=20, C=64, L=256
    # ==========================================================================
    print("\n[TEST 2] Varying Kernel Size (N=20, C=64, L=256, B=8)")
    results_k = []
    for K in [1, 3, 5, 7, 11, 15]:
        print(f"  K={K}...", end=" ", flush=True)
        r = run_single_benchmark(N=20, C=64, L=256, K=K, B=8, iters=100, device=device)
        results_k.append(r)
        print(f"G={r.grouped_speedup:.2f}x S={r.sequential_speedup:.2f}x -> {r.best_strategy}")

    print_results_table(results_k, "Kernel Size Sweep (N=20, C=64, L=256)")
    all_results.extend(results_k)

    # ==========================================================================
    # TEST 3: Vary channels with fixed N=20, K=3, L=256
    # ==========================================================================
    print("\n[TEST 3] Varying Channels (N=20, K=3, L=256, B=8)")
    results_c = []
    for C in [16, 32, 64, 128, 256]:
        print(f"  C={C}...", end=" ", flush=True)
        r = run_single_benchmark(N=20, C=C, L=256, K=3, B=8, iters=100, device=device)
        results_c.append(r)
        print(f"G={r.grouped_speedup:.2f}x S={r.sequential_speedup:.2f}x -> {r.best_strategy}")

    print_results_table(results_c, "Channel Sweep (N=20, K=3, L=256)")
    all_results.extend(results_c)

    # ==========================================================================
    # TEST 4: Vary sequence length with fixed N=20, K=3, C=64
    # ==========================================================================
    print("\n[TEST 4] Varying Sequence Length (N=20, K=3, C=64, B=8)")
    results_l = []
    for L in [64, 128, 256, 512, 1024, 2048]:
        print(f"  L={L}...", end=" ", flush=True)
        r = run_single_benchmark(N=20, C=64, L=L, K=3, B=8, iters=100, device=device)
        results_l.append(r)
        print(f"G={r.grouped_speedup:.2f}x S={r.sequential_speedup:.2f}x -> {r.best_strategy}")

    print_results_table(results_l, "Sequence Length Sweep (N=20, K=3, C=64)")
    all_results.extend(results_l)

    # ==========================================================================
    # TEST 5: Small N boundary test (find exact crossover)
    # ==========================================================================
    print("\n[TEST 5] Crossover Boundary (K=3, C=64, L=256)")
    results_boundary = []
    for N in range(2, 20):
        r = run_single_benchmark(N=N, C=64, L=256, K=3, B=8, iters=100, device=device)
        results_boundary.append(r)
        marker = "<<<" if r.best_strategy == "GROUPED" and (
                    not results_boundary[:-1] or results_boundary[-2].best_strategy == "SEQUENTIAL") else ""
        print(f"  N={N:2d}: G={r.grouped_speedup:.2f}x S={r.sequential_speedup:.2f}x -> {r.best_strategy} {marker}")

    # ==========================================================================
    # TEST 6: Batch size impact
    # ==========================================================================
    print("\n[TEST 6] Batch Size Impact (N=20, K=3, C=64, L=256)")
    results_b = []
    for B in [1, 2, 4, 8, 16, 32]:
        print(f"  B={B}...", end=" ", flush=True)
        r = run_single_benchmark(N=20, C=64, L=256, K=3, B=B, iters=100, device=device)
        results_b.append(r)
        print(f"G={r.grouped_speedup:.2f}x S={r.sequential_speedup:.2f}x -> {r.best_strategy}")

    print_results_table(results_b, "Batch Size Sweep (N=20, K=3, C=64, L=256)")

    # ==========================================================================
    # TEST 7: Real-world configs (WaveNet-like, TCN-like)
    # ==========================================================================
    print("\n[TEST 7] Real-World Configurations")
    real_world = [
        # (N, C, L, K, B, name)
        (8, 64, 16000, 3, 4, "WaveNet-small (8 experts, 1s audio)"),
        (16, 128, 8000, 3, 4, "WaveNet-medium (16 experts)"),
        (32, 64, 4000, 7, 8, "TCN (32 layers, dilated)"),
        (64, 32, 1024, 3, 16, "Small TCN (64 layers)"),
        (100, 64, 256, 3, 8, "Large ensemble (100 models)"),
    ]

    results_rw = []
    for N, C, L, K, B, name in real_world:
        print(f"  {name}...", end=" ", flush=True)
        try:
            r = run_single_benchmark(N=N, C=C, L=L, K=K, B=B, iters=50, device=device)
            results_rw.append(r)
            print(f"G={r.grouped_speedup:.2f}x S={r.sequential_speedup:.2f}x -> {r.best_strategy}")
        except Exception as e:
            print(f"FAILED: {e}")

    if results_rw:
        print_results_table(results_rw, "Real-World Configurations")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    # Count wins
    grouped_wins = sum(1 for r in all_results if r.best_strategy == "GROUPED")
    seq_wins = sum(1 for r in all_results if r.best_strategy == "SEQUENTIAL")

    print(f"\nStrategy Wins: GROUPED={grouped_wins}, SEQUENTIAL={seq_wins}")

    # Find crossover from boundary test
    crossover_n = None
    for r in results_boundary:
        if r.best_strategy == "GROUPED":
            crossover_n = r.n
            break

    print(f"\nCrossover N (K=3, C=64, L=256): {crossover_n}")

    # Recommendations
    print("\n" + "-" * 90)
    print("RECOMMENDATIONS")
    print("-" * 90)

    if crossover_n:
        print(f"  - Use GROUPED for N >= {crossover_n}")
        print(f"  - Use SEQUENTIAL for N < {crossover_n}")
        print(
            f"  - Current threshold (_GROUPED_THRESHOLD=10) is {'GOOD' if crossover_n <= 10 else 'TOO LOW - should be ' + str(crossover_n)}")

    # Best speedups achieved
    best_grouped = max(all_results, key=lambda r: r.grouped_speedup)
    print(f"\n  Best GROUPED speedup: {best_grouped.grouped_speedup:.2f}x "
          f"(N={best_grouped.n}, K={best_grouped.kernel_size}, C={best_grouped.channels}, L={best_grouped.seq_len})")

    best_overall = max(all_results, key=lambda r: max(r.grouped_speedup, r.sequential_speedup))
    best_speed = max(best_overall.grouped_speedup, best_overall.sequential_speedup)
    print(f"  Best overall speedup: {best_speed:.2f}x "
          f"(N={best_overall.n}, K={best_overall.kernel_size}, C={best_overall.channels}, L={best_overall.seq_len})")


if __name__ == '__main__':
    main()