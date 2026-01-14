"""
WideCompiler.core.benchmark.benchmark_runner

Generic benchmark execution engine.

Knows nothing about specific primitives - just runs BenchmarkJob and produces BenchmarkResult.

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional

import torch
import torch.nn as nn

from .benchmark_schema import BenchmarkJob, BenchmarkResult, SingleResult, SweepParams


def time_fn(
        fn: Callable,
        warmup: int,
        iters: int,
        device: str,
) -> float:
    """
    Time a function.

    Returns:
        Mean time in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        fn()

    if device == 'cuda':
        torch.cuda.synchronize()

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()

    if device == 'cuda':
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) / iters * 1000


def check_correctness(
        ref_outputs: List[torch.Tensor],
        test_output: torch.Tensor,
        unpack_fn: Callable,
        n: int,
        rtol: float,
        atol: float,
) -> tuple[bool, float]:
    """
    Check if wide output matches reference.

    Returns:
        (correct, max_error)
    """
    test_outputs = unpack_fn(test_output, n)

    max_error = 0.0
    for ref, test in zip(ref_outputs, test_outputs):
        diff = (ref - test).abs().max().item()
        max_error = max(max_error, diff)

    correct = max_error < atol + rtol * max(
        max(o.abs().max().item() for o in ref_outputs),
        1e-6
    )

    return correct, max_error


def run(
        job: BenchmarkJob,
        verbose: bool = True,
) -> BenchmarkResult:
    """
    Execute a benchmark job.

    Args:
        job: BenchmarkJob with sweep params and factories
        verbose: Print progress

    Returns:
        BenchmarkResult with all measurements
    """
    device = job.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        if verbose:
            print("CUDA not available, using CPU")

    if verbose:
        print(f"Benchmark: {job.name}")
        print(f"Primitive: {job.primitive}")
        print(f"Device: {device}")
        print(f"Strategies: {job.strategies}")
        print("=" * 60)

    started_at = datetime.now().isoformat()
    t_start = time.perf_counter()

    results: List[SingleResult] = []
    errors: List[str] = []

    # Iterate sweep
    n_values = job.sweep.n_values
    param_combos = list(job.sweep.param_grid())
    total = len(n_values) * len(param_combos)

    idx = 0
    for n in n_values:
        for params in param_combos:
            idx += 1

            if verbose:
                params_str = ", ".join(f"{k}={v}" for k, v in params.items())
                print(f"[{idx}/{total}] N={n}, {params_str}...", end=" ", flush=True)

            try:
                result = run_single(job, n, params, device)
                results.append(result)

                if verbose:
                    best = result.best_strategy
                    speedup = result.best_speedup
                    print(f"{best}={speedup:.2f}x")

            except Exception as e:
                error_msg = f"N={n}, {params}: {e}"
                errors.append(error_msg)
                if verbose:
                    print(f"ERROR: {e}")

    # Compute summary
    duration_s = time.perf_counter() - t_start
    completed_at = datetime.now().isoformat()

    strategy_wins = count_strategy_wins(results)
    crossover_n = find_crossover(results, job.strategies)
    best_speedup, best_config = find_best(results)

    # Determine recommended threshold
    # If grouped starts winning at crossover_n, recommend that
    recommended = crossover_n if crossover_n else job.sweep.n_values[0]

    result = BenchmarkResult(
        name=job.name,
        primitive=job.primitive,
        strategies=job.strategies,
        sweep=job.sweep,
        results=results,
        crossover_n=crossover_n,
        recommended_threshold=recommended,
        strategy_wins=strategy_wins,
        best_speedup=best_speedup,
        best_config=best_config,
        device=device if device == 'cpu' else torch.cuda.get_device_name(),
        started_at=started_at,
        completed_at=completed_at,
        duration_s=duration_s,
        errors=errors,
    )

    if verbose:
        print("=" * 60)
        print(result.summary())

    return result


def run_single(
        job: BenchmarkJob,
        n: int,
        params: Dict[str, Any],
        device: str,
) -> SingleResult:
    """
    Run benchmark for single (n, params) configuration.
    """
    # Merge params with device
    full_params = {**params, 'device': device}

    # Create N modules
    modules = [job.model_factory(**params).to(device).eval() for _ in range(n)]

    # Create single inputs for baseline
    # Input factory should return shape for ONE model (will be replicated)
    sample_input = job.input_factory(1, **full_params)
    single_shape = sample_input.shape
    single_inputs = [torch.randn(*single_shape, device=device) for _ in range(n)]

    # Create packed input for wide
    packed_input = job.pack_fn(single_inputs)

    timings: Dict[str, float] = {}

    with torch.no_grad():
        # Baseline: N separate forward passes
        timings['baseline'] = time_fn(
            lambda: [m(x) for m, x in zip(modules, single_inputs)],
            job.warmup,
            job.iters,
            device,
        )

        # Reference outputs for correctness
        ref_outputs = [m(x) for m, x in zip(modules, single_inputs)]

        # Each strategy (skip baseline)
        correct = True
        max_error = 0.0

        for strategy in job.strategies:
            if strategy == 'baseline':
                continue

            wide = job.wide_factory(modules, strategy).to(device).eval()

            timings[strategy] = time_fn(
                lambda w=wide, x=packed_input: w(x),
                job.warmup,
                job.iters,
                device,
            )

            # Check correctness (once per strategy)
            test_output = wide(packed_input)
            strat_correct, strat_error = check_correctness(
                ref_outputs, test_output, job.unpack_fn, n,
                job.rtol, job.atol
            )
            correct = correct and strat_correct
            max_error = max(max_error, strat_error)

    # Compute speedups
    baseline_ms = timings['baseline']
    speedups = {s: baseline_ms / t for s, t in timings.items()}

    # Find best non-baseline strategy
    non_baseline = {s: sp for s, sp in speedups.items() if s != 'baseline'}
    if non_baseline:
        best_strategy = max(non_baseline, key=non_baseline.get)
        best_speedup = non_baseline[best_strategy]
    else:
        best_strategy = 'baseline'
        best_speedup = 1.0

    return SingleResult(
        n=n,
        params=params,
        timings=timings,
        speedups=speedups,
        best_strategy=best_strategy,
        best_speedup=best_speedup,
        correct=correct,
        max_error=max_error,
    )


def count_strategy_wins(results: List[SingleResult]) -> Dict[str, int]:
    """Count how many times each strategy was best."""
    wins: Dict[str, int] = {}
    for r in results:
        s = r.best_strategy
        wins[s] = wins.get(s, 0) + 1
    return wins


def find_crossover(
        results: List[SingleResult],
        strategies: List[str],
) -> Optional[int]:
    """
    Find N where strategy changes from one to another.

    Looks for first N where a non-sequential strategy wins.
    """
    # Sort by n
    sorted_results = sorted(results, key=lambda r: r.n)

    # Find first result where grouped/einsum wins over sequential
    for r in sorted_results:
        if r.best_strategy in ('grouped', 'einsum'):
            return r.n

    return None


def find_best(results: List[SingleResult]) -> tuple[float, Dict[str, Any]]:
    """Find best speedup and its config."""
    if not results:
        return 1.0, {}

    best = max(results, key=lambda r: r.best_speedup)
    return best.best_speedup, {'n': best.n, **best.params}


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'run',
    'run_single',
    'time_fn',
    'check_correctness',
]