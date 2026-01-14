"""
WideCompiler.core.benchmark.benchmark_runner

Execute benchmark jobs and time functions.

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

import time
from typing import Callable, List, Optional
import torch
from torch import Tensor

from .benchmark_schema import BenchmarkJob, BenchmarkResult, SingleResult


def time_fn(
    fn: Callable,
    iterations: int = 10,
    warmup: int = 3,
    device: str = 'cuda',
) -> float:
    """
    Time a function with warmup and synchronization.

    Args:
        fn: Function to time (no args)
        iterations: Number of timed iterations
        warmup: Warmup iterations (not timed)
        device: 'cuda' or 'cpu'

    Returns:
        Average time in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        fn()

    if device == 'cuda':
        torch.cuda.synchronize()

    # Time
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / iterations) * 1000  # ms


def run_single(
    job: BenchmarkJob,
    n: int,
    strategy: str,
    params: dict,
    device: str = 'cuda',
    warmup: int = 3,
    iterations: int = 10,
    validate: bool = True,
) -> SingleResult:
    """
    Run a single benchmark configuration.

    Args:
        job: BenchmarkJob with factories
        n: Number of parallel models
        strategy: Strategy name ('baseline', 'grouped', etc.)
        params: Sweep params for this run
        device: 'cuda' or 'cpu'
        warmup: Warmup iterations
        iterations: Timing iterations
        validate: Run correctness validation (default True)

    Returns:
        SingleResult with timing and validation status
    """
    # Create N models
    models = [job.model_factory(**params).to(device).eval() for _ in range(n)]

    # Create input
    sample = job.input_factory(n=n, device=device, **params)
    inputs = [sample.clone() for _ in range(n)]

    # Baseline: sequential execution - get outputs for validation
    with torch.no_grad():
        baseline_outputs = [m(inp) for m, inp in zip(models, inputs)]

    def baseline_fn():
        for i, m in enumerate(models):
            m(inputs[i])

    baseline_ms = time_fn(baseline_fn, iterations=iterations, warmup=warmup, device=device)

    # Strategy-specific
    if strategy == 'baseline':
        return SingleResult(
            n=n,
            strategy='baseline',
            params=params,
            time_ms=baseline_ms,
            baseline_ms=baseline_ms,
            speedup=1.0,
            valid=True,
            validation_msg="OK",
        )

    # Build wide model with strategy
    wide_model = job.wide_factory(models, strategy).to(device).eval()
    packed = job.pack_fn(inputs)

    # Run validation if requested and validate_fn exists
    is_valid = True
    validation_msg = "OK"

    if validate and hasattr(job, 'validate_fn') and job.validate_fn is not None:
        with torch.no_grad():
            wide_output = wide_model(packed)

        is_valid, validation_msg = job.validate_fn(wide_output, baseline_outputs)

        if not is_valid:
            # Return immediately with validation failure - no timing
            return SingleResult(
                n=n,
                strategy=strategy,
                params=params,
                time_ms=float('inf'),
                baseline_ms=baseline_ms,
                speedup=0.0,
                valid=False,
                validation_msg=validation_msg,
            )

    def wide_fn():
        wide_model(packed)

    wide_ms = time_fn(wide_fn, iterations=iterations, warmup=warmup, device=device)
    speedup = baseline_ms / wide_ms if wide_ms > 0 else 0

    return SingleResult(
        n=n,
        strategy=strategy,
        params=params,
        time_ms=wide_ms,
        baseline_ms=baseline_ms,
        speedup=speedup,
        valid=is_valid,
        validation_msg=validation_msg,
    )


def run(
    job: BenchmarkJob,
    device: str = 'cuda',
    verbose: bool = True,
    warmup: int = 3,
    iterations: int = 10,
    validate: bool = True,
) -> BenchmarkResult:
    """
    Run complete benchmark sweep.

    Args:
        job: BenchmarkJob defining the sweep
        device: 'cuda' or 'cpu'
        verbose: Print progress (True=progress updates, False=silent)
        warmup: Warmup iterations per config
        iterations: Timing iterations per config
        validate: Run correctness validation (default True)

    Returns:
        BenchmarkResult with all measurements
    """
    if device == 'cuda' and not torch.cuda.is_available():
        if verbose:
            print("CUDA not available, falling back to CPU")
        device = 'cpu'

    results: List[SingleResult] = []
    sweep = job.sweep
    validation_failures = []

    # Generate all param combinations
    param_combos = _generate_param_combos(sweep)

    total_configs = len(sweep.n_values) * len(job.strategies) * len(param_combos)
    current = 0

    if verbose:
        print(f"Running {job.name}: {total_configs} configs")

    for n in sweep.n_values:
        if verbose:
            print(f"  N={n}", end=" ", flush=True)

        n_results = []

        for params in param_combos:
            for strategy in job.strategies:
                current += 1

                try:
                    result = run_single(
                        job, n, strategy, params,
                        device=device,
                        warmup=warmup,
                        iterations=iterations,
                        validate=validate,
                    )
                    results.append(result)
                    n_results.append(result)

                    # Track validation failures
                    if not result.valid:
                        validation_failures.append((n, strategy, result.validation_msg))

                except Exception as e:
                    if verbose:
                        import traceback
                        print(f"\n[ERROR] {strategy} failed:")
                        traceback.print_exc()
                        print()

        # Print best speedup for this N (only valid results)
        if verbose:
            non_baseline = [r for r in n_results if r.strategy != 'baseline' and r.valid]
            invalid = [r for r in n_results if not r.valid]

            if non_baseline:
                best = max(non_baseline, key=lambda r: r.speedup)
                status = f"→ {best.speedup:.2f}x ({best.strategy})"
                if invalid:
                    status += f" [{len(invalid)} invalid]"
                print(status)
            elif invalid:
                print(f"→ ALL INVALID ({len(invalid)} failures)")
            else:
                print()

    # Print validation summary
    if verbose and validation_failures:
        print(f"\n⚠ Validation failures ({len(validation_failures)}):")
        for n, strategy, msg in validation_failures[:5]:  # Show first 5
            print(f"  N={n}, {strategy}: {msg}")
        if len(validation_failures) > 5:
            print(f"  ... and {len(validation_failures) - 5} more")

    return BenchmarkResult(
        name=job.name,
        primitive=job.primitive,
        device=device,
        results=results,
    )


def _generate_param_combos(sweep) -> List[dict]:
    """Generate all parameter combinations from sweep."""
    import itertools

    # Get all sweep attributes except n_values
    # Keep plural names to match factory function signatures
    param_names = []
    param_values = []

    for attr in ['batch_sizes', 'channels', 'kernel_sizes', 'seq_lengths',
                 'heights', 'widths', 'd_model', 'n_heads', 'hidden_sizes',
                 'vocab_sizes', 'embedding_dims', 'num_groups']:
        vals = getattr(sweep, attr, None)
        if vals:
            param_names.append(attr)  # Keep original name
            param_values.append(vals)

    if not param_names:
        return [{}]

    combos = []
    for values in itertools.product(*param_values):
        combos.append(dict(zip(param_names, values)))

    return combos


__all__ = ['run', 'run_single', 'time_fn']