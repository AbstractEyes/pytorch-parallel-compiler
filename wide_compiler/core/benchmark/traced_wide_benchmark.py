"""
WideCompiler.core.benchmark.traced_wide_benchmark

Benchmarking and profiling utilities for TracedWideModel.
Test any model with the Wide traced compilation system.

Usage:
    from wide_compiler.core.traced_benchmark import benchmark_model

    # Benchmark a model class
    results = benchmark_model(MyModel, sample_input, n=10)

    # Or with existing instances
    results = benchmark_models(models, sample_input)

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Union, Type
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch import Tensor
import torch._dynamo

try:
    from .traced_wide import TracedWideModel
    from .wide_model import pack_inputs, unpack_outputs
    from .config import WideConfig, get_default_config
except ImportError:
    from wide_compiler.core.traced_wide import TracedWideModel
    from wide_compiler.core.wide_model import pack_inputs, unpack_outputs
    from wide_compiler.core.config import WideConfig, get_default_config


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class TimingResult:
    """Timing results for a single configuration."""
    name: str
    time_ms: float
    speedup_vs_baseline: float = 1.0
    speedup_vs_compiled: float = 1.0


@dataclass
class StageProfile:
    """Profile for a single stage in the Wide model."""
    name: str
    op_type: str
    time_ms: float
    percentage: float


@dataclass
class MemoryResult:
    """Memory usage results."""
    wide_forward_mb: float
    baseline_forward_mb: float
    difference_mb: float
    difference_pct: float


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    # Config
    n: int
    batch_size: int
    input_shape: tuple
    device: str

    # Correctness
    max_diff: float
    correct: bool

    # Timings
    timings: List[TimingResult] = field(default_factory=list)
    best_speedup: float = 1.0

    # Stage breakdown
    stages: List[StageProfile] = field(default_factory=list)
    total_stages: int = 0

    # Memory
    memory: Optional[MemoryResult] = None

    # Compilation
    graph_breaks: int = 0
    graphs_compiled: int = 0
    compile_success: bool = True

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 70,
            "BENCHMARK RESULTS",
            "=" * 70,
            f"Config: N={self.n}, B={self.batch_size}, shape={self.input_shape}",
            f"Device: {self.device}",
            "",
            f"Correctness: {'✓' if self.correct else '✗'} (max_diff={self.max_diff:.2e})",
            "",
            "Timings:",
        ]

        for t in self.timings:
            lines.append(f"  {t.name:<35} {t.time_ms:<10.2f} ms  {t.speedup_vs_compiled:.2f}x vs compiled")

        lines.append(f"\nBest speedup: {self.best_speedup:.2f}x")

        if self.stages:
            lines.append(f"\nTop 5 stages (of {self.total_stages}):")
            for s in self.stages[:5]:
                lines.append(f"  {s.name:<25} {s.op_type:<20} {s.time_ms:.3f} ms ({s.percentage:.1f}%)")

        if self.memory:
            lines.append(f"\nMemory:")
            lines.append(f"  Wide forward:     {self.memory.wide_forward_mb:.1f} MB")
            lines.append(f"  Baseline forward: {self.memory.baseline_forward_mb:.1f} MB")
            lines.append(
                f"  Difference:       {self.memory.difference_mb:+.1f} MB ({self.memory.difference_pct:+.1f}%)")

        lines.append(f"\nCompilation: {self.graphs_compiled} graphs, {self.graph_breaks} breaks")

        return "\n".join(lines)


# =============================================================================
# BENCHMARKING UTILITIES
# =============================================================================

def _get_device() -> str:
    """Get current device."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _benchmark_fn(fn: Callable, num_iters: int = 100, warmup: int = 20,
                  device: str = 'cuda') -> float:
    """Benchmark a function, return average time in seconds."""
    for _ in range(warmup):
        fn()

    if device == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iters):
        fn()

    if device == 'cuda':
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) / num_iters


def _profiled_forward(model: TracedWideModel, x: Tensor, device: str = 'cuda') -> Dict[str, float]:
    """Run forward with per-stage timing."""
    values = {model._input_name: x}
    timings = {}

    # Pre-populate attrs
    for attr_name in model._attr_names:
        safe_name = attr_name.replace('.', '_')
        buffer_name = f'_attr_{safe_name}'
        if hasattr(model, buffer_name):
            values[attr_name] = getattr(model, buffer_name)

    for node_name in model._execution_order:
        if node_name in model._attr_names:
            continue
        if node_name not in model.stage_info:
            continue

        safe_name = node_name.replace('.', '_')
        op = model.stages[safe_name] if safe_name in model.stages else None
        if op is None:
            continue

        # Gather args
        args = []
        for arg_name in model._node_args[node_name]:
            if isinstance(arg_name, str) and arg_name in values:
                args.append(values[arg_name])
            else:
                args.append(arg_name)

        # Gather kwargs
        kwargs = {}
        for k, v in model._node_kwargs.get(node_name, {}).items():
            if isinstance(v, str) and v in values:
                kwargs[k] = values[v]
            else:
                kwargs[k] = v

        # Time this stage
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if kwargs:
            values[node_name] = op(*args, **kwargs)
        elif len(args) == 1:
            values[node_name] = op(args[0])
        else:
            values[node_name] = op(*args)

        if device == 'cuda':
            torch.cuda.synchronize()
        timings[node_name] = (time.perf_counter() - t0) * 1000  # ms

    return timings


# =============================================================================
# MAIN BENCHMARK FUNCTIONS
# =============================================================================

def benchmark_models(
        models: List[nn.Module],
        sample_input: Tensor,
        config: Optional[WideConfig] = None,
        num_iters: int = 50,
        warmup: int = 10,
        verbose: bool = True,
) -> BenchmarkResult:
    """
    Benchmark a list of model instances with TracedWideModel.

    Args:
        models: List of N model instances (must be identical architecture)
        sample_input: Sample input tensor (single model input shape)
        config: WideConfig for compilation settings
        num_iters: Number of benchmark iterations
        warmup: Number of warmup iterations
        verbose: Print progress and results

    Returns:
        BenchmarkResult with all metrics
    """
    if config is None:
        config = get_default_config()

    n = len(models)
    device = _get_device()
    batch_size = sample_input.shape[0]
    input_shape = tuple(sample_input.shape[1:])

    if verbose:
        print("=" * 70)
        print("TracedWideModel Benchmark")
        print(f"Device: {device}")
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Config: N={n}, B={batch_size}, shape={input_shape}")
        print("=" * 70)

    # Ensure models on device and eval mode
    for m in models:
        m.to(device).eval()
    sample_input = sample_input.to(device)

    # Build Wide model
    if verbose:
        print("\nBuilding TracedWideModel...")

    try:
        wide_model = TracedWideModel.from_models(models, sample_input).to(device).eval()
        if verbose:
            print(f"✓ Built {len(wide_model.stages)} stages")
    except Exception as e:
        if verbose:
            print(f"✗ Build failed: {e}")
        return BenchmarkResult(
            n=n, batch_size=batch_size, input_shape=input_shape, device=device,
            max_diff=float('inf'), correct=False, compile_success=False
        )

    # Create test inputs
    inputs = [torch.randn_like(sample_input) for _ in range(n)]
    packed = pack_inputs(inputs)

    # Correctness check
    if verbose:
        print("\nVerifying correctness...")

    with torch.inference_mode():
        ref_outs = [models[i](inputs[i]) for i in range(n)]
        wide_out = wide_model(packed)
        wide_outs = unpack_outputs(wide_out, n)

    max_diff = max((ref_outs[i] - wide_outs[i]).abs().max().item() for i in range(n))
    correct = max_diff < config.validate_rtol

    if verbose:
        print(f"Max diff: {max_diff:.2e} {'✓' if correct else '✗'}")

    # Compile models
    timings = []

    if verbose:
        print("\nCompiling models...")

    torch._dynamo.reset()

    # Compile baseline
    models_compiled = [torch.compile(m, mode=config.compile_mode) for m in models]
    with torch.inference_mode():
        for _ in range(warmup):
            _ = [models_compiled[i](inputs[i]) for i in range(n)]

    # Compile wide
    compile_success = True
    try:
        wide_compiled = torch.compile(wide_model, mode=config.compile_mode)
        with torch.inference_mode():
            for _ in range(warmup):
                _ = wide_compiled(packed)
    except Exception as e:
        if verbose:
            print(f"Wide compile failed: {e}")
        wide_compiled = wide_model
        compile_success = False

    # Benchmark
    if verbose:
        print("\nBenchmarking...")

    with torch.inference_mode():
        t_baseline_eager = _benchmark_fn(
            lambda: [models[i](inputs[i]) for i in range(n)],
            num_iters, warmup, device
        )
        t_baseline_compiled = _benchmark_fn(
            lambda: [models_compiled[i](inputs[i]) for i in range(n)],
            num_iters, warmup, device
        )
        t_wide_eager = _benchmark_fn(
            lambda: wide_model(packed),
            num_iters, warmup, device
        )
        t_wide_compiled = _benchmark_fn(
            lambda: wide_compiled(packed),
            num_iters, warmup, device
        )

    timings = [
        TimingResult("N×Model (eager)", t_baseline_eager * 1000, 1.0, t_baseline_compiled / t_baseline_eager),
        TimingResult("N×Model (compiled)", t_baseline_compiled * 1000, t_baseline_eager / t_baseline_compiled, 1.0),
        TimingResult("Wide (eager)", t_wide_eager * 1000, t_baseline_eager / t_wide_eager,
                     t_baseline_compiled / t_wide_eager),
        TimingResult("Wide (compiled)", t_wide_compiled * 1000, t_baseline_eager / t_wide_compiled,
                     t_baseline_compiled / t_wide_compiled),
    ]

    best_speedup = t_baseline_compiled / t_wide_compiled

    if verbose:
        print(f"\n{'Method':<35} {'Time (ms)':<12} {'vs Compiled'}")
        print("-" * 60)
        for t in timings:
            print(f"{t.name:<35} {t.time_ms:<12.2f} {t.speedup_vs_compiled:.2f}x")

    # Stage profiling
    stages = []
    if verbose:
        print("\nProfiling stages...")

    with torch.inference_mode():
        for _ in range(warmup):
            _ = wide_model(packed)

        all_timings = []
        for _ in range(5):
            stage_times = _profiled_forward(wide_model, packed, device)
            all_timings.append(stage_times)

    # Average timings
    if all_timings and all_timings[0]:
        avg_timings = {}
        for key in all_timings[0].keys():
            avg_timings[key] = sum(t.get(key, 0) for t in all_timings) / len(all_timings)

        total_time = sum(avg_timings.values())
        sorted_timings = sorted(avg_timings.items(), key=lambda x: -x[1])

        for node_name, t in sorted_timings:
            safe_name = node_name.replace('.', '_')
            op = wide_model.stages[safe_name] if safe_name in wide_model.stages else None
            op_type = type(op).__name__ if op else "?"
            pct = (t / total_time * 100) if total_time > 0 else 0
            stages.append(StageProfile(node_name, op_type, t, pct))

    if verbose and stages:
        print(f"\nTop 5 slowest stages:")
        print(f"{'Stage':<25} {'Op Type':<20} {'Time (ms)':<12}")
        print("-" * 60)
        for s in stages[:5]:
            print(f"{s.name:<25} {s.op_type:<20} {s.time_ms:<12.4f}")

    # Memory profiling
    memory = None
    if device == 'cuda':
        if verbose:
            print("\nProfiling memory...")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        with torch.inference_mode():
            torch.cuda.reset_peak_memory_stats()
            _ = wide_model(packed)
            mem_wide = torch.cuda.max_memory_allocated() / 1024 ** 2

        with torch.inference_mode():
            torch.cuda.reset_peak_memory_stats()
            _ = [models[i](inputs[i]) for i in range(n)]
            mem_baseline = torch.cuda.max_memory_allocated() / 1024 ** 2

        diff = mem_wide - mem_baseline
        diff_pct = (diff / mem_baseline * 100) if mem_baseline > 0 else 0
        memory = MemoryResult(mem_wide, mem_baseline, diff, diff_pct)

        if verbose:
            print(f"Wide: {mem_wide:.1f} MB, Baseline: {mem_baseline:.1f} MB ({diff:+.1f} MB)")

    # Graph break analysis
    if verbose:
        print("\nAnalyzing graph breaks...")

    torch._dynamo.reset()
    graph_breaks = 0
    graphs_compiled = 0

    try:
        explanation = torch._dynamo.explain(wide_model)(packed)
        graph_breaks = explanation.graph_break_count
        graphs_compiled = explanation.graph_count
        if verbose:
            print(f"Graph breaks: {graph_breaks}, Graphs: {graphs_compiled}")
    except Exception as e:
        if verbose:
            print(f"Analysis failed: {e}")

    result = BenchmarkResult(
        n=n,
        batch_size=batch_size,
        input_shape=input_shape,
        device=device,
        max_diff=max_diff,
        correct=correct,
        timings=timings,
        best_speedup=best_speedup,
        stages=stages,
        total_stages=len(wide_model.stages),
        memory=memory,
        graph_breaks=graph_breaks,
        graphs_compiled=graphs_compiled,
        compile_success=compile_success,
    )

    if verbose:
        print("\n" + result.summary())

    return result


def benchmark_model(
        model_class: Type[nn.Module],
        sample_input: Tensor,
        n: int = 10,
        model_args: tuple = (),
        model_kwargs: Optional[Dict[str, Any]] = None,
        config: Optional[WideConfig] = None,
        num_iters: int = 50,
        warmup: int = 10,
        verbose: bool = True,
) -> BenchmarkResult:
    """
    Benchmark a model class with TracedWideModel.

    Args:
        model_class: Model class to instantiate
        sample_input: Sample input tensor
        n: Number of model copies to create
        model_args: Args to pass to model constructor
        model_kwargs: Kwargs to pass to model constructor
        config: WideConfig for compilation settings
        num_iters: Number of benchmark iterations
        warmup: Number of warmup iterations
        verbose: Print progress

    Returns:
        BenchmarkResult with all metrics
    """
    if model_kwargs is None:
        model_kwargs = {}

    device = _get_device()

    if verbose:
        print(f"Creating {n} instances of {model_class.__name__}...")

    models = [model_class(*model_args, **model_kwargs).to(device).eval() for _ in range(n)]

    return benchmark_models(
        models, sample_input, config, num_iters, warmup, verbose
    )


def profile_with_torch_profiler(
        wide_model: TracedWideModel,
        packed_input: Tensor,
        num_iters: int = 10,
        verbose: bool = True,
) -> Optional[Any]:
    """
    Profile Wide model with torch.profiler.

    Returns the profiler object for further analysis.
    """
    try:
        from torch.profiler import profile, ProfilerActivity, record_function
    except ImportError:
        if verbose:
            print("torch.profiler not available")
        return None

    device = _get_device()

    activities = [ProfilerActivity.CPU]
    if device == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    # Warmup
    with torch.inference_mode():
        for _ in range(10):
            _ = wide_model(packed_input)

    with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
    ) as prof:
        with torch.inference_mode():
            for _ in range(num_iters):
                with record_function("wide_forward"):
                    _ = wide_model(packed_input)

    if verbose:
        sort_key = "cuda_time_total" if device == 'cuda' else "cpu_time_total"
        print(prof.key_averages().table(sort_by=sort_key, row_limit=20))

    return prof


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark models with TracedWideModel')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50', 'mlp', 'resblock'],
                        help='Model to benchmark')
    parser.add_argument('--n', type=int, default=10, help='Number of model copies')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--iters', type=int, default=50, help='Benchmark iterations')
    parser.add_argument('--profile', action='store_true', help='Run torch.profiler')

    args = parser.parse_args()

    device = _get_device()
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    torch._dynamo.config.cache_size_limit = 128

    # Select model
    if args.model == 'resnet18':
        try:
            from torchvision.models import resnet18

            model_class = resnet18
            model_kwargs = {'weights': None}
            sample = torch.randn(args.batch, 3, 224, 224, device=device)
        except ImportError:
            print("torchvision required for resnet18")
            exit(1)

    elif args.model == 'resnet50':
        try:
            from torchvision.models import resnet50

            model_class = resnet50
            model_kwargs = {'weights': None}
            sample = torch.randn(args.batch, 3, 224, 224, device=device)
        except ImportError:
            print("torchvision required for resnet50")
            exit(1)

    elif args.model == 'mlp':
        class MLP(nn.Module):
            def __init__(self, d=256):
                super().__init__()
                self.fc1 = nn.Linear(d, d * 2)
                self.fc2 = nn.Linear(d * 2, d)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))


        model_class = MLP
        model_kwargs = {'d': 256}
        sample = torch.randn(args.batch, 256, device=device)

    elif args.model == 'resblock':
        class ResBlock(nn.Module):
            def __init__(self, c=64):
                super().__init__()
                self.conv1 = nn.Conv2d(c, c, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(c)
                self.conv2 = nn.Conv2d(c, c, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(c)

            def forward(self, x):
                identity = x
                out = torch.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                return torch.relu(out + identity)


        model_class = ResBlock
        model_kwargs = {'c': 64}
        sample = torch.randn(args.batch, 64, 32, 32, device=device)

    # Run benchmark
    result = benchmark_model(
        model_class,
        sample,
        n=args.n,
        model_kwargs=model_kwargs,
        num_iters=args.iters,
    )

    # Optional torch.profiler
    if args.profile and result.compile_success:
        print("\n" + "=" * 70)
        print("torch.profiler Analysis")
        print("=" * 70)

        models = [model_class(**model_kwargs).to(device).eval() for _ in range(args.n)]
        wide_model = TracedWideModel.from_models(models, sample).to(device).eval()
        inputs = [torch.randn_like(sample) for _ in range(args.n)]
        packed = pack_inputs(inputs)

        profile_with_torch_profiler(wide_model, packed)