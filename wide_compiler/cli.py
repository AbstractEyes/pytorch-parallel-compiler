"""
WideCompiler CLI

Benchmark and debug utilities for Wide model compilation.

Usage:
    # Benchmark primitives
    python -m wide_compiler benchmark gru -p quick        # eager (no compile)
    python -m wide_compiler benchmark gru -p quick -c    # compiled (default)
    python -m wide_compiler benchmark gru -c reduce-overhead
    python -m wide_compiler benchmark linear -p full -c

    # Benchmark TracedWideModel with sample models
    python -m wide_compiler benchmark mlp --n 100
    python -m wide_compiler benchmark resblock --n 50 -c

    # Other commands
    python -m wide_compiler test
    python -m wide_compiler trace --model resblock
    python -m wide_compiler info

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

import argparse
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx

try:
    from .core import (
        TracedWideModel,
        pack_inputs,
        unpack_outputs,
        print_trace,
    )
    from .core.registry import list_registered
    from .core.benchmark import list_primitives
except ImportError:
    from wide_compiler.core import (
        TracedWideModel,
        pack_inputs,
        unpack_outputs,
        print_trace,
    )
    from wide_compiler.core.registry import list_registered
    from wide_compiler.core.benchmark import list_primitives


# =============================================================================
# SAMPLE MODELS
# =============================================================================

class MLP(nn.Module):
    """Simple MLP for testing."""
    def __init__(self, d_in=64, d_hidden=128, d_out=64):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class DeepMLP(nn.Module):
    """Deeper MLP."""
    def __init__(self, d=64, layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d, d) for _ in range(layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = F.gelu(layer(x))
        return x


class ResBlock(nn.Module):
    """ResBlock with skip connection."""
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class ConvNet(nn.Module):
    """Simple ConvNet."""
    def __init__(self, channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


MODELS = {
    'mlp': (MLP, lambda: torch.randn(4, 64)),
    'deep_mlp': (DeepMLP, lambda: torch.randn(4, 64)),
    'resblock': (ResBlock, lambda: torch.randn(4, 64, 16, 16)),
    'convnet': (ConvNet, lambda: torch.randn(4, 3, 32, 32)),
}


# =============================================================================
# COMMANDS
# =============================================================================

def cmd_test(args):
    """Run correctness tests."""
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Device: {device}")
    print("=" * 60)

    passed = 0
    failed = 0

    for name, (model_cls, sample_fn) in MODELS.items():
        print(f"\nTesting {name}...")

        try:
            N = 20
            models = [model_cls().to(device).eval() for _ in range(N)]
            sample = sample_fn().to(device)

            wide_model = TracedWideModel.from_models(models, sample).to(device).eval()

            # Verify
            B = 8
            if name in ('resblock', 'convnet'):
                inputs = [torch.randn_like(sample[:B]) for _ in range(N)]
            else:
                inputs = [torch.randn(B, 64, device=device) for _ in range(N)]

            packed = pack_inputs(inputs)

            with torch.no_grad():
                separate = [models[i](inputs[i]) for i in range(N)]
                wide_out = wide_model(packed)
                wide_outs = unpack_outputs(wide_out, N)

            max_diff = max((separate[i] - wide_outs[i]).abs().max().item() for i in range(N))

            if max_diff < 1e-3:
                print(f"  ✓ {name}: max_diff={max_diff:.8f}")
                passed += 1
            else:
                print(f"  ✗ {name}: max_diff={max_diff:.8f} (too high)")
                failed += 1

        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


def cmd_trace(args):
    """Show FX trace for a model."""
    if args.model not in MODELS:
        print(f"Unknown model: {args.model}")
        print(f"Available: {', '.join(MODELS.keys())}")
        return 1

    model_cls, sample_fn = MODELS[args.model]
    model = model_cls()

    print(f"Model: {args.model}")
    print("=" * 60)

    try:
        traced = fx.symbolic_trace(model)
        print(print_trace(traced))

        if args.tabular:
            try:
                print("\nDetailed graph:")
                traced.graph.print_tabular()
            except ImportError:
                print("(Install 'tabulate' for detailed view)")
    except Exception as e:
        print(f"Tracing failed: {e}")
        return 1

    return 0


def cmd_benchmark(args):
    """Benchmark Wide primitives or TracedWideModel."""
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'

    # Check if this is a primitive benchmark or TracedWideModel benchmark
    primitive = args.primitive
    PRIMITIVES = list_primitives()  # Get list from registry

    # If primitive is a known primitive, use primitive benchmark mode
    if primitive in PRIMITIVES:
        return cmd_benchmark_primitive(args, device)

    # TracedWideModel mode: use --model or primitive as model name
    model_name = args.model or primitive or 'mlp'
    if model_name not in MODELS:
        # Maybe they meant a primitive?
        if primitive:
            print(f"Unknown primitive or model: {primitive}")
            print(f"Primitives: {', '.join(sorted(PRIMITIVES))}")
            print(f"Sample models: {', '.join(MODELS.keys())}")
        else:
            print(f"Unknown model: {model_name}")
            print(f"Available: {', '.join(MODELS.keys())}")
        return 1

    return cmd_benchmark_traced(args, model_name, device)


def cmd_benchmark_primitive(args, device: str):
    """Benchmark a Wide primitive with the benchmark framework."""
    try:
        from .core.benchmark.benchmark_runner import run
        from .core.benchmark.benchmark_schema import CompilationMode, compilation_available
        from .core.benchmark.benchmark_registry import get_primitive
    except ImportError:
        try:
            from wide_compiler.core.benchmark.benchmark_runner import run
            from wide_compiler.core.benchmark.benchmark_schema import CompilationMode, compilation_available
            from wide_compiler.core.benchmark.benchmark_registry import get_primitive
        except ImportError as e:
            print(f"Benchmark framework not found. Install with benchmark support.")
            print(f"Error: {e}")
            return 1

    primitive = args.primitive
    preset = args.preset

    # Map compile string to enum
    compile_map = {
        'default': CompilationMode.DEFAULT,
        'eager': CompilationMode.EAGER,
        'reduce-overhead': CompilationMode.REDUCE_OVERHEAD,
        'max-autotune': CompilationMode.MAX_AUTOTUNE,
    }
    compilation = compile_map.get(args.compile, CompilationMode.EAGER)

    # Load primitive and create benchmark job using registry
    try:
        primitive_class = get_primitive(primitive)
        job = primitive_class.benchmark_job(preset)
    except KeyError as e:
        print(f"Unknown primitive: {primitive}")
        print(f"Available primitives: {', '.join(sorted(list_primitives()))}")
        return 1
    except ImportError as e:
        print(f"Could not import benchmark dependencies for {primitive}:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        return 1
    except AttributeError as e:
        print(f"Primitive {primitive} does not support benchmarking:")
        print(f"  {e}")
        return 1
    except Exception as e:
        print(f"Could not create benchmark job for {primitive}:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print compilation info
    compile_available = compilation_available()
    if compilation == CompilationMode.EAGER:
        actual_mode = "eager"
    elif compile_available:
        actual_mode = f"compiled ({compilation.value})"
    else:
        actual_mode = "eager (compile unavailable)"

    print(f"Primitive: {primitive}")
    print(f"Preset: {preset}")
    print(f"Device: {device}")
    print(f"Compilation: {actual_mode}")
    print()

    # Run benchmark
    result = run(
        job,
        device=device,
        verbose=not args.quiet,
        warmup=args.warmup,
        iterations=args.iters,
        validate=not args.no_validate,
        compilation=compilation,
    )

    # Print results
    print(result.summary())
    print(result.top_table(args.top))

    # Save if requested
    if args.output:
        result.save(args.output)
        print(f"\nSaved to {args.output}")

    return 0


def cmd_benchmark_traced(args, model_name: str, device: str):
    """Benchmark TracedWideModel with sample models (MLP, ResBlock, etc.)."""
    model_cls, sample_fn = MODELS[model_name]
    N = args.n or 100
    B = args.batch

    print(f"Benchmark: {model_name} (TracedWideModel)")
    print(f"Device: {device}")
    print(f"N models: {N}")
    print(f"Batch size: {B}")
    print("=" * 60)

    # Create models
    models = [model_cls().to(device).eval() for _ in range(N)]
    sample = sample_fn().to(device)

    # Build wide model
    print("\nBuilding wide model...")
    wide_model = TracedWideModel.from_models(models, sample).to(device).eval()
    print(wide_model.summary())

    # Prepare inputs
    if model_name in ('resblock', 'convnet'):
        inputs = [torch.randn(B, *sample.shape[1:], device=device) for _ in range(N)]
    else:
        inputs = [torch.randn(B, 64, device=device) for _ in range(N)]
    packed = pack_inputs(inputs)

    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        for _ in range(5):
            wide_model(packed)
            models[0](inputs[0])

    if device == 'cuda':
        torch.cuda.synchronize()

    iters = args.iters
    compile_mode = args.compile

    # Determine if we should run compiled based on --compile flag
    run_compiled = compile_mode != 'eager' and device == 'cuda'

    # Eager benchmark
    print("\n--- Eager Mode ---")
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(iters):
            for i in range(N):
                models[i](inputs[i])
        if device == 'cuda':
            torch.cuda.synchronize()
        seq_time = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(iters):
            wide_model(packed)
        if device == 'cuda':
            torch.cuda.synchronize()
        wide_time = time.perf_counter() - start

    print(f"Sequential ({N}x): {seq_time*1000:.2f}ms")
    print(f"Wide:              {wide_time*1000:.2f}ms")
    print(f"Speedup:           {seq_time/wide_time:.2f}x")

    # Compiled benchmark
    if run_compiled:
        try:
            # 'default' means no mode arg (use torch.compile defaults)
            if compile_mode == 'default':
                print(f"\n--- Compiled (default) ---")
                compiled_wide = torch.compile(wide_model)
                compiled_models = [torch.compile(m) for m in models]
            else:
                print(f"\n--- Compiled ({compile_mode}) ---")
                compiled_wide = torch.compile(wide_model, mode=compile_mode)
                compiled_models = [torch.compile(m, mode=compile_mode) for m in models]

            # Warmup compiled
            with torch.no_grad():
                for _ in range(10):
                    compiled_wide(packed)
                    for i, m in enumerate(compiled_models):
                        m(inputs[i])

            if device == 'cuda':
                torch.cuda.synchronize()

            with torch.no_grad():
                # Baseline compiled
                start = time.perf_counter()
                for _ in range(iters):
                    for i, m in enumerate(compiled_models):
                        m(inputs[i])
                if device == 'cuda':
                    torch.cuda.synchronize()
                seq_compiled_time = time.perf_counter() - start

                # Wide compiled
                start = time.perf_counter()
                for _ in range(iters):
                    compiled_wide(packed)
                if device == 'cuda':
                    torch.cuda.synchronize()
                wide_compiled_time = time.perf_counter() - start

            print(f"Sequential compiled: {seq_compiled_time*1000:.2f}ms")
            print(f"Wide compiled:       {wide_compiled_time*1000:.2f}ms")
            print(f"Speedup (compiled):  {seq_compiled_time/wide_compiled_time:.2f}x")

        except Exception as e:
            print(f"Compile failed: {e}")

    return 0


def cmd_info(args):
    """Show library info."""
    print("WideCompiler")
    print("=" * 60)
    print()
    print("Fuse N identical models into a single Wide model.")
    print("Uses grouped ops for compile-friendly batched execution.")
    print()
    print("Registered primitives:")
    for name in sorted(list_registered()):
        print(f"  - {name}")
    print()
    print("Benchmarkable primitives:")
    print("  - gru, lstm, linear, conv2d, conv1d, attention, embedding")
    print()
    print("Compilation modes (-c / --compile):")
    print("  -c               Use default compilation (torch.compile())")
    print("  -c default       Same as above")
    print("  -c eager         No compilation (default if -c not specified)")
    print("  -c reduce-overhead  Optimized for low latency")
    print("  -c max-autotune  Maximum optimization (slow compile)")
    print()

    # System info
    print("System:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}", end="")
    if torch.cuda.is_available():
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()

    # Check torch.compile
    try:
        major = int(torch.__version__.split('.')[0])
        compile_available = major >= 2
    except:
        compile_available = False
    print(f"  torch.compile: {'available' if compile_available else 'not available'}")
    print()
    print("Typical speedups:")
    print("  - Eager: 1-2x")
    print("  - Compiled (CUDA): 2-10x")
    print()
    print("Usage:")
    print("  python -m wide_compiler benchmark gru -p quick        # eager")
    print("  python -m wide_compiler benchmark gru -p quick -c     # compiled (default)")
    print("  python -m wide_compiler benchmark gru -c reduce-overhead")
    print("  python -m wide_compiler benchmark mlp --n 100 -c      # TracedWideModel")
    print()
    print("GitHub: https://github.com/AbstractEyes/wide-compiler")
    return 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog='wide_compiler',
        description='WideCompiler - Fuse N models into one Wide model'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # test
    test_parser = subparsers.add_parser('test', help='Run correctness tests')
    test_parser.add_argument('--cpu', action='store_true', help='Force CPU')

    # trace
    trace_parser = subparsers.add_parser('trace', help='Show FX trace for a model')
    trace_parser.add_argument('--model', '-m', default='mlp', help='Model name')
    trace_parser.add_argument('--tabular', '-t', action='store_true', help='Show tabular view')

    # benchmark <primitive> - NEW: benchmark Wide primitives with compilation
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark Wide primitives')
    bench_parser.add_argument('primitive', nargs='?', default=None,
                              help='Primitive (gru, lstm, linear, conv2d, attention) or sample model (mlp, resblock, convnet)')
    bench_parser.add_argument('-p', '--preset', default='quick', help='Preset (quick, ci, full)')
    bench_parser.add_argument('-c', '--compile', nargs='?', const='default', default='eager',
                              choices=['default', 'eager', 'reduce-overhead', 'max-autotune'],
                              help='Compilation mode. -c alone uses default, or specify mode.')
    bench_parser.add_argument('--model', '-m', default=None, help='Sample model name (mlp, resblock, convnet, deep_mlp)')
    bench_parser.add_argument('--n', type=int, default=None, help='Number of models for TracedWideModel')
    bench_parser.add_argument('--batch', '-b', type=int, default=32, help='Batch size')
    bench_parser.add_argument('--iters', '-i', type=int, default=100, help='Iterations')
    bench_parser.add_argument('--warmup', '-w', type=int, default=3, help='Warmup iterations')
    bench_parser.add_argument('--top', type=int, default=8, help='Show top N results')
    bench_parser.add_argument('--no-validate', action='store_true', help='Skip validation')
    bench_parser.add_argument('-q', '--quiet', action='store_true', help='Quiet mode')
    bench_parser.add_argument('-o', '--output', help='Save results to JSON file')
    bench_parser.add_argument('--cpu', action='store_true', help='Force CPU')

    # info
    subparsers.add_parser('info', help='Show library info')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == 'test':
        return cmd_test(args)
    elif args.command == 'trace':
        return cmd_trace(args)
    elif args.command == 'benchmark':
        return cmd_benchmark(args)
    elif args.command == 'info':
        return cmd_info(args)

    return 0


if __name__ == '__main__':
    sys.exit(main())