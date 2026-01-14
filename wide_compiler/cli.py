"""
WideCompiler CLI

Benchmark and debug utilities for Wide model compilation.

Usage:
    python -m wide_compiler test
    python -m wide_compiler benchmark conv1d
    python -m wide_compiler benchmark conv1d --preset quick
    python -m wide_compiler benchmark all
    python -m wide_compiler trace --model resblock
    python -m wide_compiler info

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx

try:
    from .core import (
        TracedWideModel,
        print_trace,
    )
    from .core.ensemble_util import pack_inputs, unpack_outputs
    from .core.registry import list_registered
except ImportError:
    from wide_compiler.core import (
        TracedWideModel,
        print_trace,
    )
    from wide_compiler.core.ensemble_util import pack_inputs, unpack_outputs
    from wide_compiler.core.registry import list_registered


# =============================================================================
# SAMPLE MODELS (for test/trace commands)
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
                print(f"  OK {name}: max_diff={max_diff:.8f}")
                passed += 1
            else:
                print(f"  FAIL {name}: max_diff={max_diff:.8f} (too high)")
                failed += 1

        except Exception as e:
            print(f"  FAIL {name}: {e}")
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
    """Benchmark primitive strategies."""
    try:
        from .core.benchmark import benchmark, benchmark_all, list_primitives
    except ImportError:
        from wide_compiler.core.benchmark import benchmark, benchmark_all, list_primitives

    device = 'cpu' if args.cpu else 'cuda'
    available = list_primitives()

    # Handle 'all'
    if args.primitive == 'all':
        results = benchmark_all(
            preset=args.preset,
            device=device,
            verbose=True,
        )

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"  Crossover N: {result.crossover_n}")
            print(f"  Best speedup: {result.best_speedup:.2f}x")
            print(f"  Strategy wins: {result.strategy_wins}")

        if args.output:
            import json
            combined = {name: r.to_dict() for name, r in results.items()}
            with open(args.output, 'w') as f:
                json.dump(combined, f, indent=2)
            print(f"\nSaved to {args.output}")

        return 0

    # Single primitive
    primitive = args.primitive

    if primitive not in available:
        print(f"Unknown primitive: '{primitive}'")
        if available:
            print(f"Available: {', '.join(sorted(available))}")
        else:
            print("No primitives with benchmark interface registered.")
        print("\nUse 'all' to run all benchmarks:")
        print("  wide_compiler benchmark all")
        return 1

    # Parse overrides
    overrides = {}
    if args.n_values:
        overrides['n_values'] = [int(x) for x in args.n_values.split(',')]

    result = benchmark(
        primitive,
        preset=args.preset,
        device=device,
        verbose=True,
        **overrides,
    )

    # Print table
    print("\n" + result.table())

    # Save if requested
    if args.output:
        result.save(args.output)
        print(f"\nSaved to {args.output}")

    return 0


def cmd_info(args):
    """Show library info."""
    print("WideCompiler")
    print("=" * 60)
    print()
    print("Fuse N identical models into a single Wide model.")
    print("Uses grouped ops for compile-friendly batched execution.")
    print()

    print("Registered primitives (for compilation):")
    for name in sorted(list_registered()):
        print(f"  - {name}")
    print()

    # Show benchmark primitives
    try:
        from .core.benchmark import list_primitives
    except ImportError:
        try:
            from wide_compiler.core.benchmark import list_primitives
        except ImportError:
            list_primitives = None

    if list_primitives:
        available = list_primitives()
        if available:
            print("Benchmarkable primitives:")
            for name in sorted(available):
                print(f"  - {name}")
            print()

    print("Usage:")
    print("  wide_compiler test              # Run correctness tests")
    print("  wide_compiler benchmark conv1d  # Benchmark Conv1d strategies")
    print("  wide_compiler benchmark all     # Benchmark all primitives")
    print("  wide_compiler trace -m mlp      # Show FX trace")
    print("  wide_compiler info              # This help")
    print()
    print("GitHub: https://github.com/AbstractEyes/pytorch-parallel-compiler")
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
    trace_parser.add_argument('--model', '-m', default='mlp',
                             help='Model name (mlp, deep_mlp, resblock, convnet)')
    trace_parser.add_argument('--tabular', '-t', action='store_true',
                             help='Show tabular view')

    # benchmark - positional primitive argument
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark primitive strategies')
    bench_parser.add_argument('primitive', nargs='?', default='all',
                             help='Primitive to benchmark (conv1d, conv2d, linear, all)')
    bench_parser.add_argument('--preset', '-p', default='full',
                             help='Sweep preset (quick, full, ci)')
    bench_parser.add_argument('--n-values', '-n',
                             help='Override N values (comma-separated, e.g., "4,8,16,32")')
    bench_parser.add_argument('--output', '-o',
                             help='Save results to JSON file')
    bench_parser.add_argument('--cpu', action='store_true', help='Force CPU')

    # info
    subparsers.add_parser('info', help='Show library info')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        'test': cmd_test,
        'trace': cmd_trace,
        'benchmark': cmd_benchmark,
        'info': cmd_info,
    }

    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())