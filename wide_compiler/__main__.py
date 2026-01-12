"""
WideCompiler CLI

Usage:
    python -m wide_compiler --help
    python -m wide_compiler benchmark --n 100 --model mlp
    python -m wide_compiler trace --model resblock
    python -m wide_compiler test

Copyright 2025 AbstractPhil
Apache License 2.0
"""

import argparse
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx

from wide_compiler import (
    TracedWideModel,
    pack_inputs,
    unpack_outputs,
    print_trace,
)


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
    """Benchmark wide model speedup."""
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'

    if args.model not in MODELS:
        print(f"Unknown model: {args.model}")
        print(f"Available: {', '.join(MODELS.keys())}")
        return 1

    model_cls, sample_fn = MODELS[args.model]
    N = args.n
    B = args.batch

    print(f"Benchmark: {args.model}")
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
    if args.model in ('resblock', 'convnet'):
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

    # Eager benchmark
    print("\n--- Eager Mode ---")
    iters = args.iters

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
    if args.compile and device == 'cuda':
        print("\n--- Compiled (reduce-overhead) ---")
        try:
            compiled_wide = torch.compile(wide_model, mode='reduce-overhead')
            compiled_single = torch.compile(models[0], mode='reduce-overhead')

            # Warmup compiled
            with torch.no_grad():
                for _ in range(5):
                    compiled_wide(packed)
                    compiled_single(inputs[0])

            if device == 'cuda':
                torch.cuda.synchronize()

            with torch.no_grad():
                start = time.perf_counter()
                for _ in range(iters):
                    compiled_single(inputs[0])
                if device == 'cuda':
                    torch.cuda.synchronize()
                single_time = time.perf_counter() - start

                start = time.perf_counter()
                for _ in range(iters):
                    compiled_wide(packed)
                if device == 'cuda':
                    torch.cuda.synchronize()
                wide_time = time.perf_counter() - start

            print(f"Single compiled x{iters}: {single_time*1000:.2f}ms")
            print(f"Wide compiled x{iters}:   {wide_time*1000:.2f}ms")
            print(f"Effective speedup:     {(single_time * N)/wide_time:.2f}x (vs N sequential)")

        except Exception as e:
            print(f"Compile failed: {e}")

    return 0


def cmd_info(args):
    """Show library info."""

    print(f"WideCompiler")
    print("=" * 60)
    print()
    print("Fuse N identical models into a single Wide model.")
    print("Uses grouped ops for compile-friendly batched execution.")
    print()
    print("Supported layers:")
    print("  - Linear (via grouped Conv1d)")
    print("  - Conv1d, Conv2d (grouped)")
    print("  - BatchNorm1d, BatchNorm2d")
    print("  - LayerNorm")
    print("  - Embedding")
    print("  - All F.* functionals (relu, gelu, etc.)")
    print("  - Binary ops (+, -, *, /, @)")
    print()
    print("Typical speedups:")
    print("  - Eager: 2-5x")
    print("  - Compiled (CUDA): 20-40x")
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
    trace_parser.add_argument('--model', '-m', default='mlp', help='Model name')
    trace_parser.add_argument('--tabular', '-t', action='store_true', help='Show tabular view')

    # benchmark
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark speedup')
    bench_parser.add_argument('--model', '-m', default='mlp', help='Model name')
    bench_parser.add_argument('--n', type=int, default=100, help='Number of models')
    bench_parser.add_argument('--batch', '-b', type=int, default=32, help='Batch size')
    bench_parser.add_argument('--iters', '-i', type=int, default=100, help='Iterations')
    bench_parser.add_argument('--compile', '-c', action='store_true', help='Also benchmark compiled')
    bench_parser.add_argument('--cpu', action='store_true', help='Force CPU')

    # info
    info_parser = subparsers.add_parser('info', help='Show library info')

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