"""
Benchmark script for A100 GPU.

Run this on an A100 to generate official benchmark results for v0.7.0.
Results will be saved to benchmark_results_a100.json with timestamp.

Usage:
    python benchmark_a100.py                    # Quick preset (recommended)
    python benchmark_a100.py --preset full      # Full sweep (takes longer)
    python benchmark_a100.py --compile          # With torch.compile (recommended for A100)
"""

import argparse
import json
import torch
from datetime import datetime
from wide_compiler.core.benchmark import benchmark_multi

# All 24 primitives
PRIMITIVES = [
    'linear',
    'conv1d',
    'conv2d',
    'conv3d',
    'convtranspose1d',
    'convtranspose2d',
    'batchnorm1d',
    'batchnorm2d',
    'batchnorm3d',
    'layernorm',
    'groupnorm',
    'instancenorm2d',
    'rmsnorm',
    'ada_layer_norm_zero_single',
    'embedding',
    'mlp_embedder',
    'attention',
    'multiheadcrossattention',
    'gru',
    'lstm',
    'rnn',
    'prelu',
    'dropout',
    'adaptiveavgpool2d',
]

# All 5 blocks
BLOCKS = [
    'mlp_block',
    'attention_block',
    'joint_attention',
    'double_stream_block',
    'single_stream_block',
]

ALL_TARGETS = PRIMITIVES + BLOCKS


def print_gpu_info():
    """Print GPU information."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return False

    print("="*80)
    print("GPU INFORMATION")
    print("="*80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print()

    device_name = torch.cuda.get_device_name(0).lower()
    if 'a100' not in device_name:
        print(f"WARNING: Expected A100, got {torch.cuda.get_device_name(0)}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False

    return True


def format_result_table(results, targets):
    """Format results as markdown table."""
    lines = []
    lines.append("\n| Primitive/Block | N=4 | N=8 | N=16 | N=32 | Best Strategy |")
    lines.append("|-----------------|-----|-----|------|------|---------------|")

    for name in targets:
        result = results.get(name)
        if result is None or result.best_result is None:
            lines.append(f"| {name:15s} | N/A | N/A | N/A  | N/A  | - |")
            continue

        # Extract speedups at each N
        speedups = {}
        for r in result.all_results:
            if r.valid:
                n = r.n
                if n not in speedups or r.speedup > speedups[n][0]:
                    speedups[n] = (r.speedup, r.strategy)

        n4 = f"{speedups.get(4, (0, ''))[0]:.2f}x" if 4 in speedups else "N/A"
        n8 = f"{speedups.get(8, (0, ''))[0]:.2f}x" if 8 in speedups else "N/A"
        n16 = f"{speedups.get(16, (0, ''))[0]:.2f}x" if 16 in speedups else "N/A"
        n32 = f"{speedups.get(32, (0, ''))[0]:.2f}x" if 32 in speedups else "N/A"

        best = result.best_result
        best_strat = best.strategy

        lines.append(f"| {name:15s} | {n4:5s} | {n8:5s} | {n16:6s} | {n32:6s} | {best_strat} |")

    return "\n".join(lines)


def save_results(results, args):
    """Save results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    preset_str = f"_{args.preset}" if args.preset != 'quick' else ""
    compile_str = "_compiled" if args.compile else ""
    filename = f"benchmark_results_a100{preset_str}{compile_str}_{timestamp}.json"

    # Convert results to serializable format
    output = {
        'timestamp': timestamp,
        'gpu': torch.cuda.get_device_name(0),
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
        'preset': args.preset,
        'compiled': args.compile,
        'results': {}
    }

    for name, result in results.items():
        if result is None:
            output['results'][name] = None
            continue

        output['results'][name] = {
            'all_results': [
                {
                    'n': r.n,
                    'strategy': r.strategy,
                    'speedup': r.speedup,
                    'time_ms': r.time_ms,
                    'baseline_ms': r.baseline_ms,
                    'valid': r.valid,
                }
                for r in result.all_results
            ],
            'best': {
                'n': result.best_result.n,
                'strategy': result.best_result.strategy,
                'speedup': result.best_result.speedup,
                'time_ms': result.best_result.time_ms,
                'baseline_ms': result.best_result.baseline_ms,
            } if result.best_result else None
        }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to: {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(description='Benchmark WideCompiler on A100')
    parser.add_argument('--preset', default='quick', choices=['quick', 'full', 'ci'],
                        help='Benchmark preset (default: quick)')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile (recommended for A100)')
    parser.add_argument('--primitives-only', action='store_true',
                        help='Only benchmark primitives')
    parser.add_argument('--blocks-only', action='store_true',
                        help='Only benchmark blocks')
    args = parser.parse_args()

    # Check GPU
    if not print_gpu_info():
        return

    # Determine targets
    if args.primitives_only:
        targets = PRIMITIVES
        print(f"Benchmarking 24 primitives only")
    elif args.blocks_only:
        targets = BLOCKS
        print(f"Benchmarking 5 blocks only")
    else:
        targets = ALL_TARGETS
        print(f"Benchmarking all 29 components (24 primitives + 5 blocks)")

    print(f"Preset: {args.preset}")
    print(f"Compile: {'enabled' if args.compile else 'disabled'}")
    print()

    # Confirm
    response = input("Start benchmark? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Run benchmarks
    print("\n" + "="*80)
    print("RUNNING BENCHMARKS")
    print("="*80)
    print("This may take 5-30 minutes depending on preset...\n")

    results = benchmark_multi(
        targets,
        preset=args.preset,
        device='cuda',
        verbose=True,
        compile_mode='default' if args.compile else None
    )

    # Print summary tables
    print("\n" + "="*80)
    print("PRIMITIVES RESULTS")
    print("="*80)
    print(format_result_table(results, PRIMITIVES))

    if not args.primitives_only:
        print("\n" + "="*80)
        print("BLOCKS RESULTS")
        print("="*80)
        print(format_result_table(results, BLOCKS))

    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)

    passed = sum(1 for r in results.values() if r and r.best_result and r.best_result.valid)
    failed = sum(1 for r in results.values() if r and (not r.best_result or not r.best_result.valid))
    no_results = sum(1 for r in results.values() if not r)

    print(f"Passed:      {passed}/{len(targets)}")
    print(f"Failed:      {failed}/{len(targets)}")
    print(f"No results:  {no_results}/{len(targets)}")

    # Top performers
    top_primitives = []
    for name in PRIMITIVES:
        result = results.get(name)
        if result and result.best_result and result.best_result.valid:
            top_primitives.append((name, result.best_result.speedup, result.best_result.n))

    top_primitives.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 10 Primitives:")
    for i, (name, speedup, n) in enumerate(top_primitives[:10], 1):
        print(f"  {i:2d}. {name:30s} {speedup:6.2f}x @ N={n}")

    if not args.primitives_only:
        top_blocks = []
        for name in BLOCKS:
            result = results.get(name)
            if result and result.best_result and result.best_result.valid:
                top_blocks.append((name, result.best_result.speedup, result.best_result.n))

        top_blocks.sort(key=lambda x: x[1], reverse=True)

        print("\nTop Blocks:")
        for i, (name, speedup, n) in enumerate(top_blocks, 1):
            print(f"  {i}. {name:30s} {speedup:6.2f}x @ N={n}")

    # Save results
    filename = save_results(results, args)

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"Results saved to: {filename}")
    print("\nNext steps:")
    print("1. Review the results above")
    print("2. Update README.md and CLAUDE.md with A100 numbers")
    print("3. Archive the JSON file for reference")


if __name__ == '__main__':
    main()
