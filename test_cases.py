"""
Test cases for all WideCompiler primitives and blocks.

This file provides reusable test cases for benchmarking and validation.
Can be run standalone or imported for programmatic testing.

Usage:
    python test_cases.py                    # Run all tests
    python test_cases.py --primitives       # Test primitives only
    python test_cases.py --blocks           # Test blocks only
    python test_cases.py --preset quick     # Use quick preset (default: quick)
"""

from wide_compiler.core.benchmark import benchmark_multi

# All primitive benchmarks (24 total)
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

# All block benchmarks (5 total)
BLOCKS = [
    'mlp_block',
    'attention_block',
    'joint_attention',
    'double_stream_block',
    'single_stream_block',
]

# Known issues (validation passes but slower than baseline at low N)
KNOWN_SLOWDOWNS = {
    'gru': 'Slower than baseline until N>=32 (cudnn optimization)',
    'lstm': 'Slower than baseline until N>=32 (cudnn optimization)',
    'rnn': 'Slower than baseline until N<32 (cudnn optimization)',
    'prelu': 'Slower than baseline at N=8,16 (kernel launch overhead)',
    'batchnorm3d': 'Always slower (no grouped implementation available)',
}


def run_tests(targets, preset='quick', device='cuda', verbose=True):
    """
    Run benchmark tests on specified targets.

    Args:
        targets: List of primitive/block names to test
        preset: 'quick', 'full', or 'ci'
        device: 'cuda' or 'cpu'
        verbose: Print progress

    Returns:
        Dict[str, BenchmarkResult]
    """
    print('='*80)
    print(f'TESTING {len(targets)} TARGETS (preset={preset}, device={device})')
    print('='*80)

    results = benchmark_multi(targets, preset=preset, device=device, verbose=verbose)

    # Analyze results
    passed = []
    failed = []
    no_results = []
    slowdowns = []

    for name in targets:
        result = results.get(name)
        if result is None:
            no_results.append(name)
        elif result.best_result:
            best = result.best_result
            if best.valid:
                if best.speedup < 1.0:
                    slowdowns.append((name, best.speedup, best.n))
                else:
                    passed.append((name, best.speedup, best.n))
            else:
                failed.append((name, 'validation_failed'))
        else:
            no_results.append(name)

    # Print summary
    print('\n' + '='*80)
    print('RESULTS SUMMARY')
    print('='*80)

    print(f'\nPASSED ({len(passed)}):')
    for name, speedup, n in sorted(passed):
        print(f'  {name:30s} {speedup:6.2f}x @ N={n}')

    if slowdowns:
        print(f'\nSLOWDOWNS ({len(slowdowns)}):')
        for name, speedup, n in sorted(slowdowns):
            note = f' ({KNOWN_SLOWDOWNS[name]})' if name in KNOWN_SLOWDOWNS else ''
            print(f'  {name:30s} {speedup:6.2f}x @ N={n}{note}')

    if failed:
        print(f'\nFAILED ({len(failed)}):')
        for name, reason in failed:
            print(f'  {name:30s} {reason}')

    if no_results:
        print(f'\nNO RESULTS ({len(no_results)}):')
        for name in no_results:
            print(f'  {name}')

    # Overall status
    total = len(targets)
    success = len(passed) + len(slowdowns)
    print(f'\n{"="*80}')
    print(f'OVERALL: {success}/{total} passed ({success*100//total}%)')
    print(f'{"="*80}')

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test WideCompiler primitives and blocks')
    parser.add_argument('--primitives', action='store_true', help='Test primitives only')
    parser.add_argument('--blocks', action='store_true', help='Test blocks only')
    parser.add_argument('--preset', default='quick', choices=['quick', 'full', 'ci'],
                        help='Benchmark preset (default: quick)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    args = parser.parse_args()

    # Determine targets
    if args.primitives and not args.blocks:
        targets = PRIMITIVES
    elif args.blocks and not args.primitives:
        targets = BLOCKS
    else:
        targets = PRIMITIVES + BLOCKS

    # Run tests
    run_tests(targets, preset=args.preset, device=args.device, verbose=not args.quiet)


if __name__ == '__main__':
    main()
