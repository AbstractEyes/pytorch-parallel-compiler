"""
Benchmark script for A100 GPU - Colab-friendly.

This script is designed to run in Google Colab with minimal setup.
Just copy-paste this entire file into a Colab cell and run.

No arguments needed - runs quick preset by default.
"""

import torch
import json
from datetime import datetime

# Check GPU
print("="*80)
print("GPU INFORMATION")
print("="*80)
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"PyTorch Version: {torch.__version__}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print()

# Import after GPU check
from wide_compiler.core.benchmark import benchmark_multi

# All targets
PRIMITIVES = [
    'linear', 'conv1d', 'conv2d', 'conv3d', 'convtranspose1d', 'convtranspose2d',
    'batchnorm1d', 'batchnorm2d', 'batchnorm3d', 'layernorm', 'groupnorm',
    'instancenorm2d', 'rmsnorm', 'ada_layer_norm_zero_single',
    'embedding', 'mlp_embedder', 'attention', 'multiheadcrossattention',
    'gru', 'lstm', 'rnn', 'prelu', 'dropout', 'adaptiveavgpool2d',
]

BLOCKS = [
    'mlp_block', 'attention_block', 'joint_attention',
    'double_stream_block', 'single_stream_block',
]

ALL_TARGETS = PRIMITIVES + BLOCKS

print("="*80)
print("RUNNING BENCHMARKS (Quick Preset)")
print("="*80)
print(f"Total: {len(ALL_TARGETS)} components (24 primitives + 5 blocks)")
print("This will take ~5-10 minutes...\n")

# Run benchmarks
results = benchmark_multi(ALL_TARGETS, preset='quick', device='cuda', verbose=True)

# Format results table
def format_table(results, targets):
    lines = ["\n| Primitive/Block | N=4 | N=8 | N=16 | N=32 | Best Strategy |"]
    lines.append("|-----------------|-----|-----|------|------|---------------|")

    for name in targets:
        result = results.get(name)
        if result is None or result.best_result is None:
            lines.append(f"| {name:15s} | N/A | N/A | N/A  | N/A  | - |")
            continue

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

        best_strat = result.best_result.strategy
        lines.append(f"| {name:15s} | {n4:5s} | {n8:5s} | {n16:6s} | {n32:6s} | {best_strat} |")

    return "\n".join(lines)

# Print tables
print("\n" + "="*80)
print("PRIMITIVES RESULTS")
print("="*80)
print(format_table(results, PRIMITIVES))

print("\n" + "="*80)
print("BLOCKS RESULTS")
print("="*80)
print(format_table(results, BLOCKS))

# Statistics
print("\n" + "="*80)
print("STATISTICS")
print("="*80)

passed = sum(1 for r in results.values() if r and r.best_result and r.best_result.valid)
failed = sum(1 for r in results.values() if r and (not r.best_result or not r.best_result.valid))

print(f"Passed:  {passed}/{len(ALL_TARGETS)}")
print(f"Failed:  {failed}/{len(ALL_TARGETS)}")

# Top 10
top_all = []
for name, result in results.items():
    if result and result.best_result and result.best_result.valid:
        top_all.append((name, result.best_result.speedup, result.best_result.n, result.best_result.strategy))

top_all.sort(key=lambda x: x[1], reverse=True)

print("\nTop 10 Overall:")
for i, (name, speedup, n, strategy) in enumerate(top_all[:10], 1):
    print(f"  {i:2d}. {name:30s} {speedup:6.2f}x @ N={n:2d} ({strategy})")

# Save JSON
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"benchmark_results_a100_quick_{timestamp}.json"

output = {
    'timestamp': timestamp,
    'gpu': torch.cuda.get_device_name(0),
    'cuda_version': torch.version.cuda,
    'pytorch_version': torch.__version__,
    'preset': 'quick',
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

# Download instructions for Colab
print("\n" + "="*80)
print("COLAB: Download Results")
print("="*80)
print("Run this to download the JSON file:")
print(f"  from google.colab import files")
print(f"  files.download('{filename}')")

print("\n" + "="*80)
print("DONE!")
print("="*80)
