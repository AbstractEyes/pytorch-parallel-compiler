# CLAUDE.md - WideCompiler Quick Reference

VERSION = 0.4.0

Read this first when working on this codebase.

## What This Project Does

Fuses N identical PyTorch models into ONE wide model. Instead of N sequential forward passes, one batched forward pass using grouped operations.

**Key insight:** Reshape N models into batch dimension, run single fused kernel.

## Core Concept

```
N separate models:        Wide model:
[Model_0] → out_0        [WideModel] → [out_0, out_1, ..., out_N]
[Model_1] → out_1         (single forward pass)
...
[Model_N] → out_N
```

**Memory layout:** `[B, N*C, ...]` - N models' channels concatenated.

## Entry Point

```python
import wide_compiler

# This is the main API
wide = wide_compiler.compile(models, sample_input)
```

All roads lead to `TracedWideModel.from_models()` in `core/traced_wide.py`.

## File Map

```
wide_compiler/
├── __init__.py          → Package exports
├── api.py               → compile(), WideBuilder, pack(), unpack()
├── cli.py               → CLI commands (test, benchmark, trace, info)
├── __main__.py          → Delegates to cli.main()
└── core/
    ├── config.py        → WideConfig dataclass
    ├── registry.py      → Maps 'Linear' → WideLinear.from_modules
    ├── traced_wide.py   → FX tracing, graph execution (THE CORE)
    ├── wide_model.py    → pack_inputs(), unpack_outputs(), tree utils
    ├── benchmark/       → Primitive benchmarking system (NEW)
    │   ├── __init__.py
    │   ├── benchmark_api.py      → run_benchmark(), list_primitives()
    │   ├── benchmark_runner.py   → Execution engine
    │   ├── benchmark_schema.py   → BenchmarkJob, SweepParams, results
    │   └── benchmark_registry.py → Auto-discovers primitives
    └── primitives/      → One file per Wide op
        ├── wide_attention.py   → MHA via batched SDPA (11x speedup)
        ├── wide_linear.py      → Linear via einsum
        ├── wide_conv1d.py      → Conv1d with groups=N
        ├── wide_conv2d.py      → Conv2d with groups=N
        ├── wide_embedding.py   → Batched index lookup (6x speedup)
        ├── wide_layernorm.py   → Per-group normalization
        ├── wide_batchnorm_1d.py
        └── wide_batchnorm_2d.py
```

## How It Works

### 1. Trace (traced_wide.py)
```python
traced = fx.symbolic_trace(template_model)
# Captures: call_module, call_function, call_method
```

### 2. Build Wide Ops
```python
for node in traced.graph.nodes:
    if node.op == 'call_module':
        modules = [m.get_submodule(path) for m in models]
        wide_op = WIDE_BUILDERS[module_type](modules)
```

### 3. Graph Execution
```python
def forward(self, x):
    values = {self._input_name: x}
    for node_name in self._execution_order:
        args = [values[arg] for arg in self._node_args[node_name]]
        values[node_name] = self.stages[node_name](*args)
    return values[self._output_name]
```

## Wide Primitives Pattern

Every primitive follows this pattern:

```python
class WideLinear(nn.Module):
    """N parallel Linear layers fused."""
    
    def __init__(self, n, in_features, out_features, strategy='auto'):
        self.weight = nn.Parameter(torch.empty(n, out_features, in_features))
        self._strategy = strategy
    
    def forward(self, x):
        # x: [B, N*D_in] → [B, N*D_out]
        if self._use_einsum:
            return self._forward_einsum(x)
        return self._forward_sequential(x)
    
    @classmethod
    def from_modules(cls, modules: List[nn.Linear], strategy='auto'):
        # Stack weights from N modules
```

**Critical:** Each primitive has multiple strategies. AUTO selects the fastest.

## Strategy Pattern (NEW in 0.4.0)

Each primitive defines strategies with different performance tradeoffs:

| Primitive | Strategies | Default |
|-----------|------------|---------|
| WideAttention | fused, sequential | fused |
| WideLinear | einsum, sequential | einsum |
| WideEmbedding | indexed, gather, sequential | indexed |
| WideConv1d | grouped, sequential | grouped |
| WideConv2d | grouped, channels_last, sequential | grouped |

```python
# Override strategy
wide = WideLinear.from_modules(modules, strategy='einsum')
```

## Benchmark System (NEW in 0.4.0)

Each primitive defines its own benchmark interface:

```python
class WideAttention(nn.Module):
    BENCHMARK_STRATEGIES = ['baseline', 'fused', 'sequential']
    BENCHMARK_SWEEPS = {
        'quick': SweepParams(n_values=[4,8,16,32], ...),
        'full': SweepParams(n_values=[2,4,8,16,32,64], ...),
    }
    
    @classmethod
    def benchmark_job(cls, preset='full'):
        return BenchmarkJob(...)
    
    @staticmethod
    def _bench_model(**params): ...
    @staticmethod
    def _bench_input(**params): ...
    @classmethod
    def _bench_wide(cls, modules, strategy): ...
```

Run via CLI:
```bash
wide_compiler benchmark attention -p quick
wide_compiler benchmark all -s  # Save results
```

## Key Speedups

| Primitive | Best Speedup | Why |
|-----------|--------------|-----|
| **WideAttention** | 11.4x | Batched Flash Attention |
| **WideLinear** | 12.8x | Fused einsum |
| **WideEmbedding** | 6.4x | Batched index lookup |
| **WideConv1d** | 3.2x | Grouped convolution |
| **WideConv2d** | 2.5x | Grouped convolution |

## CLI

```bash
# Benchmark primitives
wide_compiler benchmark              # List available
wide_compiler benchmark attention    # Benchmark one
wide_compiler benchmark all          # Benchmark all
wide_compiler benchmark conv2d -p quick -s  # Quick preset, save results

# Other commands
wide_compiler test                   # Correctness tests
wide_compiler trace -m resblock      # Show FX graph
wide_compiler info                   # Library info
```

## Pack / Unpack

```python
# Pack: List of [B, C, ...] → [B, N*C, ...]
packed = wide_compiler.pack(inputs)

# Unpack: [B, N*C, ...] → List of [B, C, ...]
outputs = wide_compiler.unpack(output, n)
```

## Config

```python
WideConfig(
    compile=True,
    compile_mode='reduce-overhead',
    validate=True,
    debug=True,
)

# Presets
WideConfig.fast()   # Compiled, no validation
WideConfig.debug()  # Verbose, strict
```

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `TracedWideModel` | traced_wide.py | Main output - the fused model |
| `WideConfig` | config.py | Configuration dataclass |
| `WideRegistry` | registry.py | Maps module types to builders |
| `BenchmarkJob` | benchmark_schema.py | Defines a benchmark sweep |
| `WideAttention` | wide_attention.py | N parallel MHA (11x speedup) |

## Adding a New Primitive

1. Create `primitives/wide_foo.py`:
```python
class WideFoo(nn.Module):
    BENCHMARK_STRATEGIES = ['baseline', 'fast', 'sequential']
    BENCHMARK_SWEEPS = {...}
    
    def __init__(self, n, ...): ...
    def forward(self, x): ...
    
    @classmethod
    def from_modules(cls, modules): ...
    
    @classmethod
    def benchmark_job(cls, preset='full'): ...
```

2. Add to `primitives/__init__.py`
3. Add to `benchmark_registry.py` imports
4. Register in `registry.py`

## Debugging

1. **Benchmark errors?** Run with verbose to see stack trace
2. **Wrong outputs?** Check `pack_inputs` produces `[B, N*C, ...]`
3. **Strategy selection?** Print `wide.strategy` to see which was chosen
4. **Slow first run?** Warmup iterations. Benchmark after 5+ runs.

## Quick Test

```python
import torch
from wide_compiler.core.primitives import WideAttention

# Create N attention modules
n, d_model, n_heads = 8, 256, 8
modules = [torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True).cuda() 
           for _ in range(n)]

# Fuse them
wide = WideAttention.from_modules(modules, strategy='fused')
print(wide)  # WideAttention(8x[d=256, h=8], strategy=fused)

# Test
x = torch.randn(4, 128, n * d_model).cuda()  # [B, T, N*D]
out = wide(x)
print(out.shape)  # [4, 128, 2048]
```

---

**TL;DR:** `wide_compiler.compile(models, sample)` → FX traces → builds Wide ops with auto-selected strategies → returns `TracedWideModel`. Each primitive benchmarks itself via `wide_compiler benchmark <name>`.