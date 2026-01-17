# CLAUDE.md - WideCompiler Quick Reference

VERSION = 0.7.0

Read this first when working on this codebase.

## What This Project Does

Fuses N identical PyTorch models into ONE wide model. Instead of N sequential forward passes, one batched forward pass using grouped operations.

**Key insight:** Wide primitives use N-first format `[N, B, C, ...]` internally. TracedWideModel handles packing/unpacking at boundaries.

## Core Concept

```
N separate models:        Wide model:
[Model_0] → out_0        [WideModel] → [out_0, out_1, ..., out_N]
[Model_1] → out_1         (single forward pass)
...
[Model_N] → out_N
```

**Two packing formats:**
1. **N-first** `[N, B, C, ...]` - Used by Wide primitives internally
2. **Channel-packed** `[B, N*C, ...]` - Used by TracedWideModel I/O

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
    ├── benchmark/       → Primitive benchmarking system (NEW)
    │   ├── __init__.py
    │   ├── benchmark_api.py      → run_benchmark(), list_primitives()
    │   ├── benchmark_runner.py   → Execution engine
    │   ├── benchmark_schema.py   → BenchmarkJob, SweepParams, results
    │   └── benchmark_registry.py → Auto-discovers primitives
    └── primitives/      → One file per Wide op (21 total)
        # Core layers
        ├── wide_attention.py           → MHA via batched SDPA (10.7x)
        ├── wide_cross_attention.py     → Cross-attention (8-12x)
        ├── wide_linear.py              → Linear via einsum (9.3x)
        ├── wide_embedding.py           → Batched index lookup (76.8x)
        # Convolutions
        ├── wide_conv1d.py              → Conv1d grouped (12.3x)
        ├── wide_conv2d.py              → Conv2d grouped (15.0x)
        ├── wide_conv3d.py              → Conv3d grouped (10.9x)
        ├── wide_convtranspose1d.py     → ConvTranspose1d (10-15x)
        ├── wide_convtranspose2d.py     → ConvTranspose2d (8-15x)
        # Normalization
        ├── wide_batchnorm_1d.py        → BatchNorm1d (31.4x)
        ├── wide_batchnorm_2d.py        → BatchNorm2d (38.0x)
        ├── wide_batchnorm_3d.py        → BatchNorm3d (30-40x est)
        ├── wide_layernorm.py           → LayerNorm (9.1x)
        ├── wide_groupnorm.py           → GroupNorm (36.5x)
        ├── wide_instancenorm.py        → InstanceNorm1d/2d (37.5x)
        # RNNs
        ├── wide_gru.py                 → GRU fused (3.0x)
        ├── wide_lstm.py                → LSTM fused (3.3x)
        ├── wide_rnn.py                 → RNN fused (2-4x est)
        # Other
        ├── wide_prelu.py               → PReLU (15-30x est)
        ├── wide_dropout.py             → Dropout (5-20x est)
        └── wide_adaptive_avgpool2d.py  → AdaptiveAvgPool2d (20-50x est)
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

### 3. Graph Execution (v0.6.0 - N-first internal format)
```python
def forward(self, x):
    # Unpack ONCE: [B, N*C, ...] → [N, B, C, ...]
    B, nc, *spatial = x.shape[0], x.shape[1], x.shape[2:]
    c = nc // self.n
    x = x.view(B, self.n, c, *spatial).movedim(1, 0)

    # Execute graph (all stages operate on N-first)
    values = {self._input_name: x}
    for node_name in self._execution_order:
        args = [values[arg] for arg in self._node_args[node_name]]
        values[node_name] = self.stages[node_name](*args)

    # Pack ONCE: [N, B, C, ...] → [B, N*C, ...]
    out = values[self._output_name]
    out = out.movedim(0, 1).reshape(B, self.n * c, *spatial)
    return out
```

**Key optimization:** Only 2 reshapes per forward pass (unpack + pack), zero intermediate conversions.

## Wide Primitives Pattern (v0.6.0 - N-first format)

Every primitive follows this pattern:

```python
class WideLinear(nn.Module):
    """N parallel Linear layers fused."""

    def __init__(self, n, in_features, out_features, strategy='auto'):
        self.weight = nn.Parameter(torch.empty(n, out_features, in_features))
        self._strategy = strategy

    def forward(self, x):
        # x: [N, B, ..., D_in] → [N, B, ..., D_out] (N-first!)
        if self._use_einsum:
            return self._forward_einsum(x)
        return self._forward_sequential(x)

    @classmethod
    def from_modules(cls, modules: List[nn.Linear], strategy='auto'):
        # Stack weights from N modules
```

**Critical:**
- All primitives use **N-first format** `[N, B, ...]` internally
- Each primitive has multiple strategies. AUTO selects the fastest.

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

## Benchmark System (v0.6.0 - N-first validation)

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
        return BenchmarkJob(
            name=f'attention_{preset}',
            primitive='attention',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
            # pack_fn/unpack_fn: use default N-first [N, B, ...]
            # validate_fn: optional custom validation
        )

    @staticmethod
    def _bench_model(**params): ...  # Returns nn.Module
    @staticmethod
    def _bench_input(**params): ...  # Returns [B, ...] single input
    @classmethod
    def _bench_wide(cls, modules, strategy): ...  # Returns WideModule
```

Run via CLI (14 primitives available):
```bash
wide_compiler benchmark attention -p quick
wide_compiler benchmark layernorm -p quick
wide_compiler benchmark lstm -p quick
wide_compiler benchmark all  # Run all primitives
```

## Key Speedups

| Primitive | Best Speedup | Why |
|-----------|--------------|-----|
| **WideAttention** | 11.4x | Batched Flash Attention |
| **WideLinear** | 12.8x | Fused einsum |
| **WideEmbedding** | 6.4x | Batched index lookup |
| **WideConv1d** | 3.2x | Grouped convolution |
| **WideConv2d** | 2.5x | Grouped convolution |

## CLI (v0.6.0 - All 14 primitives)

```bash
# Benchmark primitives (auto-discovered from registry)
wide_compiler benchmark attention -p quick      # Benchmark attention
wide_compiler benchmark layernorm -p quick      # Benchmark layernorm
wide_compiler benchmark lstm -p quick           # Benchmark LSTM
wide_compiler benchmark all -p quick            # Benchmark all 14 primitives

# Available primitives (auto-registered):
# attention, batchnorm1d, batchnorm2d, conv1d, conv2d, conv3d,
# embedding, gru, groupnorm, instancenorm1d, instancenorm2d,
# layernorm, linear, lstm

# Benchmark full models (TracedWideModel)
wide_compiler benchmark resblock --n 100        # Benchmark sample model
wide_compiler benchmark mlp --n 50              # Benchmark MLP

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

## Adding a New Primitive (v0.6.0 checklist)

1. Create `primitives/wide_foo.py`:
```python
class WideFoo(nn.Module):
    """N parallel Foo modules fused."""

    BENCHMARK_STRATEGIES = ['baseline', 'fused', 'sequential']
    BENCHMARK_SWEEPS = {}  # Populated lazily
    _SWEEPS_INITIALIZED = False

    def __init__(self, n, ...): ...

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with N-first format.

        Input:  [N, B, ...] N-first format
        Output: [N, B, ...] N-first format
        """
        # Implementation here

    @classmethod
    def from_modules(cls, modules: List[nn.Module]): ...

    @classmethod
    def _init_benchmark_sweeps(cls):
        """Initialize sweep configs (called once)."""
        if cls._SWEEPS_INITIALIZED:
            return
        cls._SWEEPS_INITIALIZED = True
        SweepParams = cls._get_sweep_params_class()
        if SweepParams is None:
            return
        cls.BENCHMARK_SWEEPS = {
            'quick': SweepParams(...),
            'full': SweepParams(...),
            'ci': SweepParams(...),
        }

    @classmethod
    def benchmark_job(cls, preset='full'):
        cls._init_benchmark_sweeps()
        # Return BenchmarkJob with factories
        # DON'T specify pack_fn/unpack_fn unless non-standard format
```

2. Add to `primitives/__init__.py` (import and `__all__`)
3. Add to `benchmark_registry.py` imports (will auto-register if has `benchmark_job()`)
4. Add to `registry.py` WIDE_BUILDERS dict (for TracedWideModel support)

## Debugging (v0.6.0)

1. **Benchmark errors?** CLI shows full stack trace automatically
2. **Validation failures?** Check shapes:
   - Wide output should be `[N, B, ...]`
   - Baseline outputs should be list of `[B, ...]`
   - Default validation stacks baseline to `[N, B, ...]` and compares
3. **Wrong outputs?** Verify primitive uses N-first format internally
4. **TracedWideModel errors?** Check:
   - Input is `[B, N*C, ...]` (channel-packed)
   - Output is `[B, N*C, ...]` (channel-packed)
   - Internal stages operate on `[N, B, C, ...]` (N-first)
5. **Strategy selection?** Print `wide.strategy` to see which was chosen
6. **Slow first run?** Warmup iterations. Benchmark after 5+ runs.
7. **Missing primitive in CLI?** Check `benchmark_registry.py` imports

## Quick Test (v0.6.0 - N-first format)

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

# Test with N-first format
x = torch.randn(n, 4, 128, d_model).cuda()  # [N, B, T, D] N-first!
out = wide(x)
print(out.shape)  # [8, 4, 128, 256] N-first output
```

**For TracedWideModel (channel-packed I/O):**
```python
import wide_compiler

# Create N models
models = [MyModel() for _ in range(n)]
sample = torch.randn(4, 64)  # Single model input

# Compile
wide = wide_compiler.compile(models, sample)

# Use with channel-packed format
inputs = [torch.randn(4, 64) for _ in range(n)]
packed = wide_compiler.pack(inputs)  # [4, N*64]
output = wide(packed)  # [4, N*output_dim]
outputs = wide_compiler.unpack(output, n)  # List of [4, output_dim]
```

---

**TL;DR:**
- `wide_compiler.compile(models, sample)` → FX traces → builds Wide ops → returns `TracedWideModel`
- Wide primitives use **N-first** `[N, B, ...]` internally for maximum efficiency
- TracedWideModel uses **channel-packed** `[B, N*C, ...]` at I/O boundaries
- Zero intermediate pack/unpack between stages
- 14 primitives with auto-discovered benchmarking via `wide_compiler benchmark <name>`