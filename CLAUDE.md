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
├── test_cases.py        → Reusable test suite for all 29 components
└── core/
    ├── config.py        → WideConfig dataclass
    ├── registry.py      → Maps 'Linear' → WideLinear.from_modules (24 primitives)
    ├── traced_wide.py   → FX tracing, graph execution (THE CORE)
    ├── ensemble_util.py → pack_inputs(), unpack_outputs()
    ├── benchmark/       → Primitive benchmarking system
    │   ├── __init__.py
    │   ├── benchmark_api.py      → run_benchmark(), list_primitives()
    │   ├── benchmark_runner.py   → Execution engine
    │   ├── benchmark_schema.py   → BenchmarkJob, SweepParams, results
    │   └── benchmark_registry.py → Auto-discovers primitives
    ├── blocks/          → Flux-style composite blocks (5 total)
    │   ├── wide_mlp.py                 → MLP block (2.9x @ N=32)
    │   ├── wide_attention.py           → Attention block (7.9x @ N=16)
    │   ├── wide_joint_attention.py     → Dual-stream attention (9.0x @ N=32)
    │   ├── wide_double_stream_block.py → Flux double-stream (5.2x @ N=8)
    │   └── wide_single_stream_block.py → Flux single-stream (3.4x @ N=8)
    └── primitives/      → One file per Wide op (24 total)
        # Core layers
        ├── wide_linear.py              → Linear via einsum (9.7x)
        ├── wide_embedding.py           → Batched index lookup (9.1x)
        ├── wide_mlp_embedder.py        → MLP with timestep embedding (14.8x)
        # Attention
        ├── wide_attention.py           → MHA via batched SDPA (3.5x)
        ├── wide_cross_attention.py     → Cross-attention (15.7x)
        # Convolutions
        ├── wide_conv1d.py              → Conv1d grouped (5.3x)
        ├── wide_conv2d.py              → Conv2d grouped (2.9x)
        ├── wide_conv3d.py              → Conv3d grouped (1.9x)
        ├── wide_convtranspose1d.py     → ConvTranspose1d (5.4x)
        ├── wide_convtranspose2d.py     → ConvTranspose2d (3.8x)
        # Normalization
        ├── wide_rmsnorm.py             → RMSNorm (21.0x) ⭐ NEW
        ├── wide_batchnorm_1d.py        → BatchNorm1d (11.5x)
        ├── wide_ada_layer_norm_zero_single.py → AdaLayerNormZero (9.7x)
        ├── wide_instancenorm.py        → InstanceNorm1d/2d (7.7x)
        ├── wide_groupnorm.py           → GroupNorm (4.9x)
        ├── wide_layernorm.py           → LayerNorm (4.7x)
        ├── wide_batchnorm_2d.py        → BatchNorm2d (3.4x)
        ├── wide_batchnorm_3d.py        → BatchNorm3d (0.9x slower)
        # RNNs
        ├── wide_rnn.py                 → RNN fused (1.0x break-even)
        ├── wide_lstm.py                → LSTM fused (0.7x slower)
        ├── wide_gru.py                 → GRU fused (0.5x slower)
        # Other
        ├── wide_dropout.py             → Dropout (73.2x) ⭐ EXTREME
        ├── wide_adaptive_avgpool2d.py  → AdaptiveAvgPool2d (2.2x)
        └── wide_prelu.py               → PReLU (1.0x break-even)
```

## How It Works

### 1. Trace (traced_wide.py)
```python
traced = fx.symbolic_trace(template_model)
# Captures: call_module, call_function, call_method
```

### 2. Build Wide Ops (v0.7.0 - Registry-based)
```python
for node in traced.graph.nodes:
    if node.op == 'call_module':
        modules = [m.get_submodule(path) for m in models]
        module_type = type(modules[0]).__name__

        # Dynamic lookup via registry (24 primitives)
        registry = get_registry()
        builder = registry.get_builder(module_type)
        if builder:
            wide_op = builder(modules)
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

## Key Speedups (v0.7.0 - Top 10)

| Primitive | Best Speedup | Why |
|-----------|--------------|-----|
| **WideDropout** | 73.2x | Shared random state across N models |
| **WideRMSNorm** | 21.0x | Batched normalization |
| **WideMultiheadCrossAttention** | 15.7x | Dual-stream batched attention |
| **WideMLPEmbedder** | 14.8x | Fused Linear + timestep projection |
| **WideBatchNorm1d** | 11.5x | Batched batch normalization |
| **WideAdaLayerNormZeroSingle** | 9.7x | Fused adaptive normalization |
| **WideLinear** | 9.7x | Batched einsum |
| **WideEmbedding** | 9.1x | Batched index lookup |
| **WideJointAttention** | 9.0x | Dual-stream attention block |
| **WideAttentionBlock** | 7.9x | QKV + SDPA + out proj |

## CLI (v0.7.0 - All 29 primitives + blocks)

```bash
# Benchmark primitives (auto-discovered from registry)
wide_compiler benchmark rmsnorm -p quick       # RMSNorm (21x)
wide_compiler benchmark dropout -p quick       # Dropout (73x)
wide_compiler benchmark multiheadcrossattention -p quick  # Cross-attn (15.7x)

# Benchmark blocks
wide_compiler benchmark mlp_block -p quick
wide_compiler benchmark attention_block -p quick
wide_compiler benchmark joint_attention -p quick
wide_compiler benchmark double_stream_block -p quick
wide_compiler benchmark single_stream_block -p quick

# Available primitives (24 total):
# linear, conv1d, conv2d, conv3d, convtranspose1d, convtranspose2d,
# batchnorm1d, batchnorm2d, batchnorm3d, layernorm, groupnorm,
# instancenorm2d, rmsnorm, ada_layer_norm_zero_single,
# embedding, mlp_embedder, attention, multiheadcrossattention,
# gru, lstm, rnn, prelu, dropout, adaptiveavgpool2d

# Other commands
wide_compiler test                   # Correctness tests
wide_compiler trace -m mlp           # Show FX graph
wide_compiler info                   # Library info

# Test suite
python test_cases.py                 # All 29 components
python test_cases.py --primitives    # 24 primitives
python test_cases.py --blocks        # 5 blocks
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

## I/O Formats (v0.7.0)

### Primitives (N-first)
| Primitive | Input | Output |
|-----------|-------|--------|
| WideLinear | `[N, B, ..., Din]` | `[N, B, ..., Dout]` |
| WideConv2d | `[N, B, C, H, W]` | `[N, B, Cout, Hout, Wout]` |
| WideAttention | `[N, B, T, D]` | `[N, B, T, D]` |
| WideRMSNorm | `[N, B, ..., D]` | `[N, B, ..., D]` |
| WideEmbedding | `[N, B, T]` (indices) | `[N, B, T, D]` |
| WideGRU | `[N, B, T, Din]` | `[N, B, T, H], [N, B, H]` |
| WideDropout | `[N, B, ...]` | `[N, B, ...]` |

### Blocks (N-first)
| Block | Input | Output |
|-------|-------|--------|
| WideMLP | `[N, B, T, D]` | `[N, B, T, D]` |
| WideAttention (block) | `[N, B, T, D]` + optional rope | `[N, B, T, D]` |
| WideJointAttention | `[N,B,Ttxt,D], [N,B,Timg,D]` | `[N,B,Ttxt,D], [N,B,Timg,D]` |
| WideDoubleStreamBlock | `[N,B,Ttxt,D], [N,B,Timg,D]` + rope | `[N,B,Ttxt,D], [N,B,Timg,D]` |
| WideSingleStreamBlock | `[N,B,T,D], [N,B,Demb]` + rope | `[N,B,T,D]` |

### TracedWideModel (Channel-packed I/O)
- **Input**: `[B, N*C, ...]` (channel-packed)
- **Output**: `[B, N*C, ...]` (channel-packed)
- **Internal**: All stages use N-first `[N, B, C, ...]`

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `TracedWideModel` | traced_wide.py | Main output - the fused model |
| `WideConfig` | config.py | Configuration dataclass |
| `WideRegistry` | registry.py | Maps module types to builders (24 primitives) |
| `BenchmarkJob` | benchmark_schema.py | Defines a benchmark sweep |
| `WideRMSNorm` | wide_rmsnorm.py | N parallel RMSNorm (21x speedup) ⭐ |
| `WideDropout` | wide_dropout.py | N parallel Dropout (73x speedup) ⭐ |
| `WideDoubleStreamBlock` | wide_double_stream_block.py | Flux double-stream transformer block |

## Adding a New Primitive (v0.7.0 checklist)

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
4. Add to `registry.py` auto_register_primitives() function
5. Add to `core/__init__.py` exports

## Adding a New Block (v0.7.0 checklist)

1. Create `blocks/wide_foo_block.py`:
```python
class WideFooBlock(nn.Module):
    """N parallel Foo blocks fused."""

    BENCHMARK_STRATEGIES = ['baseline', 'fused', 'sequential']

    def __init__(self, n, ...): ...

    def forward(self, x: Tensor) -> Tensor:
        """Input/Output: [N, B, ...] N-first format"""
        if self._strategy == 'fused':
            return self._forward_fused(x)
        return self._forward_sequential(x)

    @classmethod
    def from_modules(cls, modules: List[nn.Module]): ...

    @classmethod
    def benchmark_job(cls, preset='full'): ...
```

2. Add to `blocks/__init__.py` (import and `__all__`)
3. Add to `benchmark_registry.py` imports
4. Blocks are NOT auto-registered in registry.py (composite structures)

## Debugging (v0.7.0)

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
8. **Registry not finding primitive?** Check `registry.py` auto_register_primitives()

## Known Issues (v0.7.0)

1. **RNN Slowdowns** - cuDNN implementations are faster than batched versions
   - GRU: 0.5x @ N=32 (slower)
   - LSTM: 0.7x @ N=32 (slower)
   - RNN: 1.0x @ N=32 (break-even)
   - **Solution**: Use sequential execution for RNNs

2. **BatchNorm3d Slowdown** - No grouped implementation available
   - Always slower (0.7-0.9x)
   - **Solution**: Avoid if possible or use LayerNorm

3. **PReLU Break-even** - Kernel launch overhead at low N
   - 1.0x @ N=4
   - **Solution**: Only use at N>8

4. **Attention Degradation** - Performance drops at very high N
   - Best @ N=8 (3.5x), drops to 1.3x @ N=32
   - **Solution**: Keep N<=16 for attention-heavy models

## Quick Test (v0.7.0 - RMSNorm example)

```python
import torch
from wide_compiler.core.primitives import WideRMSNorm

# Create N RMSNorm modules
n, d_model = 8, 256
modules = [torch.nn.RMSNorm(d_model).cuda() for _ in range(n)]

# Fuse them
wide = WideRMSNorm.from_modules(modules, strategy='batched')
print(wide)  # WideRMSNorm(8x[d=256], strategy=batched)

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

## Blocks (v0.7.0 - Flux Architecture Support)

WideCompiler now includes composite blocks for modern architectures like Flux:

```python
from wide_compiler.core.blocks import (
    WideAttention,           # Flash Attention with optional RoPE
    WideJointAttention,      # Joint attention for multi-modal (text+image)
    WideMLP,                 # MLP with overridable activation
    WideDoubleStreamBlock,   # Flux double-stream transformer block
    WideSingleStreamBlock,   # Flux single-stream transformer block
)

# Example: Flux-style joint attention
n = 8
txt = torch.randn(n, 4, 64, 256)   # [N, B, L, hidden_size] text
img = torch.randn(n, 4, 256, 256)  # [N, B, S, hidden_size] image
vec = torch.randn(n, 4, 256)       # [N, B, emb_size] conditioning

joint_attn = WideJointAttention(n=n, hidden_size=256, num_heads=8, head_dim=32)
txt_out, img_out = joint_attn(txt, img, rope=None)

# Example: Complete double-stream block
block = WideDoubleStreamBlock(n=n, hidden_size=256, num_heads=8, head_dim=32)
txt_out, img_out = block(txt, img, vec, rope=None)
```

**Block Features:**
- **WideAttention**: Self-attention with Flash Attention (SDPA) and optional RoPE
- **WideJointAttention**: Two-stream attention (text+image) for MMDiT architectures
- **WideMLP**: Feed-forward network with configurable activation (GELU, SiLU, ReLU, etc.)
- **WideDoubleStreamBlock**: Complete Flux double-stream transformer (adaptive norm + joint attn + MLPs)
- **WideSingleStreamBlock**: Complete Flux single-stream transformer (adaptive norm + self-attn + gated MLP)

All blocks use N-first format `[N, B, ...]` internally and support both 'fused' and 'sequential' strategies.

## Version History

### v0.7.0 (Current)
- 24 primitives (11 new)
- 5 Flux-style blocks
- Registry-based TracedWideModel
- Comprehensive I/O documentation
- Test suite (test_cases.py)

### v0.6.0
- N-first internal format
- 13 primitives
- Benchmark system

### v0.5.0
- WideGRU, WideLSTM
- Initial RNN support

---

**TL;DR:**
- `wide_compiler.compile(models, sample)` → FX traces → builds Wide ops via registry → returns `TracedWideModel`
- Wide primitives use **N-first** `[N, B, ...]` internally for maximum efficiency
- TracedWideModel uses **channel-packed** `[B, N*C, ...]` at I/O boundaries
- Zero intermediate pack/unpack between stages
- 24 primitives + 5 blocks with auto-discovered benchmarking
- RMSNorm (21x), Dropout (73x), CrossAttention (15.7x) are top new additions
- Test suite: `python test_cases.py`