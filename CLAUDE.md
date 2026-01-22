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

## Project Structure

```
pytorch-parallel-compiler/
├── CLAUDE.md              → This file - quick reference
├── README.md              → Project overview & examples
├── LICENSE                → Apache 2.0
├── pyproject.toml         → Package config (version 0.7.0)
├── requirements.txt       → Dependencies
├── test_cases.py          → Test suite runner (29 components)
├── benchmarks/            → Benchmark output JSON files
└── wide_compiler/         → Main package
    ├── __init__.py        → Package exports
    ├── api.py             → compile(), WideBuilder, pack(), unpack()
    ├── cli.py             → CLI commands (test, benchmark, trace, info)
    ├── __main__.py        → Delegates to cli.main()
    └── core/
        ├── __init__.py        → Core exports
        ├── config.py          → WideConfig dataclass
        ├── registry.py        → Maps 'Linear' → WideLinear.from_modules (24 primitives)
        ├── traced_wide.py     → FX tracing, graph execution (THE CORE)
        ├── ensemble_util.py   → pack_inputs(), unpack_outputs()
        ├── autotune/          → Performance tuning utilities
        │   ├── run.py                     → Auto-tuning runner
        │   ├── linear_difference_check.py → Linear accuracy tests
        │   ├── conv2d_difference_check.py → Conv2d accuracy tests
        │   ├── conv2d_grouped_vs_sequential.py
        │   └── einsum_conv2d_speed_test.py
        ├── strategy/          → Strategy selection utilities
        │   └── inductor.py    → torch.compile integration
        ├── benchmark/         → Primitive benchmarking system
        │   ├── benchmark_api.py      → run_benchmark(), list_primitives()
        │   ├── benchmark_runner.py   → Execution engine
        │   ├── benchmark_schema.py   → BenchmarkJob, SweepParams, results
        │   ├── benchmark_registry.py → Auto-discovers primitives
        │   ├── traced_wide_benchmark.py → Full model benchmarking
        │   └── SCHEMA.md             → Benchmark schema documentation
        ├── blocks/            → Flux-style composite blocks (5 total)
        │   ├── wide_mlp.py                 → MLP block (2.9x @ N=32)
        │   ├── wide_attention.py           → Attention block (7.9x @ N=16)
        │   ├── wide_joint_attention.py     → Dual-stream attention (9.0x @ N=32)
        │   ├── wide_double_stream_block.py → Flux double-stream (5.2x @ N=8)
        │   └── wide_single_stream_block.py → Flux single-stream (3.4x @ N=8)
        └── primitives/        → One file per Wide op (25 files, 24 auto-registered)
            # Core layers
            ├── wide_linear.py              → Linear via einsum (9.7x)
            ├── wide_embedding.py           → Batched index lookup (9.1x)
            ├── wide_mlp_embedder.py        → MLP with timestep embedding (14.8x)
            ├── wide_rotary_embedding.py    → RoPE support (helper, not auto-registered)
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
            ├── wide_rmsnorm.py             → RMSNorm (21.0x)
            ├── wide_batchnorm_1d.py        → BatchNorm1d (11.5x)
            ├── wide_batchnorm_2d.py        → BatchNorm2d (3.4x)
            ├── wide_batchnorm_3d.py        → BatchNorm3d (0.9x slower)
            ├── wide_layernorm.py           → LayerNorm (4.7x)
            ├── wide_groupnorm.py           → GroupNorm (4.9x)
            ├── wide_instancenorm.py        → InstanceNorm1d/2d (7.7x)
            ├── wide_ada_layer_norm_zero_single.py → AdaLayerNormZero (9.7x)
            # RNNs
            ├── wide_rnn.py                 → RNN fused (1.0x break-even)
            ├── wide_lstm.py                → LSTM fused (0.7x slower)
            ├── wide_gru.py                 → GRU fused (0.5x slower)
            # Other
            ├── wide_dropout.py             → Dropout (73.2x)
            ├── wide_adaptive_avgpool2d.py  → AdaptiveAvgPool2d (2.2x)
            └── wide_prelu.py               → PReLU (1.0x break-even)
```

## Dependencies

**Core (pyproject.toml):**
- Python >= 3.9
- torch >= 2.1
- numpy >= 1.23

**Dev/Optional (requirements.txt):**
- pytest >= 7.0
- tabulate >= 0.9 (CLI tabular output)
- huggingface_hub, datasets, safetensors (optional integrations)
- PyYAML >= 6.0

## Development Workflow

### Installation

```bash
# From source
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Full test suite (29 components)
python test_cases.py

# Primitives only (24)
python test_cases.py --primitives

# Blocks only (5)
python test_cases.py --blocks

# With specific preset
python test_cases.py --preset quick
```

### Running Benchmarks

```bash
# Single primitive
wide_compiler benchmark linear -p quick

# All primitives
wide_compiler benchmark all

# Save results
wide_compiler benchmark all -o results.json
```

### CLI Commands

```bash
wide_compiler test      # Correctness tests
wide_compiler benchmark # Performance benchmarks
wide_compiler trace     # Show FX graph
wide_compiler info      # Library info
```

## How It Works

### 1. Trace (traced_wide.py)
```python
traced = fx.symbolic_trace(template_model)
# Captures: call_module, call_function, call_method
```

### 2. Build Wide Ops (Registry-based)
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

### 3. Graph Execution (N-first internal format)
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

## Wide Primitives Pattern

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

## Numerical Accuracy Notes

PyTorch batched operations are NOT guaranteed to match sequential operations bitwise.

**Expected tolerances:**
- NCHW grouped vs sequential: ~1e-6 relative error (IEEE fp32)
- NHWC grouped vs sequential: 0.0 error (most accurate)
- With TF32: ~1e-4 relative error
- Einsum vs sequential linear: ~1e-6 relative error (fp32)

**To disable TF32 for stricter accuracy:**
```python
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
```

## Strategy Pattern

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

## Registry System

The registry maps PyTorch module types to Wide primitives:

```python
from wide_compiler.core.registry import get_registry, register

# Check what's registered
registry = get_registry()
print(registry.list_registered())

# Register custom primitive
@register('MyModule')
class WideMyModule(nn.Module):
    @classmethod
    def from_modules(cls, modules): ...

# Build Wide version from modules
wide = registry.build(modules)
```

**24 Auto-registered primitives:**
- Linear → WideLinear
- Conv1d/2d/3d → WideConv1d/2d/3d
- ConvTranspose1d/2d → WideConvTranspose1d/2d
- BatchNorm1d/2d/3d → WideBatchNorm1d/2d/3d
- LayerNorm → WideLayerNorm
- GroupNorm → WideGroupNorm
- InstanceNorm1d/2d → WideInstanceNorm1d/2d
- RMSNorm → WideRMSNorm
- AdaLayerNormZeroSingle → WideAdaLayerNormZeroSingle
- Embedding → WideEmbedding
- MLPEmbedder → WideMLPEmbedder
- MultiheadAttention → WideAttention
- GRU/LSTM/RNN → WideGRU/WideLSTM/WideRNN
- PReLU → WidePReLU
- Dropout → WideDropout
- AdaptiveAvgPool2d → WideAdaptiveAvgPool2d

**Not auto-registered:**
- WideMultiheadCrossAttention (requires explicit intent)
- WideRotaryEmbedding (helper, used internally)
- All blocks (composite structures)

## Benchmark System

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
        )
```

Run via CLI:
```bash
wide_compiler benchmark attention -p quick
wide_compiler benchmark layernorm -p full
wide_compiler benchmark all
```

## Key Speedups (Top 10)

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

## I/O Formats

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
| `WideRMSNorm` | wide_rmsnorm.py | N parallel RMSNorm (21x speedup) |
| `WideDropout` | wide_dropout.py | N parallel Dropout (73x speedup) |
| `WideDoubleStreamBlock` | wide_double_stream_block.py | Flux double-stream transformer block |

## Adding a New Primitive (Checklist)

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
4. Add to `registry.py` `auto_register_primitives()` function
5. Add to `core/__init__.py` exports

## Adding a New Block (Checklist)

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

## Debugging

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
8. **Registry not finding primitive?** Check `registry.py` `auto_register_primitives()`

## Known Issues

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

## Quick Test

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

## Blocks (Flux Architecture Support)

WideCompiler includes composite blocks for modern architectures like Flux:

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

### v0.7.0 (Current - January 2025)
- 24 auto-registered primitives (11 new)
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
- RMSNorm (21x), Dropout (73x), CrossAttention (15.7x) are top speedups
- Test suite: `python test_cases.py`
