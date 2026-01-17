# WideCompiler

### Version 0.6.0

Compile-friendly batched model execution. Fuse N identical models into a single Wide model for massive speedups.

## What it does

Instead of running N models sequentially:
```python
outputs = [model(x) for model in models]  # N kernel launches
```

Fuse them into one:
```python
import wide_compiler

wide = wide_compiler.compile(models, sample_input)
output = wide(packed_input)  # 1 kernel launch
```

**Speedups:** 2-40x depending on model type, N, and compilation mode.

## What's New in 0.6.0

- **N-First Internal Format** - Wide primitives use `[N, B, ...]` format internally for optimal performance
- **Zero Intermediate Pack/Unpack** - Only 2 reshapes per forward pass (at boundaries), zero conversions between stages
- **13 Registered Primitives** - All primitives auto-discovered via registry, accessible via CLI
- **`benchmark all` Command** - Benchmark all primitives in one command with consolidated results
- **Auto-save with `-s` Flag** - Save benchmark results with auto-generated timestamps
- **Fixed TracedWideModel** - Correctly handles spatial inputs (images `[B, N*C, H, W]`)
- **Comprehensive Error Messages** - Full stack traces for easier debugging
- **Unified Benchmark System** - All primitives use consistent N-first validation protocol

## What's New in 0.5.0

- **WideGRU** - N parallel GRUs via einsum fusion (**8x speedup** at N=32)
- **WideLSTM** - N parallel LSTMs via fused projections (**3.3x speedup** at N=4)
- **Primitive Benchmarks with Compilation** - `benchmark gru -p quick -c` for torch.compile testing
- **Validation Checkmarks** - Benchmark output shows ✓ when correctness verified

## Installation

```bash
git clone https://github.com/AbstractEyes/pytorch-parallel-compiler
cd pytorch-parallel-compiler
pip install -e .
```

## Quick Start

### RNN Fusion (WideGRU / WideLSTM) - N-first format

```python
import torch
from wide_compiler.core.primitives import WideGRU

# Create 8 separate GRUs with different weights
grus = [torch.nn.GRU(64, 128, batch_first=True).cuda() for _ in range(8)]

# Fuse into single WideGRU (copies weights)
wide_gru = WideGRU.from_modules(grus)

# Compile for best performance
wide_gru = torch.compile(wide_gru, mode='reduce-overhead')

# Input: N-first format [N, B, T, input_size]
x = torch.randn(8, 4, 32, 64).cuda()  # [N=8, B=4, T=32, D=64]

# Output: N-first format [N, B, T, hidden_size], hidden state [N, B, hidden_size]
out, h_n = wide_gru(x)  # out: [8, 4, 32, 128], h_n: [8, 4, 128]
```

**Note:** Wide primitives use **N-first format** `[N, B, ...]`. For automatic packing/unpacking with channel-packed format, use `TracedWideModel` (see below).

### Full Model Fusion (TracedWideModel)

```python
import torch
import wide_compiler

# Define your model
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 64)
    
    def forward(self, x):
        return self.fc2(torch.nn.functional.relu(self.fc1(x)))

# Create N models
models = [MLP().cuda() for _ in range(100)]
sample = torch.randn(1, 64).cuda()

# Compile to Wide model
wide = wide_compiler.compile(models, sample)

# Pack inputs and run
inputs = [torch.randn(32, 64).cuda() for _ in range(100)]
packed = wide_compiler.pack(inputs)
output = wide(packed)

# Unpack outputs
outputs = wide_compiler.unpack(output, n=100)
```

## API

### Main Entry Point

```python
import wide_compiler

# From list of models
wide = wide_compiler.compile(models, sample_input)

# From single model (creates N copies with different weights)
wide = wide_compiler.compile(MyModel(), sample_input, n=100)

# With torch.compile enabled
wide = wide_compiler.compile(models, sample_input, compile_model=True)

# With validation
wide = wide_compiler.compile(models, sample_input, validate=True)

# With config
config = wide_compiler.WideConfig.fast()
wide = wide_compiler.compile(models, sample_input, config=config)
```

### Builder Pattern

```python
wide = (wide_compiler.WideBuilder(models)
    .with_sample(sample_input)
    .validate()
    .compile(mode='reduce-overhead')
    .build())
```

### Pack / Unpack

```python
# Pack N inputs: List[Tensor] → Tensor [B, N*C, ...]
packed = wide_compiler.pack(inputs)

# Unpack output: Tensor [B, N*C, ...] → List[Tensor]
outputs = wide_compiler.unpack(output, n=100)
```

### Configuration

```python
from wide_compiler import WideConfig

# Presets
config = WideConfig.default()   # Basic, no compile
config = WideConfig.fast()      # Compiled, no validation
config = WideConfig.debug()     # Verbose, strict
config = WideConfig.safe()      # With validation

# Custom
config = WideConfig(
    compile=True,
    compile_mode='reduce-overhead',  # 'default', 'max-autotune'
    validate=True,
    validate_rtol=1e-3,
    debug=True,
)
```

## CLI

### Benchmark Primitives (v0.6.0 - 13 primitives)

```bash
# Benchmark specific primitive (auto-discovered from registry)
wide_compiler benchmark attention -p quick
wide_compiler benchmark layernorm -p quick
wide_compiler benchmark lstm -p quick
wide_compiler benchmark conv2d -p quick

# All 13 available primitives:
# attention, batchnorm1d, batchnorm2d, conv1d, conv2d, conv3d,
# embedding, gru, groupnorm, instancenorm2d, layernorm, linear, lstm

# Benchmark all primitives
wide_compiler benchmark all -p quick

# With presets
wide_compiler benchmark gru -p quick     # Quick (fewer configs)
wide_compiler benchmark gru -p full      # Full sweep (default)
wide_compiler benchmark gru -p ci        # CI preset (minimal)

# With torch.compile (recommended for accurate timing)
wide_compiler benchmark gru -p quick -c                  # Default: reduce-overhead
wide_compiler benchmark gru -p quick -c max-autotune    # Max performance

# Benchmark full models (TracedWideModel)
wide_compiler benchmark resblock --n 100     # Benchmark sample model
wide_compiler benchmark mlp --n 50           # Benchmark MLP

# Other options
wide_compiler benchmark conv1d -t 20          # Show top 20 results
wide_compiler benchmark conv1d -q             # Quiet mode (no progress)
wide_compiler benchmark conv1d -s             # Auto-save with timestamp
wide_compiler benchmark conv1d -o results.json  # Save to specific file
wide_compiler benchmark gru --no-validate     # Skip validation (faster)

wide_compiler benchmark all -q -s -c          # All primitives, quiet, auto-save, compiled
```

### Other Commands

```bash
# Run correctness tests
wide_compiler test

# Show FX trace for built-in models
wide_compiler trace -m mlp
wide_compiler trace -m resblock

# Show library info
wide_compiler info
```

## Supported Layers (v0.6.0 - 13 primitives)

| Layer | Wide Version | Format | Strategies | Best Speedup (A100) |
|-------|--------------|--------|------------|---------------------|
| `nn.Embedding` | `WideEmbedding` | N-first | indexed, gather, sequential | **76.8x** @ N=32 |
| `nn.BatchNorm2d` | `WideBatchNorm2d` | N-first | wide | **38.0x** @ N=32 |
| `nn.InstanceNorm2d` | `WideInstanceNorm2d` | N-first | fused, sequential | **37.5x** @ N=32 |
| `nn.GroupNorm` | `WideGroupNorm` | N-first | fused, sequential | **36.5x** @ N=32 |
| `nn.BatchNorm1d` | `WideBatchNorm1d` | N-first | wide | **31.4x** @ N=32 |
| `nn.Conv2d` | `WideConv2d` | N-first | grouped, channels_last, sequential | **15.0x** @ N=16 |
| `nn.Conv1d` | `WideConv1d` | N-first | grouped, sequential | **12.3x** @ N=32 |
| `nn.Conv3d` | `WideConv3d` | N-first | grouped, sequential | **10.9x** @ N=16 |
| `nn.MultiheadAttention` | `WideAttention` | N-first | fused, sequential | **10.7x** @ N=32 |
| `nn.Linear` | `WideLinear` | N-first | einsum, sequential | **9.3x** @ N=32 |
| `nn.LayerNorm` | `WideLayerNorm` | N-first | wide | **9.1x** @ N=32 |
| `nn.LSTM` | `WideLSTM` | N-first | fused | **3.3x** @ N=32 |
| `nn.GRU` | `WideGRU` | N-first | fused (einsum) | **3.0x** @ N=32 |
| `F.relu`, `F.gelu`, etc. | `FunctionalOp` | agnostic | — | — |
| `+`, `-`, `*`, `/`, `@` | `BinaryOp` | agnostic | — | — |

**All primitives operate on N-first format `[N, B, ...]` internally for optimal performance.**

*Benchmarks: A100 GPU, torch.compile (default mode), quick preset. See [benchmarks/](benchmarks/) for full results.*

## Primitive Benchmarks (A100, compiled)

All benchmarks: A100 GPU, `torch.compile(mode='default')`, quick preset (N=[4,8,16,32]).

### Top Performers

| Primitive | N=4 | N=8 | N=16 | N=32 | Crossover | Best Strategy |
|-----------|-----|-----|------|------|-----------|---------------|
| **Embedding** | 4.0x | 8.5x | 15.4x | **76.8x** | N=4 | indexed |
| **BatchNorm2d** | 4.6x | 9.3x | 14.7x | **38.0x** | N=4 | wide |
| **InstanceNorm2d** | 4.6x | 9.1x | 18.4x | **37.5x** | N=4 | fused |
| **GroupNorm** | 4.2x | 8.9x | 17.4x | **36.5x** | N=4 | fused |
| **BatchNorm1d** | 4.0x | 7.9x | 15.6x | **31.4x** | N=4 | wide |

### Conv Layers

| Primitive | N=4 | N=8 | N=16 | N=32 | Crossover | Best Strategy |
|-----------|-----|-----|------|------|-----------|---------------|
| **Conv2d** | 3.8x | 7.4x | **15.0x** | 13.4x | N=4 | grouped |
| **Conv1d** | 3.2x | 5.5x | 8.6x | **12.3x** | N=4 | grouped |
| **Conv3d** | 3.8x | 7.6x | **10.9x** | — | N=4 | grouped |

### Attention & Linear

| Primitive | N=4 | N=8 | N=16 | N=32 | Crossover | Best Strategy |
|-----------|-----|-----|------|------|-----------|---------------|
| **Attention** | 4.4x | 6.4x | 8.2x | **10.7x** | N=4 | fused |
| **Linear** | 1.1x | 2.4x | 4.4x | **9.3x** | N=4 | einsum |
| **LayerNorm** | 1.3x | 2.5x | 5.0x | **9.1x** | N=4 | wide |

### RNN Layers

| Primitive | N=8 | N=16 | N=32 | Crossover | Best Strategy | Notes |
|-----------|-----|------|------|-----------|---------------|-------|
| **LSTM** | 0.9x | 1.7x | **3.3x** | N=16 | fused | High overhead at low N |
| **GRU** | 0.7x | 1.5x | **3.0x** | N=16 | fused | High overhead at low N |

> **RNN Note:** Use `WideGRU`/`WideLSTM` only for N ≥ 16. At N < 16, sequential execution is faster due to kernel launch overhead.

### Key Takeaways

1. **Normalization layers** achieve 30-80x speedups (highest gains in WideCompiler)
2. **Embedding** sees extreme speedups at high N due to batched indexing
3. **Conv layers** scale well, best at N=16-32
4. **Attention** provides consistent 4-11x speedup across all N
5. **RNN layers** require N ≥ 16 to overcome overhead

## End-to-End Benchmarks

| Model | N | Batch | Mode | Speedup |
|-------|---|-------|------|---------|
| **MLP** (2 linear) | 100 | 32 | eager | **40x** |
| **MLP** (2 linear) | 20 | 32 | compile | **3.2x** |
| **Deep MLP** (8 linear) | 20 | 32 | eager | **6.2x** |
| **ResBlock** | 20 | 8 | eager | **4.9x** |
| **ResBlock** | 20 | 8 | compile | **2.9x** |
| **ResNet18** | 10 | 8 | eager | **2.3x** |
| **ResNet18** | 10 | 8 | compile | **1.9x** |

## How it Works (v0.6.0)

1. **FX Tracing** - `torch.fx.symbolic_trace` captures the computation graph
2. **Wide Primitives** - Each layer replaced with N-first fused equivalent:
   - `GRU` → Einsum fusion for input projections `[N, B, T, I] @ [N, I, 3H]`
   - `LSTM` → Fused projections `[N, B, T, I] @ [N, I, 4H]`
   - `Linear` → Batched einsum `[N, B, I] @ [N, I, O]`
   - `Conv2d` → Grouped convolution on `[N, B, C, H, W]`
   - `Attention` → Reshape N→batch, single Flash Attention call
   - All normalization layers operate on N-first `[N, B, C, ...]`
3. **Strategy Selection** - Each primitive auto-selects optimal strategy based on N
4. **Compile-Friendly** - 0 graph breaks, all native PyTorch ops
5. **Optimal Data Flow** - Only 2 reshapes per forward pass:
   ```
   Input [B, N*C, ...] → Unpack → [N, B, C, ...]
     ↓ All stages operate on N-first (zero intermediate conversions)
   Output [N, B, C, ...] → Pack → [B, N*C, ...]
   ```

**Why N-first format is optimal:**
```python
# All primitives operate on [N, B, ...] format
# Data flows through without any intermediate packing
# Only reshape at boundaries (input/output)
# Result: Maximum kernel fusion, minimum overhead
```

**Why WideAttention is so fast:**
```python
# Input: [N, B, T, D] (N-first)
# Reshape: [N, B, H, T, Dh] → [N*B, H, T, Dh]
# Run single F.scaled_dot_product_attention (Flash Attention)
# All N models processed in ONE kernel call
# Output: [N, B, T, D] (stays N-first for next stage)
```

## Project Structure

```
wide_compiler/
├── __init__.py
├── __main__.py
├── api.py                    # compile(), WideBuilder
├── cli.py                    # CLI commands
└── core/
    ├── config.py             # WideConfig
    ├── registry.py           # Primitive registration
    ├── traced_wide.py        # FX tracing + TracedWideModel
    ├── benchmark/            # Benchmark system
    │   ├── __init__.py
    │   ├── benchmark_api.py
    │   ├── benchmark_runner.py
    │   ├── benchmark_schema.py
    │   └── benchmark_registry.py
    └── primitives/
        ├── wide_gru.py         # 8x speedup (einsum fusion)
        ├── wide_lstm.py        # 3x speedup (block-diagonal)
        ├── wide_attention.py   # 11x speedup
        ├── wide_linear.py
        ├── wide_conv1d.py
        ├── wide_conv2d.py
        ├── wide_conv3d.py
        ├── wide_batchnorm_1d.py
        ├── wide_batchnorm_2d.py
        ├── wide_groupnorm.py
        ├── wide_instancenorm.py
        ├── wide_layernorm.py
        └── wide_embedding.py
```

## Limitations

### General
- **Identical architecture required** - All N models must have same structure
- **Static shapes** - FX tracing requires fixed tensor shapes
- **No dynamic control flow** - `if`/`for` based on tensor values won't trace

### RNN Primitives (WideGRU, WideLSTM)
- **Single layer only** - `num_layers=1` currently required
- **Unidirectional only** - `bidirectional=False` required
- **batch_first only** - `batch_first=True` required

### WideLSTM Performance
- **N=4:** 3.3x speedup
- **N>4:** Use WideGRU instead

## Use Cases

- **Ensemble models** - Run N ensemble members in parallel
- **Hyperparameter search** - Evaluate N configurations simultaneously  
- **Population-based training** - Evolve N agents together
- **Monte Carlo dropout** - N stochastic forward passes
- **Transformer ensembles** - N attention heads across models
- **RNN ensembles** - N sequence models in parallel (NEW)

## License

Apache License 2.0

## Author

AbstractPhil - [HuggingFace](https://huggingface.co/AbstractPhil) | [GitHub](https://github.com/AbstractEyes)