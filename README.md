# WideCompiler

### Version 0.5.0

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

## What's New in 0.5.0

- **WideGRU** - N parallel GRUs via einsum fusion (**8x speedup** at N=32)
- **WideLSTM** - N parallel LSTMs via block-diagonal weights (**3.3x speedup** at N=4)
- **Primitive Benchmarks with Compilation** - `benchmark gru -p quick -c` for torch.compile testing
- **Validation Checkmarks** - Benchmark output shows ✓ when correctness verified

## What's New in 0.4.0

- **WideAttention** - N parallel attention in single Flash Attention call (**11x speedup**)
- **Benchmark System** - Per-primitive benchmarking with strategy comparison
- **Improved WideEmbedding** - Batched indexing strategy (**6x speedup**)
- **Strategy Selection** - Each primitive auto-selects optimal execution strategy

## Installation

```bash
git clone https://github.com/AbstractEyes/pytorch-parallel-compiler
cd pytorch-parallel-compiler
pip install -e .
```

## Quick Start

### RNN Fusion (WideGRU / WideLSTM)

```python
import torch
from wide_compiler import WideGRU

# Create 8 separate GRUs with different weights
grus = [torch.nn.GRU(64, 128, batch_first=True).cuda() for _ in range(8)]

# Fuse into single WideGRU (copies weights)
wide_gru = WideGRU.from_modules(grus)

# Compile for best performance
wide_gru = torch.compile(wide_gru, mode='reduce-overhead')

# Input: concatenate along feature dim [B, T, N*input_size]
x = torch.randn(4, 32, 8 * 64).cuda()  # [4, 32, 512]

# Output: [B, T, N*hidden_size] = [4, 32, 1024]
out, h_n = wide_gru(x)
```

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

### Benchmark Primitives

```bash
# List available primitives
wide_compiler benchmark

# Benchmark specific primitive
wide_compiler benchmark gru
wide_compiler benchmark lstm
wide_compiler benchmark attention
wide_compiler benchmark linear

# Benchmark all primitives
wide_compiler benchmark all

# With presets
wide_compiler benchmark gru -p smoke     # Fastest (2 configs)
wide_compiler benchmark gru -p quick     # Quick (fewer configs)
wide_compiler benchmark gru -p full      # Full sweep (default)
wide_compiler benchmark gru -p ci        # CI preset (minimal)

# With torch.compile (recommended for accurate timing)
wide_compiler benchmark gru -p quick -c                  # Default: reduce-overhead
wide_compiler benchmark gru -p quick -c max-autotune    # Max performance

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

## Supported Layers

| Layer | Wide Version | Strategies | Best Speedup |
|-------|--------------|------------|--------------|
| `nn.GRU` | `WideGRU` | fused (einsum) | **7.8x** |
| `nn.LSTM` | `WideLSTM` | fused (block-diag) | **3.3x** (N=4 only) |
| `nn.MultiheadAttention` | `WideAttention` | fused, sequential | **11.4x** |
| `nn.Linear` | `WideLinear` | einsum, sequential | **12.8x** |
| `nn.Embedding` | `WideEmbedding` | indexed, gather, sequential | **6.4x** |
| `nn.Conv1d` | `WideConv1d` | grouped, sequential | **3.2x** |
| `nn.Conv2d` | `WideConv2d` | grouped, channels_last, sequential | **2.5x** |
| `nn.Conv3d` | `WideConv3d` | grouped, sequential | **~2x** |
| `nn.GroupNorm` | `WideGroupNorm` | fused, sequential | **~2x** |
| `nn.InstanceNorm1d/2d` | `WideInstanceNorm1d/2d` | fused, sequential | **~2x** |
| `nn.LayerNorm` | `WideLayerNorm` | wide | **1.8x** |
| `nn.BatchNorm1d` | `WideBatchNorm1d` | wide | **1.5x** |
| `nn.BatchNorm2d` | `WideBatchNorm2d` | wide | **1.5x** |
| `F.relu`, `F.gelu`, etc. | Passthrough | — | — |
| `+`, `-`, `*`, `/`, `@` | `BinaryOp` | — | — |

## Primitive Benchmarks (A100)

### GRU (NEW in 0.5.0)

| N | Fused | Baseline | Notes |
|---|-------|----------|-------|
| 8 | **5.0x** | 1.0x | Einsum fusion |
| 16 | **6.2x** | 1.0x | |
| 32 | **7.8x** | 1.0x | Scales well with N |

### LSTM (NEW in 0.5.0)

| N | Fused | Baseline |
|---|-------|----------|
| 4 | **3.3x** | 1.0x |
| 8 | 0.5x | 1.0x |
| 16 | 1.0x | 1.0x |
| 32 | 0.8x | 1.0x |

> Use WideGRU for N > 4.

### Attention (NEW in 0.4.0)

| N | Fused | Sequential | Baseline |
|---|-------|------------|----------|
| 4 | **3.91x** | 1.72x | 1.0x |
| 8 | **8.33x** | 1.77x | 1.0x |
| 16 | **11.23x** | 1.72x | 1.0x |
| 32 | **11.40x** | 1.80x | 1.0x |

### Embedding (Improved in 0.4.0)

| N | Indexed | Gather | Sequential |
|---|---------|--------|------------|
| 4 | **1.29x** | 1.09x | 0.85x |
| 8 | **2.38x** | 2.30x | 0.87x |
| 16 | 4.35x | **4.41x** | 0.88x |
| 32 | **6.42x** | 5.93x | 0.92x |

### Linear

| N | Einsum | Sequential |
|---|--------|------------|
| 20 | **3.0x** | 1.0x |
| 50 | **5.1x** | 1.0x |
| 100 | **12.8x** | 1.0x |

### Conv2d

| N | Grouped | Channels Last | Sequential |
|---|---------|---------------|------------|
| 8+ | **2-2.5x** | ~2x | 1.0x |

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

## How it Works

1. **FX Tracing** - `torch.fx.symbolic_trace` captures the computation graph
2. **Wide Primitives** - Each layer replaced with fused equivalent:
   - `GRU` → Einsum fusion for input projections
   - `LSTM` → Block-diagonal weight matrices (N≤4)
   - `Linear` → Batched einsum
   - `Conv2d` → Grouped convolution  
   - `Attention` → Reshape N→batch, single Flash Attention call
   - `BatchNorm` → Single BN over N*C channels
   - `GroupNorm` → Single GN with N*num_groups
   - `InstanceNorm` → Single IN over N*C channels
3. **Strategy Selection** - Each primitive auto-selects optimal strategy based on N
4. **Compile-Friendly** - 0 graph breaks, all native PyTorch ops

**Why WideGRU is fast:**
```python
# Einsum fuses N input projections: [B,T,I] @ [N,I,3H] → [B,T,N,3H]
# One kernel for all N models' input gates
```

**Why WideAttention is so fast:**
```python
# Reshape: [N, B, H, T, D] → [N*B, H, T, D]
# Run single F.scaled_dot_product_attention (Flash Attention)
# All N models processed in ONE kernel call
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