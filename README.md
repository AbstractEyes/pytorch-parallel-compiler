# WideCompiler

### Version 0.7.0

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

**Speedups:** 2-73x depending on model type, N, and compilation mode.

## What's New in 0.7.0

- **24 Primitives** - Added 11 new primitives: RMSNorm (21x), AdaLayerNormZero (9.7x), MLPEmbedder (14.8x), CrossAttention (15.7x), ConvTranspose1d/2d, BatchNorm3d, RNN, PReLU, Dropout (73x), AdaptiveAvgPool2d
- **5 Flux-Style Blocks** - WideMLP, WideAttention, WideJointAttention, WideDoubleStreamBlock, WideSingleStreamBlock for transformer architectures
- **Registry-Based TracedWideModel** - Fully dynamic primitive lookup, no more hardcoded WIDE_BUILDERS dict
- **RMSNorm Support** - Native PyTorch RMSNorm with 21x speedup at N=32
- **Comprehensive Benchmarks** - All 29 components (24 primitives + 5 blocks) tested and validated
- **I/O Shape Documentation** - Clear input/output formats for every primitive and block

## Installation

```bash
git clone https://github.com/AbstractEyes/pytorch-parallel-compiler
cd pytorch-parallel-compiler
pip install -e .
```

## Quick Start

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

### Using Wide Blocks (Flux-style)

```python
import torch
from wide_compiler.core.blocks import WideMLP, WideAttention

# Create N separate MLP blocks
mlps = [...]  # Your N MLP modules
wide_mlp = WideMLP.from_modules(mlps, strategy='fused')

# Input: N-first format [N, B, T, D]
x = torch.randn(8, 4, 128, 256).cuda()
out = wide_mlp(x)  # [8, 4, 128, 256]

# Create N attention blocks
attns = [...]  # Your N attention modules
wide_attn = WideAttention.from_modules(attns, strategy='fused')

# With RoPE embeddings
rope = torch.randn(128, 64).cuda()  # Positional embeddings
out = wide_attn(x, rope=rope)  # [8, 4, 128, 256]
```

### Using Wide Primitives Directly (N-first format)

```python
import torch
from wide_compiler.core.primitives import WideRMSNorm, WideLinear

# Create N separate RMSNorm layers
norms = [torch.nn.RMSNorm(256).cuda() for _ in range(8)]
wide_norm = WideRMSNorm.from_modules(norms, strategy='batched')

# Input: N-first format [N, B, T, D]
x = torch.randn(8, 4, 128, 256).cuda()
out = wide_norm(x)  # [8, 4, 128, 256]
```

**Note:** Wide primitives use **N-first format** `[N, B, ...]`. For automatic packing/unpacking with channel-packed format, use `TracedWideModel` (see above).

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

### Benchmark Primitives (v0.7.0 - 24 primitives)

```bash
# Benchmark specific primitive
wide_compiler benchmark rmsnorm -p quick
wide_compiler benchmark dropout -p quick
wide_compiler benchmark multiheadcrossattention -p quick

# All 24 primitives:
# linear, conv1d, conv2d, conv3d, convtranspose1d, convtranspose2d,
# batchnorm1d, batchnorm2d, batchnorm3d, layernorm, groupnorm,
# instancenorm2d, rmsnorm, ada_layer_norm_zero_single,
# embedding, mlp_embedder, attention, multiheadcrossattention,
# gru, lstm, rnn, prelu, dropout, adaptiveavgpool2d

# Benchmark blocks (5 total)
wide_compiler benchmark mlp_block -p quick
wide_compiler benchmark attention_block -p quick
wide_compiler benchmark joint_attention -p quick
wide_compiler benchmark double_stream_block -p quick
wide_compiler benchmark single_stream_block -p quick

# With presets
wide_compiler benchmark rmsnorm -p quick     # Quick (fewer configs)
wide_compiler benchmark rmsnorm -p full      # Full sweep (default)
wide_compiler benchmark rmsnorm -p ci        # CI preset (minimal)

# With torch.compile
wide_compiler benchmark rmsnorm -p quick -c

# Other options
wide_compiler benchmark rmsnorm -t 20          # Show top 20 results
wide_compiler benchmark rmsnorm -s             # Auto-save with timestamp
wide_compiler benchmark rmsnorm -o results.json  # Save to specific file
```

### Run Test Suite

```bash
# Run all 29 components (24 primitives + 5 blocks)
python test_cases.py

# Primitives only
python test_cases.py --primitives

# Blocks only
python test_cases.py --blocks

# With different presets
python test_cases.py --preset full
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

## Supported Layers (v0.7.0 - 24 primitives)

### Linear & Embedding

| Layer | Wide Version | I/O Format | Strategies | Best Speedup |
|-------|--------------|------------|------------|--------------|
| `nn.Embedding` | `WideEmbedding` | `[N,B,T]→[N,B,T,D]` | indexed, gather, sequential | **9.1x** @ N=32 |
| `nn.Linear` | `WideLinear` | `[N,B,...,Din]→[N,B,...,Dout]` | einsum, sequential | **9.7x** @ N=32 |
| `MLPEmbedder` | `WideMLPEmbedder` | `[N,B,D]→[N,B,Dout]` | fused, sequential | **14.8x** @ N=32 |

### Convolution Layers

| Layer | Wide Version | I/O Format | Strategies | Best Speedup |
|-------|--------------|------------|------------|--------------|
| `nn.Conv1d` | `WideConv1d` | `[N,B,C,L]→[N,B,Cout,Lout]` | grouped, sequential | **5.3x** @ N=32 |
| `nn.Conv2d` | `WideConv2d` | `[N,B,C,H,W]→[N,B,Cout,Hout,Wout]` | grouped, channels_last, sequential | **2.9x** @ N=16 |
| `nn.Conv3d` | `WideConv3d` | `[N,B,C,D,H,W]→[N,B,Cout,Dout,Hout,Wout]` | grouped, sequential | **1.9x** @ N=16 |
| `nn.ConvTranspose1d` | `WideConvTranspose1d` | `[N,B,C,L]→[N,B,Cout,Lout]` | grouped, sequential | **5.4x** @ N=32 |
| `nn.ConvTranspose2d` | `WideConvTranspose2d` | `[N,B,C,H,W]→[N,B,Cout,Hout,Wout]` | grouped, channels_last, sequential | **3.8x** @ N=32 |

### Normalization Layers

| Layer | Wide Version | I/O Format | Strategies | Best Speedup |
|-------|--------------|------------|------------|--------------|
| `nn.RMSNorm` | `WideRMSNorm` | `[N,B,...,D]→[N,B,...,D]` | batched, sequential | **21.0x** @ N=32 |
| `nn.BatchNorm1d` | `WideBatchNorm1d` | `[N,B,C]→[N,B,C]` | wide | **11.5x** @ N=32 |
| `AdaLayerNormZeroSingle` | `WideAdaLayerNormZeroSingle` | `[N,B,D],[N,B,Demb]→[N,B,D],gate` | fused, sequential | **9.7x** @ N=32 |
| `nn.InstanceNorm2d` | `WideInstanceNorm2d` | `[N,B,C,H,W]→[N,B,C,H,W]` | fused, sequential | **7.7x** @ N=16 |
| `nn.GroupNorm` | `WideGroupNorm` | `[N,B,C,...]→[N,B,C,...]` | fused, sequential | **4.9x** @ N=16 |
| `nn.LayerNorm` | `WideLayerNorm` | `[N,B,...,D]→[N,B,...,D]` | wide | **4.7x** @ N=32 |
| `nn.BatchNorm2d` | `WideBatchNorm2d` | `[N,B,C,H,W]→[N,B,C,H,W]` | wide | **3.4x** @ N=16 |
| `nn.BatchNorm3d` | `WideBatchNorm3d` | `[N,B,C,D,H,W]→[N,B,C,D,H,W]` | wide | 0.9x @ N=4 (slower) |

### Attention Layers

| Layer | Wide Version | I/O Format | Strategies | Best Speedup |
|-------|--------------|------------|------------|--------------|
| `MultiheadCrossAttention` | `WideMultiheadCrossAttention` | `[N,B,Tq,D],[N,B,Tkv,D]→[N,B,Tq,D]` | fused, sequential | **15.7x** @ N=32 |
| `nn.MultiheadAttention` | `WideAttention` | `[N,B,T,D]→[N,B,T,D]` | fused, sequential | **3.5x** @ N=8 |

### RNN Layers

| Layer | Wide Version | I/O Format | Strategies | Best Speedup | Notes |
|-------|--------------|------------|------------|--------------|-------|
| `nn.RNN` | `WideRNN` | `[N,B,T,Din]→[N,B,T,H],[N,B,H]` | fused, sequential | 1.0x @ N=32 | Break-even |
| `nn.LSTM` | `WideLSTM` | `[N,B,T,Din]→[N,B,T,H],[N,B,H],[N,B,H]` | fused, sequential | 0.7x @ N=32 | Slower (cuDNN) |
| `nn.GRU` | `WideGRU` | `[N,B,T,Din]→[N,B,T,H],[N,B,H]` | fused, sequential | 0.5x @ N=32 | Slower (cuDNN) |

### Other Layers

| Layer | Wide Version | I/O Format | Strategies | Best Speedup | Notes |
|-------|--------------|------------|------------|--------------|-------|
| `nn.Dropout` | `WideDropout` | `[N,B,...]→[N,B,...]` | independent, shared, sequential | **73.2x** @ N=32 | Extreme speedup |
| `nn.AdaptiveAvgPool2d` | `WideAdaptiveAvgPool2d` | `[N,B,C,Hin,Win]→[N,B,C,Hout,Wout]` | batched, sequential | **2.2x** @ N=8 |
| `nn.PReLU` | `WidePReLU` | `[N,B,C,...]→[N,B,C,...]` | wide, sequential | 1.0x @ N=4 | Break-even |
| `F.relu`, `F.gelu`, etc. | `FunctionalOp` | agnostic | — | — | Passthrough |
| `+`, `-`, `*`, `/`, `@` | `BinaryOp` | agnostic | — | — | Element-wise |

**All primitives operate on N-first format `[N, B, ...]` internally for optimal performance.**

*Benchmarks: A100 GPU, eager mode (no compile), quick preset. See test_cases.py for full results.*

## Flux-Style Blocks (v0.7.0 - 5 blocks)

Higher-level composite blocks for transformer architectures.

| Block | I/O Format | Components | Best Speedup |
|-------|------------|------------|--------------|
| `WideMLP` | `[N,B,T,D]→[N,B,T,D]` | 2x Linear + activation | **2.9x** @ N=32 |
| `WideAttention` | `[N,B,T,D]→[N,B,T,D]` | QKV proj + SDPA + out proj | **7.9x** @ N=16 |
| `WideJointAttention` | `[N,B,Ttxt,D],[N,B,Timg,D]→[N,B,Ttxt,D],[N,B,Timg,D]` | Dual-stream attention | **9.0x** @ N=32 |
| `WideDoubleStreamBlock` | `[N,B,Ttxt,D],[N,B,Timg,D]→[N,B,Ttxt,D],[N,B,Timg,D]` | JointAttn + 2x MLP + norms | **5.2x** @ N=8 |
| `WideSingleStreamBlock` | `[N,B,T,D],[N,B,Demb]→[N,B,T,D]` | AdaLN + Attn + MLP | **3.4x** @ N=8 |

### Block Usage

```python
from wide_compiler.core.blocks import WideDoubleStreamBlock

# Create N double-stream blocks
blocks = [...]  # Your N DoubleStreamBlock modules
wide_block = WideDoubleStreamBlock.from_modules(blocks, strategy='fused')

# Input: Two streams (text and image)
txt = torch.randn(8, 4, 64, 256).cuda()   # [N, B, Ttxt, D]
img = torch.randn(8, 4, 256, 256).cuda()  # [N, B, Timg, D]
rope = torch.randn(320, 128).cuda()       # [Ttxt+Timg, head_dim]

# Forward through block
txt_out, img_out = wide_block(txt, img, rope=rope)
```

## Primitive Benchmarks (A100, compiled)

All benchmarks: A100 GPU, torch.compile (default mode), quick preset (N=[4,8,16,32]).

### Top Performers

| Primitive | Best Speedup | Strategy |
|-----------|--------------|----------|
| **Dropout** | **173.7x** | shared |
| **BatchNorm1d** | **36.7x** | wide |
| **BatchNorm2d** | **35.8x** | wide |
| **Embedding** | **27.1x** | indexed/gather |
| **BatchNorm3d** | **23.5x** | wide |
| **InstanceNorm2d** | **21.3x** | fused |
| **RMSNorm** | **20.8x** | batched |
| **MultiheadCrossAttention** | **17.8x** | fused |
| **AdaptiveAvgPool2d** | **15.2x** | batched |
| **AdaLayerNormZeroSingle** | **15.2x** | fused |

### Linear & Embedding

| Primitive | Best Speedup | Strategy |
|-----------|--------------|----------|
| **MLPEmbedder** | **14.8x** | fused |
| **Linear** | **8.8x** | einsum |

### Convolution Layers

| Primitive | Best Speedup | Strategy |
|-----------|--------------|----------|
| **Conv1d** | **12.1x** | grouped |
| **PReLU** | **12.0x** | wide |
| **Conv2d** | **6.2x** | grouped |
| **ConvTranspose2d** | **5.7x** | grouped |
| **ConvTranspose1d** | **5.3x** | grouped |
| **Conv3d** | **4.4x** | grouped |

### Attention Layers

| Primitive | Best Speedup | Strategy |
|-----------|--------------|----------|
| **Attention** | **9.6x** | fused |

### Other Layers

| Primitive | Best Speedup | Strategy | Notes |
|-----------|--------------|----------|-------|
| **GroupNorm** | **12.9x** | fused | |
| **LayerNorm** | **9.8x** | wide | |
| **RNN** | **5.6x** | fused | Speedup improved with compilation |
| **LSTM** | **3.3x** | fused | Speedup improved with compilation |
| **GRU** | **2.9x** | fused | Speedup improved with compilation |

### Key Takeaways (A100 Compiled)

1. **Dropout** achieves extreme speedups (173.7x) with shared random state
2. **BatchNorm layers** see massive gains with compilation (23-37x)
3. **Embedding** scales exceptionally well (27.1x)
4. **RMSNorm** provides 20.8x speedup, outperforms LayerNorm (9.8x) by 2.1x
5. **CrossAttention** scales to 17.8x with compilation
6. **RNN layers** benefit from compilation (GRU: 2.9x, LSTM: 3.3x, RNN: 5.6x)
7. **Conv1d** reaches 12.1x with compilation vs 5.3x eager
8. **Compilation is critical** - most primitives see 2-5x additional speedup

## Block Benchmarks (A100, compiled)

| Block | N=4 | N=8 | N=16 | N=32 | Components |
|-------|-----|-----|------|------|------------|
| **AttentionBlock** | 3.7x | 6.1x | **10.7x** | 10.3x | QKV proj + SDPA + norm |
| **DoubleStreamBlock** | 3.2x | **6.8x** | — | — | JointAttn + 2xMLP + norms |
| **JointAttention** | 2.4x | 3.9x | 5.2x | **5.5x** | Dual-stream QKV + concat attn |
| **MLPBlock** | 2.1x | 2.9x | 3.6x | **3.9x** | 2x Linear + activation |
| **SingleStreamBlock** | 2.0x | **3.5x** | — | — | AdaLN + Attn + MLP |

## How it Works (v0.7.0)

1. **FX Tracing** - `torch.fx.symbolic_trace` captures the computation graph
2. **Registry Lookup** - Each layer dynamically resolved via global registry (24 primitives registered)
3. **Wide Primitives** - Each layer replaced with N-first fused equivalent:
   - `Linear` → Batched einsum `[N, B, I] @ [N, O, I]` (transposed weight)
   - `Conv2d` → Grouped convolution on `[N*B, C, H, W]` with `groups=N`
   - `Attention` → Reshape N→batch, single Flash Attention call
   - `RMSNorm` → Batched normalization `[N, B, D]`
   - All primitives operate on N-first `[N, B, ...]`
4. **Strategy Selection** - Each primitive auto-selects optimal strategy based on N
5. **Compile-Friendly** - 0 graph breaks, all native PyTorch ops
6. **Optimal Data Flow** - Only 2 reshapes per forward pass:
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

## Project Structure

```
wide_compiler/
├── __init__.py
├── __main__.py
├── api.py                    # compile(), WideBuilder, pack(), unpack()
├── cli.py                    # CLI commands
├── test_cases.py             # Reusable test suite (NEW)
└── core/
    ├── config.py             # WideConfig
    ├── registry.py           # Dynamic primitive registration (24 primitives)
    ├── traced_wide.py        # FX tracing + TracedWideModel
    ├── ensemble_util.py      # pack_inputs(), unpack_outputs()
    ├── benchmark/            # Benchmark system
    │   ├── benchmark_api.py      # High-level API
    │   ├── benchmark_runner.py   # Execution engine
    │   ├── benchmark_schema.py   # BenchmarkJob, results
    │   └── benchmark_registry.py # Auto-discovery
    ├── blocks/               # Flux-style composite blocks (NEW)
    │   ├── wide_mlp.py
    │   ├── wide_attention.py
    │   ├── wide_joint_attention.py
    │   ├── wide_double_stream_block.py
    │   └── wide_single_stream_block.py
    └── primitives/           # 24 wide primitives
        ├── wide_linear.py
        ├── wide_conv1d.py, wide_conv2d.py, wide_conv3d.py
        ├── wide_convtranspose1d.py, wide_convtranspose2d.py (NEW)
        ├── wide_batchnorm_1d.py, wide_batchnorm_2d.py, wide_batchnorm_3d.py (NEW)
        ├── wide_layernorm.py
        ├── wide_groupnorm.py
        ├── wide_instancenorm.py
        ├── wide_rmsnorm.py (NEW)
        ├── wide_ada_layer_norm_zero_single.py (NEW)
        ├── wide_embedding.py
        ├── wide_mlp_embedder.py (NEW)
        ├── wide_attention.py
        ├── wide_cross_attention.py (NEW)
        ├── wide_gru.py, wide_lstm.py, wide_rnn.py (NEW)
        ├── wide_prelu.py (NEW)
        ├── wide_dropout.py (NEW)
        └── wide_adaptive_avgpool2d.py (NEW)
```

## Limitations

### General
- **Identical architecture required** - All N models must have same structure
- **Static shapes** - FX tracing requires fixed tensor shapes
- **No dynamic control flow** - `if`/`for` based on tensor values won't trace

### Known Slowdowns
- **RNN layers** - cuDNN optimizations make sequential execution faster
- **BatchNorm3d** - No grouped implementation, always slower
- **PReLU** - High kernel launch overhead at low N
- **Attention** - Performance degrades at N>16 (memory bandwidth)

### RNN Primitives (WideGRU, WideLSTM, WideRNN)
- **Single layer only** - `num_layers=1` currently required
- **Unidirectional only** - `bidirectional=False` required
- **batch_first only** - `batch_first=True` required
- **Slower than cuDNN** - Not recommended unless N>32

## Use Cases

- **Ensemble models** - Run N ensemble members in parallel
- **Hyperparameter search** - Evaluate N configurations simultaneously
- **Population-based training** - Evolve N agents together
- **Monte Carlo dropout** - N stochastic forward passes
- **Transformer ensembles** - N attention heads across models
- **Flux-style diffusion** - Parallel text/image streams with WideDoubleStreamBlock

## License

Apache License 2.0

## Author

AbstractPhil - [HuggingFace](https://huggingface.co/AbstractPhil) | [GitHub](https://github.com/AbstractEyes)
