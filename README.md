# pytorch-parallel-compiler

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

**Speedups:** 3-40x eager (model/N dependent), 1.5-2x vs compiled baseline

## Installation

```bash
git clone https://github.com/AbstractEyes/pytorch-parallel-compiler
cd pytorch-parallel-compiler
pip install -e .
```

## Quick Start

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

**Compile modes:**
- `default` - Standard Inductor compilation. Wide gives ~2x vs this baseline.
- `reduce-overhead` - CUDA graphs. Baseline is faster, so Wide's relative benefit drops to ~1.1-1.2x.
- No compile (eager) - Wide gives biggest benefit here (3-40x).

### Custom Primitives

```python
import wide_compiler

@wide_compiler.register('MyCustomLayer')
class WideMyCustomLayer(torch.nn.Module):
    @classmethod
    def from_modules(cls, modules):
        # Build wide version from N modules
        ...

# Check registered primitives
print(wide_compiler.list_registered())
# ['Linear', 'Conv2d', 'Conv1d', 'BatchNorm2d', 'BatchNorm1d', 'LayerNorm', 'Embedding', 'MyCustomLayer']
```

## CLI

```bash
# Run correctness tests
python -m wide_compiler test

# Benchmark speedup
python -m wide_compiler benchmark --model mlp --n 100 --compile

# Show FX trace
python -m wide_compiler trace --model resblock

# Show info
python -m wide_compiler info
```

**Available models:** `mlp`, `deep_mlp`, `resblock`, `convnet`

**Benchmark options:**
```bash
python -m wide_compiler benchmark \
    --model mlp \
    --n 100 \
    --batch 32 \
    --iters 100 \
    --compile \
    --cpu  # Force CPU
```

## How it Works

1. **FX Tracing** - `torch.fx.symbolic_trace` captures the computation graph
2. **Wide Primitives** - Each layer replaced with grouped equivalent:
   - `Linear` → Grouped `Conv1d` or einsum (auto-selected)
   - `Conv2d` → `Conv2d` with `groups=N`  
   - `BatchNorm` → `BatchNorm` over `N*C` channels
3. **Graph Execution** - Dict-based value lookup replays ops respecting dataflow
4. **Compile-Friendly** - 0 graph breaks, all native PyTorch ops

**Why it's fast:**
- N sequential models = N × num_ops kernel launches
- Wide model = num_ops kernel launches (same ops, bigger tensors)
- Kernel launch overhead dominates for small ops → Wide wins big

## Supported Layers

| Layer | Wide Version | Strategy |
|-------|--------------|----------|
| `nn.Linear` | `WideLinear` | Einsum (N<10) or grouped Conv1d (N≥10) |
| `nn.Conv1d` | `WideConv1d` | Grouped convolution |
| `nn.Conv2d` | `WideConv2d` | Grouped (N≥10) or sequential (N<10) |
| `nn.BatchNorm1d` | `WideBatchNorm1d` | Single BN over N*C |
| `nn.BatchNorm2d` | `WideBatchNorm2d` | Single BN over N*C |
| `nn.LayerNorm` | `WideLayerNorm` | Per-group normalization |
| `nn.Embedding` | `WideEmbedding` | Concatenated tables |
| `F.relu`, `F.gelu`, etc. | Passthrough | Elementwise ops work directly |
| `+`, `-`, `*`, `/`, `@` | `BinaryOp` | Captured via FX |

## Benchmarks

Speedup depends on model type, N, and whether baseline uses `torch.compile`:

**MLP** (N=100, batch=32, RTX 4090):
```
Eager:           40x speedup (1013ms → 25ms)
```

**ResBlock** (N=50, batch=8, RTX 4090):
```
Eager:           3.5x speedup (1351ms → 383ms)
```

**ResNet18** (N=10, batch=8, A100):
```
Eager:           2.3x speedup (25ms → 11ms)
vs Compiled:     1.9x speedup (17ms → 9ms)
```

**Why the variance?**
- **MLP N=100**: 100 models × small ops = massive kernel launch overhead. Wide eliminates this.
- **ResBlock/ResNet**: Conv-heavy, GPU-bound. Less launch overhead to eliminate.
- **vs Compiled**: `torch.compile` already reduces launch overhead via CUDA graphs, so Wide's relative benefit is smaller.

## Limitations

- **Identical architecture required** - All N models must have same structure
- **Static shapes** - FX tracing requires fixed tensor shapes
- **No dynamic control flow** - `if`/`for` based on tensor values won't trace
- **Attention not yet supported** - MultiheadAttention needs custom Wide version
- **Method calls with args** - e.g. `x.flatten(1)` requires careful handling in FX

## Known Working Models

| Model | Status | Notes |
|-------|--------|-------|
| MLP | ✓ | Best speedups (40x for N=100) |
| ResBlock | ✓ | 3-5x speedup |
| ResNet18 | ✓ | 2x speedup, all 69 stages traced |
| ConvNet | ✓ | Works with BN, pooling |
| Transformer | ⚠ | Attention not yet supported |

## Use Cases

- **Ensemble models** - Run N ensemble members in parallel
- **Hyperparameter search** - Evaluate N configurations simultaneously
- **Population-based training** - Evolve N agents together
- **Monte Carlo dropout** - N stochastic forward passes

## Project Structure

```
wide_compiler/
├── __init__.py          # Package exports
├── __main__.py          # CLI entry point
├── api.py               # Main API: compile(), WideBuilder
├── cli.py               # CLI commands: test, benchmark, trace, info
└── core/
    ├── config.py        # WideConfig
    ├── registry.py      # Primitive registration
    ├── traced_wide.py   # FX tracing + TracedWideModel
    ├── wide_model.py    # Tree traversal, pack/unpack
    └── primitives/
        ├── wide_linear.py     # Einsum/Conv1d strategies
        ├── wide_conv1d.py
        ├── wide_conv2d.py     # Grouped/Sequential strategies
        ├── wide_batchnorm_1d.py
        ├── wide_batchnorm_2d.py
        ├── wide_layernorm.py
        └── wide_embedding.py
```

## License

Apache License 2.0

## Author

AbstractPhil - [HuggingFace](https://huggingface.co/AbstractPhil) | [GitHub](https://github.com/AbstractEyes)