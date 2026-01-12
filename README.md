# pytorch-parallel-compiler

Compile-friendly batched model execution. Fuse N identical models into a single Wide model for massive speedups.

## What it does

Instead of running N models sequentially:
```python
outputs = [model(x) for model in models]  # N kernel launches
```

WideCompiler fuses them into one:
```python
wide_model = TracedWideModel.from_models(models, sample)
output = wide_model(packed_input)  # 1 kernel launch
```

**Speedups:** 2-5x eager, **20-40x compiled** (CUDA)

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import torch
from wide_compiler import TracedWideModel, pack_inputs, unpack_outputs

# Your model
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 64)
    
    def forward(self, x):
        return self.fc2(torch.nn.functional.relu(self.fc1(x)))

# Create N identical models (different weights)
N = 100
models = [MLP().cuda() for _ in range(N)]

# Build wide model (requires sample input for FX tracing)
sample = torch.randn(1, 64).cuda()
wide_model = TracedWideModel.from_models(models, sample).cuda()

# Prepare inputs
inputs = [torch.randn(32, 64).cuda() for _ in range(N)]
packed = pack_inputs(inputs)  # [B, N*C] layout

# Run
output = wide_model(packed)

# Unpack if needed
outputs = unpack_outputs(output, N)  # List of N tensors

# Compile for max speed
compiled = torch.compile(wide_model, mode='reduce-overhead')
```

## CLI

```bash
# Run correctness tests
python -m wide_compiler test

# Show FX trace for a model
python -m wide_compiler trace --model resblock

# Benchmark speedup
python -m wide_compiler benchmark --model mlp --n 100 --compile

# Show info
python -m wide_compiler info
```

Available test models: `mlp`, `deep_mlp`, `resblock`, `convnet`

## How it works

1. **FX Tracing** - Uses `torch.fx.symbolic_trace` to capture the computation graph
2. **Wide Primitives** - Replaces each layer with grouped equivalents:
   - `Linear` → `Conv1d` with `groups=N`
   - `Conv2d` → `Conv2d` with `groups=N`
   - `BatchNorm` → `BatchNorm` over `N*C` channels
3. **Graph Execution** - Replays ops in traced order, respecting dataflow (handles residuals)
4. **Compile-Friendly** - All ops are native PyTorch, no custom CUDA kernels

## Supported Layers

| Layer | Wide Version | Notes |
|-------|--------------|-------|
| `nn.Linear` | `WideLinear` | Via grouped Conv1d |
| `nn.Conv1d` | `WideConv1d` | Grouped convolution |
| `nn.Conv2d` | `WideConv2d` | Grouped convolution |
| `nn.BatchNorm1d` | `WideBatchNorm1d` | Single BN over N*C |
| `nn.BatchNorm2d` | `WideBatchNorm2d` | Single BN over N*C |
| `nn.LayerNorm` | `WideLayerNorm` | Per-group normalization |
| `nn.Embedding` | `WideEmbedding` | Concatenated tables |
| `F.relu`, `F.gelu`, etc. | Passthrough | Elementwise ops work directly |
| `+`, `-`, `*`, `/`, `@` | `BinaryOp` | Captured via FX |

## Limitations

- **Identical architecture required** - All N models must have the same structure
- **Static shapes** - FX tracing requires fixed tensor shapes
- **No dynamic control flow** - `if`/`for` based on tensor values won't trace
- **Attention not yet supported** - MultiheadAttention needs custom Wide version

## Benchmarks

MLP (N=100, batch=32, CUDA):
```
Eager:    2-3x speedup
Compiled: 35-40x speedup
```

ResBlock (N=20, batch=8, CUDA):
```
Eager:    ~5x speedup  
Compiled: ~10x speedup
```

## Use Cases

- **Ensemble models** - Run N ensemble members in parallel
- **Hyperparameter search** - Evaluate N configurations simultaneously  
- **Population-based training** - Evolve N agents together
- **Monte Carlo dropout** - N stochastic forward passes

## License

Apache License 2.0

## Author

AbstractPhil - [HuggingFace](https://huggingface.co/AbstractPhil) | [GitHub](https://github.com/AbstractEyes)