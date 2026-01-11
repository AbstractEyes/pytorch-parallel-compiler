# WideCompiler

Type-agnostic compilation optimizer for PyTorch models.

## Usage

```python
import wide_compiler

# Compile any model
compiled = wide_compiler.compile(model, mode='hybrid')

# Use normally
output = compiled(input)
loss.backward()  # Works
```

## Modes

| Mode | Speedup | Training | Description |
|------|---------|----------|-------------|
| `default` | ~5x | ✅ | Safe compilation |
| `hybrid` | ~10-15x | ✅ | CUDAGraphs + wide_forward |
| `reduce-overhead` | ~20x | ❌ | Full CUDAGraphs (inference) |
| `inference` | ~20x | ❌ | Alias for reduce-overhead |

## How It Works

1. **Analyze** - Detect parallel regions
2. **Organize** - Structure into wide_forward calls
3. **Protect** - Wrap parallel with `@torch.compiler.disable`
4. **Compile** - Apply `torch.compile`

## Structured Models

```python
from wide_compiler import CompiledModel, ParallelBlock

model = CompiledModel("my_model")
layer = model.add_layer("encoder")
layer.add_sequential(nn.Linear(64, 128))
layer.add_parallel([nn.Linear(128, 128) for _ in range(4)])
layer.add_sequential(nn.Linear(128, 64))

compiled = model.compile(mode='hybrid')
```

## Hierarchy

```
Block  → Layer  → Model
  │        │        │
  ├─ CompiledBlock  ├─ CompiledLayer  ├─ CompiledModel
  └─ ParallelBlock  └─ ParallelLayer  └─ ParallelModel
```

- **Compiled*** - Sequential execution
- **Parallel*** - Parallel execution (wide_forward eligible)

## Analysis

```python
analysis = wide_compiler.analyze(model)
print(f"Parallel regions: {analysis.num_parallel}")
print(wide_compiler.summary(model))
```

## Requirements

- Python 3.8+
- PyTorch 2.0+

## License

Apache 2.0