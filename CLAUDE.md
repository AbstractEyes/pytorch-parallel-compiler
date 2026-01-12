# CLAUDE.md - WideCompiler Quick Reference

VERSION = 0.2.0

Read this first when working on this codebase.

## What This Project Does

Fuses N identical PyTorch models into ONE wide model. Instead of N sequential forward passes, one batched forward pass using grouped convolutions.

**Key insight:** `nn.Conv1d(groups=N)` processes N independent channels in one kernel launch.

## Core Concept

```
N separate models:        Wide model:
[Model_0] → out_0        [WideModel] → [out_0, out_1, ..., out_N]
[Model_1] → out_1         (single forward pass)
...
[Model_N] → out_N
```

**Memory layout:** `[B, N*C, ...]` - N models' channels interleaved.

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
├── __init__.py      → Package exports (try/except for inline debug)
├── api.py           → compile(), WideBuilder, pack(), unpack()
├── cli.py           → CLI commands (test, benchmark, trace, info)
├── __main__.py      → Delegates to cli.main()
└── core/
    ├── config.py    → WideConfig dataclass
    ├── registry.py  → Maps 'Linear' → WideLinear.from_modules
    ├── traced_wide.py   → FX tracing, graph execution (THE CORE)
    ├── wide_model.py    → pack_inputs(), unpack_outputs(), tree utils
    └── primitives/      → One file per Wide op
        ├── wide_linear.py      → Linear via grouped Conv1d
        ├── wide_conv2d.py      → Conv2d with groups=N
        └── ...
```

## How It Works

### 1. Trace (traced_wide.py)
```python
traced = fx.symbolic_trace(template_model)
# Captures: call_module, call_function, call_method
# Graph shows dataflow including residual connections
```

### 2. Build Wide Ops
```python
for node in traced.graph.nodes:
    if node.op == 'call_module':
        modules = [m.get_submodule(path) for m in models]
        wide_op = WIDE_BUILDERS[module_type](modules)  # e.g., WideLinear.from_modules
```

### 3. Graph Execution (handles residuals)
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
    def __init__(self, n, in_features, out_features):
        # Build grouped op in __init__, NOT reshape in forward
        self.op = nn.Conv1d(n * in_features, n * out_features, 1, groups=n)
    
    def forward(self, x):
        return self.op(x)  # No reshapes!
    
    @classmethod
    def from_modules(cls, modules: List[nn.Linear]) -> 'WideLinear':
        # Stack weights from N modules into grouped conv
```

**Critical:** All reshaping happens at construction. Forward is just `self.op(x)`.

## Registry

```python
# Auto-registers on import via auto_register_primitives()
WIDE_BUILDERS = {
    'Linear': WideLinear.from_modules,
    'Conv2d': WideConv2d.from_modules,
    ...
}

# User can add custom:
@wide_compiler.register('MyOp')
class WideMyOp:
    @classmethod
    def from_modules(cls, modules): ...
```

## Pack / Unpack

```python
# Pack: List of [B, C, ...] → [B, N*C, ...]
packed = pack_inputs(inputs)  # torch.stack then reshape

# Unpack: [B, N*C, ...] → List of [B, C, ...]
outputs = unpack_outputs(output, n)  # reshape then index
```

## Config

```python
WideConfig(
    compile=True,           # torch.compile the result
    compile_mode='reduce-overhead',
    validate=True,          # Check outputs match
    debug=True,             # Print build info
)

# Presets
WideConfig.fast()   # Compiled, no validation
WideConfig.debug()  # Verbose, strict
```

## CLI

```bash
python -m wide_compiler test                    # Correctness tests
python -m wide_compiler benchmark -n 100 -c     # Benchmark with compile
python -m wide_compiler trace --model resblock  # Show FX graph
```

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `TracedWideModel` | traced_wide.py | Main output - the fused model |
| `WideConfig` | config.py | Configuration dataclass |
| `WideRegistry` | registry.py | Maps module types to builders |
| `WideLinear` etc. | primitives/ | Grouped op implementations |

## Debugging

1. **Trace not working?** Model has dynamic control flow. FX can't trace `if tensor.sum() > 0`.

2. **Wrong outputs?** Check `pack_inputs` produces `[B, N*C, ...]`. Print shapes.

3. **Slow compiled?** First few runs are warmup. Benchmark after 5+ iterations.

4. **Unknown module type?** Either:
   - Register it: `@wide_compiler.register('MyModule')`
   - Set `fallback_passthrough=True` in config (wraps as FunctionalOp)

## Speedup Sources

1. **Kernel launch overhead** - 1 launch vs N launches
2. **Memory coalescing** - Grouped conv accesses memory efficiently
3. **torch.compile** - Fuses ops, uses CUDA graphs

Typical: **2-5x eager, 20-40x compiled**

## Extension Points

1. **New primitive:** Add to `primitives/`, register in `registry.py`
2. **New config option:** Add field to `WideConfig`
3. **Graph optimization:** Transform `traced.graph` before building
4. **Custom backend:** Use `strategy/` folder (placeholder for inductor hooks)

## Import Pattern

All files use try/except for inline debugging:

```python
try:
    from .core import TracedWideModel
except ImportError:
    from wide_compiler.core import TracedWideModel
```

This allows running files directly: `python wide_compiler/cli.py test`

## Quick Test

```python
import torch
import wide_compiler

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 64)
    def forward(self, x):
        return self.fc2(torch.nn.functional.relu(self.fc1(x)))

models = [MLP() for _ in range(10)]
sample = torch.randn(1, 64)
wide = wide_compiler.compile(models, sample)
print(wide.summary())
```

---

**TL;DR:** `wide_compiler.compile(models, sample)` → FX traces → builds Wide ops from registry → returns `TracedWideModel` with graph-based forward.