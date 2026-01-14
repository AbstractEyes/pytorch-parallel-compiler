"""
WideCompiler Benchmark System - Unified Architecture
=====================================================

CORE PRINCIPLE: Primitives own their benchmark configs.
No central config file. Edit the primitive, edit its benchmarks.

ARCHITECTURE
------------

    ┌─────────────────────────────────────────────────────────────┐
    │                         API / CLI                           │
    │                                                             │
    │  benchmark('conv1d')                                        │
    │  benchmark('conv1d', preset='quick')                        │
    │  benchmark_custom(model_cls, input_shape, n_values)         │
    │                                                             │
    │  $ wide_compiler benchmark conv1d                           │
    │  $ wide_compiler benchmark conv1d --preset quick            │
    │  $ wide_compiler benchmark --all                            │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                        REGISTRY                             │
    │                                                             │
    │  Maps name -> primitive class                               │
    │                                                             │
    │  'conv1d'  -> WideConv1d                                    │
    │  'conv2d'  -> WideConv2d                                    │
    │  'linear'  -> WideLinear                                    │
    │                                                             │
    │  Each class implements BenchmarkProvider protocol           │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                        RUNNER                               │
    │                                                             │
    │  Generic execution - knows nothing about specific ops       │
    │                                                             │
    │  run(job: BenchmarkJob) -> BenchmarkResult                  │
    │                                                             │
    │  For each n in job.sweep.n_values:                          │
    │      For each params in job.sweep.param_grid():             │
    │          modules = [job.model_factory(**params) for _ in n] │
    │          inputs = job.input_factory(n, **params)            │
    │          time baseline, time each strategy                  │
    │          check correctness                                  │
    │          record result                                      │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   PRIMITIVE (e.g. WideConv1d)               │
    │                                                             │
    │  class WideConv1d(nn.Module):                               │
    │                                                             │
    │      # Normal module code...                                │
    │      def __init__(...): ...                                 │
    │      def forward(x): ...                                    │
    │      def from_modules(modules, strategy): ...               │
    │                                                             │
    │      # ============ BENCHMARK INTERFACE ============        │
    │                                                             │
    │      BENCHMARK_SWEEPS = {                                   │
    │          'quick': SweepParams(                              │
    │              n_values=[4, 8, 16],                           │
    │              channels=[64],                                 │
    │              kernel_sizes=[3],                              │
    │              seq_lengths=[256],                             │
    │          ),                                                 │
    │          'full': SweepParams(                               │
    │              n_values=[2,4,6,8,10,12,16,20,32,64],          │
    │              channels=[32, 64, 128],                        │
    │              kernel_sizes=[1, 3, 5, 7],                     │
    │              seq_lengths=[64, 256, 1024],                   │
    │          ),                                                 │
    │      }                                                      │
    │                                                             │
    │      BENCHMARK_STRATEGIES = ['baseline', 'grouped', 'seq']  │
    │                                                             │
    │      @classmethod                                           │
    │      def benchmark_job(cls, preset='full') -> BenchmarkJob  │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘


UNIFIED SCHEMA (schema.py)
--------------------------

Minimal, uniform across all primitives:

```python
@dataclass
class SweepParams:
    '''Parameter ranges. Use only fields that apply.'''
    n_values: List[int]
    batch_sizes: List[int] = field(default_factory=lambda: [8])
    
    # Dimension params (use what applies)
    channels: List[int] = field(default_factory=list)
    features: List[int] = field(default_factory=list)
    seq_lengths: List[int] = field(default_factory=list)
    spatial_sizes: List[int] = field(default_factory=list)
    kernel_sizes: List[int] = field(default_factory=list)
    embed_dims: List[int] = field(default_factory=list)
    vocab_sizes: List[int] = field(default_factory=list)
    
    def param_grid(self) -> Iterator[Dict[str, Any]]:
        '''Yield all (non-n) parameter combinations.'''


@dataclass 
class BenchmarkJob:
    '''Everything needed to run a benchmark.'''
    name: str
    strategies: List[str]
    sweep: SweepParams
    
    # Factories (the primitive provides these)
    model_factory: Callable      # (**params) -> nn.Module
    input_factory: Callable      # (n, batch, **params) -> Tensor
    wide_factory: Callable       # (modules, strategy) -> WideModule
    pack_fn: Callable            # (List[Tensor]) -> Tensor
    unpack_fn: Callable          # (Tensor, n) -> List[Tensor]
    
    # Settings
    device: str = 'cuda'
    warmup: int = 20
    iters: int = 100


@dataclass
class SingleResult:
    '''One (n, params) measurement.'''
    n: int
    params: Dict[str, Any]
    timings: Dict[str, float]    # strategy -> ms
    speedups: Dict[str, float]   # strategy -> speedup vs baseline
    best_strategy: str
    best_speedup: float
    correct: bool


@dataclass
class BenchmarkResult:
    '''Complete output. Serializable.'''
    name: str
    results: List[SingleResult]
    crossover_n: Optional[int]
    recommended_threshold: int
    strategy_wins: Dict[str, int]
    best_speedup: float
    device: str
    duration_s: float
    
    def save(self, path): ...
    def load(cls, path): ...
    def summary(self) -> str: ...
```


PRIMITIVE IMPLEMENTATION PATTERN
--------------------------------

Every primitive follows this exact pattern:

```python
# wide_conv1d.py

class WideConv1d(nn.Module):
    '''N parallel Conv1d as single op.'''
    
    # ========================
    # NORMAL MODULE CODE
    # ========================
    
    def __init__(self, n, in_ch, out_ch, kernel, stride=1, padding=0, 
                 dilation=1, bias=True, strategy=None):
        ...
    
    def forward(self, x):
        ...
    
    @classmethod
    def from_modules(cls, modules, strategy=None):
        ...
    
    
    # ========================
    # BENCHMARK INTERFACE
    # ========================
    
    BENCHMARK_SWEEPS = {
        'quick': SweepParams(
            n_values=[4, 8, 16, 32],
            batch_sizes=[8],
            channels=[64],
            kernel_sizes=[3],
            seq_lengths=[256],
        ),
        'full': SweepParams(
            n_values=[2, 4, 6, 8, 10, 12, 16, 20, 32, 64],
            batch_sizes=[8],
            channels=[32, 64, 128],
            kernel_sizes=[1, 3, 5, 7],
            seq_lengths=[64, 256, 1024],
        ),
        'ci': SweepParams(
            n_values=[4, 16],
            batch_sizes=[8],
            channels=[64],
            kernel_sizes=[3],
            seq_lengths=[256],
        ),
    }
    
    BENCHMARK_STRATEGIES = ['baseline', 'grouped', 'sequential']
    
    @classmethod
    def benchmark_job(cls, preset: str = 'full', **overrides) -> BenchmarkJob:
        '''Get benchmark job for this primitive.'''
        sweep = cls.BENCHMARK_SWEEPS[preset]
        
        # Apply any overrides
        if overrides:
            sweep = replace(sweep, **overrides)
        
        return BenchmarkJob(
            name=f'conv1d_{preset}',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
            pack_fn=cls._bench_pack,
            unpack_fn=cls._bench_unpack,
        )
    
    # --- Factory methods ---
    
    @staticmethod
    def _bench_model(channels, kernel_sizes, **_):
        return nn.Conv1d(channels, channels, kernel_sizes, 
                         padding=kernel_sizes//2)
    
    @staticmethod
    def _bench_input(n, batch_sizes, channels, seq_lengths, device, **_):
        return torch.randn(batch_sizes, n * channels, seq_lengths, 
                          device=device)
    
    @classmethod
    def _bench_wide(cls, modules, strategy):
        strat = Conv1dStrategy[strategy.upper()]
        return cls.from_modules(modules, strategy=strat)
    
    @staticmethod
    def _bench_pack(inputs):
        # inputs: List[Tensor[B, C, L]]
        stacked = torch.stack(inputs, dim=1)  # [B, N, C, L]
        B, N, C, L = stacked.shape
        return stacked.view(B, N * C, L)
    
    @staticmethod
    def _bench_unpack(output, n):
        B, NC, L = output.shape
        C = NC // n
        return output.view(B, n, C, L).unbind(1)
```


API (benchmark_api.py)
----------------------

```python
def benchmark(
    name: str,
    preset: str = 'full',
    device: str = 'cuda',
    verbose: bool = True,
    **overrides,
) -> BenchmarkResult:
    '''
    Run benchmark for a registered primitive.
    
    Args:
        name: 'conv1d', 'conv2d', 'linear', etc.
        preset: 'quick', 'full', 'ci'
        device: 'cuda' or 'cpu'
        verbose: Print progress
        **overrides: Override sweep params (e.g., n_values=[4,8])
    
    Returns:
        BenchmarkResult with all measurements
    
    Example:
        result = benchmark('conv1d')
        result = benchmark('conv1d', preset='quick')
        result = benchmark('conv1d', n_values=[8, 16, 32])
    '''
    cls = REGISTRY[name]
    job = cls.benchmark_job(preset, **overrides)
    job.device = device
    return run(job, verbose=verbose)


def benchmark_custom(
    model_class: type,
    input_shape: Tuple[int, ...],
    n_values: List[int],
    name: str = 'custom',
    device: str = 'cuda',
    **model_kwargs,
) -> BenchmarkResult:
    '''
    Benchmark an arbitrary model class.
    
    Args:
        model_class: nn.Module subclass
        input_shape: Shape for single model input (B, ...)
        n_values: List of N values to test
        name: Name for this benchmark
        device: 'cuda' or 'cpu'
        **model_kwargs: Arguments to pass to model_class()
    
    Example:
        class Expert(nn.Module):
            def __init__(self, d=256):
                super().__init__()
                self.fc1 = nn.Linear(d, d*4)
                self.fc2 = nn.Linear(d*4, d)
            def forward(self, x):
                return self.fc2(F.gelu(self.fc1(x)))
        
        result = benchmark_custom(Expert, (32, 256), [8, 16, 32, 64])
    '''
    ...
```


CLI (cli.py additions)
----------------------

```python
@cli.command()
@click.argument('names', nargs=-1)
@click.option('--preset', default='full', help='quick/full/ci')
@click.option('--device', default='cuda')
@click.option('--output', '-o', help='Save results to JSON')
@click.option('--all', 'run_all', is_flag=True, help='Run all benchmarks')
def benchmark(names, preset, device, output, run_all):
    '''Run benchmarks.
    
    Examples:
        wide_compiler benchmark conv1d
        wide_compiler benchmark conv1d conv2d --preset quick
        wide_compiler benchmark --all --preset ci
    '''
    if run_all:
        names = list(REGISTRY.keys())
    
    for name in names:
        result = benchmark_api.benchmark(name, preset=preset, device=device)
        click.echo(result.summary())
        
        if output:
            result.save(output)
```


RUNNER (runner.py)
------------------

```python
def run(job: BenchmarkJob, verbose: bool = True) -> BenchmarkResult:
    '''Execute a benchmark job.'''
    
    results = []
    
    for n in job.sweep.n_values:
        for params in job.sweep.param_grid():
            
            # Create N modules
            modules = [job.model_factory(**params).to(job.device) 
                      for _ in range(n)]
            
            # Create inputs
            single_inputs = [torch.randn(*input_shape, device=job.device) 
                            for _ in range(n)]
            packed_input = job.pack_fn(single_inputs)
            
            timings = {}
            
            # Baseline: N separate calls
            timings['baseline'] = time_fn(
                lambda: [m(x) for m, x in zip(modules, single_inputs)],
                job.warmup, job.iters, job.device
            )
            
            # Each strategy
            for strategy in job.strategies:
                if strategy == 'baseline':
                    continue
                    
                wide = job.wide_factory(modules, strategy).to(job.device)
                timings[strategy] = time_fn(
                    lambda: wide(packed_input),
                    job.warmup, job.iters, job.device
                )
            
            # Compute speedups
            baseline_ms = timings['baseline']
            speedups = {s: baseline_ms / t for s, t in timings.items()}
            best = max(speedups.items(), key=lambda x: x[1])
            
            results.append(SingleResult(
                n=n,
                params=params,
                timings=timings,
                speedups=speedups,
                best_strategy=best[0],
                best_speedup=best[1],
                correct=True,  # TODO: check
            ))
    
    return BenchmarkResult(
        name=job.name,
        results=results,
        ...
    )
```


FILE STRUCTURE
--------------

```
wide_compiler/
├── api.py                          # Main API (add benchmark())
├── cli.py                          # CLI (add benchmark command)
└── core/
    ├── benchmark/
    │   ├── __init__.py             # Exports
    │   ├── schema.py               # SweepParams, BenchmarkJob, Results
    │   ├── runner.py               # run(job) -> result
    │   ├── api.py                  # benchmark(), benchmark_custom()
    │   └── registry.py             # REGISTRY = {'conv1d': WideConv1d, ...}
    │
    └── primitives/
        ├── wide_linear.py          # Has BENCHMARK_SWEEPS, benchmark_job()
        ├── wide_conv1d.py          # Has BENCHMARK_SWEEPS, benchmark_job()
        ├── wide_conv2d.py          # Has BENCHMARK_SWEEPS, benchmark_job()
        └── ...


KEY POINTS
----------

1. PRIMITIVES OWN THEIR SWEEPS
   - Edit wide_conv1d.py to change Conv1d benchmarks
   - No separate config file to maintain
   - Sweeps live next to the code they test

2. UNIFORM INTERFACE
   - Every primitive has BENCHMARK_SWEEPS dict
   - Every primitive has BENCHMARK_STRATEGIES list  
   - Every primitive has benchmark_job() classmethod
   - Same factory method signatures

3. MINIMAL SCHEMA
   - SweepParams: just parameter lists
   - BenchmarkJob: sweep + factories
   - SingleResult: one measurement
   - BenchmarkResult: all measurements + summary

4. FLEXIBLE API
   - benchmark('conv1d')  # defaults
   - benchmark('conv1d', preset='quick')  # preset
   - benchmark('conv1d', n_values=[8,16])  # override
   - benchmark_custom(MyModel, ...)  # arbitrary

5. SERIALIZABLE OUTPUT
   - result.save('benchmark.json')
   - result = BenchmarkResult.load('benchmark.json')
   - result.summary() for human-readable
"""