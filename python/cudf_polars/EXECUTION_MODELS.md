# Execution Models in cudf-polars

This document explains the different execution models available in the cudf-polars streaming executor.

## Overview

The cudf-polars streaming executor supports two fundamentally different execution models:

1. **Task-Based Execution** (`engine="tasks"`) - The traditional execution model
2. **RapidsMPF Execution** (`engine="rapidsmpf"`) - New asynchronous coroutine-based execution

## Task-Based Execution (Default)

The task-based execution model represents queries as task graphs. You can run on a single GPU or distributed across multiple GPUs:

### Single-GPU

```python
import polars as pl

engine = pl.GPUEngine(
    executor="streaming",
    executor_options={
        "engine": "tasks",  # Default
        "cluster": "single",  # Default
    }
)

lf = pl.scan_parquet("data.parquet")
result = lf.collect(engine=engine)
```

**Environment Variable:**
```bash
export CUDF_POLARS__EXECUTOR__ENGINE=tasks
export CUDF_POLARS__EXECUTOR__CLUSTER=single
```

### Multi-GPU (Distributed)

```python
import polars as pl
from dask.distributed import Client

# Start Dask cluster
client = Client()

engine = pl.GPUEngine(
    executor="streaming",
    executor_options={
        "engine": "tasks",
        "cluster": "distributed",
    }
)

lf = pl.scan_parquet("data.parquet")
result = lf.collect(engine=engine)
```

**Environment Variable:**
```bash
export CUDF_POLARS__EXECUTOR__ENGINE=tasks
export CUDF_POLARS__EXECUTOR__CLUSTER=distributed
```

## RapidsMPF Execution (Experimental)

The RapidsMPF execution model uses asynchronous coroutines instead of task graphs. This execution model is experimental and requires the `rapidsmpf` package to be installed.

```python
import polars as pl

engine = pl.GPUEngine(
    executor="streaming",
    executor_options={
        "engine": "rapidsmpf",
        # Note: Currently single-GPU only
    }
)

lf = pl.scan_parquet("data.parquet")
result = lf.collect(engine=engine)
```

**Environment Variable:**
```bash
export CUDF_POLARS__EXECUTOR__ENGINE=rapidsmpf
```

### Installation

To use the RapidsMPF execution model, install rapidsmpf:

```bash
# For single-GPU execution
pip install rapidsmpf

# For distributed execution (if needed in the future)
pip install "rapidsmpf[dask]"
```

### Key Differences

| Feature | Task-Based | RapidsMPF |
|---------|-----------|-----------|
| Execution Model | Task graph | Asynchronous coroutines |
| Cluster Support | Single or Distributed | Single (currently) |
| Shuffle Method | Tasks or RapidsMPF | RapidsMPF (required) |
| Multi-GPU | Yes (distributed) | Coming soon |
| Maturity | Stable | Experimental |

## Migration Guide

### From Task-Based to RapidsMPF

**Before:**
```python
engine = pl.GPUEngine(
    executor="streaming",
    executor_options={
        "cluster": "single",
    }
)
```

**After:**
```python
engine = pl.GPUEngine(
    executor="streaming",
    executor_options={
        "engine": "rapidsmpf",
    }
)
```

### Deprecated `scheduler` Parameter

If you're using the old `scheduler` parameter, you'll see a deprecation warning:

**Old (deprecated):**
```python
engine = pl.GPUEngine(
    executor="streaming",
    executor_options={
        "scheduler": "synchronous",  # Deprecated
    }
)
```

**New:**
```python
engine = pl.GPUEngine(
    executor="streaming",
    executor_options={
        "cluster": "single",
    }
)
```

Mapping:
- `scheduler="synchronous"` → `cluster="single"`
- `scheduler="distributed"` → `cluster="distributed"`

## Making RapidsMPF the Default (Future)

To make RapidsMPF the default execution model in the future, change the default in `StreamingExecutor`:

```python
# In config.py, line ~521:
engine: Engine = dataclasses.field(
    default_factory=_make_default_factory(
        f"{_env_prefix}__ENGINE",
        Engine.__call__,
        default=Engine.RAPIDSMPF,  # Changed from Engine.TASKS
    )
)
```

Or set the environment variable globally:

```bash
export CUDF_POLARS__EXECUTOR__ENGINE=rapidsmpf
```

## Performance Tuning

### Task-Based Execution

```python
engine = pl.GPUEngine(
    executor="streaming",
    executor_options={
        "engine": "tasks",
        "cluster": "single",
        "target_partition_size": 100_000_000,  # 100MB partitions
        "shuffle_method": "tasks",
    }
)
```

### RapidsMPF Execution

```python
engine = pl.GPUEngine(
    executor="streaming",
    executor_options={
        "engine": "rapidsmpf",
        "shuffle_method": "rapidsmpf",  # Set automatically
        "rapidsmpf_spill": False,  # Enable spilling if needed
    }
)
```

## Troubleshooting

### "The rapidsmpf streaming engine requires rapidsmpf"

Install the rapidsmpf package:
```bash
pip install rapidsmpf
```

### "The rapidsmpf streaming engine does not support task-based shuffling"

Don't explicitly set `shuffle_method="tasks"` when using `engine="rapidsmpf"`:

```python
# Wrong
engine = pl.GPUEngine(
    executor="streaming",
    executor_options={
        "engine": "rapidsmpf",
        "shuffle_method": "tasks",  # Error!
    }
)

# Correct
engine = pl.GPUEngine(
    executor="streaming",
    executor_options={
        "engine": "rapidsmpf",
        # shuffle_method is set to "rapidsmpf" automatically
    }
)
```

## API Reference

See `cudf_polars.utils.config` for detailed API documentation:

- `Engine` enum: Execution model selection
- `Scheduler` enum: Scheduler selection (task-based only)
- `StreamingExecutor`: Configuration dataclass
