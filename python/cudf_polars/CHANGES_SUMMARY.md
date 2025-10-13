# Summary of Execution Model Configuration Updates

## Changes Made to `config.py`

### 1. Enhanced Documentation

#### `Engine` Enum (lines 141-156)
- **Updated**: Clarified that this controls the "execution model" not just "engine"
- **Added**: Detailed explanation of task-based vs. coroutine-based execution
- Made it clear that:
  - `Engine.TASKS` uses task graphs where `cluster` determines execution (single/distributed)
  - `Engine.RAPIDSMPF` uses asynchronous coroutine networks

#### NEW: `Cluster` Enum (lines 159-174)
- **Added**: New enum to specify single-GPU or distributed execution
- Applies to BOTH task-based and rapidsmpf execution models
- **Values**:
  - `Cluster.SINGLE` for single-GPU execution
  - `Cluster.DISTRIBUTED` for multi-GPU distributed execution

#### `Scheduler` Enum (lines 176-188)
- **Deprecated**: Entire enum in favor of `Cluster`
- **Backward compatibility**: Still exists but emits deprecation warning when used
- **Mapping**:
  - `Scheduler.SYNCHRONOUS` → `Cluster.SINGLE`
  - `Scheduler.DISTRIBUTED` → `Cluster.DISTRIBUTED`

#### `StreamingExecutor` Docstring (lines 437-463)
- **Added**: New `cluster` parameter documentation
- **Enhanced**: `engine` parameter documentation with detailed explanation of both models
- **Deprecated**: `scheduler` parameter documentation
- **Updated**: All references from "scheduler" to "cluster" throughout docstring

### 2. New `cluster` Field and Backward Compatibility (lines 544-627)

Added `cluster` field to `StreamingExecutor`:
```python
cluster: Cluster = dataclasses.field(
    default_factory=_make_default_factory(
        f"{_env_prefix}__CLUSTER",
        Cluster.__call__,
        default=Cluster.SINGLE,
    )
)
```

Deprecated `scheduler` field (now `Scheduler | None`):
- When user provides `scheduler`, a deprecation warning is emitted
- Values are automatically mapped to `cluster`:
  - `"synchronous"` → `Cluster.SINGLE`
  - `"distributed"` → `Cluster.DISTRIBUTED`
- All internal logic now uses `self.cluster` instead of `self.scheduler`

### 3. Public API Export (lines 41-52)

Added `Cluster` and `Engine` to `__all__`:
```python
from cudf_polars.utils.config import Cluster, Engine
```

Kept `Scheduler` for backward compatibility (deprecated).

## Supporting Documentation Created

### 1. `EXECUTION_MODELS.md`
Comprehensive guide covering:
- Overview of both execution models
- Usage examples for each model
- Environment variable configuration
- Migration guide
- Performance tuning
- Troubleshooting
- API reference

### 2. `examples/execution_model_selection.py`
Practical Python script demonstrating:
- How to check if RapidsMPF is available
- How to select execution models programmatically
- Fallback logic when RapidsMPF is not installed
- Environment variable usage

## Current State

### Task-Based Execution (Default)
```python
# Single-GPU
engine = pl.GPUEngine(
    executor="streaming",
    executor_options={
        "engine": "tasks",  # Default
        "cluster": "single",  # Default
    }
)

# Multi-GPU
engine = pl.GPUEngine(
    executor="streaming",
    executor_options={
        "engine": "tasks",
        "cluster": "distributed",
    }
)
```

### RapidsMPF Execution (Experimental)
```python
engine = pl.GPUEngine(
    executor="streaming",
    executor_options={
        "engine": "rapidsmpf",
    }
)
```

## How Users Choose Between Models

### Method 1: Keyword Arguments (Recommended)
```python
import polars as pl
from cudf_polars.utils.config import Engine

engine = pl.GPUEngine(
    executor="streaming",
    executor_options={
        "engine": Engine.RAPIDSMPF,  # or "rapidsmpf"
    }
)
```

### Method 2: Environment Variables
```bash
export CUDF_POLARS__EXECUTOR__ENGINE=rapidsmpf
# or
export CUDF_POLARS__EXECUTOR__ENGINE=tasks
export CUDF_POLARS__EXECUTOR__CLUSTER=single
```

### Method 3: Programmatic with Fallback
```python
from cudf_polars.utils.config import rapidsmpf_single_available

if rapidsmpf_single_available():
    engine_config = {"engine": "rapidsmpf"}
else:
    engine_config = {"engine": "tasks", "cluster": "single"}

engine = pl.GPUEngine(executor="streaming", executor_options=engine_config)
```

### Backward Compatibility (Deprecated)

Old code using `scheduler` will still work with a deprecation warning:
```python
# This still works but emits a warning
engine = pl.GPUEngine(
    executor="streaming",
    executor_options={
        "scheduler": "synchronous",  # Automatically mapped to cluster="single"
    }
)
```

## Future: Making RapidsMPF the Default

When RapidsMPF is ready to become the default, make this change:

### Option 1: Change Code Default
In `config.py` around line 521:
```python
engine: Engine = dataclasses.field(
    default_factory=_make_default_factory(
        f"{_env_prefix}__ENGINE",
        Engine.__call__,
        default=Engine.RAPIDSMPF,  # Changed from Engine.TASKS
    )
)
```

### Option 2: Environment Variable
Set globally for all users:
```bash
export CUDF_POLARS__EXECUTOR__ENGINE=rapidsmpf
```

### Option 3: Gradual Rollout
1. Add a deprecation warning when `engine` is not explicitly set
2. Announce the upcoming default change
3. After a release cycle or two, change the default
4. Users who want to keep task-based execution can explicitly set `engine="tasks"`

## Validation Checklist

- [x] `Engine` enum clearly distinguishes execution models
- [x] NEW `Cluster` enum for single vs distributed execution
- [x] `Scheduler` enum deprecated with backward compatibility
- [x] `Cluster` field added to `StreamingExecutor`
- [x] Backward compatibility for `scheduler` parameter with deprecation warning
- [x] All internal logic updated to use `cluster` instead of `scheduler`
- [x] `Cluster` and `Engine` added to public API (`__all__`)
- [x] Clear documentation in docstrings
- [x] User-facing documentation created and updated
- [x] Example code provided and updated
- [x] Environment variable configuration documented (`CLUSTER` and legacy `SCHEDULER`)
- [x] Path to making RapidsMPF default is clear

## Testing Recommendations

### Unit Tests to Add
1. Test that `engine="rapidsmpf"` works correctly
2. Test that `engine="rapidsmpf"` + `shuffle_method="tasks"` raises error
3. Test that `engine="rapidsmpf"` requires rapidsmpf to be installed
4. Test environment variable `CUDF_POLARS__EXECUTOR__ENGINE=rapidsmpf`
5. Test new `cluster` parameter: `cluster="single"` and `cluster="distributed"`
6. Test environment variable `CUDF_POLARS__EXECUTOR__CLUSTER=single|distributed`
7. Test backward compatibility: `scheduler="synchronous"` maps to `cluster="single"`
8. Test backward compatibility: `scheduler="distributed"` maps to `cluster="distributed"`
9. Test deprecation warning is emitted when using `scheduler` parameter
10. Test that `cluster` works with both `engine="tasks"` and `engine="rapidsmpf"`

### Integration Tests to Add
1. Run actual queries with both `engine="tasks"` and `engine="rapidsmpf"`
2. Verify results are identical between execution models
3. Test fallback behavior when rapidsmpf is not available
4. Test multi-GPU scenarios (when supported)

## Next Steps

1. **Review**: Have the team review these changes
2. **Test**: Add unit and integration tests as outlined above
3. **Document**: Update main README or user guide to reference `EXECUTION_MODELS.md`
4. **Announce**: When RapidsMPF is stable, announce it in release notes
5. **Migrate**: Provide migration guide for users to adopt RapidsMPF
6. **Default**: After sufficient testing and user feedback, make RapidsMPF the default

## Questions to Consider

1. **Multi-GPU Support**: Will RapidsMPF support multi-GPU in the future?
   - If yes, will it use its own scheduler or integrate with Dask?
   - Update documentation accordingly when decided

2. **Performance**: What are the performance characteristics?
   - Add benchmarking section to documentation
   - Provide guidance on when to use each model

3. **Feature Parity**: Are there any operations that work in one model but not the other?
   - Document any limitations clearly
   - Provide feature comparison matrix

4. **Spilling**: Does `rapidsmpf_spill` work with single-GPU RapidsMPF?
   - Currently seems limited to distributed scheduler
   - Clarify or extend support

5. **Configuration Validation**: Should certain combinations be invalid?
   - Currently: `engine="rapidsmpf"` + `shuffle_method="tasks"` → Error ✓
   - Any other invalid combinations?
