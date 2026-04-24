(cudf-polars-options)=
# Configuration Options

{class}`~cudf_polars.experimental.rapidsmpf.frontend.options.StreamingOptions` is the recommended
way to configure the streaming multi-GPU engines. Build one and pass it to `RayEngine.from_options()`
to construct a {class}`~cudf_polars.experimental.rapidsmpf.frontend.ray.RayEngine`:

```python
import polars as pl
from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions
from cudf_polars.experimental.rapidsmpf.frontend.ray import RayEngine

opts = StreamingOptions(
    num_streaming_threads=8,
    fallback_mode="silent",
    spill_device_limit="70%",
)

with RayEngine.from_options(opts) as engine:
    result = (
        pl.scan_parquet("/data/*.parquet")
          .filter(pl.col("amount") > 100)
          .group_by("customer_id")
          .agg(pl.col("amount").sum())
          .collect(engine=engine)
    )
```

{class}`~cudf_polars.experimental.rapidsmpf.frontend.options.StreamingOptions` covers three
categories of fields:

| Category    | Scope                                                                                  | Env var prefix             |
| ----------- | -------------------------------------------------------------------------------------- | -------------------------- |
| `rapidsmpf` | RapidsMPF runtime, e.g. threads, CUDA streams, spilling, pinned memory, log level      | `RAPIDSMPF_`              |
| `executor`  | Query execution and partitioning, e.g. `max_rows_per_partition`, `fallback_mode`, ...  | `CUDF_POLARS__EXECUTOR__` |
| `engine`    | `pl.GPUEngine` kwargs, e.g. Parquet, memory resource, CUDA streams, hardware binding   | `CUDF_POLARS__`           |

The `engine` category surfaces the same knobs as plain `pl.GPUEngine(...)` — for example,
`parquet_options` and `memory_resource_config`. Configure these settings through
{class}`~cudf_polars.experimental.rapidsmpf.frontend.options.StreamingOptions` rather than
passing them to `pl.GPUEngine(...)` directly.

The `rapidsmpf` category adds knobs for the streaming runtime that have no equivalent on the plain
`pl.GPUEngine`. See the [RapidsMPF configuration reference][rapidsmpf-config] for the underlying
meaning of each `RAPIDSMPF_*` field.

Every option has a corresponding environment variable. When an option is not set explicitly, its
value is read from the environment variable if present; otherwise the underlying library applies
its built-in default. Boolean environment variables accept `{"1", "true", "yes", "y"}` as true
and `{"0", "false", "no", "n"}` as false.


## Building from a dictionary

{meth}`~cudf_polars.experimental.rapidsmpf.frontend.options.StreamingOptions.from_dict` accepts a
flat dict of field names. Unknown keys raise `TypeError`; `None` values leave the field
unspecified:

```python
opts = StreamingOptions.from_dict({
    "num_streaming_threads": 8,
    "fallback_mode": "silent",
})
```

This is convenient when options come from a config file or CLI.

## Engine keyword arguments

Each engine ({class}`~cudf_polars.experimental.rapidsmpf.frontend.ray.RayEngine`,
{class}`~cudf_polars.experimental.rapidsmpf.frontend.dask.DaskEngine`, or
{class}`~cudf_polars.experimental.rapidsmpf.frontend.spmd.SPMDEngine`) accepts
`rapidsmpf_options`, `executor_options`, and `engine_options` as raw keyword arguments.
We recommend using this only when you need fine-grained control that doesn't fit the
{class}`~cudf_polars.experimental.rapidsmpf.frontend.options.StreamingOptions` schema.
Otherwise, prefer the engine's `from_options` constructor with
{class}`~cudf_polars.experimental.rapidsmpf.frontend.options.StreamingOptions`.

For the in-memory engine,
{class}`~cudf_polars.experimental.rapidsmpf.frontend.options.StreamingOptions` does not apply —
pass keyword arguments to `pl.GPUEngine(...)` directly (see below).

## In-memory Engine Options

The in-memory GPU engine is configured by passing keyword arguments to
[`polars.GPUEngine`][polars-gpuengine]:

```python
import polars as pl

engine = pl.GPUEngine(parquet_options={"chunked": True})
```

See the [Polars GPU support guide][polars-gpu] for the full in-memory usage story.


## Options Reference

Environment variables follow these patterns:

* `rapidsmpf`: `RAPIDSMPF_<OPTION_NAME>` (e.g. `RAPIDSMPF_NUM_STREAMING_THREADS`)
* `executor`: `CUDF_POLARS__EXECUTOR__<OPTION_NAME>` (e.g. `CUDF_POLARS__EXECUTOR__FALLBACK_MODE`)
* `engine`: `CUDF_POLARS__<OPTION_NAME>` (e.g. `CUDF_POLARS__RAISE_ON_FAIL`; nested prefixes for structured options)

### Category: `rapidsmpf`

See the [RapidsMPF configuration reference][rapidsmpf-config] for the full list of fields and defaults.

### Category: `executor`

| Field                    | Description                                                                                                         | Default                      |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| `num_py_executors`       | Number of workers for the internal Python `ThreadPoolExecutor`.                                                     | `8`                          |
| `fallback_mode`          | Fallback behavior (`"warn"`, `"raise"`, `"silent"`).                                                                | `"warn"`                     |
| `max_rows_per_partition` | Maximum number of rows per partition.                                                                               | `1_000_000`                  |
| `broadcast_join_limit`   | Maximum number of partitions for broadcast joins.                                                                   | auto                         |
| `target_partition_size`  | Target I/O partition size in bytes. `0` means auto.                                                                 | auto                         |
| `dynamic_planning`       | Dynamic planning configuration, dict or {class}`~cudf_polars.utils.config.DynamicPlanningOptions`. `None` disables. | enabled                      |
| `unique_fraction`        | Per-column uniqueness estimate (0–1).                                                                               | `{}` (i.e. `1.0` per column) |
| `sink_to_directory`      | Whether `.sink_*()` writes its output as a directory. Currently, only `True` is supported (see [Sink behavior](usage.md#sink-behavior)). | `True`  |

### Category: `engine`

| Field                    | Description                                                                                                                                            | Default                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------- |
| `raise_on_fail`          | Raise an error instead of falling back to CPU execution.                                                                                               | `False`                   |
| `parquet_options`        | Parquet configuration, dict or {class}`~cudf_polars.utils.config.ParquetOptions`.                                                                      | —                         |
| `memory_resource_config` | RMM configuration, dict or {class}`~cudf_polars.utils.config.MemoryResourceConfig`.                                                                    | —                         |
| `cuda_stream_policy`     | CUDA stream policy (`"default"`, `"pool"`, or a configuration dict).                                                                                   | —                         |
| `hardware_binding`       | Hardware binding policy. Pass a {class}`~cudf_polars.experimental.rapidsmpf.frontend.hardware_binding.HardwareBindingPolicy` for fine-grained control. | `HardwareBindingPolicy()` |
| `allow_gpu_sharing`      | When `False` (default), the engine raises if multiple ranks share the same physical GPU.                                                               | `False`                   |

<!-- Reference links -->
[polars-gpu]: https://docs.pola.rs/user-guide/gpu-support/
[polars-gpuengine]: https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.lazyframe.engine_config.GPUEngine.html
[rapidsmpf-config]: https://docs.rapids.ai/api/rapidsmpf/nightly/configuration/
