(cudf-polars-options)=
# Configuration Options

{class}`~cudf_polars.experimental.rapidsmpf.frontend.options.StreamingOptions` is the recommended
way to configure the streaming multi-GPU engines
({class}`~cudf_polars.experimental.rapidsmpf.frontend.ray.RayEngine`,
{class}`~cudf_polars.experimental.rapidsmpf.frontend.dask.DaskEngine`,
{class}`~cudf_polars.experimental.rapidsmpf.frontend.spmd.SPMDEngine`).
Build one and pass it to `Engine.from_options()` to construct a
{class}`~cudf_polars.experimental.rapidsmpf.frontend.ray.RayEngine`:

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
| `rapidsmpf` | RapidsMPF runtime, e.g. threads, CUDA streams, spilling, pinned memory, log level      | `RAPIDSMPF_*`              |
| `executor`  | Query execution and partitioning, e.g. `max_rows_per_partition`, `fallback_mode`, ...  | `CUDF_POLARS__EXECUTOR__*` |
| `engine`    | `pl.GPUEngine` kwargs, e.g. Parquet, memory resource, CUDA streams, hardware binding   | `CUDF_POLARS__*`           |

The `engine` category surfaces the same knobs as plain `pl.GPUEngine(...)` — for example,
`parquet_options` and `memory_resource_config`. You do not need to set them separately when using
{class}`~cudf_polars.experimental.rapidsmpf.frontend.options.StreamingOptions`.

The `rapidsmpf` category adds streaming-runtime knobs that have no equivalent on the plain
`pl.GPUEngine`. See the [RapidsMPF configuration reference][rapidsmpf-config] for the underlying
meaning of each `RAPIDSMPF_*` field.

Every option has a corresponding environment variable. When an option is not set explicitly, its
value is read from the environment variable if present; otherwise the underlying library applies
its built-in default. Boolean environment variables accept `{"1", "true", "yes", "y"}` as true
and `{"0", "false", "no", "n"}` as false.

All fields default to an `UNSPECIFIED` sentinel with the same precedence as the low-level
options: explicit value → environment variable → built-in default. Calling
`Engine.from_options(opts)` splits the fields into three dictionaries
(`to_rapidsmpf_options()`, `to_executor_options()`, `to_engine_options()`) and routes each to the
right destination.

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

## When to use StreamingOptions vs. raw keyword arguments

* Use {class}`~cudf_polars.experimental.rapidsmpf.frontend.options.StreamingOptions` +
  `Engine.from_options(opts)` for typical streaming workflows — it is typed, centralized, and
  handles routing automatically.
* Use the raw `Engine(rapidsmpf_options=..., executor_options=..., engine_options=...)`
  constructor only when you need fine-grained control that doesn't fit the
  {class}`~cudf_polars.experimental.rapidsmpf.frontend.options.StreamingOptions` schema.
* For the in-memory engine,
  {class}`~cudf_polars.experimental.rapidsmpf.frontend.options.StreamingOptions` does not apply —
  pass keyword arguments to `pl.GPUEngine(...)` directly (see below).

## GPUEngine options

The in-memory GPU engine is configured by passing keyword arguments to
[`polars.GPUEngine`][polars-gpuengine]:

```python
import polars as pl

engine = pl.GPUEngine(parquet_options={"chunked": True})
```

See the [Polars GPU support guide][polars-gpu] for the full in-memory usage story.

<!-- Reference links -->
[polars-gpu]: https://docs.pola.rs/user-guide/gpu-support/
[polars-gpuengine]: https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.lazyframe.engine_config.GPUEngine.html
[rapidsmpf-config]: https://docs.rapids.ai/api/rapidsmpf/nightly/configuration/
