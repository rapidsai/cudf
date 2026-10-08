(cudf-polars-profiling)=
# Profiling and Tracing

## Streaming Statistics

When a query runs on a streaming engine
({class}`~cudf_polars.engine.ray.RayEngine`,
{class}`~cudf_polars.engine.dask.DaskEngine`,
{class}`~cudf_polars.engine.spmd.SPMDEngine`, or the default
`engine="gpu"`), the underlying streaming runtime can record detailed per-rank statistics:
shuffle byte counts, allgather participation, memory-pool high-water marks, and more. See the
[underlying statistics reference][rapidsmpf-stats] for the full list of metrics.

Statistics collection is off by default. Enable it by setting `statistics=True` on
{class}`~cudf_polars.engine.options.StreamingOptions` (or exporting
`RAPIDSMPF_STATISTICS=1`), then call `gather_statistics()` on the engine to pull the per-rank
records:

```python
import polars as pl
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.engine.ray import RayEngine

opts = StreamingOptions(statistics=True)

with RayEngine.from_options(opts) as engine:
    result = (
        pl.scan_parquet("/data/*.parquet")
          .group_by("customer_id")
          .agg(pl.col("amount").sum())
          .collect(engine=engine)
    )

    per_rank = engine.gather_statistics(clear=True)
    for rank, stats in enumerate(per_rank):
        print(f"rank {rank}:\n{stats}")
```

`gather_statistics(*, clear=False)` returns a list of `rapidsmpf.statistics.Statistics` objects,
one per rank, in rank order. Passing `clear=True` resets each rank's counters after the gather —
useful when you want to scope statistics to a single query.

Use `global_statistics(*, clear=False)` when you only need the cluster-wide picture. It gathers
and merges the per-rank statistics into a single `Statistics` (counts and values summed, maxima
reduced with `max`). Capture it inside the engine context, then print after exit:

```python
import polars as pl
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.engine.ray import RayEngine

opts = StreamingOptions(statistics=True)

with RayEngine.from_options(opts) as engine:
    result = pl.scan_parquet("/data/*.parquet").collect(engine=engine)
    total = engine.global_statistics(clear=True)
print(total)
```


## I/O Statistics

`kvikio_statistics=True` turns on [KvikIO I/O statistics][kvikio-stats] on every rank, which
report what storage did. It is separate from `statistics`, so you can collect either on its own.
`gather_io_summary()` returns one `kvikio.Summary` per rank, keyed by rank index:

```python
import polars as pl
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.engine.ray import RayEngine

opts = StreamingOptions(kvikio_statistics=True)

with RayEngine.from_options(opts) as engine:
    pl.scan_parquet("/data/*.parquet").collect(engine=engine)

    for rank, summary in engine.gather_io_summary().items():
        print(f"--- rank {rank} ---")
        print(summary)
```

`clear=True` restarts each rank's measured span after reading, scoping the next gather to
whatever follows. A rank that is not counting is absent, so the result is empty unless
`kvikio_statistics=True` is set. That is distinct from a zeroed summary, which means the rank
was counting and did no I/O.

Printing a summary gives KvikIO's own report:

```text
KvikIO I/O summary
  wall time            122.55 ms
  busy time            18.40 ms (15.02 % of the wall time)
  busy bandwidth       66.44 MB/s
  operations           12 (12 read, 0 write)
  mean duration        3.90 ms
  bytes                1.17 MiB of 1.17 MiB requested (1.17 MiB read, 0 B written)
  errors               0
  backend POSIX        1.17 MiB in 12 ops, 46.83 ms, 26.11 MB/s
  backend GDS          unused
  backend MMAP         unused
  backend REMOTE_HTTP  unused
  backend REMOTE_HDFS  unused
```

Every row is also an attribute, `s.bytes_read`, `s.busy_ns` and so on. See the
[KvikIO statistics reference][kvikio-stats] for the full set, and [busy time and
bandwidth][kvikio-busy] for how the busy figures are measured.

### What is and is not counted

Counting happens per process, so what a summary covers depends on what else shares that
process. With {class}`~cudf_polars.engine.ray.RayEngine` and
{class}`~cudf_polars.engine.dask.DaskEngine` each rank has a process to itself, so a summary
covers only cudf-polars[^shared-worker]. With {class}`~cudf_polars.engine.spmd.SPMDEngine`
cudf-polars shares your script's process, so KvikIO operations your own code performs are
counted too.

Some I/O never reaches the monitor. On a system with working GDS the cuFile asynchronous API
reports nothing, the batch API reports nothing, and anything cudf-polars reads outside KvikIO is
invisible.

[^shared-worker]: A Dask worker can host more than one rank if you run several engines, or other
    Dask work, against one cluster. Neither is a recommended setup, and the summaries would be
    mixed together.


## GPU Profiling

For streaming queries, we recommend profiling with [NVIDIA NSight Systems][nsight]. `cudf-polars`
includes [nvtx][nvtx] annotations to help you understand where time is being spent. Streaming
engines do not support `LazyFrame.profile`, since `profile` requires a single in-memory pass.

If you specifically need [`LazyFrame.profile`](https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.profile.html),
the in-memory engine supports it. This is useful for small queries during development:

```python
import polars as pl
q = pl.scan_parquet("ny-taxi/2024/*.parquet").filter(pl.col("total_amount") > 15.0)
profile = q.profile(engine=pl.GPUEngine(executor="in-memory"))
```

The result is `(result_df, timings_df)`, see the Polars docs link above for the schema.

## Tracing

Streaming node evaluation is exported through Quent. Enable a Quent context to
record each physical node evaluation as an `Evaluate` lifecycle with timestamps,
input and output byte counts, processor and I/O-channel usage, failures, and
incoming chunk metadata.

By default, `Evaluate` events also include the shape and byte size of every
input dataframe and the output dataframe. Set
`CUDF_POLARS_LOG_TRACES_DATAFRAMES=0` before importing `cudf_polars` to omit
these dataframe details. Aggregate input and output byte counts are always
recorded.

The legacy structlog tracing path now records only asynchronous I/O tasks. Set
`CUDF_POLARS_LOG_TRACES=1` before starting the process to emit records with
`scope="io_task"`. These records include their query and actor identifiers,
sequence number, timing, reservation and estimated-output byte counts, and IR
type.

You can configure this output using [structlog]'s
[configuration][structlog-configure] and enrich the records with
[context variables][structlog-context].

[nsight]: https://developer.nvidia.com/nsight-systems
[nvtx]: https://nvidia.github.io/NVTX/
[kvikio-stats]: inv:kvikio:std:doc:#statistics
[kvikio-busy]: <inv:kvikio:std:label:#statistics:busy time and bandwidth>
[rapidsmpf-stats]: inv:rapidsmpf:std:doc:#statistics
[structlog]: https://www.structlog.org/en/stable/
[structlog-configure]: https://www.structlog.org/en/stable/configuration.html
[structlog-context]: https://www.structlog.org/en/stable/contextvars.html
