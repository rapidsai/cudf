# Usage

`cudf-polars` enables GPU acceleration for Polars' LazyFrame API by executing logical plans with cuDF and pylibcudf. It requires minimal code changes and works by specifying a GPU engine during execution.

For a high-level overview of GPU support in Polars, see the [Polars GPU support guide](https://docs.pola.rs/user-guide/gpu-support/).

## Getting Started

Use `cudf-polars` by calling `.collect(engine="gpu")` or `.sink_<method>(engine="gpu")` on a LazyFrame:

```python
import polars as pl

q = pl.scan_parquet("ny-taxi/2024/*.parquet").filter(pl.col("total_amount") > 15.0)
result = q.collect(engine="gpu")
```

Alternatively, you can create a `GPUEngine` instance with custom configuration:

```python
import polars as pl

engine = pl.GPUEngine(raise_on_fail=True)

q = pl.scan_parquet("ny-taxi/2024/*.parquet").filter(pl.col("total_amount") > 15.0)
result = q.collect(engine=engine)
```

With `raise_on_fail=True`, the query will raise an exception if it cannot be run on the GPU instead of transparently falling back to polars CPU. See more [engine options](engine_options.md).

## GPU Profiling

The `streaming` executor does not support profiling query execution through the `LazyFrame.profile` method. With the default `synchronous` scheduler for the `streaming` executor, we recommend using [NVIDIA NSight Systems](https://developer.nvidia.com/nsight-systems) to profile your queries.
cudf-polars includes [nvtx](https://nvidia.github.io/NVTX/) annotations to help you understand where time is being spent.

With the `distributed` scheduler for the `streaming` executor, we recommend using Dask's [built-in diagnostics](https://docs.dask.org/en/stable/diagnostics-distributed.html).

Finally, the `"in-memory"` *does* support [`LazyFrame.profile`](https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.profile.html).

```python
import polars as pl
q = pl.scan_parquet("ny-taxi/2024/*.parquet").filter(pl.col("total_amount") > 15.0)
profile = q.profile(engine=pl.GPUEngine(executor="in-memory"))
```

The result is a tuple containing 2 materialized DataFrames - the first with the query result and the second with profiling information of each node that is executed.
```python
print(profile[0])
```
```
shape: (32_439_327, 19)
┌──────────┬──────────────────────┬───────────────────────┬─────────────────┬───┬───────────────────────┬──────────────┬──────────────────────┬─────────────┐
│ VendorID ┆ tpep_pickup_datetime ┆ tpep_dropoff_datetime ┆ passenger_count ┆ … ┆ improvement_surcharge ┆ total_amount ┆ congestion_surcharge ┆ Airport_fee │
│ ---      ┆ ---                  ┆ ---                   ┆ ---             ┆   ┆ ---                   ┆ ---          ┆ ---                  ┆ ---         │
│ i32      ┆ datetime[μs]         ┆ datetime[μs]          ┆ i64             ┆   ┆ f64                   ┆ f64          ┆ f64                  ┆ f64         │
╞══════════╪══════════════════════╪═══════════════════════╪═════════════════╪═══╪═══════════════════════╪══════════════╪══════════════════════╪═════════════╡
│ 2        ┆ 2024-01-01 00:57:55  ┆ 2024-01-01 01:17:43   ┆ 1               ┆ … ┆ 1.0                   ┆ 22.7         ┆ 2.5                  ┆ 0.0         │
│ 1        ┆ 2024-01-01 00:03:00  ┆ 2024-01-01 00:09:36   ┆ 1               ┆ … ┆ 1.0                   ┆ 18.75        ┆ 2.5                  ┆ 0.0         │
│ 1        ┆ 2024-01-01 00:17:06  ┆ 2024-01-01 00:35:01   ┆ 1               ┆ … ┆ 1.0                   ┆ 31.3         ┆ 2.5                  ┆ 0.0         │
│ 1        ┆ 2024-01-01 00:36:38  ┆ 2024-01-01 00:44:56   ┆ 1               ┆ … ┆ 1.0                   ┆ 17.0         ┆ 2.5                  ┆ 0.0         │
│ 1        ┆ 2024-01-01 00:46:51  ┆ 2024-01-01 00:52:57   ┆ 1               ┆ … ┆ 1.0                   ┆ 16.1         ┆ 2.5                  ┆ 0.0         │
│ …        ┆ …                    ┆ …                     ┆ …               ┆ … ┆ …                     ┆ …            ┆ …                    ┆ …           │
│ 2        ┆ 2024-12-31 23:05:43  ┆ 2024-12-31 23:18:15   ┆ null            ┆ … ┆ 1.0                   ┆ 24.67        ┆ null                 ┆ null        │
│ 2        ┆ 2024-12-31 23:02:00  ┆ 2024-12-31 23:22:14   ┆ null            ┆ … ┆ 1.0                   ┆ 15.25        ┆ null                 ┆ null        │
│ 2        ┆ 2024-12-31 23:17:15  ┆ 2024-12-31 23:17:34   ┆ null            ┆ … ┆ 1.0                   ┆ 24.46        ┆ null                 ┆ null        │
│ 1        ┆ 2024-12-31 23:14:53  ┆ 2024-12-31 23:35:13   ┆ null            ┆ … ┆ 1.0                   ┆ 32.88        ┆ null                 ┆ null        │
│ 1        ┆ 2024-12-31 23:15:33  ┆ 2024-12-31 23:36:29   ┆ null            ┆ … ┆ 1.0                   ┆ 28.57        ┆ null                 ┆ null        │
└──────────┴──────────────────────┴───────────────────────┴─────────────────┴───┴───────────────────────┴──────────────┴──────────────────────┴─────────────┘
```

```python
print(profile[1])
```
```
shape: (3, 3)
┌────────────────────┬───────┬────────┐
│ node               ┆ start ┆ end    │
│ ---                ┆ ---   ┆ ---    │
│ str                ┆ u64   ┆ u64    │
╞════════════════════╪═══════╪════════╡
│ optimization       ┆ 0     ┆ 416    │
│ gpu-ir-translation ┆ 416   ┆ 741    │
│ Scan               ┆ 813   ┆ 233993 │
└────────────────────┴───────┴────────┘
```

## Tracing

cudf-polars can optionally trace execution of the various nodes of each node in
the query plan by setting the environment variable ``CUDF_POLARS_TRACE`` to a
true value ("1", "true", "y", "yes") before starting your process. This will
capture and log information about each node before and after it executes, including information on

- The type of the node being executed (e.g. `Scan`, `Select`, `Join`, `Groupby`, etc.)
- The shape and size (in memory) of each input and output DataFrame
- The GPU memory usage, as reported by [nvml], before and after executing the ndoe
- A start and stop clock, which can be used to measure the duration of the node's execution

The implementation uses [structlog] to build log records. You can configure the
output using structlog's [configuration][structlog-configure] and enrich the
records with [context variables][structlog-context].


```
>>> df = pl.DataFrame({"a": ["a", "a", "b"], "b": [1, 2, 3]}).lazy()
>>> df.group_by("a").agg(pl.col("b").min().alias("min"), pl.col("b").max().alias("max")).collect(engine="gpu")
2025-09-10 07:44:01 [info     ] Execute IR      count_frames_input=0 count_frames_output=1 ... type=DataFrameScan
2025-09-10 07:44:01 [info     ] Execute IR      count_frames_input=1 count_frames_output=1 ... type=GroupBy
shape: (2, 3)
┌─────┬─────┬─────┐
│ a   ┆ min ┆ max │
│ --- ┆ --- ┆ --- │
│ str ┆ i64 ┆ i64 │
╞═════╪═════╪═════╡
│ b   ┆ 3   ┆ 3   │
│ a   ┆ 1   ┆ 2   │
└─────┴─────┴─────┘
```

[nvml]: https://developer.nvidia.com/management-library-nvml
[rmm-stats]: https://docs.rapids.ai/api/rmm/stable/guide/#memory-statistics-and-profiling
[structlog]: https://www.structlog.org/
[structlog-configure](https://www.structlog.org/en/stable/configuration.html)
[structlog-context](https://www.structlog.org/en/stable/contextvars.html)
