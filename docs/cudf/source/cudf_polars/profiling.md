(cudf-polars-profiling)=
# Profiling and Tracing

## RapidsMPF Statistics

When a query runs on a streaming engine
({class}`~cudf_polars.experimental.rapidsmpf.frontend.ray.RayEngine`,
{class}`~cudf_polars.experimental.rapidsmpf.frontend.dask.DaskEngine`, or
{class}`~cudf_polars.experimental.rapidsmpf.frontend.spmd.SPMDEngine`), the underlying RapidsMPF
runtime can record detailed per-rank statistics — shuffle byte counts, allgather participation,
memory-pool high-water marks, and more. See the [RapidsMPF statistics reference][rapidsmpf-stats] for the full list of metrics.

Statistics collection is off by default. Enable it by setting `statistics=True` on
{class}`~cudf_polars.experimental.rapidsmpf.frontend.options.StreamingOptions` (or exporting
`RAPIDSMPF_STATISTICS=1`), then call `gather_statistics()` on the engine to pull the per-rank
records:

```python
import polars as pl
from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions
from cudf_polars.experimental.rapidsmpf.frontend.ray import RayEngine

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
reduced with `max`, formatters taken from rank 0):

```python
total = engine.global_statistics(clear=True)
print(total)
```


## GPU Profiling

The streaming engines do not support profiling query execution through the `LazyFrame.profile`
method. We recommend profiling streaming queries with [NVIDIA NSight Systems][nsight];
`cudf-polars` includes [nvtx][nvtx] annotations to help you understand where time is being spent.

The in-memory engine *does* support [`LazyFrame.profile`](https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.profile.html):

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

cudf-polars can optionally trace execution of each node in the query plan. To enable tracing, set
the environment variable ``CUDF_POLARS_LOG_TRACES`` to a true value ("1", "true", "y", "yes")
before starting your process.

cudf-polars logs traces at three scopes (levels):

1. `plan`: These generally happen once per query. This will include things like the (serialized)
   query plan.
2. `actor`: (rapidsmpf runtime only). There will be roughly one `actor` trace per node in the
   logical plan.
3. `evaluate_ir_node`: Logs the evaluation of a physical node in the query plan. Note that one
   logical node might expand to more than one physical nodes.

Each trace includes a `scope` key indicating which level that trace belongs to. `actor`-scoped
nodes will be nested under a `plan`-scoped node. When using the rapidsmpf runtime,
`evaluate_ir_node`-scoped nodes will be nested under an `actor`-scoped node.

### Schemas

The different scopes have different schemas. Fields in **bold** are required / always present.

#### scope=plan

| Field Name | Type  | Description |
| ---------- | ----- | ----------- |
| **scope**  | Literal["plan"] | The string literal `"plan"`. Useful for distinguishing from other types of traces. |
| **cudf_polars_query_id** | UUID4 | A unique identifier for the polars query being executed. All traces logged as part of this query use this ID. |
| **plan**   | `PlanObject` | A serialized representation of the query plan. |
| **event**  | String | A message like "Query Plan" |

#### scope=actor

`actor`-scoped traces will only appear with the rapidsmpf runtime.

| Field Name | Type  | Description |
| ---------- | ----- | ----------- |
| **scope**      | Literal["actor"] | The string literal `"actor"`. Useful for distinguishing from other types of traces. |
| **cudf_polars_query_id** | UUID4 | A unique identifier for the polars query being executed. All traces logged as part of this query use this ID. |
| **start**      | int   | A nanosecond-resolution counter indicating when the actor started. Note: actors generally start early in the query and suspend waiting for data. |
| **stop**      | int   | A nanosecond-resolution counter indicating when the actor completed. |
| **event**      | String | A message like "Streaming Actor". |
| **actor_ir_type** | String | The type of the actor, like `"Scan"`. |
| **actor_ir_id**   | int    | A unique identifier for the actor. All traces logged under this actor will include this value. |
| chunk_count | int | A counter for how many table chunks have been processed by this actor at the time of logging. |
| duplicated | bool | Whether the output rows are duplicated across ranks (e.g. after an allgather). |
| row_count       | int  | Total row count produced by this node during execution. |

#### scope=evaluate_ir_node

| Field Name | Type  | Description |
| ---------- | ----- | ----------- |
| **scope** | `Literal["evaluate_ir_node"]` | The string literal `"evaluate_ir_node"`. Useful for distinguishing from other types of traces. |
| **cudf_polars_query_id** | UUID4 | A unique identifier for the polars query being executed. All traces logged as part of this query use this ID. |
| **type**       | string | The name of the IR node |
| **start**      | int    | A nanosecond-precision counter indicating when this node started executing |
| **stop**       | int    | A nanosecond-precision counter indicating when this node finished executing |
| **overhead_duration**   | int    | The overhead, in nanoseconds, added by tracing |
| `count_frames_{phase}` | int | The number of dataframes for the input / output `phase`. This metric can be disabled by setting `CUDF_POLARS_LOG_TRACES_DATAFRAMES=0`. |
| `frames_{phase}` | `list[dict]` | A list with dictionaries with "shape" and "size" fields, one per input dataframe, for the input / output `phase`. This metric can be disabled by setting `CUDF_POLARS_LOG_TRACES_DATAFRAMES=0`. |
| `total_bytes_{phase}` | int | The sum of the size (in bytes) of the dataframes for the input / output `phase`. This metric can be disabled by setting `CUDF_POLARS_LOG_TRACES_MEMORY=0`. |
| `rmm_current_bytes_{phase}` | int | The current number of bytes allocated by RMM Memory Resource used by cudf-polars for the input / output `phase`. This metric can be disabled by setting `CUDF_POLARS_LOG_TRACES_MEMORY=0`. |
| `rmm_current_count_{phase}` | int | The current number of allocations made by RMM Memory Resource used by cudf-polars for the input / output `phase`. This metric can be disabled by setting `CUDF_POLARS_LOG_TRACES_MEMORY=0`. |
| `rmm_peak_bytes_{phase}` | int | The peak number of bytes allocated by RMM Memory Resource used by cudf-polars for the input / output `phase`. This metric can be disabled by setting `CUDF_POLARS_LOG_TRACES_MEMORY=0`. |
| `rmm_peak_count_{phase}` | int | The peak number of allocations made by RMM Memory Resource used by cudf-polars for the input / output `phase`. This metric can be disabled by setting `CUDF_POLARS_LOG_TRACES_MEMORY=0`. |
| `rmm_total_bytes_{phase}` | int | The total number of bytes allocated by RMM Memory Resource used by cudf-polars for the input / output `phase`. This metric can be disabled by setting `CUDF_POLARS_LOG_TRACES_MEMORY=0`. |
| `rmm_total_count_{phase}` | int | The total number of allocations made by RMM Memory Resource used by cudf-polars for the input / output `phase`. This metric can be disabled by setting `CUDF_POLARS_LOG_TRACES_MEMORY=0`. |
| `nvml_current_bytes_{phase}` | int | The device memory usage of this process, as reported by NVML, for the input / output `phase`. This metric can be disabled by setting `CUDF_POLARS_LOG_TRACES_MEMORY=0`. |
| actor_ir_id   | int    | A unique identifier for the parent actor (rapidsmpf runtime only). |

Setting `CUDF_POLARS_LOG_TRACES=1` enables all the metrics. Depending on the query, the overhead
from collecting the memory or dataframe metrics can be measurable. You can disable some metrics
through additional environment variables. For example, to disable the memory related metrics, set:

```
CUDF_POLARS_LOG_TRACES=1 CUDF_POLARS_LOG_TRACES_MEMORY=0
```

And to disable the memory and dataframe metrics, which essentially leaves just the duration
metrics, set
```
CUDF_POLARS_LOG_TRACES=1 CUDF_POLARS_LOG_TRACES_MEMORY=0 CUDF_POLARS_LOG_TRACES_DATAFRAMES=0
```

Note that tracing still needs to be enabled with `CUDF_POLARS_LOG_TRACES=1`.

The implementation uses [structlog] to build log records. You can configure the output using
structlog's [configuration][structlog-configure] and enrich the records with
[context variables][structlog-context].

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

[nsight]: https://developer.nvidia.com/nsight-systems
[nvtx]: https://nvidia.github.io/NVTX/
[rapidsmpf-stats]: https://docs.rapids.ai/api/rapidsmpf/nightly/statistics/
[structlog]: https://www.structlog.org/
[structlog-configure]: https://www.structlog.org/en/stable/configuration.html
[structlog-context]: https://www.structlog.org/en/stable/contextvars.html
