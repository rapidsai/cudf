# Usage

`cudf-polars` enables GPU acceleration for Polars' LazyFrame API by executing logical plans with cuDF and pylibcudf. It requires minimal code changes and works by specifying a GPU execution engine during collection or profiling.

## Getting Started

Use `cudf-polars` by calling `.collect(engine="gpu")` or `.profile(engine="gpu")` on a LazyFrame:

```python
import polars as pl

q = pl.scan_parquet("ny-taxi/2024/*.parquet").filter(pl.col("total_amount") > 15.0)
result = q.collect(engine="gpu")
```

Alternatively, you can create a GPUEngine instance with custom configuration:

```python
import polars as pl

engine = pl.GPUEngine(raise_on_fail=True)

q = pl.scan_parquet("ny-taxi/2024/*.parquet").filter(pl.col("total_amount") > 15.0)
result = q.collect(engine=engine)
```

With `raise_on_fail=True`, the query will raise an exception if it cannot be run on the GPU instead of transparently falling back to polars CPU. See more [engine options](engine_options.md).

## GPU Profiling

You can profile your code by passing `engine="gpu"` or `engine=pl.GPUEngine(...)`

```python
import polars as pl

q = pl.scan_parquet("ny-taxi/2024/*.parquet").filter(pl.col("total_amount") > 15.0)
profile = q.profile(engine="gpu")
```

The result is a tuple containing the materialized DataFrame and a DataFrame that contains profiling information of each node that is executed.
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
