# Multi-GPU with Dask-cuDF

cuDF is a single-GPU library. For Multi-GPU cuDF solutions we use
[Dask](https://dask.org/) and the [dask-cudf
package](https://github.com/rapidsai/cudf/tree/main/python/dask_cudf),
which is able to scale cuDF across multiple GPUs on a single machine,
or multiple GPUs across many machines in a cluster.

[Dask DataFrame](http://docs.dask.org/en/latest/dataframe.html) was
originally designed to scale Pandas, orchestrating many Pandas
DataFrames spread across many CPUs into a cohesive parallel DataFrame.
Because cuDF currently implements only a subset of the Pandas API, not
all Dask DataFrame operations work with cuDF.

The following is tested and expected to work:

## What works

- Data ingestion

  - `dask_cudf.read_csv`
  - Use standard Dask ingestion with Pandas, then convert to cuDF (For
    Parquet and other formats this is often decently fast)

- Linear operations

  - Element-wise operations: `df.x + df.y`, `df ** 2`
  - Assignment: `df['z'] = df.x + df.y`
  - Row-wise selections: `df[df.x > 0]`
  - Loc: `df.loc['2001-01-01': '2005-02-02']`
  - Date time/string accessors: `df.timestamp.dt.dayofweek`
  - ... and most similar operations in this category that are already
    implemented in cuDF

- Reductions

  - Like `sum`, `mean`, `max`, `count`, and so on on
    `Series` objects
  - Support for reductions on full dataframes
  - `std`
  - Custom reductions with
    [dask.dataframe.reduction](https://docs.dask.org/en/latest/generated/dask.dataframe.Series.reduction.html)

- Groupby aggregations

  - On single columns: `df.groupby('x').y.max()`
  - With custom aggregations:
  - groupby standard deviation
  - grouping on multiple columns
  - groupby agg for multiple outputs

- Joins:

  - On full unsorted columns: `left.merge(right, on='id')`
    (expensive)
  - On sorted indexes:
    `left.merge(right, left_index=True, right_index=True)` (fast)
  - On large and small dataframes: `left.merge(cudf_df, on='id')`
    (fast)

- Rolling operations

- Converting to and from other forms

  - Dask + Pandas to Dask + cuDF
    `df.map_partitions(cudf.DataFrame.from_pandas)`
  - Dask + cuDF to Dask + Pandas
    `df.map_partitions(lambda df: df.to_pandas())`
  - cuDF to Dask + cuDF:
    `dask.dataframe.from_pandas(df, npartitions=20)`
  - Dask + cuDF to cuDF: `df.compute()`

Additionally all generic Dask operations, like `compute`, `persist`,
`visualize` and so on work regardless.

## Developing the API

Above we mention the following:

> and most similar operations in this category that are already
> implemented in cuDF

This is because it is difficult to create a comprehensive list of
operations in the cuDF and Pandas libraries. The API is large enough to
be difficult to track effectively. For any operation that operates
row-wise like `fillna` or `query` things will likely, but not
certainly work. If operations don't work it is often due to a slight
inconsistency between Pandas and cuDF that is generally easy to fix. We
encourage users to look at the [cuDF issue
tracker](https://github.com/rapidsai/cudf/issues) to see if their
issue has already been reported and, if not, [raise a new
issue](https://github.com/rapidsai/cudf/issues/new).

## Navigating the API

This project reuses the [Dask
DataFrame](https://docs.dask.org/en/latest/dataframe.html) project,
which was originally designed for Pandas, with the newer library cuDF.
Because we use the same Dask classes for both projects there are often
methods that are implemented for Pandas, but not yet for cuDF. As a
result users looking at the full Dask DataFrame API can be misleading,
and often lead to frustration when operations that are advertised in the
Dask API do not work as expected with cuDF. We apologize for this in
advance.
