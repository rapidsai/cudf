(cudf-polars-legacy)=
# Legacy APIs

```{note}
The APIs on this page are superseded by the new streaming engines
({class}`~cudf_polars.experimental.rapidsmpf.frontend.ray.RayEngine`,
{class}`~cudf_polars.experimental.rapidsmpf.frontend.dask.DaskEngine`,
{class}`~cudf_polars.experimental.rapidsmpf.frontend.spmd.SPMDEngine`) described in {doc}`usage`.
They are retained here for users on existing deployments. New workflows should use the engines
documented on the main Usage page.
```

## Selecting the streaming executor with `executor="streaming"`

The streaming executor used to be selected explicitly via
`pl.GPUEngine(executor="streaming", executor_options={...})`:

```python
import polars as pl

engine = pl.GPUEngine(
    executor="streaming",
    executor_options={"max_rows_per_partition": 500_000},
)
```

New workflows should drop the `executor=` keyword and use
{class}`~cudf_polars.experimental.rapidsmpf.frontend.options.StreamingOptions` with one of the
streaming engines instead — see {doc}`options` and {doc}`usage`. Every field that used to live in
`executor_options` has an equivalent on
{class}`~cudf_polars.experimental.rapidsmpf.frontend.options.StreamingOptions`.

## Legacy multi-GPU streaming via `cluster="distributed"`

Prior to the introduction of
{class}`~cudf_polars.experimental.rapidsmpf.frontend.dask.DaskEngine`, multi-GPU streaming was
selected by passing `cluster="distributed"` in `executor_options`:

```python
import polars as pl

engine = pl.GPUEngine(
    executor="streaming",
    executor_options={"cluster": "distributed"},
)
```

This path requires a number of additional dependencies — in particular
[Dask](https://www.dask.org/) and [Dask-CUDA](https://docs.rapids.ai/api/dask-cuda/nightly/). We
also recommend the Dask Distributed plugin of [UCXX](https://github.com/rapidsai/ucxx) and
[RapidsMPF](https://github.com/rapidsai/rapidsmpf) for high-performance networking.

To install all of these into a conda environment:

```
conda install -c rapidsai -c conda-forge \
    cudf-polars rapidsmpf dask-cuda distributed-ucxx
```

The multi-GPU engine uses the currently active Dask client to carry out the partitioned
execution, so a typical multi-GPU driver looks like:

```python
from dask_cuda import LocalCUDACluster

client = LocalCUDACluster(...).get_client()

q = ...
engine = pl.GPUEngine(
    executor="streaming",
    executor_options={"cluster": "distributed"},
)
result = q.collect(engine=engine)
```

```{warning}
If you request a `"distributed"` cluster but do not have a cluster deployed, `collect`ing the
query will fail.
```

```{note}
If part of a query does not support execution with multiple partitions, but is supported by the
in-memory GPU engine, cudf-polars concatenates the inputs and executes using a single partition.
```

### Sink behavior

When the `"distributed"` cluster option is active, sink operations like `df.sink_parquet("my_path")`
will always produce a directory containing one or more files. It is not currently possible to
disable this behavior.

When the `"single"` cluster option is active, sink operations will generate a single file by
default. However, you may opt into the distributed sink behavior by adding
`{"sink_to_directory": True}` to your `executor_options` dictionary.
