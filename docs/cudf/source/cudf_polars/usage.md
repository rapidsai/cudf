(cudf-polars-usage)=
# Usage

`cudf-polars` enables GPU acceleration for Polars' `LazyFrame` API by executing logical plans
with cuDF. It requires only minimal code changes: create a GPU engine, then pass that engine to
each `collect()` call.

```python
import polars as pl
from cudf_polars.experimental.rapidsmpf.frontend.ray import RayEngine

with RayEngine() as engine:
    result = (
        pl.scan_parquet("/data/dataset/*.parquet")
        .filter(pl.col("amount") > 100)
        .group_by("customer_id")
        .agg(pl.col("amount").sum())
        .collect(engine=engine)
    )
    print(result)
```
In the example above, we use {class}`~cudf_polars.experimental.rapidsmpf.frontend.ray.RayEngine`,
which is the preferred engine for GPU acceleration. Other engines are also available, including
{doc}`DaskEngine <dask_engine>` and {doc}`SPMDEngine <spmd_engine>`. All of these engines provide
multi-GPU and multi-node execution. They partition inputs and stream data through the query
graph, allowing execution to scale beyond device memory and across multiple GPUs out of the box.

For the most lightweight streaming setup, a single GPU with no Ray, Dask, or `rrun` launcher,
see the [Single-GPU setup](spmd_engine.md#single-gpu-setup).

As a separate alternative, `cudf-polars` also provides an *in-memory* engine. This engine uses a
simpler execution model: each Polars operation is translated into a corresponding GPU operation
and executed on a single GPU. See [Polars GPU Support](https://docs.pola.rs/user-guide/gpu-support/).


## Configuring `RayEngine`

{class}`~cudf_polars.experimental.rapidsmpf.frontend.ray.RayEngine` with no arguments uses
built-in defaults: it calls `ray.init()` to start a local [Ray][ray-docs] cluster and creates one
GPU worker per visible GPU.

For custom configuration, build a
{class}`~cudf_polars.experimental.rapidsmpf.frontend.options.StreamingOptions` and use
`RayEngine.from_options()`:

```python
import polars as pl
from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions
from cudf_polars.experimental.rapidsmpf.frontend.ray import RayEngine

opts = StreamingOptions(num_streaming_threads=8, fallback_mode="silent")

with RayEngine.from_options(opts) as engine:
    result = pl.scan_parquet("/data/dataset/*.parquet").collect(engine=engine)
```

See {doc}`options` for the available fields.

```{note}
{class}`~cudf_polars.experimental.rapidsmpf.frontend.ray.RayEngine` is **instance-based**: you
must create one and hand it to `.collect(engine=engine)`. Prefer the context-manager form so the
Ray cluster and GPU workers are torn down automatically.
```

## Attaching to an existing Ray cluster

For multi-machine runs, start a Ray cluster separately (for example with `ray start` on each
node) and attach to it from your driver script. When Ray is already initialized,
{class}`~cudf_polars.experimental.rapidsmpf.frontend.ray.RayEngine` connects to the running
cluster and leaves it untouched on exit:

```python
import ray
import polars as pl
from cudf_polars.experimental.rapidsmpf.frontend.ray import RayEngine

ray.init(address="auto")  # attach to a running cluster
with RayEngine() as engine:
    result = (
        pl.scan_parquet("s3://bucket/*.parquet")
            .group_by("customer_id")
            .agg(pl.col("amount").sum())
            .collect(engine=engine)
    )
```

{class}`~cudf_polars.experimental.rapidsmpf.frontend.ray.RayEngine` creates one rank per GPU in
the Ray cluster and bootstraps a UCXX communicator across them. It raises `RuntimeError` if
created inside an `rrun` cluster or if no GPUs are available.

## Using `RayEngine` in a Jupyter notebook

In a notebook, the context-manager form is inconvenient because a `with` block cannot span
multiple cells — the engine would be torn down at the end of the cell that created it. Instead,
construct {class}`~cudf_polars.experimental.rapidsmpf.frontend.ray.RayEngine` once and reuse it
across cells, then call `engine.shutdown()` when you are done:

```python
# Cell 1: start the engine
from cudf_polars.experimental.rapidsmpf.frontend.ray import RayEngine

engine = RayEngine()
```

```python
# Cell 2: run a query
import polars as pl

result = (
    pl.scan_parquet("/data/*.parquet")
      .group_by("customer_id")
      .agg(pl.col("amount").sum())
      .collect(engine=engine)
)
result
```

```python
# Cell 3: run another query reusing the same engine
other = pl.scan_parquet("/data/other/*.parquet").collect(engine=engine)
```

```python
# Final cell: tear everything down
engine.shutdown()
```

`engine.shutdown()` stops the rank actors and, if the engine started Ray itself, also calls
`ray.shutdown()`. It is idempotent, so calling it twice is safe.

## Cluster diagnostics

{meth}`~cudf_polars.experimental.rapidsmpf.frontend.ray.RayEngine.gather_cluster_info` returns
placement information for every rank actor:

```python
with RayEngine() as engine:
    print(f"cluster has {engine.nranks} ranks")
    for i, info in enumerate(engine.gather_cluster_info()):
        print(
            f"rank {i}: hostname={info['hostname']}, pid={info['pid']}, "
            f"CUDA_VISIBLE_DEVICES={info['cuda_visible_devices']}"
        )
```

[ray-docs]: https://docs.ray.io/
