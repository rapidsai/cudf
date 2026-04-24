(cudf-polars-usage)=
# Usage

`cudf-polars` enables GPU acceleration for Polars' `LazyFrame` API by executing logical plans
with cuDF. It requires only minimal code changes: create a GPU engine, then pass that engine to
each `collect()` call.

```python
import polars as pl
from cudf_polars.experimental.rapidsmpf.frontend.ray import RayEngine

query = (
    pl.scan_parquet("/data/dataset/*.parquet")
    .filter(pl.col("amount") > 100)
    .group_by("customer_id")
    .agg(pl.col("amount").sum())
)

with RayEngine() as engine:
    result = query.collect(engine=engine)
print(result)
```
In the example above, we use {class}`~cudf_polars.experimental.rapidsmpf.frontend.ray.RayEngine`,
which is the preferred engine for GPU acceleration. Other engines are also available, including
{doc}`DaskEngine <dask_engine>` and {doc}`SPMDEngine <spmd_engine>`. All of these engines provide
multi-GPU and multi-node execution. They partition inputs and stream data through the query
graph, allowing execution to scale beyond device memory and across multiple GPUs out of the box.

For the simplest streaming setup, a single GPU with no Ray, Dask, or `rrun` launcher,
see [Single-GPU setup](spmd_engine.md#single-gpu-setup).

As an alternative, `cudf-polars` also provides an *in-memory* engine. This engine uses a
simpler execution model: each Polars operation is translated into a corresponding GPU operation
and executed on a single GPU. The in-memory engine works well for datasets that fit within GPU
memory, but does not scale beyond it. See [Polars GPU Support](https://docs.pola.rs/user-guide/gpu-support/).


## Configuring `RayEngine`

{class}`~cudf_polars.experimental.rapidsmpf.frontend.ray.RayEngine` starts a local [Ray][ray-docs] cluster and creates one GPU worker per visible GPU.

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
`RayEngine` is an object you create and pass to `.collect(engine=engine)`. Prefer the
context-manager form so the Ray cluster and GPU workers are torn down automatically.
```

## Attaching to an existing Ray cluster

For multi-node runs, start a Ray cluster separately (for example with `ray start` on each
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

## Manual Engine Lifetime Control

When you need to control the engine lifetime explicitly. For example, in a Jupyter notebook
where a `with` block cannot span multiple cells, construct `RayEngine` once and reuse it,
then call `engine.shutdown()` when you are done:

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

`engine.shutdown()` stops the GPU worker processes (rank actors) and, if the engine started Ray itself,
also calls `ray.shutdown()`. It is idempotent, so calling it twice is safe.

## Sink behavior

When a streaming engine is used, sink operations such as `df.sink_parquet("my_path")` always produce
a directory containing one or more files. It is not currently possible to disable this behavior, and
setting `sink_to_directory=False` raises a `ValueError`.

The in-memory engine, by contrast, follows standard Polars semantics and writes to a single file at
the specified path.


## Cluster diagnostics

{meth}`~cudf_polars.experimental.rapidsmpf.frontend.ray.RayEngine.gather_cluster_info` returns
a list of {class}`~cudf_polars.experimental.rapidsmpf.frontend.core.ClusterInfo` — one per rank
actor — with fields `hostname`, `pid`, `cuda_visible_devices`, and `gpu_uuid`:

```python
with RayEngine() as engine:
    print(f"cluster has {engine.nranks} ranks")
    for i, info in enumerate(engine.gather_cluster_info()):
        print(
            f"rank {i}: hostname={info.hostname}, pid={info.pid}, "
            f"cuda_visible_devices={info.cuda_visible_devices}, "
            f"gpu_uuid={info.gpu_uuid}"
        )
# rank 0: hostname=node-0, pid=12345, cuda_visible_devices=0, gpu_uuid=GPU-abc123...
# rank 1: hostname=node-0, pid=12346, cuda_visible_devices=1, gpu_uuid=GPU-def456...
```

[ray-docs]: https://docs.ray.io/
