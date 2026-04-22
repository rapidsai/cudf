(cudf-polars-dask-engine)=
# Dask

{class}`~cudf_polars.experimental.rapidsmpf.frontend.dask.DaskEngine` runs the streaming executor
on a [Dask distributed][dask-distributed] cluster — one Dask worker per GPU, coordinated by a
single client process. Partitions are streamed through the query graph and collective operations
(shuffles, all-gathers, joins) run across workers over a shared UCXX communicator.

```python
import polars as pl
from cudf_polars.experimental.rapidsmpf.frontend.dask import DaskEngine

with DaskEngine() as engine:
    result = (
        pl.scan_parquet("/data/dataset/*.parquet")
          .filter(pl.col("amount") > 100)
          .group_by("customer_id")
          .agg(pl.col("amount").sum())
          .collect(engine=engine)
    )
    print(result)
```

With no arguments, {class}`~cudf_polars.experimental.rapidsmpf.frontend.dask.DaskEngine` creates a
`distributed.LocalCluster` with one worker per visible GPU, a `distributed.Client`, and
bootstraps a UCXX communicator across all workers. On exit, everything it created is torn down.

## Configuring `DaskEngine`

For custom configuration, build a
{class}`~cudf_polars.experimental.rapidsmpf.frontend.options.StreamingOptions` and use
`DaskEngine.from_options()`:

```python
import polars as pl
from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions
from cudf_polars.experimental.rapidsmpf.frontend.dask import DaskEngine

opts = StreamingOptions(num_streaming_threads=8, fallback_mode="silent")

with DaskEngine.from_options(opts) as engine:
    result = pl.scan_parquet("/data/dataset/*.parquet").collect(engine=engine)
```

See {doc}`options` for the available fields.

## Bring your own Dask client

Pass an existing `distributed.Client` via `dask_client=` to attach to an already-running
scheduler:

```python
from distributed import Client
import polars as pl
from cudf_polars.experimental.rapidsmpf.frontend.dask import DaskEngine

with Client("scheduler-address:8786") as dc:
    with DaskEngine(dask_client=dc) as engine:
        result = pl.scan_parquet("/data/*.parquet").collect(engine=engine)
```

When you supply the client, {class}`~cudf_polars.experimental.rapidsmpf.frontend.dask.DaskEngine`
leaves it (and the cluster) alone on exit.

### Pre-configured GPU clusters

Some Dask launchers — notably `dask_cuda.LocalCUDACluster` — already pin CPU affinity and set
`CUDA_VISIBLE_DEVICES` per worker. Disable the built-in hardware binding via
{class}`~cudf_polars.experimental.rapidsmpf.frontend.hardware_binding.HardwareBindingPolicy` to
avoid conflicts:

```python
from cudf_polars.experimental.rapidsmpf.frontend.dask import DaskEngine
from cudf_polars.experimental.rapidsmpf.frontend.hardware_binding import (
    HardwareBindingPolicy,
)

with DaskEngine(
    dask_client=dc,
    engine_options={
        "hardware_binding": HardwareBindingPolicy(enabled=False),
    },
) as engine:
    ...
```

### Manually launched workers

When launching workers yourself (for example on a multi-node HPC cluster), use the built-in nanny
preload to assign one GPU per worker. The preload sets `CUDA_VISIBLE_DEVICES` on each worker
before the process spawns:

```bash
# On each node — launch one worker per GPU with a single thread each:
dask worker SCHEDULER:8786 --nworkers N --nthreads 1 \
    --preload-nanny cudf_polars.experimental.rapidsmpf.frontend.dask
```

Then connect from the client:

```python
from distributed import Client
from cudf_polars.experimental.rapidsmpf.frontend.dask import DaskEngine

with Client("SCHEDULER:8786") as dc:
    with DaskEngine(dask_client=dc) as engine:
        result = lf.collect(engine=engine)
```

Hardware binding (CPU affinity, NUMA, network) is handled automatically by
{class}`~cudf_polars.experimental.rapidsmpf.frontend.dask.DaskEngine`; the nanny preload only
deals with GPU assignment.

See the [Dask CLI deployment guide][dask-cli] for more on `dask worker` options.

## Cluster diagnostics

{meth}`~cudf_polars.experimental.rapidsmpf.frontend.dask.DaskEngine.gather_cluster_info` returns
placement information for every worker:

```python
with DaskEngine() as engine:
    print(f"cluster has {engine.nranks} workers")
    for info in engine.gather_cluster_info():
        print(
            f"hostname={info['hostname']}, pid={info['pid']}, "
            f"CUDA_VISIBLE_DEVICES={info['cuda_visible_devices']}"
        )
```

{class}`~cudf_polars.experimental.rapidsmpf.frontend.dask.DaskEngine` raises `RuntimeError` if
created inside an `rrun` cluster.

[dask-distributed]: https://distributed.dask.org/
[dask-cli]: https://docs.dask.org/en/latest/deploying-cli.html
