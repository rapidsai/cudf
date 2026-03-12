# Multi-GPU Polars

Multi-GPU Polars extends Polars query execution to multiple GPUs.

Multi-process (mp) execution distributes a query across several GPU workers. Each
worker owns a disjoint fragment of the data and participates in collective operations
(shuffles, all-gathers, joins) to produce a globally correct result.

The entry point in all cases is the Polars `GPUEngine` with `executor="streaming"`.
The `cluster` option selects the execution model:

| `cluster` value | Description                                 | Status          |
| --------------- | ------------------------------------------- | --------------- |
| `"single"`      | Single-GPU, in-process execution            | Stable (legacy) |
| `"distributed"` | Multi-GPU via Dask Distributed              | Stable (legacy) |
| `"spmd"`        | Multi-GPU via SPMD with the `rrun` launcher | Experimental    |
| `"ray"`         | Multi-GPU via Ray actors                    | Experimental    |

This document describes the two experimental multi-GPU modes. Both rely on RapidsMPF
for shuffle and collective communication.

* [SPMD cluster mode](#spmd-cluster-mode)
* [Ray cluster mode](#ray-cluster-mode)

---

# SPMD cluster mode

In SPMD (Single Program, Multiple Data) execution, the same Python script is launched
multiple times simultaneously, once per GPU, using the `rrun` launcher bundled with
RapidsMPF. Each process is assigned a GPU and receives a **rank**. Ranks communicate
through a UCXX-based communicator established at startup.

Each rank runs an independent Python process and owns its local data. File-based
sources (`scan_parquet`, `scan_csv`, etc.) are automatically partitioned so that
different ranks read different file or row-group ranges. In-memory `DataFrame`
objects are already rank-local, so each rank processes its own copy.

## Prerequisites

* RapidsMPF (`rapidsmpf`) installed
* UCXX available (usually installed with RapidsMPF)
* `rrun` launcher available (`rrun --help` should succeed)

## Running in SPMD mode

`spmd_execution()` is the primary entry point for SPMD execution. It is a context
manager imported from `cudf_polars.experimental.rapidsmpf.spmd`. On entry it:

1. Bootstraps a UCXX communicator connecting all ranks.
2. Creates a RapidsMPF streaming `Context` that owns GPU memory and a CUDA stream pool.
3. Constructs and yields a `pl.GPUEngine` bound to that context.

All resources are released when the context exits.

`spmd_execution()` must run inside an `rrun` cluster. It raises `RuntimeError`
if `rapidsmpf.bootstrap.is_running_with_rrun()` returns `False`.

```python
# launch with: rrun -n 4 python my_script.py
import polars as pl
from cudf_polars.experimental.rapidsmpf.spmd import (
    spmd_execution,
    allgather_polars_dataframe,
)

with spmd_execution() as (comm, ctx, engine):
    result = (
        pl.scan_parquet("/data/dataset/*.parquet")
        .filter(pl.col("amount") > 100)
        .group_by("customer_id")
        .agg(pl.col("amount").sum())
        .collect(engine=engine)
    )

    full = allgather_polars_dataframe(
        comm=comm,
        ctx=ctx,
        local_df=result,
        op_id=0,
    )
```

The context manager yields:

* `comm` — `rapidsmpf.communicator.Communicator`
* `ctx` — `rapidsmpf.streaming.core.context.Context`
* `engine` — `pl.GPUEngine` configured for SPMD execution

Pass `engine` to every `LazyFrame.collect()` inside the context block.

## Collecting distributed results

`collect()` returns a rank-local result. Use
`allgather_polars_dataframe()` to gather all fragments:

```python
full = allgather_polars_dataframe(
    comm=comm,
    ctx=ctx,
    local_df=result,
    op_id=0,
)
```

`op_id` is a unique integer that identifies this collective operation across ranks.
All ranks must call the same collective with the same `op_id`. Otherwise the program
will deadlock.

The result is a `pl.DataFrame` containing rows from all ranks, ordered by rank.

## Query symmetry requirement

All ranks must execute the **same sequence of queries in the same order**. Collective
operations are matched using internal operation IDs. If one rank executes a collective
that another rank does not, the program will deadlock.

In practice:

* Avoid rank-conditional `collect()` calls
* Avoid branches that change the query graph
* Keep the driver script deterministic

## Passing options

`executor_options` and `engine_kwargs` accept pass-through dictionaries:

```python
with spmd_execution(
    executor_options={
        "max_rows_per_partition": 500_000,
        "rapidsmpf_spill": True,
    },
    parquet_options={"use_rapidsmpf_native": True},  # forwarded via **engine_kwargs
) as (comm, ctx, engine):
    ...
```

`executor_options` keys map to `StreamingExecutor` fields. Any additional keyword
arguments to `spmd_execution()` (such as `parquet_options`) are forwarded directly
to `pl.GPUEngine` as `**engine_kwargs`.

The keys `"runtime"`, `"cluster"`, and `"spmd"` in `executor_options`, and
`"memory_resource"` and `"executor"` in `engine_kwargs`, are reserved.

---

# Ray cluster mode

Ray mode uses a single client process that drives execution across multiple GPU
workers. Internally, the system uses the concept of **ranks**, similar to MPI ranks.
Each rank corresponds to one GPU worker and participates in collective operations
through a shared UCXX communicator.

In the Ray implementation, each rank is implemented as a **Ray actor**, with one
actor created per available GPU. Each rank owns its GPU, memory resource,
communicator endpoint, and RapidsMPF streaming context.

The client sends the query plan to all ranks. The ranks execute the pipeline
collectively through UCXX, and their outputs are streamed back and concatenated on
the client.

Unlike SPMD mode, the driver script runs as a normal Python program. There is no
`rrun` launcher and no symmetry requirement for the driver code.

## Prerequisites

* Ray (`ray`) installed
* RapidsMPF and UCXX available on all GPU nodes

## Running in Ray mode

`ray_execution()` is imported from `cudf_polars.experimental.rapidsmpf.ray`. It:

1. Calls `ray.init()` if Ray is not already running
2. Creates one `RankActor` per GPU
3. Bootstraps a UCXX communicator across the actors
4. Yields a `pl.GPUEngine` and a `RayClient`

Actors are shut down on exit. If the context started Ray, it also calls
`ray.shutdown()`.

```python
import polars as pl
from cudf_polars.experimental.rapidsmpf.ray import ray_execution

with ray_execution() as (ray_client, engine):
    result = (
        pl.scan_parquet("/data/dataset/*.parquet")
        .filter(pl.col("amount") > 100)
        .group_by("customer_id")
        .agg(pl.col("amount").sum())
        .collect(engine=engine)
    )

print(result)
```

The context manager yields:

* `ray_client` — cluster diagnostics and utilities
* `engine` — `pl.GPUEngine` configured for Ray execution

## Ray lifecycle

If Ray is already initialized, `ray_execution()` attaches to the existing cluster and
does not call `ray.shutdown()` on exit.

```python
import ray
import polars as pl
from cudf_polars.experimental.rapidsmpf.ray import ray_execution

ray.init(address="auto")

try:
    with ray_execution() as (ray_client, engine):
        result = pl.scan_parquet(...).collect(engine=engine)
finally:
    ray.shutdown()
```

`ray_execution()` raises `RuntimeError` if called inside an `rrun` cluster or if no
GPUs are available.

## Cluster diagnostics

`RayClient.gather_cluster_info()` returns placement information for all rank actors:

```python
with ray_execution() as (ray_client, engine):
    for i, info in enumerate(ray_client.gather_cluster_info()):
        print(
            f"rank {i}: hostname={info['hostname']}, pid={info['pid']}, "
            f"CUDA_VISIBLE_DEVICES={info['cuda_visible_devices']}"
        )
```

Each entry includes `pid`, `hostname`, `cuda_visible_devices`, and `node_id`.

## Passing options

`executor_options`, `engine_kwargs`, and `ray_init_kwargs` accept pass-through
dictionaries:

```python
with ray_execution(
    executor_options={"max_rows_per_partition": 500_000},
    engine_kwargs={"raise_on_fail": True},
    ray_init_kwargs={"num_cpus": 4},
) as (ray_client, engine):
    ...
```

The keys `"runtime"`, `"cluster"`, `"spmd"`, and `"ray_client"` in
`executor_options`, and `"memory_resource"` and `"executor"` in `engine_kwargs`,
are reserved.
