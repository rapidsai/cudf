# Multi-GPU Polars

Multi-GPU Polars extends Polars query execution to multiple GPUs.

Multi-GPU execution distributes a query across several GPU workers. Each worker
owns a disjoint fragment of the data and participates in collective operations
such as shuffles, all-gathers, and joins to produce a globally correct result.

The entry point in all cases is the Polars `GPUEngine` with `executor="streaming"`.
The `cluster` option selects the execution model:

| `cluster`       | Description                                          | Status            |
| --------------- | ---------------------------------------------------- | ----------------- |
| `"single"`      | Single-GPU, in-process execution                     | Stable (legacy)   |
| `"distributed"` | Multi-GPU via [Dask Distributed][dask-distributed]   | Stable (legacy)   |
| `"ray"`         | Multi-GPU via [Ray][ray-docs] actors                 | Preview (new API) |
| `"spmd"`        | Multi-GPU via [SPMD][spmd-wiki]                      | Preview (new API) |

Two preview execution modes are available:

* **Ray mode** — a single-client model where a driver program coordinates GPU
  workers implemented as Ray actors.
* **SPMD mode** — each GPU runs the same script as an independent process.
  When launched with `rrun` a full UCXX communicator connects the ranks.
  Without `rrun` it falls back to a single-rank communicator with no external
  dependencies, which is useful for local development and testing.

This document describes these two execution modes.

* [Ray execution mode](#ray-execution-mode)
* [SPMD execution mode](#spmd-execution-mode)

---

## Ray execution mode

Ray mode uses a single client process that drives execution across multiple ranks.
Each rank corresponds to one GPU worker and participates in collective operations
through a shared UCXX communicator.

In the Ray implementation each rank is implemented as a [**Ray actor**][ray-actors],
with one actor created per available GPU.

Conceptually the system looks like this:

```
                 ┌──────────────────────────────┐
                 │        User script           │
                 │   (single client process)    │
                 │  LazyFrame.collect(engine=…) │
                 └──────────────┬───────────────┘
                                │ IR dispatched to all actors
               ┌────────────────|─────────────────┐
               ↓                ↓                 ↓
        ┌─────────────┐  ┌─────────────┐   ┌─────────────┐
        │  RankActor  │  │  RankActor  │   │  RankActor  │
        │   rank 0    │  │   rank 1    │   │  rank N-1   │
        │   run IR    │  │   run IR    │   │   run IR    │
        └──────┬──────┘  └──────┬──────┘   └──────┬──────┘
               ↓                ↓                 ↓
┌────────────────────────────────────────────────────────────────┐
│                     RapidsMPF streaming engine                 │
│   shuffle / all-gather · UCXX communicator · RMM GPU memory    │
└────────────────────────────────────────────────────────────────┘
               ↑                ↑                 ↑
             GPU 0            GPU 1            GPU N-1
```

The client broadcasts the query plan to all ranks. The ranks execute the pipeline
collectively through UCXX, and their outputs are streamed back and concatenated on
the client process.

The driver script runs as a normal Python program with no `rrun` launcher. Query
symmetry is handled automatically: the client serializes the complete query plan and
broadcasts it to all actors, so every rank always executes the same query.

### Prerequisites

* Ray (`ray`) installed
* RapidsMPF and UCXX available on all GPU nodes

### Running in Ray mode

`ray_execution()` is imported from `cudf_polars.experimental.rapidsmpf.frontend.ray`. It:

1. Calls `ray.init()` if Ray is not already running
2. Creates one `RankActor` per GPU
3. Bootstraps a UCXX communicator across the actors
4. Yields a `pl.GPUEngine` and a `RayClient`

Actors are shut down on exit. If the context started Ray, it also calls
`ray.shutdown()`.

```python
import polars as pl
from cudf_polars.experimental.rapidsmpf.frontend.ray import ray_execution

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

### Ray lifecycle

If Ray is already initialized, `ray_execution()` attaches to the existing cluster and
does not call `ray.shutdown()` on exit.

```python
import ray
import polars as pl
from cudf_polars.experimental.rapidsmpf.frontend.ray import ray_execution

ray.init(address="auto")

try:
    with ray_execution() as (ray_client, engine):
        result = pl.scan_parquet(...).collect(engine=engine)
finally:
    ray.shutdown()
```

`ray_execution()` raises `RuntimeError` if called inside an `rrun` cluster or if no
GPUs are available.

### Cluster diagnostics

`RayClient.gather_cluster_info()` returns placement information for all rank actors:

```python
with ray_execution() as (ray_client, engine):
    print(f"cluster has {ray_client.nranks} ranks")
    for i, info in enumerate(ray_client.gather_cluster_info()):
        print(
            f"rank {i}: hostname={info['hostname']}, pid={info['pid']}, "
            f"CUDA_VISIBLE_DEVICES={info['cuda_visible_devices']}"
        )
```

Each entry includes `pid`, `hostname`, `cuda_visible_devices`, and `node_id`.

### Passing options

`rapidsmpf_options`, `executor_options`, `engine_options`, and `ray_init_options` accept
pass-through dictionaries:

```python
from rapidsmpf.integrations.cudf_polars import Options

with ray_execution(
    rapidsmpf_options=Options(num_streaming_threads=8),
    executor_options={
        "max_rows_per_partition": 500_000,
        "rapidsmpf_py_executor_max_workers": 2,
    },
    engine_options={"raise_on_fail": True},
    ray_init_options={"num_cpus": 4},
) as (ray_client, engine):
    ...
```

`rapidsmpf_options` is an `Options` object passed to the RapidsMPF `Context` on each
worker. If not provided, `ray_execution()` constructs a default `Options` with
`num_streaming_threads=4`.

`executor_options` is forwarded directly to `pl.GPUEngine` as its `executor_options`
argument; user-supplied keys are merged with reserved entries set by `ray_execution()`.
Any additional keyword arguments to `ray_execution()` are also forwarded to `pl.GPUEngine`.

Notable `executor_options` keys:

* `"rapidsmpf_py_executor_max_workers"` (default: `1`) — number of threads in the Python
  `ThreadPoolExecutor` that drives the RapidsMPF actor network on each worker.

Reserved keys:

* `executor_options`: `"runtime"`, `"cluster"`, `"spmd"`, `"ray_context"`
* `engine_options`: `"memory_resource"`, `"executor"`

---

## SPMD execution mode

In SPMD (Single Program, Multiple Data) execution, the same Python script runs once
per GPU and each process owns its local data fragment. Collective operations
(shuffles, all-gathers, joins) coordinate across processes to produce a globally
consistent result.

`spmd_execution()` selects the communicator automatically:

* **With `rrun`** — the `rrun` launcher starts one process per GPU and
  `spmd_execution()` bootstraps a UCXX communicator across all ranks.
* **Without `rrun`** — `spmd_execution()` falls back to a single-rank
  communicator that requires no external communication library (no UCXX,
  Ray, or Dask). This mode is useful for local development, unit tests,
  and single-GPU pipelines.

File-based sources (`scan_parquet`, `scan_csv`, etc.) are automatically partitioned
so that different ranks read different file or row-group ranges. In-memory
`DataFrame` objects are already rank-local, so each rank processes its own copy.

Conceptually the setup looks like this:

```
       rank 0               rank 1       ...      rank N-1
┌─────────────────┐  ┌─────────────────┐     ┌─────────────────┐
│   User script   │  │   User script   │     │   User script   │
│ (same code on   │  │ (same code on   │     │ (same code on   │
│  every rank)    │  │  every rank)    │     │  every rank)    │
└────────┬────────┘  └────────┬────────┘     └────────┬────────┘
         │                    │                       │
┌────────┴────────────────────┴───────────────────────┴────────┐
│              LazyFrame.collect(engine=engine)                │
└────────┬────────────────────┬───────────────────────┬────────┘
         ↓                    ↓                       ↓
┌─────────────────┐  ┌─────────────────┐     ┌─────────────────┐
│     run IR      │  │     run IR      │     │     run IR      │
└────────┬────────┘  └────────┬────────┘     └────────┬────────┘
         │                    │                       │
         ↓                    ↓                       ↓
┌────────────────────────────────────────────────────────────────┐
│                     RapidsMPF streaming engine                 │
│   shuffle / all-gather · UCXX communicator · RMM GPU memory    │
└────────────────────────────────────────────────────────────────┘
         ↑                    ↑                       ↑
      GPU 0                GPU 1                   GPU N-1
```

After `collect`, results are **rank-local**. To assemble the full dataset on
every rank, call `allgather_polars_dataframe()`.

### Prerequisites

* RapidsMPF (`rapidsmpf`) installed
* UCXX available when using `rrun` for multi-GPU execution
  (usually installed with RapidsMPF; not required for single-rank use)
* `rrun` launcher available for multi-GPU use (`rrun --help` should succeed)

### Running in SPMD mode

`spmd_execution()` is the primary entry point for SPMD execution. It is a context
manager imported from `cudf_polars.experimental.rapidsmpf.frontend.spmd`. On entry it:

1. Bootstraps a communicator: UCXX when running under `rrun`, otherwise a
   single-rank communicator that requires no external library.
2. Creates a RapidsMPF streaming `Context` that owns GPU memory and a CUDA stream pool.
3. Constructs and yields a `pl.GPUEngine` bound to that context.

All resources are released when the context exits.

```python
# multi-GPU launch: rrun -n 4 python my_script.py
# single-GPU (no rrun needed): python my_script.py
import polars as pl
from cudf_polars.experimental.rapidsmpf.collectives.common import reserve_op_id
from cudf_polars.experimental.rapidsmpf.frontend.spmd import (
    allgather_polars_dataframe,
    spmd_execution,
)

with spmd_execution() as (comm, ctx, engine):
    result = (
        pl.scan_parquet("/data/dataset/*.parquet")
        .filter(pl.col("amount") > 100)
        .group_by("customer_id")
        .agg(pl.col("amount").sum())
        .collect(engine=engine)
    )

    with reserve_op_id() as op_id:
        full = allgather_polars_dataframe(
            comm=comm,
            ctx=ctx,
            local_df=result,
            op_id=op_id,
        )
```

The context manager yields:

* `comm` — [`rapidsmpf.communicator.Communicator`][rapidsmpf-communicator]
* `ctx` — [`rapidsmpf.streaming.core.context.Context`][rapidsmpf-context]
* `engine` — {class}`~polars.lazyframe.engine_config.GPUEngine`

Pass `engine` to every `LazyFrame.collect()` or `sink*()` call inside the context block.

### Query symmetry requirement

All ranks must execute the **same sequence of queries in the same order**. Collective
operations are matched using internal operation IDs. If one rank executes a collective
that another rank does not, the program will deadlock.

In practice:

* Avoid rank-conditional `collect()` or `sink*()` calls
* Avoid branches that change the query graph
* Keep the driver script deterministic

**Example that works correctly:**

```python
# Every rank executes the same query in the same order.
with spmd_execution() as (comm, ctx, engine):
    result = (
        pl.scan_parquet("/data/*.parquet")
        .filter(pl.col("amount") > 100)
        .group_by("customer_id")
        .agg(pl.col("amount").sum())
        .collect(engine=engine)
    )
```

**Example that deadlocks:**

```python
# Rank 0 executes a group_by collective; other ranks do not.
# The collective IDs go out of sync → deadlock.
with spmd_execution() as (comm, ctx, engine):
    df = pl.scan_parquet("/data/*.parquet")
    if comm.rank == 0:        # DON'T DO THIS
        df = df.group_by("customer_id").agg(pl.col("amount").sum())
    result = df.collect(engine=engine)
```

### Collecting distributed results

`collect()` returns a rank-local result. Use
`allgather_polars_dataframe()` to gather all fragments:

```python
from cudf_polars.experimental.rapidsmpf.collectives.common import reserve_op_id
from cudf_polars.experimental.rapidsmpf.frontend.spmd import (
    allgather_polars_dataframe,
    spmd_execution,
)

with spmd_execution() as (comm, ctx, engine):
    with reserve_op_id() as op_id:
        full = allgather_polars_dataframe(
            comm=comm,
            ctx=ctx,
            local_df=result,
            op_id=op_id,
        )
```

`op_id` identifies this collective across ranks — all ranks must pass the same value.
Use `reserve_op_id()` to obtain a safe ID. It draws from the same pool that cudf-polars uses internally for shuffle
and join collectives, so there is no risk of collision. Do not pass hardcoded integers: they
may silently collide with an ID already reserved by an active collective inside `collect()`.

The result is guaranteed to be a `pl.DataFrame` containing rows from all ranks in rank order
(rank 0 first, then rank 1, …, rank N-1).

### Passing options

`rapidsmpf_options`, `executor_options`, and `engine_options` accept pass-through
arguments:

```python
import rmm
from rapidsmpf.integrations.cudf_polars import Options

with spmd_execution(
    rapidsmpf_options=Options(num_streaming_threads=8),
    executor_options={
        "max_rows_per_partition": 500_000,
        "rapidsmpf_spill": True,
        "rapidsmpf_py_executor_max_workers": 2,
    },
    engine_options={"parquet_options": {"use_rapidsmpf_native": True}},
) as (comm, ctx, engine):
    ...
```

**Memory resource:** `spmd_execution` captures `rmm.mr.get_current_device_resource()`
at entry, wraps it in `RmmResourceAdaptor` (so libcudf temporary allocations and the
RapidsMPF `Context` share the same resource), sets the wrapped resource as current, and
restores the original resource on exit. To use a custom allocator, call
`rmm.mr.set_current_device_resource(your_mr)` **before** entering `spmd_execution()`.
Do not pre-wrap it in `RmmResourceAdaptor`.

`rapidsmpf_options` is an `Options` object passed to the RapidsMPF `Context`. Defaults
to `None` (uses RapidsMPF defaults).

`executor_options` is forwarded directly to `pl.GPUEngine` as its `executor_options`
argument; user-supplied keys are merged with reserved entries set by `spmd_execution()`.

`engine_options` is forwarded as keyword arguments to `pl.GPUEngine`. For example,
pass `engine_options={"parquet_options": {"use_rapidsmpf_native": True}}` to enable
native Parquet reads.

Notable `executor_options` keys:

* `"rapidsmpf_py_executor_max_workers"` (default: `1`) — number of threads in the Python
  `ThreadPoolExecutor` that drives the RapidsMPF actor network.

Reserved keys:

* `executor_options`: `"runtime"`, `"cluster"`, `"spmd_context"`
* `engine_options`: `"memory_resource"`, `"executor"`

<!-- Reference links -->
[dask-distributed]: https://distributed.dask.org/
[spmd-wiki]: https://en.wikipedia.org/wiki/Single_program,_multiple_data
[ray-docs]: https://docs.ray.io/
[ray-actors]: https://docs.ray.io/en/latest/ray-core/actors.html
[rapidsmpf-communicator]: https://docs.rapids.ai/api/rapidsmpf/stable/api/communicator/
[rapidsmpf-context]: https://docs.rapids.ai/api/rapidsmpf/stable/api/streaming/context/
[polars-gpuengine]: https://docs.pola.rs/api/python/stable/reference/api/polars.GPUEngine.html
