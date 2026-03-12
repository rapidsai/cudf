# cudf-polars Multi-GPU Design

This document describes the multi-GPU execution architecture of cudf-polars.
For user-facing setup instructions, see
`cudf-polars-mp.md`.

- [1. Architecture Overview](#1-architecture-overview)
- [2. SPMD Mode](#2-spmd-mode)
- [3. Ray Mode](#3-ray-mode)
- [4. Comparison](#4-comparison)
- [5. Hardware Mapping (GPU Pinning)](#5-hardware-mapping-gpu-pinning)

---

## 1. Architecture Overview

Both modes share the same bottom two layers (RapidsMPF engine and SPMD
cluster) and differ only in how the user script interacts with them.

### SPMD mode

The user script runs on **all N workers simultaneously**. There is no separate
client — every process is both client and worker.

```
       rank 0               rank 1       ...      rank N-1
┌─────────────────┐  ┌─────────────────┐     ┌─────────────────┐
│   User script   │  │   User script   │     │   User script   │
│ (same code on   │  │ (same code on   │     │ (same code on   │
│  every rank)    │  │  every rank)    │     │  every rank)    │
└────────┬────────┘  └────────┬────────┘     └────────┬────────┘
         │                    │                       │
         │     LazyFrame.collect(engine=engine)       │
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

Results are rank-local after `collect`. Call `allgather_polars_dataframe()` to
assemble the full dataset on every rank.

### Ray mode

A single client script dispatches work to N `RankActor` Ray actors (one per
GPU). The client never touches a GPU directly.

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

Per-rank output fragments are concatenated on the client before being returned.
No `allgather` step is needed.

### Key insight — why client modes exist

In SPMD mode every process runs the same code, which is unfamiliar to users
accustomed to single-process or Dask-style client/worker workflows. Client
frontends such as Ray let users write a normal single-process script while the
cluster handles distribution transparently. The underlying engine is unchanged;
only the dispatch layer differs. Future frontends (Dask, custom clients) can
target the same SPMD cluster without modifying the engine.

---

## 2. SPMD Mode

### Execution model

The user script is launched with `rrun -n N python script.py`. `rrun` starts N
identical processes, each pinned to one GPU. There is no separate client
process — every process runs the full script, acting simultaneously as client
and worker on its rank-local data.

Because every rank runs independent Python, a `pl.DataFrame` is always
*rank-local*: it holds only that rank's fragment of the distributed dataset.
File-based sources (`scan_parquet`, `scan_csv`) distribute work automatically
— the engine assigns disjoint file- or row-group ranges to each rank.

### Bootstrapping

`spmd_execution()` calls `bootstrap.create_ucxx_comm(type=BackendType.AUTO)`.
Under `rrun`, `BackendType.AUTO` resolves to the `rrun`-native bootstrap
mechanism, connecting all N ranks without additional configuration.

### GPU assignment

`rrun` sets `CUDA_VISIBLE_DEVICES` for each process automatically. Each rank
sees exactly one GPU. No user action is required.

### Query symmetry requirement

All ranks must issue the **same** sequence of Polars queries in the **same**
order. Collective operations (shuffles, all-gathers, joins) are matched across
ranks by a monotonically increasing operation ID. If one rank calls a
collective that another does not, all ranks will deadlock. Driver logic must be
fully deterministic: avoid rank-conditional `collect` calls, early exits, or
branching that causes different ranks to execute different query graphs.

### Entry point

```python
from cudf_polars.experimental.rapidsmpf.spmd import (
    spmd_execution,
    allgather_polars_dataframe,
)
```

`spmd_execution()` is a context manager that yields `(comm, ctx, engine)`:

| Object   | Type           | Purpose                                     |
|----------|----------------|---------------------------------------------|
| `comm`   | `Communicator` | Active RapidsMPF communicator               |
| `ctx`    | `Context`      | RapidsMPF context (GPU memory, stream pool) |
| `engine` | `pl.GPUEngine` | Polars GPU engine wired to `comm` and `ctx` |

### Collecting results

Each `collect(engine=engine)` returns a rank-local `pl.DataFrame`. To
assemble the full result on every rank, call `allgather_polars_dataframe()`:

```python
full = allgather_polars_dataframe(
    comm=comm, ctx=ctx, local_df=result, op_id=0
)
```

`op_id` must be the same on every rank and must be unique across all
`allgather_polars_dataframe` calls in the script.

### Example

```python
# Launch with: rrun -n 4 python spmd_script.py
import polars as pl
from cudf_polars.experimental.rapidsmpf.spmd import (
    spmd_execution,
    allgather_polars_dataframe,
)

with spmd_execution() as (comm, ctx, engine):
    result = (
        pl.scan_parquet("/data/dataset/*.parquet")
        .filter(pl.col("value") > 0)
        .group_by("category")
        .agg(pl.col("value").sum())
        .collect(engine=engine)
    )
    # result is rank-local; gather all fragments onto every rank
    full = allgather_polars_dataframe(
        comm=comm, ctx=ctx, local_df=result, op_id=0
    )
    # full is now identical on every rank
    print(full)
```

### Options pass-through

```python
with spmd_execution(
    executor_options={"rapidsmpf_py_executor_max_workers": 2},
    parquet_options={"use_rapidsmpf_native": True},
) as (comm, ctx, engine):
    ...
```

Reserved `executor_options` keys: `"runtime"`, `"cluster"`, `"spmd"`.
Reserved `engine_kwargs` keys: `"memory_resource"`, `"executor"`.

---

## 3. Ray Mode

### Execution model

The user runs a single client script. `ray_execution()` creates N `RankActor`
Ray remote actors — one per available GPU. The actors form a private SPMD
cluster; the client dispatches Polars IR to them and receives concatenated
results directly, with no `allgather` step needed.

### Bootstrapping

1. `ray_execution()` creates N `RankActor` instances (each requesting
   `num_gpus=1` from Ray's resource scheduler).
2. Root actor (rank 0) calls `setup_root()` → returns its UCXX address.
3. All N actors (including the root) call `setup_worker(root_ucxx_address)`
   **concurrently**. Non-root actors connect to the root; the root skips
   communicator creation and proceeds directly to the barrier. All ranks must
   reach the barrier simultaneously.
4. After `setup_worker` completes, each actor holds a fully initialized
   `Communicator` and `Context`.

### GPU assignment

Ray's resource scheduler assigns `num_gpus=1` to each `RankActor` before the
actor process starts, setting `CUDA_VISIBLE_DEVICES` automatically. The
`RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` environment variable prevents Ray from
overriding `CUDA_VISIBLE_DEVICES` to empty on the client process (which has
zero GPUs assigned).

For hardware placement details, see [Section 5](#5-hardware-mapping-gpu-pinning).

### Entry point

```python
from cudf_polars.experimental.rapidsmpf.ray import ray_execution
```

`ray_execution()` is a context manager that yields `(ray_client, engine)`:

| Object       | Type           | Purpose                                |
|--------------|----------------|----------------------------------------|
| `ray_client` | `RayClient`    | Client handle to the actor cluster     |
| `engine`     | `pl.GPUEngine` | Polars GPU engine backed by Ray actors |

### Results

`collect(engine=engine)` dispatches the query to all actors, concatenates
their per-rank output fragments on the client, and returns a single
`pl.DataFrame`. No `allgather` is needed.

### Ray lifecycle

`ray_execution()` calls `ray.init()` on entry if Ray is not already
initialized, and `ray.shutdown()` on exit. If `ray.is_initialized()` returns
`True` before entry, the caller manages the cluster lifetime.

### Diagnostics

```python
for i, info in enumerate(ray_client.gather_cluster_info()):
    print(f"rank {i}: {info}")
# Each info dict contains: pid, hostname, cuda_visible_devices, node_id
```

### Example

```python
# Launch with: python ray_script.py
import polars as pl
from cudf_polars.experimental.rapidsmpf.ray import ray_execution

with ray_execution() as (ray_client, engine):
    print(ray_client.gather_cluster_info())  # verify actor placement

    result = (
        pl.scan_parquet("/data/dataset/*.parquet")
        .filter(pl.col("value") > 0)
        .group_by("category")
        .agg(pl.col("value").sum())
        .collect(engine=engine)
    )
    # result is the full concatenated output, returned directly to the client
    print(result)
```

### Options pass-through

```python
with ray_execution(
    executor_options={"rapidsmpf_py_executor_max_workers": 2},
    engine_kwargs={"parquet_options": {"use_rapidsmpf_native": True}},
    ray_init_kwargs={"address": "auto"},
) as (ray_client, engine):
    ...
```

Reserved `executor_options` keys: `"runtime"`, `"cluster"`, `"spmd"`,
`"ray_client"`. Reserved `engine_kwargs` keys: `"memory_resource"`,
`"executor"`.

---

## 4. Comparison

|                  | SPMD                             | Ray                               | Future (Dask / custom) |
|------------------|----------------------------------|-----------------------------------|------------------------|
| Driver           | Script runs on **all N workers** | Single client process             | Single client process  |
| Launch           | `rrun -n N python script.py`     | `python script.py`                | `python script.py`     |
| GPU pinning      | `rrun` auto-pins each rank       | Ray scheduler (one actor per GPU) | Depends on frontend    |
| Result delivery  | Rank-local; `allgather` needed   | Concatenated, returned to client  | Returned to client     |
| Query symmetry   | Required (all ranks same order)  | Not required                      | Not required           |
| Extra dependency | `rrun` / RapidsMPF               | `ray`                             | `dask` / none          |

**Note on future frontends.** Dask and custom clients are not yet implemented,
but the architecture supports them. A new frontend only needs to:

1. Manage a pool of SPMD workers (one per GPU).
2. Bootstrap the UCXX communicator across those workers.
3. Dispatch pickled IR + `partition_info` + `collective_id_map` to all workers
   concurrently and collect their output fragments.

No changes to the underlying RapidsMPF streaming engine are required.

---

## 5. Hardware Mapping (GPU Pinning)

### Launch model

Hardware mapping depends on how ranks are **launched**.

Two launch models are supported:

* **SPMD mode**, where ranks are started directly by the `rrun` launcher.
* **Ray mode**, where ranks run inside Ray actors scheduled by the Ray runtime.

`rrun` is a lightweight process launcher designed for GPU workloads. It starts
multiple ranks, assigns GPUs, and optionally applies topology-aware bindings.

A typical SPMD program is started with:

```bash
rrun -n 4 python script.py
```

This launches four identical Python processes. Each process becomes a **rank**
in the SPMD program and runs the same code independently.

Because `rrun` performs the hardware setup **before the program starts**, it
must be used to launch the application. It cannot attach to or configure
processes that are already running.

Other execution modes may require such functionality. For example, in **Ray
mode** the ranks run inside actors created by the Ray runtime rather than by
`rrun`, so GPU assignment is handled by Ray instead.

---

### SPMD mode

In SPMD mode, `rrun` assigns one GPU to each rank.

By default it detects all GPUs on the node, but a specific list can be
provided:

```bash
rrun -n 4 -g 0,1,2,3 python script.py
```

Each rank receives a single GPU via `CUDA_VISIBLE_DEVICES`. Inside the process
this GPU always appears as **device 0**, so CUDA programs require no special
configuration.

If more ranks than GPUs are launched, multiple ranks will share a GPU.

`rrun` can also apply topology-aware bindings so that each rank runs close to
its GPU. This includes CPU affinity, NUMA memory locality, and network-device
selection. Bindings are enabled by default and can be disabled with:

```bash
rrun -n 4 --bind-to none python script.py
```

If topology discovery is unavailable, bindings are skipped automatically.

`rrun` also sets environment variables used by RapidsMPF to bootstrap the
communication backend.

When running under Slurm (for example with `srun`), Slurm launches the ranks
across nodes while `rrun` performs the same local setup for each rank.

---

### Ray mode

In Ray mode, ranks run as Ray actors scheduled by the Ray runtime rather than
being launched by `rrun`.

Each rank is defined with:

```python
@ray.remote(num_gpus=1)
```

Ray's scheduler selects a node with a free GPU, launches the actor there, and
sets `CUDA_VISIBLE_DEVICES` before the Python code starts. As in SPMD mode,
each rank therefore sees its assigned GPU as **device 0**.

Unlike SPMD mode, the processes already exist when RapidsMPF code begins
executing. Ray creates the worker processes and then runs the user code inside
them. This means the `rrun` launcher cannot perform hardware setup ahead of
time.

#### Future Request for `rrun`
To support such execution models, `rrun` will need a **library API** that can
configure an already-running process. Conceptually, the workflow would look
like:

1. Ray launches one worker per GPU.
2. Each worker determines which GPU it has been assigned.
3. The worker calls a new `rrun` API to apply the hardware bindings locally.

This API would configure the process in-place, for example by applying CPU
affinity, NUMA bindings, and other topology-aware settings for the specified
GPU. The GPU could be specified by index or by a stable identifier such as a
GPU UUID.

This capability is not currently provided by `rrun`, which today only supports
processes that it launches itself, see [Launch model](#launch-model).
