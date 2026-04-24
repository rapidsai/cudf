(cudf-polars-engines)=
# Engines

## What is an engine?

`cudf-polars` executes Polars `LazyFrame` queries on GPU. You select GPU execution by passing an
`engine=` argument to `.collect()` or `.sink_*()`. The `engine` you pass decides *how* the
query runs: whether it streams through partitioned inputs or fits everything in device memory,
whether it runs in-process or distributes work across a cluster of GPU workers, and which
cluster backend coordinates those workers.

## Execution modes

### In-memory

`engine="gpu"` (or `engine=pl.GPUEngine()`) runs the query on a single GPU, materializing
intermediates in device memory. This is the simplest path and matches the mental model of
OSS Polars' CPU engine: one process, one device, no streaming.

Use this when the data fits comfortably in device memory. On GPUs with Unified Virtual Memory,
cudf-polars can spill past device memory, but at a performance cost.

### Streaming

Streaming engines partition their inputs (Parquet files or in-memory `DataFrame`s) and stream
those partitions through the query graph. This lets queries scale past device memory, and — by
distributing partitions across a cluster of GPU workers — across multiple GPUs and multiple
nodes.

{class}`~cudf_polars.experimental.rapidsmpf.frontend.ray.RayEngine` with no arguments uses every
GPU visible to the process, so on a single node with N GPUs it runs the query on all N of them
without any extra configuration. Launching a multi-node cluster simply means pointing the
engine at that cluster; the user-facing code is the same.

## Cluster backends

All streaming engines run the same streaming executor. They differ only in how the cluster of
GPU workers is provisioned and coordinated:

| Engine                                                                | Cluster model                                           | Runtime dependency              | Typical use                                                                   |
| --------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------- | ----------------------------------------------------------------------------- |
| {class}`~cudf_polars.experimental.rapidsmpf.frontend.ray.RayEngine`   | Single-client driver; one Ray actor per GPU             | [Ray][ray-docs]                 | Works from a laptop to a cloud cluster. No separate cluster setup needed.     |
| {class}`~cudf_polars.experimental.rapidsmpf.frontend.dask.DaskEngine` | Single-client driver; one Dask worker per GPU           | [Dask distributed][dask]        | Teams with an existing Dask deployment or a preferred Dask launcher.          |
| {class}`~cudf_polars.experimental.rapidsmpf.frontend.spmd.SPMDEngine` | Same script runs once per GPU, joined by a communicator | RapidsMPF (+ UCXX under `rrun`) | HPC / SPMD launchers such as `rrun`. Single-rank mode needs no cluster at all.|

All three produce equivalent query results. Pick by deployment fit, not by performance.

## Where to go next

- {doc}`usage` — tutorial that walks through running your first GPU query end-to-end.
- {doc}`other_engines` — per-engine reference pages for DaskEngine and SPMDEngine.
- {doc}`options` — the `StreamingOptions` configuration object and every field it surfaces.

[ray-docs]: https://docs.ray.io/
[dask]: https://distributed.dask.org/
