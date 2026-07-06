<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# cudf-benchmarks

End-to-end workload benchmarks (TPC-H / TPC-DS) for cudf-polars and cudf.pandas.

These are whole-engine benchmarks that run at scale and compare against CPU engines
(polars, DuckDB, pandas).

The benchmarks ship as two wheels so they work on both GPU and CPU-only machines:

- `cudf-benchmarks` — CUDA-suffixed (e.g. `cudf-benchmarks-cu12`). Its extras install the
  matching-nightly RAPIDS packages, so the benchmark version is tied to the engine it measures.
- `cudf-benchmarks-cpu` — an unsuffixed, CUDA-free wheel (polars, DuckDB, pandas) for machines
  with no GPU.

GPU extras on `cudf-benchmarks`:

| Extra | Installs |
|-------|----------|
| `polars` | cudf-polars, rapidsmpf, kvikio, duckdb, tpchgen-cli |
| `pandas` | cudf, pandas, tpchgen-cli |
| `dask` | rapids-dask-dependency (multi-GPU / multi-node, with `polars`) |
| `ray` | ray (multi-GPU / multi-node, with `polars`) |

## Running the benchmarks

The docs walk through installation, data generation, running, and tuning for each engine:

- [cudf-polars PDS-H / PDS-DS benchmarks](../../docs/cudf/source/cudf_polars/benchmarks.md)
- [cudf.pandas PDS-H benchmarks](../../docs/cudf/source/cudf_pandas/benchmarks.md)
