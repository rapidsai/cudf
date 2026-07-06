<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# cudf-benchmarks

End-to-end workload benchmarks (TPC-H / TPC-DS) for cudf-polars and cudf.pandas.

These are whole-engine benchmarks that run at scale and compare against CPU engines
(polars, DuckDB, pandas). They live in their own package so they can run on CPU-only
machines without importing CUDA.

Install it from a checkout of the cuDF repository. The extras install only non-RAPIDS
packages; the GPU packages (cudf, cudf-polars, rapidsmpf, kvikio) are prerequisites you
install yourself.

| Extra | Installs | Prerequisites (bring your own) |
|-------|----------|--------------------------------|
| `cpu` | polars, duckdb, pandas, tpchgen-cli | none |
| `polars` | duckdb, tpchgen-cli | cudf-polars (brings polars), rapidsmpf, kvikio |
| `pandas` | tpchgen-cli | cudf (brings pandas) |
| `dask` | dask, distributed | via `polars` |
| `ray` | ray | via `polars` |

## Running the benchmarks

The docs walk through installation, data generation, running, and tuning for each engine:

- [cudf-polars PDS-H / PDS-DS benchmarks](../../docs/cudf/source/cudf_polars/benchmarks.md)
- [cudf.pandas PDS-H benchmarks](../../docs/cudf/source/cudf_pandas/benchmarks.md)
