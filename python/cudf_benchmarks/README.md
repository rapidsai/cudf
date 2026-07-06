<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# cudf-benchmarks

End-to-end workload benchmarks (TPC-H / TPC-DS) for cudf-polars and cudf.pandas.

These are whole-engine benchmarks that run at scale and compare against CPU engines
(polars, DuckDB, pandas). They live in their own package so they can run on CPU-only
machines without importing CUDA.

## Install

Install it from a checkout of the cuDF repository:

```shell
git clone https://github.com/rapidsai/cudf.git
cd cudf
```

The package is a single pure wheel and pins no CUDA-suffixed dependency. GPU packages
(cudf, cudf-polars, rapidsmpf, kvikio) are prerequisites you install yourself; the
extras below install only the non-RAPIDS pieces. Pick the extra that matches what you want
to run:

```shell
# CPU-only machine or CI (no GPU, no CUDA import)
pip install -e python/cudf_benchmarks[cpu]

# Single-GPU polars + CPU baselines (bring cudf-polars, rapidsmpf, kvikio yourself)
pip install -e python/cudf_benchmarks[polars]

# Multi-GPU / multi-node polars
pip install -e python/cudf_benchmarks[polars,ray]     # or [polars,dask]

# cudf.pandas benchmarks (bring cudf yourself)
pip install -e python/cudf_benchmarks[pandas]
```

To install the GPU prerequisites, follow the [RAPIDS installation guide](https://docs.rapids.ai/install)
(for `[polars]`, install `cudf-polars`; for `[pandas]`, install `cudf`). The full benchmark
docs linked below walk through this step by step.

| Extra | Installs | Prerequisites (bring your own) |
|-------|----------|--------------------------------|
| `cpu` | polars, duckdb, pandas, tpchgen-cli | none |
| `polars` | duckdb, tpchgen-cli | cudf-polars (brings polars), rapidsmpf, kvikio |
| `pandas` | tpchgen-cli | cudf (brings pandas) |
| `dask` | dask, distributed | via `polars` |
| `ray` | ray | via `polars` |

## Run

```shell
python -m cudf_benchmarks.polars.pdsh ...
python -m cudf_benchmarks.polars.pdsds ...
python -m cudf_benchmarks.pandas.pdsh ...
```

The polars benchmarks run on a CPU-only machine with `--frontend polars-cpu` or
`--frontend duckdb`. The pandas benchmarks currently require `cudf` even for
`--executor cpu`.

## Full instructions

These reproduce published results end to end, with data generation, tuning options, multi-GPU
runs, and the results format:

- [cudf-polars PDS-H / PDS-DS benchmarks](https://docs.rapids.ai/api/cudf/stable/cudf_polars/benchmarks/)
- [cudf.pandas PDS-H benchmarks](https://docs.rapids.ai/api/cudf/stable/cudf_pandas/benchmarks/)
