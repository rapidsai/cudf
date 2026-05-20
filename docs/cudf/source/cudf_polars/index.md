# Polars GPU engine

cuDF provides GPU-accelerated execution engines for Python users of the Polars Lazy API. The
engines support most of the core expressions and data types as well as a growing set of more
advanced dataframe manipulations and data file formats. When a GPU engine is selected, Polars
converts expressions into an optimized query plan and determines whether the plan is supported
on the GPU. If it is not, the execution transparently falls back to the standard Polars engine
and runs on the CPU.

## Install

Follow the [RAPIDS installation guide](https://docs.rapids.ai/install) and pick the
`cudf-polars` package for your CUDA and Python versions. For example, with conda:

```bash
conda install -c rapidsai -c conda-forge -c nvidia cudf-polars
```

Or with pip (CUDA 13 wheels; use `cudf-polars-cu12` for CUDA 12):

```bash
pip install cudf-polars-cu13
```

## Quick start

{class}`~cudf_polars.engine.ray.RayEngine` with no arguments uses
every GPU visible to the process, so the same code runs on one GPU and scales to multi-GPU /
multi-node setups automatically:

```python
import polars as pl
from cudf_polars.engine.ray import RayEngine

query = (
    pl.scan_parquet("/data/dataset/*.parquet")
    .filter(pl.col("amount") > 100)
    .group_by("customer_id")
    .agg(pl.col("amount").sum())
)

with RayEngine() as engine:
    result = query.collect(engine=engine)
```

See {doc}`usage` for the full tutorial, {doc}`engines` for a conceptual overview of the
available engines, and {doc}`options` for the
{class}`~cudf_polars.engine.options.StreamingOptions` configuration.

## Benchmark

```{note}
The following benchmarks were performed with the `POLARS_GPU_ENABLE_CUDA_MANAGED_MEMORY`
environment variable set to `"0"`. Using managed memory (the default) imposes a performance cost
in order to avoid out of memory errors. Peak performance can still be attained by setting the
environment variable to `0`.
```

We reproduced the [Polars Decision Support (PDS)](https://github.com/pola-rs/polars-benchmark)
benchmark to compare Polars GPU engine with the default CPU settings across several dataset sizes.
Here are the results:

```{figure} ../_static/pds_benchmark_polars.png
:width: 600px
```

You can see up to 13x speedup using the GPU engine on the compute-heavy PDS queries involving
complex aggregation and join operations. Below are the speedups for the top performing queries:

```{figure} ../_static/compute_heavy_queries_polars.png
:width: 1000px
```

*PDS-H benchmark | GPU: NVIDIA H100 PCIe | CPU: Intel Xeon W9-3495X (Sapphire Rapids) | Storage:
Local NVMe*

You can reproduce the results by visiting the [Polars Decision Support (PDS) GitHub repository](https://github.com/pola-rs/polars-benchmark).

## Learn More

The GPU engine for Polars is now available in Open Beta and the engine is undergoing rapid development.
To learn more, visit the [GPU Support page](https://docs.pola.rs/user-guide/gpu-support/) on the Polars website.

```{toctree}
:maxdepth: 1
:caption: Contents:

usage
engines
options
profiling
other_engines
api
developer_docs
```

## Launch on Google Colab

```{figure} ../_static/colab.png
:width: 200px
:target: https://nvda.ws/4eKlWZW

Try out the GPU engine for Polars in a free GPU notebook environment.
Sign in with your Google account and [launch the demo on Colab](https://nvda.ws/4eKlWZW).
```
