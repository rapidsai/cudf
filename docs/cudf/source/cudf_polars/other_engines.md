(cudf-polars-other-engines)=
# Other Engines

{class}`~cudf_polars.experimental.rapidsmpf.frontend.ray.RayEngine` (covered in {doc}`usage`) is
the preferred engine for GPU acceleration. Two sibling engines are also available:

* **{doc}`dask_engine`** — runs the streaming executor on a [Dask distributed][dask-distributed]
  cluster (one Dask worker per GPU). Use this when you already have a Dask deployment or want to
  attach to an existing `distributed.Client`.
* **{doc}`spmd_engine`** — single program, multiple data: the same script runs once per GPU,
  typically launched with `rrun`. Use this for HPC-style SPMD workflows or unit tests.

Both engines share the same streaming executor, the same
{class}`~cudf_polars.experimental.rapidsmpf.frontend.options.StreamingOptions` configuration (see
{doc}`options`), and produce equivalent results to
{class}`~cudf_polars.experimental.rapidsmpf.frontend.ray.RayEngine`.

```{toctree}
:maxdepth: 1
:hidden:

dask_engine
spmd_engine
```

[dask-distributed]: https://distributed.dask.org/
