(cudf-polars-other-engines)=
# Other Engines

The examples in {doc}`usage` use
{class}`~cudf_polars.experimental.rapidsmpf.frontend.ray.RayEngine`. `cudf-polars` ships two
other streaming engines that run the same streaming executor and produce equivalent results:

* **{doc}`dask_engine`** — runs on a [Dask distributed][dask] cluster with one Dask worker per
  GPU. Use this when you already have a Dask deployment or a preferred Dask launcher.
* **{doc}`spmd_engine`** — single program, multiple data: the same script runs once per GPU,
  typically launched with `rrun`. Single-rank mode needs no external cluster at all.

See {doc}`engines` for the conceptual comparison with `RayEngine` (cluster model, runtime
dependencies, typical use), and {doc}`options` for the shared
{class}`~cudf_polars.experimental.rapidsmpf.frontend.options.StreamingOptions` configuration.

```{toctree}
:maxdepth: 1
:hidden:

dask_engine
spmd_engine
```

[dask]: https://distributed.dask.org/
