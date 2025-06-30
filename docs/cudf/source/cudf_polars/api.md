(cudf-polars-api)=
# API

For the most part, the public API of `cudf-polars` is the polars API. All
cudf-polars specific configuration is done through options passed to
{class}`~polars.lazyframe.engine_config.GPUEngine`. The majority of the options
are passed as `**kwargs` and collected into the configuration described below


```{eval-rst}
.. automodule:: cudf_polars.utils.config
   :members:
      ConfigOptions,
      ParquetOptions,
      StreamingExecutor,
      InMemoryExecutor,
```
