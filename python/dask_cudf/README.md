# Dask-cuDF

A [cuDF](https://github.com/rapidsai/cudf)-backed [Dask-DataFrame](https://docs.dask.org/en/latest/dataframe.html) API for out-of-core and multi-GPU ETL.

## Brief Introduction

Dask is a task-based library for parallel scheduling and execution. In addition to its central scheduling machinery, the library also includes the [Dask-DataFrame](https://docs.dask.org/en/latest/dataframe.html) module, which is a scalable version of the [Pandas](https://pandas.pydata.org/) DataFrame/Series API.  Dask-cuDF builds upon Dask-DataFrame to provide a convenient API for the decomposition and processing of large cuDF DataFrame and/or Series objects.

### Documentation Links

- [10 Minutes to cuDF and Dask-cuDF](https://docs.rapids.ai/api/cudf/stable/10min.html)
- [Dask-CUDA](https://github.com/rapidsai/dask-cuda) (for multi-GPU Scaling)