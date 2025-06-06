# Experimental streaming and multi-GPU options

As well as in-memory execution, obtained by selecting `engine="gpu"`
when `collect`ing a query, the GPU engine supports streaming execution
that partitions the data in the query into chunks. To select streaming
execution, we need to pass an appropriately configured `GPUEngine`
object into `collect`.

The streaming executors work best when the inputs to your query come
from parquet files. That is, start with `scan_parquet`, not existing
Polars `DataFrame`s or CSV files.

````{note}
Streaming execution is in an experimental state and not all queries
will run in streaming mode.
````

## Single GPU streaming

The simplest case, requiring no additional dependencies, is the
`synchronous` executor. An appropriate engine is:
```python
engine = pl.GPUEngine(
    executor="streaming",
    executor_options={"scheduler": "synchronous"},
)
```

When executed with this engine, any parquet inputs are split into
"partitions" that are streamed through the query graph. We try and
pick a good default for the typical partition size (based on the
amount of GPU memory available), however, it might not be optimal. You
can configure the execution by providing more options to the executor.
For example, to configure the max number of rows in each partition:
``` executor_options={ ..., "max_rows_per_partition": 1_000_000, } ```
A million rows per partition is a reasonable default. You may find, at
the cost of higher memory footprint, that a larger value gives better
performance.

````{note}
If part of a query does not run in streaming mode, but _does_ run
using the in-memory GPU engine, then we materialize the whole input
and execute with the in-memory engine.
````

## Multi GPU streaming

Streaming utilising multiple GPUs simultaneously is supported by
setting the ``"scheduler"`` to ``"distributed"``:
```python
engine = pl.GPUEngine(
    executor="streaming",
    executor_options={"scheduler": "distributed"},
)
```

Unlike the single GPU executor, this does require a number of
additional dependencies. We currently require
[Dask](https://www.dask.org/) and
[Dask-CUDA](https://docs.rapids.ai/api/dask-cuda/nightly/) to be
installed. In addition, we recommend that
[ucxx](https://github.com/rapidsai/ucxx) and
[rapidsmpf](https://github.com/rapidsai/rapidsmpf) are installed to
take advantage of any high-performance networking.

````{note}
Unlike single-GPU streaming, if part of a query does not execute in
multi-GPU mode, we do not fall back to the in-memory GPU engine.
Instead we fall back (as normal for unsupported queries) to Polars'
CPU engine.
````

The multi-GPU engine uses the currently active Dask client to carry
out the partitioned execution, so for multi-GPU we would use something
like

```python
from dask_cuda import LocalCUDACluster

...

client = LocalCUDACluster(...).get_client()

q = ...
engine = pl.GPUEngine(
    executor="streaming",
    executor_options={"scheduler": "distributed"},
)
result = q.collect(engine=engine)
```
