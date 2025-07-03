# GPUEngine Configuration Options

The `polars.GPUEngine` object may be configured in several different ways.

## Executor

`cudf-polars` includes multiple *executors*, backends that take a Polars query and execute it to produce the result (either an in-memory `polars.DataFrame` from `.collect()` or one or more files with `.sink_<method>`). These can be specified with the `"executor"` option when you create the `GPUEngine`.

```python
import polars as pl

engine = pl.GPUEngine(executor="streaming")
query = ...

result = query.collect(engine=engine)
```

The `streaming` executor is the default, and is equivalent to passing
`engine="gpu"` or `engine=pl.GPUEngine()` to `collect`. At a high-level, the
`streaming` executor works by breaking inputs (in-memory DataFrames or parquet
files) into multiple pieces and streaming those pieces through the series of
operations needed to produce the final result.

We also provide an `in-memory` executor. This executor can be faster for very
small inputs, where the overhead of splitting inputs and executing them in
batches isn't worth it. However, each input and intermediate DataFrame must fit
in (device) memory for the `in-memory` executor to work.

In general, we recommend using the default `streaming` executor. The `streaming`
includes several configuration options, which can be provided with the `executor_options`
key when constructing the `GPUEngine`:

```python
engine = pl.GPUEngine(
    executor="streaming",  # the default
    executor_options={
        "max_rows_per_partition": 500_000,
    }
)
```

See [Configuration Reference](#cudf-polars-api) for a full list of options.

## Parquet Reader Options

Reading large parquet files can use a large amount of memory, especially when the files are compressed. This may lead to out of memory errors for some workflows. To mitigate this, the "chunked" parquet reader may be selected. When enabled, parquet files are read in chunks, limiting the peak memory usage at the cost of a small drop in performance.

To configure the parquet reader, we provide a dictionary of options to the `parquet_options` keyword of the `GPUEngine` object. Valid keys and values are:
- `chunked` indicates that chunked parquet reading is to be used. By default, chunked reading is turned on.
- [`chunk_read_limit`](https://docs.rapids.ai/api/libcudf/legacy/classcudf_1_1io_1_1chunked__parquet__reader#aad118178b7536b7966e3325ae1143a1a) controls the maximum size per chunk. By default, the maximum chunk size is unlimited.
- [`pass_read_limit`](https://docs.rapids.ai/api/libcudf/legacy/classcudf_1_1io_1_1chunked__parquet__reader#aad118178b7536b7966e3325ae1143a1a) controls the maximum memory used for decompression. The default pass read limit is 16GiB.

For example, to select the chunked reader with custom values for `pass_read_limit` and `chunk_read_limit`:
```python
engine = GPUEngine(
    parquet_options={
        'chunked': True,
        'chunk_read_limit': int(1e9),
        'pass_read_limit': int(4e9)
    }
)
result = query.collect(engine=engine)
```
Note that passing `chunked: False` disables chunked reading entirely, and thus `chunk_read_limit` and `pass_read_limit` will have no effect.

## Experimental Multi-GPU Scheduling

By default, the `"streaming"` executor uses a synchronous *scheduler* to execute
the query on a single GPU. You can instead use the `"distributed"` scheduler to
execute the query in parallel on a single node with multiple GPUs or multiple
nodes with one or more GPUs.

```{note}
The distributed scheduler is considered experimental and might change without warning.
```

```python
import polars as pl
engine = pl.GPUEngine(
    executor="streaming",
    executor_options={
        "scheduler": "distributed"
    }
)
```

When you actually compute the results (using `.collect()` or `.sink_<method>`),
you'll need to have a [Dask](http://dask.org/) Cluster active. For example,
using a [dask-cuda](https://docs.rapids.ai/api/dask-cuda/stable/)
`LocalCUDACluster` with one worker per GPU:

```python
from dask_cuda import LocalCUDACluster

cluster = LocalCUDACluster()
client = cluster.get_client()

q = ...

q.collect(engine=engine)
```

This will execute the query in parallel using all the GPUs available on your
system by default. See the
[dask-cuda](https://docs.rapids.ai/api/dask-cuda/stable/) documentation for
more.

## Disabling CUDA Managed Memory

By default the `in-memory` executor will use a [CUDA managed memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-memory-introduction) with RMM's pool allocator. On systems that don't support managed memory, a non-managed asynchronous pool
allocator is used.
Managed memory can be turned off by setting `POLARS_GPU_ENABLE_CUDA_MANAGED_MEMORY` to `0`. System requirements for managed memory can be found [here](
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#system-requirements-for-unified-memory).
