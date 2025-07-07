# GPUEngine Configuration Options

The `polars.GPUEngine` object may be configured in several different ways.

## Executor

`cudf-polars` includes multiple *executors*, backends that take a Polars query and execute it to produce the result (either an in-memory `polars.DataFrame` from `.collect()` or one or more files with `.sink_<method>`). These can be specified with the `executor` option when you create the `GPUEngine`.

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

See [Configuration Reference](#cudf-polars-api) for a full list of options, and
[Streaming Execution](#cudf-polars-streaming) for more on the streaming executor,
including multi-GPU execution.

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

## Disabling CUDA Managed Memory

By default the `in-memory` executor will use [CUDA managed memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-memory-introduction) with RMM's pool allocator. On systems that don't support managed memory, a non-managed asynchronous pool
allocator is used.
Managed memory can be turned off by setting `POLARS_GPU_ENABLE_CUDA_MANAGED_MEMORY` to `0`. System requirements for managed memory can be found [here](
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#system-requirements-for-unified-memory).
