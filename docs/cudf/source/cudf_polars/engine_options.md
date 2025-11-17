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

The `streaming` executor is the default executor as of RAPIDS 25.08, and is
equivalent to passing `engine="gpu"` or `engine=pl.GPUEngine()` to `collect`. At
a high-level, the `streaming` executor works by breaking inputs (in-memory
DataFrames or parquet files) into multiple pieces and streaming those pieces
through the series of operations needed to produce the final result.

We also provide an `in-memory` executor. This executor is often faster when the
underlying data fits comfortably in device memory, because the overhead of splitting
inputs and executing them in batches is less beneficial at this scale. With that said,
this executor must rely on [Unified Virtual Memory] (UVM) if the input and intermediate
data do not fit in device memory. The `in-memory` executor can be used with

```python
engine = pl.GPUEngine(executor="in-memory")
```

In general, we recommend starting with the default `streaming` executor, because
it scales significantly better than `in-memory`. The `streaming` executor includes
several configuration options, which can be provided with the `executor_options`
key when constructing the `GPUEngine`:

```python
engine = pl.GPUEngine(
    executor="streaming",  # the default
    executor_options={
        "max_rows_per_partition": 500_000,
    }
)
```

You can configure the default value for configuration options through
environment variables with the prefix `CUDF_POLARS__EXECUTOR__{option_name}`.
For example, the environment variable
`CUDF_POLARS__EXECUTOR__MAX_ROWS_PER_PARTITION` will set the default
`max_rows_per_partition` to use if it isn't overridden through
`executor_options`.

For boolean options, like `rapidsmpf_spill`, the values `{"1", "true", "yes", "y"}`
are considered `True` and `{"0", "false", "no", "n"}` are considered `False`.

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

You can configure the default value for configuration options through
environment variables with the prefix
`CUDF_POLARS__PARQUET_OPTIONS__{option_name}`. For example, the environment
variable `CUDF_POLARS__PARQUET_OPTIONS__CHUNKED=0` will set the default
`chunked` to `False`.

## CUDA Stream Policy

By default, all CUDA operations in `cudf-polars` are launched on the default
stream. You can configure `cudf-polars` to use multiple CUDA streams, which may
improve performance by overlapping data transfers and kernel launches.

This behavior is configured by the `cuda_stream_policy` keyword or
`CUDF_POLARS__CUDA_STREAM_POLICY` environment variable.  The valid options are

* `default`: use the default CUDA stream for all kernel launches and memory operations
* `new`: create a new CUDA stream when necessary (e.g. when reading from a file or loading an in-memory `polars.LazyFrame` object,
  or when performing a join where the inputs might be on different streams)
* `pool`: use an RMM stream pool (only supported with the rapidsmpf runtime)

The ``rapidsmpf`` runtime for the streaming executor also supports using a CUDA Stream Pool.

```python
engine = GPUEngine(
    executor="streaming",
    executor_options={
        "runtime": "rapidsmpf",
    },
    cuda_stream_policy="pool",
)
```

Or provide a dictionary with configuration options for the pool, like `cuda_stream_pool={"pool_size": 16}`.
You can also set the `CUDF_POLARS__CUDA_STREAM_POLICY` environment variable the JSON encoded configuration dictionary.

This stream pool is shared between cudf-polars and rapidsmpf.

## Memory Resource

All GPU memory allocations made by cudf-polars use an RMM Memory Resource object from {mod}`rmm.mr`. You can specify
the memory resource to use by:

1. Passing a concrete `MemoryResource` instance to {class}`~polars.lazyframe.engine_config.GPUEngine`.
2. Passing the configuration options for a Memory Resource as the `memory_resource_config` keyword argument to {class}`~polars.lazyframe.engine_config.GPUEngine`.
3. Relying on the default behavior, which creates a memory resource for you (details below).

By default, cudf-polars will create a new RMM Memory Resource for you, which is cached and reused
for each query. The type of that memory resource is hardware-dependent. GPUs that support [Unified Virtual Memory] memory,
use a {class}`rmm.mr.ManagedMemoryResource` wrapped in a {class}`rmm.mr.PoolMemoryResource` and {class}`rmm.mr.PrefetchResourceAdaptor`.
Otherwise, {class}`rmm.mr.CudaAsyncMemoryResource` is used.

Set `POLARS_GPU_ENABLE_CUDA_MANAGED_MEMORY=0` to disabled managed memory and use {class}`rmm.mr.CudaAsyncMemoryResource` instead.

Alternatively, you can customize the pool by passing the configuration for an RMM Memory Resource object as `memory_resource_config`
when creating your {class}`~polars.lazyframe.engine_config.GPUEngine`:

```python
memory_resource_config = {
    "qualname": "rmm.mr.CudaAsyncMemoryResource",
    "options": {
        "initial_pool_size": "100 MiB",
    }
}

engine = pl.GPUEngine(memory_resource_config=memory_resource_config)
```

This lets you control things like the initial pool size or release threshold.

Finally, for maximum flexibility, you can create your own memory resource object and pass it into the {class}`~polars.lazyframe.engine_config.GPUEngine`:

```python
import polars as pl
import rmm

mr = rmm.mr.CudaAsyncMemoryResource()
engine = pl.GPUEngine(memory_resource=mr)
```

Passing a concrete memory resource takes precedence over passing the `memory_resource_config` options,
which takes precedence over the default memory resource.

Note that providing a concrete memory resource isn't an option with the distributed scheduler,
because the concrete memory resource is only valid for the process in which it was created.

## Disabling CUDA Managed Memory

By default the `in-memory` executor will use [CUDA managed memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-memory-introduction) with RMM's pool allocator. On systems that don't support managed memory, a non-managed asynchronous pool
allocator is used.
Managed memory can be turned off by setting `POLARS_GPU_ENABLE_CUDA_MANAGED_MEMORY` to `0`. System requirements for managed memory can be found [here](
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#system-requirements-for-unified-memory).

[Unified Virtual Memory]: https://developer.nvidia.com/blog/unified-memory-cuda-beginners/
