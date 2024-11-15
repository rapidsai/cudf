# GPUEngine Configuration Options

The `polars.GPUEngine` object may be configured in several different ways.

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
