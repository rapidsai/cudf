# GPUEngine Configuration Options

The `polars.GPUEngine` object may be configured in several different ways.

## Parquet Reader Options

Chunked reading is controlled by passing a dictionary of options to the `GPUEngine` object. Details may be found following the links to the underlying `libcudf` reader.
- `parquet_chunked`, indicicates is chunked parquet reading is to be used, default True.
- [chunk_read_limit](https://docs.rapids.ai/api/libcudf/legacy/classcudf_1_1io_1_1chunked__parquet__reader#aad118178b7536b7966e3325ae1143a1a) controls the maximum size per chunk, default unlimited.
- [pass_read_limit](https://docs.rapids.ai/api/libcudf/legacy/classcudf_1_1io_1_1chunked__parquet__reader#aad118178b7536b7966e3325ae1143a1a) controls the maximum memory used for decompression, default 16GiB.

For example, one would pass these parameters as follows:
```python
engine = GPUEngine(
    raise_on_fail=True,
    parquet_options={
        'parquet_chunked': True,
        'chunk_read_limit': int(1e9),
        'pass_read_limit': int(4e9)
    }
)
result = query.collect(engine=engine)
```
Note that passing `parquet_chunked: False` disables chunked reading entirely, and thus `chunk_read_limit` and `pass_read_limit` will have no effect.
