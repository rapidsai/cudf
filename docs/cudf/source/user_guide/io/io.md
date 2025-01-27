# Input / Output

This page contains Input / Output related APIs in cuDF.

## I/O Supported dtypes

The following table lists are compatible cudf types for each supported
IO format.

<div class="special-table-wrapper" style="overflow:auto">

```{eval-rst}
.. table::
    :class: io-supported-types-table special-table
    :widths: 15 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10

    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+-------------------+--------+--------+---------+---------+
    |                       |       CSV       |      Parquet    |       JSON       |       ORC       |  AVRO  |        HDF        |       DLPack    |      Feather      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | Data Type             | Writer | Reader | Writer | Reader | Writer¹ | Reader | Writer | Reader | Reader | Writer² | Reader² | Writer | Reader | Writer² | Reader² |
    +=======================+========+========+========+========+=========+========+========+========+========+=========+=========+========+========+=========+=========+
    | int8                  | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ✅     | ✅     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | int16                 | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ✅     | ✅     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | int32                 | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ✅     | ✅     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | int64                 | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ✅     | ✅     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | uint8                 | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ❌     | ✅     | ❌     | ✅      | ✅      | ✅     | ✅     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | uint16                | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ❌     | ✅     | ❌     | ✅      | ✅      | ✅     | ✅     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | uint32                | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ❌     | ✅     | ❌     | ✅      | ✅      | ✅     | ✅     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | uint64                | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ❌     | ❌     | ❌     | ✅      | ✅      | ✅     | ✅     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | float32               | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ✅     | ✅     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | float64               | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ✅     | ✅     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | bool                  | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ✅     | ✅     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | str                   | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | category              | ✅     | ❌     | ❌     | ❌     | ❌      | ❌     | ❌     | ❌     | ❌     | ✅      | ✅      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | list                  | ❌     | ❌     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ❌     | ❌      | ❌      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | timedelta64[s]        | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ❌     | ❌     | ❌     | ✅      | ✅      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | timedelta64[ms]       | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ❌     | ❌     | ❌     | ✅      | ✅      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | timedelta64[us]       | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ❌     | ❌     | ❌     | ✅      | ✅      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | timedelta64[ns]       | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ❌     | ❌     | ❌     | ✅      | ✅      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | datetime64[s]         | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | datetime64[ms]        | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | datetime64[us]        | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | datetime64[ns]        | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | struct                | ❌     | ❌     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ❌     | ✅      | ✅      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | decimal32             | ✅     | ✅     | ✅     | ✅     | ✅      | ❌     | ✅     | ✅     | ❌     | ❌      | ❌      | ❌     | ❌     | ❌      | ❌      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | decimal64             | ✅     | ✅     | ✅     | ✅     | ✅      | ❌     | ✅     | ✅     | ❌     | ❌      | ❌      | ❌     | ❌     | ❌      | ❌      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | decimal128            | ✅     | ✅     | ✅     | ✅     | ✅      | ❌     | ✅     | ✅     | ❌     | ❌      | ❌      | ❌     | ❌     | ❌      | ❌      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
```

</div>

**Notes:**

- \[¹\] - Not all orientations are GPU-accelerated.
- \[²\] - Not GPU-accelerated.

## Magnum IO GPUDirect Storage Integration

Many IO APIs can use Magnum IO GPUDirect Storage (GDS) library to optimize
IO operations.  GDS enables a direct data path for direct memory access
(DMA) transfers between GPU memory and storage, which avoids a bounce
buffer through the CPU.  GDS also has a compatibility mode that allows
the library to fall back to copying through a CPU bounce buffer. The
SDK is available for download
[here](https://developer.nvidia.com/gpudirect-storage). GDS is also
included in CUDA Toolkit 11.4 and higher.

Use of GPUDirect Storage in cuDF is disabled by default, but can be
enabled through the environment variable `LIBCUDF_CUFILE_POLICY`.
This variable also controls the GDS compatibility mode.

There are four valid values for the environment variable:

- "GDS": Enable GDS use. If the cuFile library cannot be properly loaded,
fall back to the GDS compatibility mode.
- "ALWAYS": Enable GDS use. If the cuFile library cannot be properly loaded,
throw an exception.
- "KVIKIO": Enable GDS compatibility mode through [KvikIO](https://github.com/rapidsai/kvikio).
Note that KvikIO also provides the environment variable `KVIKIO_COMPAT_MODE` for GDS
control that may alter the effect of "KVIKIO" option in cuDF:
  - By default, `KVIKIO_COMPAT_MODE` is unset. In this case, cuDF enforces
    the GDS compatibility mode, and the system configuration check for GDS I/O
    is never performed.
  - If `KVIKIO_COMPAT_MODE=ON`, this is the same with the above case.
  - If `KVIKIO_COMPAT_MODE=OFF`, KvikIO enforces GDS I/O without system
    configuration check, and will error out if GDS requirements are not met. The
    only exceptional case is that if the system does not support files being
    opened with the `O_DIRECT` flag, the GDS compatibility mode will be used.
- "OFF": Completely disable GDS and kvikIO use.

If no value is set, behavior will be the same as the "KVIKIO" option.

This environment variable also affects how cuDF treats GDS errors.

- When `LIBCUDF_CUFILE_POLICY` is set to "GDS" and a GDS API call
  fails for any reason, cuDF falls back to the internal implementation
  with bounce buffers.
- When `LIBCUDF_CUFILE_POLICY` is set to "ALWAYS" and a GDS API call
fails for any reason (unlikely, given that the compatibility mode is
on), cuDF throws an exception to propagate the error to the user.
- When `LIBCUDF_CUFILE_POLICY` is set to "KVIKIO" and a KvikIO API
  call fails for any reason (unlikely, given that KvikIO implements
  its own compatibility mode) cuDF throws an exception to propagate
  the error to the user.

For more information about error handling, compatibility mode, and
tuning parameters in KvikIO see: <https://github.com/rapidsai/kvikio>

Operations that support the use of GPUDirect Storage:

- {py:func}`cudf.read_avro`
- {py:func}`cudf.read_json`
- {py:func}`cudf.read_parquet`
- {py:func}`cudf.read_orc`
- {py:meth}`cudf.DataFrame.to_csv`
- {py:func}`cudf.DataFrame.to_json`
- {py:meth}`cudf.DataFrame.to_parquet`
- {py:meth}`cudf.DataFrame.to_orc`

Several parameters that can be used to tune the performance of
GDS-enabled I/O are exposed through environment variables:

- `LIBCUDF_CUFILE_THREAD_COUNT`: Integral value, maximum number of
  parallel reads/writes per file (default 16);
- `LIBCUDF_CUFILE_SLICE_SIZE`: Integral value, maximum size of each
  GDS read/write, in bytes (default 4MB).  Larger I/O operations are
  split into multiple calls.

## nvCOMP Integration

Some types of compression/decompression can be performed using either
the [nvCOMP library](https://github.com/NVIDIA/nvcomp) or the internal
implementation.

Which implementation is used by default depends on the data format and
the compression type.  Behavior can be influenced through environment
variable `LIBCUDF_NVCOMP_POLICY`.

There are three valid values for the environment variable:

- "STABLE": Only enable the nvCOMP in places where it has been deemed
  stable for production use.
- "ALWAYS": Enable all available uses of nvCOMP, including new,
  experimental combinations.
- "OFF": Disable nvCOMP use whenever possible and use the internal
  implementations instead.

If no value is set, behavior will be the same as the "STABLE" option.

```{eval-rst}
.. table:: Current policy for nvCOMP use for different types
    :widths: 20 20 20 20 20 20 20 20 20 20

    +-----------------------+--------+--------+--------------+--------------+---------+--------+--------------+--------------+--------+
    |                       |       CSV       |            Parquet          |       JSON       |             ORC             |  AVRO  |
    +-----------------------+--------+--------+--------------+--------------+---------+--------+--------------+--------------+--------+
    | Compression Type      | Writer | Reader |    Writer    |    Reader    | Writer¹ | Reader |    Writer    |    Reader    | Reader |
    +=======================+========+========+==============+==============+=========+========+==============+==============+========+
    | Snappy                | ❌     | ❌     | Stable       | Stable       | ❌      | ❌     | Stable       | Stable       | ❌     |
    +-----------------------+--------+--------+--------------+--------------+---------+--------+--------------+--------------+--------+
    | ZSTD                  | ❌     | ❌     | Stable       | Stable       | ❌      | ❌     | Stable       | Stable       | ❌     |
    +-----------------------+--------+--------+--------------+--------------+---------+--------+--------------+--------------+--------+
    | DEFLATE               | ❌     | ❌     | ❌           | ❌           | ❌      | ❌     | Experimental | Experimental | ❌     |
    +-----------------------+--------+--------+--------------+--------------+---------+--------+--------------+--------------+--------+
    | LZ4                   | ❌     | ❌     | Stable       | Stable       | ❌      | ❌     | Stable       | Stable       | ❌     |
    +-----------------------+--------+--------+--------------+--------------+---------+--------+--------------+--------------+--------+
    | GZIP                  | ❌     | ❌     | Experimental | Experimental | ❌      | ❌     | ❌           | ❌           | ❌     |
    +-----------------------+--------+--------+--------------+--------------+---------+--------+--------------+--------------+--------+

```

## Low Memory Considerations

By default, cuDF's parquet and json readers will try to read the entire file in one pass. This can cause problems when dealing with large datasets or when running workloads on GPUs with limited memory.

To better support low memory systems, cuDF provides a "low-memory" reader for parquet and json files. This low memory reader processes data in chunks, leading to lower peak memory usage due to the smaller size of intermediate allocations.

To read a parquet or json file in low memory mode, there are [cuDF options](https://docs.rapids.ai/api/cudf/nightly/user_guide/api_docs/options/#api-options) that must be set globally prior to calling the reader. To set those options, call:
- `cudf.set_option("io.parquet.low_memory", True)` for parquet files, or
- `cudf.set_option("io.json.low_memory", True)` for json files.
