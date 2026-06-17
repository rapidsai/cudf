# cuDF C API (`libcudf_c`)

## Overview

`libcudf_c.so` is a stable C ABI layer on top of the C++ `libcudf` library. It
provides language-agnostic access to GPU DataFrames without requiring callers to
link against or understand C++ exception handling, name mangling, or STL types.

Any language with a C FFI (Rust, Python, Go, Java, Julia, ...) can load
`libcudf_c.so` and drive GPU-accelerated DataFrame operations through a clean,
versioned interface.

## Architecture

```
  ┌─────────────────────────────────────────────────────┐
  │  C++ libcudf   (libcudf.so, C++ templates + CUDA)   │
  └──────────────────────┬──────────────────────────────┘
                         │ wraps via translate_exceptions
  ┌──────────────────────▼──────────────────────────────┐
  │  C API  (libcudf_c.so, extern "C", CUDF_C_EXPORT)   │
  └──────────────────────┬──────────────────────────────┘
                         │ bindgen / cbindgen
  ┌──────────────────────▼──────────────────────────────┐
  │  cudf-sys  (Rust FFI bindings, auto-generated)       │
  └──────────────────────┬──────────────────────────────┘
                         │ safe wrappers
  ┌──────────────────────▼──────────────────────────────┐
  │  cudf  (safe Rust crate, RAII, Result<T, CudfError>) │
  └─────────────────────────────────────────────────────┘
```

## Build Instructions

Prerequisites: a working CUDA toolkit and libcudf installed (e.g. via conda
into `$CONDA_PREFIX`).

```bash
cmake -S c -B c/build -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
cmake --build c/build --parallel
```

The shared library is written to `c/build/libcudf_c.so`.

To install into `$CONDA_PREFIX`:

```bash
cmake --install c/build --prefix $CONDA_PREFIX
```

## Minimal C Usage Example

```c
#include <cudf/core/c_api.h>
#include <cudf/io/parquet.h>

int main(void) {
    /* Create a resources handle (holds a CUDA stream internally). */
    cudfResources_t res;
    if (cudfResourcesCreate(&res) != CUDF_SUCCESS) {
        fprintf(stderr, "cudfResourcesCreate: %s\n", cudfGetLastErrorText());
        return 1;
    }

    /* Set up reader options. */
    cudfParquetReaderOptions_t opts;
    if (cudfParquetReaderOptionsCreate(&opts) != CUDF_SUCCESS) {
        fprintf(stderr, "%s\n", cudfGetLastErrorText());
        cudfResourcesDestroy(res);
        return 1;
    }
    opts->filepath = "/path/to/data.parquet";
    opts->num_rows = -1; /* -1 means read all rows */

    /* Read the file. */
    cudfTable_t table;
    cudfError_t err = cudfParquetRead(opts, res, &table);
    cudfParquetReaderOptionsDestroy(opts);
    if (err != CUDF_SUCCESS) {
        fprintf(stderr, "cudfParquetRead: %s\n", cudfGetLastErrorText());
        cudfResourcesDestroy(res);
        return 1;
    }

    /* Inspect the result. */
    int64_t rows;
    cudfTableGetNumRows(table, &rows);
    printf("Loaded %lld rows\n", (long long)rows);

    /* Always destroy owning handles when done. */
    cudfTableDestroy(table);
    cudfResourcesDestroy(res);
    return 0;
}
```

## Error Handling

Every C API function returns `cudfError_t`:

```c
typedef enum { CUDF_ERROR = 0, CUDF_SUCCESS = 1 } cudfError_t;
```

`CUDF_SUCCESS` (1) means the call succeeded. `CUDF_ERROR` (0) means it failed.
On failure, call `cudfGetLastErrorText()` to retrieve a thread-local string
describing the error. The string is valid until the next C API call on the same
thread. Pass `NULL` to `cudfSetLastErrorText()` to clear it manually.

**Fatal GPU errors.** If a function returns with the underlying error being a
`fatal_cuda_error`, the GPU context associated with the resources handle is
poisoned. Do not call any further C API functions with that handle. Destroy it
with `cudfResourcesDestroy()` and create a fresh one.

## Resources and Stream Model

The `cudfResources_t` handle bundles a CUDA stream with any other per-session
GPU state. Pass it to every function that issues GPU work.

```c
/* Use the default stream (stream 0). */
cudfResources_t res;
cudfResourcesCreate(&res);

/* Attach a non-default stream. */
cudaStream_t stream;
cudaStreamCreate(&stream);
cudfStreamSet(res, stream);

/* Flush outstanding GPU work on the stream. */
cudfStreamSync(res);

/* Read back the current stream. */
cudaStream_t current;
cudfStreamGet(res, &current);
```

Note: the Parquet functions take `cudaStream_t` in their signature but the
current implementation forwards the resources handle's stream. Pass the
`cudfResources_t` cast to `cudaStream_t` (i.e. `(cudaStream_t)res`) when
calling lower-level functions directly.

## ABI Stability

`libcudf_c.so` follows the RAPIDS SOVERSION policy:

- The shared library carries a SOVERSION that increments only on
  ABI-breaking changes.
- ABI-breaking changes are reserved for major RAPIDS release boundaries
  (e.g. 25.x → 26.x).
- Additions (new functions, new enum values at the end) are compatible
  within a major release series.

Link against `libcudf_c.so` (the unversioned symlink) at build time; the
loader resolves the correct SOVERSION at runtime.

## Supported Types

The full `cudfTypeId_t` enum is defined in `c/include/cudf/types.h`. The
supported primitive type IDs are:

| ID                           | Description            |
|------------------------------|------------------------|
| `CUDF_TYPE_INT8` .. `INT64`  | Signed integers        |
| `CUDF_TYPE_UINT8` .. `UINT64`| Unsigned integers      |
| `CUDF_TYPE_FLOAT32/64`       | Floating point         |
| `CUDF_TYPE_BOOL8`            | Boolean (1 byte)       |
| `CUDF_TYPE_TIMESTAMP_*`      | Days / sec / ms / us / ns |
| `CUDF_TYPE_DURATION_*`       | Duration variants      |
| `CUDF_TYPE_STRING`           | Variable-length UTF-8  |
| `CUDF_TYPE_LIST`             | Nested list            |
| `CUDF_TYPE_STRUCT`           | Nested struct          |
| `CUDF_TYPE_DECIMAL32/64/128` | Fixed-point decimals   |
| `CUDF_TYPE_DICTIONARY32`     | Dictionary encoded     |

Decimal types require setting the `scale` field of `cudfDataType_t`. All other
types use `scale = 0`.

## Function Reference (by category)

**Resources**
- `cudfResourcesCreate(cudfResources_t*)` — allocate handle
- `cudfResourcesDestroy(cudfResources_t)` — free handle
- `cudfStreamSet(cudfResources_t, cudaStream_t)` — attach stream
- `cudfStreamGet(cudfResources_t, cudaStream_t*)` — read current stream
- `cudfStreamSync(cudfResources_t)` — synchronize stream
- `cudfVersionGet(uint16_t*, uint16_t*, uint16_t*)` — library version

**Tables and Columns**
- `cudfTableGetNumRows`, `cudfTableGetNumColumns`, `cudfTableGetColumn`
- `cudfTableDestroy`, `cudfColumnDestroy`
- `cudfColumnGetSize`, `cudfColumnGetNullCount`, `cudfColumnGetType`

**Parquet I/O**
- `cudfParquetReaderOptionsCreate` / `cudfParquetReaderOptionsDestroy`
- `cudfParquetRead(opts, stream, table*)` — read into owning table
- `cudfParquetWriterOptionsCreate` / `cudfParquetWriterOptionsDestroy`
- `cudfParquetWrite(table, opts, stream)` — write table to file
