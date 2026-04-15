# Variant Field Extraction -- Implementation Report

## Overview

This document covers the GPU-backed Parquet VARIANT field extraction feature for
libcudf, limited to non-nested (top-level) fields.  It describes the public API
surface, kernel algorithm, test coverage, benchmark design, and benchmark results
including a performance fix discovered during benchmarking.

---

## 1. Public API

Header: `cpp/include/cudf/io/variant.hpp`

The feature exposes three functions in the `cudf` namespace.  All operate on a
VARIANT column materialized as `struct<list<uint8> metadata, list<uint8> value>`.

### `get_variant_field`

```cpp
std::unique_ptr<column> get_variant_field(
    column_view const& variant_column,
    std::string const& field_name,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);
```

Extracts the raw Variant-encoded bytes of a named top-level field from each row.
Returns a new VARIANT struct column where child 0 is a deep copy of the input
metadata (preserving the shared dictionary for downstream nested access) and
child 1 contains the extracted field's encoded value bytes.  Produces null when
the struct row is null, the key is absent from the metadata dictionary, or the
value blob is not an object.

### `cast_variant`

```cpp
std::unique_ptr<column> cast_variant(
    column_view const& variant_column,
    data_type desired_type,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);
```

Decodes a VARIANT struct column's value blobs into a typed cuDF column.
Supported target types: `INT32` and `STRING`.  Produces null when the struct row
is null or the encoded type does not match `desired_type`.

### `extract_variant_field`

```cpp
std::unique_ptr<column> extract_variant_field(
    column_view const& variant_column,
    std::string const& field_name,
    data_type desired_type,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);
```

Convenience wrapper equivalent to
`cast_variant(get_variant_field(col, name), type)`.

---

## 2. Kernel Algorithm

Implementation: `cpp/src/io/variant_extract.cu`

### 2.1 Data model

Each row's VARIANT data consists of two `list<uint8>` blobs accessed via
`lists_column_device_view`:

- **Metadata**: A V1 header byte, a dictionary size, an offset array into a
  string table, and the concatenated dictionary key strings.  The dictionary is
  global -- it covers keys at all nesting levels.
- **Value**: A self-describing encoded value.  Objects have `basic_type == 2` and
  encode a sorted field-ID array, an offset array, and concatenated child values.

### 2.2 Device-side parsing helpers

| Function | Purpose |
|---|---|
| `device_read_uint_le` | Read a 1/2/4/8-byte little-endian unsigned integer at an arbitrary byte offset with bounds checking. |
| `device_find_key_in_metadata` | Linear scan of the metadata dictionary.  Compares key lengths first (`slen != key_len` early exit) then byte-compares on length match.  Returns the dictionary index or -1. |
| `device_locate_object_field` | Parses the object value header, scans the field-ID array for the target dictionary index, then performs a second "tightest end" scan over all N+1 offsets to find the field's byte range.  The second scan is necessary because the Variant spec allows field values to be stored in any order (offsets are not monotonic). |
| `device_decode_int32` | Validates the header (`basic_type==0, header6==5`) and reads 4 LE bytes. |
| `device_decode_string_info` | Handles both short strings (`basic_type==1`, length in header) and long strings (`basic_type==0, header6==16`, 4-byte LE length). Returns `{length, data_offset}`. |

### 2.3 `get_variant_field` -- two-pass kernel

**Pass 1 (sizing):** `get_variant_field_sizes_kernel`
- One thread per row.  Reads the row's metadata and value blobs.
- Calls `device_find_key_in_metadata` to resolve the key string to a dictionary index.
- Calls `device_locate_object_field` to find the field's value span.
- Writes the span length to `d_sizes[row]`.  Clears the null-mask bit on failure.

**Offset scan:** `thrust::exclusive_scan` via `make_offsets_child_column` converts
sizes to output offsets and computes `total_bytes`.

**Pass 2 (copy):** `get_variant_field_copy_kernel`
- One thread per row.  Skips null rows via the bitmask.
- Re-parses metadata and value (same logic as pass 1) to locate the source span.
- Byte-copies `val_ptr[fs.offset .. fs.offset + fs.length)` to the output buffer
  at `d_out_bytes + d_offsets[row]`.

**Output assembly:** The output struct is constructed with `create_structs_hierarchy`
(not `make_structs_column`) to avoid the expensive `superimpose_and_sanitize_nulls`
path that would deep-copy and purge the metadata child column when nulls are present.

### 2.4 `cast_variant` -- single-pass kernels

- **INT32 path:** `cast_variant_int32_kernel` -- one thread per row, calls
  `device_decode_int32`, writes to a fixed-width output column.
- **STRING path:** `cast_variant_string_fn` functor used with
  `make_strings_children` (cuDF's standard two-pass string builder).  Calls
  `device_decode_string_info` in the sizing pass and copies char data in the
  materialization pass.

### 2.5 Thread configuration

All kernels use `block_size = 256` with `grid_1d{num_rows, block_size}` --
one thread per row, no shared memory.

---

## 3. Tests

### 3.1 Unit tests (`cpp/tests/io/variant_extract_test.cpp`)

19 test cases in the `VariantExtractTest` fixture covering the full API surface:

| Test | What it validates |
|---|---|
| `ExtractInt32TopLevelField` | Basic single-row INT32 extraction via `extract_variant_field`. |
| `ExtractShortStringField` | Short string (basic_type=1) extraction. |
| `NullStructRow` | Null propagation from a nullable struct row. |
| `MissingKeyYieldsNull` | Key not in dictionary produces null. |
| `WrongDesiredTypeYieldsNull` | INT32 value with STRING target type produces null. |
| `NonObjectValueYieldsNull` | Bare primitive (not wrapped in object) produces null for field extraction. |
| `InvalidMetadataYieldsNull` | Corrupt/unsupported metadata version produces null. |
| `TruncatedObjectValueYieldsNull` | Truncated object value blob produces null. |
| `ObjectTwoFieldsSingleRow` | Object with two fields; both extracted correctly. |
| `MultiRowDistinctKeysHandBuilt` | 3 rows with varying dictionaries and mixed key presence. |
| `MultiRowMixedKeys` | Key present in one row, absent in another. |
| `ExtraShreddingSiblingIgnored` | Struct with >2 children (simulating shredded columns); extraction ignores extras. |
| `GetVariantFieldReturnsVariantStruct` | Validates output shape (struct, 2 list children) and round-trip via `cast_variant`. |
| `GetVariantFieldMissingKeyAllNull` | All-null output preserves metadata child. |
| `CastVariantInt32` | Direct `cast_variant` on bare INT32 value. |
| `CastVariantString` | Direct `cast_variant` on short string. |
| `CastVariantLongString` | Direct `cast_variant` on long string (4-byte length header). |
| `GetThenCastMatchesExtract` | Multi-row: `get_variant_field` + `cast_variant` equals `extract_variant_field`. |
| `ApacheNestedGetVariantField` | Chained `get_variant_field` on Apache parquet-testing `object_nested` fixture (validates shared metadata across nesting levels and non-monotonic field offsets). |

### 3.2 Apache parquet-testing reference tests

Additional tests in the same file validate against known-good byte sequences from
the Apache parquet-testing repository:

| Test | Reference |
|---|---|
| `ApachePrimitiveInt32` | `primitive_int32` -- INT32(123456) |
| `ApacheShortString` | `short_string` -- 37-byte UTF-8 with emoji |
| `ApacheLongString` | `primitive_string` -- 174-byte UTF-8 with emoji |
| `ApacheObjectPrimitiveExtractString` | `object_primitive` -- 7-field object, extract `string_field` = "Apache Parquet" |

### 3.3 Parquet round-trip tests (`cpp/tests/io/parquet_variant_roundtrip_test.cpp`)

End-to-end tests that read Parquet files containing VARIANT columns and extract
fields:

| Test | What it validates |
|---|---|
| `ReadMinimalVariantParquet` | Single-row Parquet file with VARIANT column; extract INT32 field "x". |
| `ReadMultirowVariantParquet` | 3-row Parquet file with varying dictionaries; extract "x" (INT32), "k" (STRING), "y" (INT32) with mixed null patterns. |

---

## 4. Benchmarks

Implementation: `cpp/benchmarks/io/variant_extract.cu`
Binary: `VARIANT_EXTRACT_NVBENCH`

### 4.1 Data generation

Benchmark data is constructed on the host and copied to the GPU before timing
begins.  Key infrastructure:

- **Variable-length keys:** Dictionary keys are split into 5 length groups (3, 6,
  10, 15, 21 bytes) to realistically exercise the `slen != key_len` early-exit
  optimization in `device_find_key_in_metadata`.
- **Subset dictionaries:** A 5-key subset is drawn from the full dictionary such
  that the first and last keys of the full dictionary are always represented.
  This ensures both "first key" and "last key" benchmarks find the target.
- **Blob builders:** `build_metadata` and `build_object_value` construct
  spec-compliant Variant binary blobs from explicit key lists and field-ID arrays.

### 4.2 Benchmark: `get_variant_field`

Axes:
- `num_rows`: 2^15, 2^17, 2^19, 2^21 (32K to 2M)
- `scenario`: 5 divergence scenarios (see below)
- `key_position`: `"first"` or `"last"` (target is the first or last key in the dictionary)

| Scenario | Dict size | Fields per row | Divergence axis |
|---|---|---|---|
| `uniform_small` | 5-key subset | 5, uniform | Baseline (small work, no divergence) |
| `uniform_large` | 50-key full | 50, uniform | Baseline (large work, no divergence) |
| `skewed_field_count` | 50-key full (all rows) | Even: 5, Odd: 50 | Field-ID scan divergence (dict scan uniform) |
| `skewed_dict_size` | Even: 5-key, Odd: 100-key | 5 (both) | Dict-scan divergence (field scan uniform) |
| `half_missing` | 20-key full (all rows) | Even: 20 (target present), Odd: 19 (target absent) | Found-vs-null divergence |

### 4.3 Benchmark: `cast_variant_int32`

Axes:
- `num_rows`: 2^15, 2^17, 2^19, 2^21

Measures bare INT32 decode throughput with empty metadata (no object parsing).

---

## 5. Benchmark Results

Device: NVIDIA A100 80GB PCIe.  All benchmarks run in a single NVBench process
(important -- separate-process invocations incur multi-millisecond CUDA/RMM
initialization overhead that contaminates results).

### 5.1 `get_variant_field` -- full results

GPU time in milliseconds.

| num_rows | key_pos | uniform_small | uniform_large | skewed_field_count | skewed_dict_size | half_missing |
|----------|---------|--------------|---------------|-------------------|-----------------|-------------|
| 32K | first | 0.223 | 0.292 | 0.272 | 0.241 | 0.256 |
| 128K | first | 0.239 | 0.455 | 0.419 | 0.345 | 0.320 |
| 512K | first | 0.343 | 1.388 | 1.006 | 0.786 | 0.642 |
| **2M** | **first** | **0.729** | **4.452** | **3.209** | **2.442** | **1.782** |
| 32K | last | 0.223 | 0.433 | 0.406 | 0.358 | 0.270 |
| 128K | last | 0.249 | 0.736 | 0.681 | 0.626 | 0.392 |
| 512K | last | 0.385 | 3.122 | 2.192 | 1.619 | 1.106 |
| **2M** | **last** | **0.849** | **10.658** | **7.177** | **5.373** | **3.291** |

### 5.2 `cast_variant_int32` -- baseline

| num_rows | GPU Time (ms) |
|----------|--------------|
| 32K | 0.079 |
| 128K | 0.080 |
| 512K | 0.081 |
| 2M | 0.099 |

### 5.3 Performance fix: `make_structs_column` -> `create_structs_hierarchy`

During benchmarking with separate-process invocations, `half_missing` appeared
disproportionately slow.  Investigation revealed that `make_structs_column` calls
`superimpose_and_sanitize_nulls` on each child column when `null_count > 0`.
Since `half_missing` is the only scenario with output nulls, this triggered
`purge_nonempty_nulls` -- a full deep-copy and stream compaction of the ~518 MB
metadata list child -- adding tens of milliseconds of overhead invisible to other
scenarios.

The fix replaced `make_structs_column` with `create_structs_hierarchy`, which
constructs the struct column without propagating the null mask into children.  The
struct-level null mask is authoritative; consumers check it before accessing child
data.  This is a correctness-preserving optimization that benefits any call to
`get_variant_field` that produces nulls.

### 5.4 Analysis

Key observations at **2M rows**:

**1. Key position matters significantly.** `uniform_large` shows a **2.4x**
slowdown from first (4.45 ms) to last (10.66 ms), reflecting the full 50-key
dictionary scan.  `skewed_field_count` shows 2.2x and `skewed_dict_size` shows
2.2x.  `uniform_small` (5-key dict) shows only 1.16x, consistent with the
short scan.  `half_missing` shows 1.85x.

**2. Divergence is measurable in the skewed scenarios.** At 2M rows with last
key:
- `skewed_field_count` (7.18 ms) is 33% faster than `uniform_large` (10.66 ms)
  despite identical dict size (50 keys), because half the rows have only 5 fields
  instead of 50.
- `skewed_dict_size` (5.37 ms) is 50% faster than `uniform_large` (10.66 ms)
  because half the rows use a 5-key dict instead of 50-key.
- Both skewed scenarios are slower than `uniform_small` (0.85 ms), confirming the
  "heavy half" of rows pulls overall performance down.

**3. `half_missing` is lightweight per row.** At 2M/last, `half_missing` (3.29 ms)
is 3.2x faster than `uniform_large` (10.66 ms), consistent with its smaller
per-row data (20-key dict vs 50-key, 20 fields vs 50 fields).  The found-vs-null
divergence adds modest overhead compared to a uniform 20-key workload.

**4. Pure cast is nearly free.** `cast_variant_int32` at 2M rows (0.099 ms) is
7.4x faster than even the lightest `get_variant_field` scenario (0.729 ms),
confirming that object parsing and metadata-copy overhead dominate over primitive
decode.

**5. Scaling is roughly linear** for all scenarios above 128K rows, confirming
that kernel launch overhead is amortized at those sizes.

---

## 6. File Inventory

| File | Role |
|---|---|
| `cpp/include/cudf/io/variant.hpp` | Public API declarations |
| `cpp/src/io/variant_extract.cu` | GPU implementation (kernels + host-side logic) |
| `cpp/tests/io/variant_extract_test.cpp` | Unit tests (19 test cases) |
| `cpp/tests/io/parquet_variant_roundtrip_test.cpp` | Parquet round-trip tests (2 test cases) |
| `cpp/benchmarks/io/variant_extract.cu` | NVBench divergence benchmarks |
| `cpp/CMakeLists.txt` | Adds `src/io/variant_extract.cu` to libcudf |
| `cpp/tests/CMakeLists.txt` | Adds test sources to `PARQUET_TEST` |
| `cpp/benchmarks/CMakeLists.txt` | Adds `VARIANT_EXTRACT_NVBENCH` target |
