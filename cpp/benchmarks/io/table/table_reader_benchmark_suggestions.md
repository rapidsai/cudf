# Benchmark Suggestions for `read_table` Implementation

## Overview
The `read_table` function implements a simple binary table format that:
- Reads a fixed 24-byte header
- Reads metadata from host memory
- Reads data to device memory (with different IO paths)
- Performs zero-copy unpacking (just creates `column_view`s)

Unlike Parquet, there's no compression, encoding, or chunked reading. The main performance factors are:
1. **IO path** (FILEPATH vs HOST_BUFFER vs DEVICE_BUFFER)
2. **Data transfer bandwidth** (host-to-device for FILEPATH/HOST_BUFFER)
3. **Metadata parsing overhead** (number of columns, nested types)
4. **Memory allocation** (device buffer allocation)

## Recommended Benchmark Cases

### 1. **IO Path Comparison** (CRITICAL)
**Purpose**: Measure performance differences between IO source types
- **FILEPATH**: Tests file I/O + host-to-device transfer
- **HOST_BUFFER**: Tests host memory read + host-to-device transfer
- **DEVICE_BUFFER**: Tests direct device memory read (should be fastest)

**Suggested benchmarks**:
- `BM_table_read_io_paths` - Compare all three IO types across different data sizes
- Focus on: 512MB, 1GB, 2GB data sizes
- Test with fixed number of columns (64) and various data types

### 2. **Data Size Scaling** (IMPORTANT)
**Purpose**: Measure IO bandwidth and unpack overhead at different scales
- Since unpack is zero-copy, this primarily tests IO bandwidth
- Should test: 64MB, 256MB, 512MB, 1GB, 2GB, 4GB

**Suggested benchmarks**:
- `BM_table_read_size_scaling` - Single data type, varying sizes
- Use fixed data type (INTEGRAL or MIXED) to isolate IO performance
- Test with DEVICE_BUFFER to measure pure unpack overhead

### 3. **Metadata Complexity** (IMPORTANT)
**Purpose**: Measure overhead of parsing metadata for different table shapes
- More columns = more metadata to parse
- Nested types (STRUCT, LIST) = more complex metadata parsing
- Wide tables vs tall tables = different metadata/data ratios

**Suggested benchmarks**:
- `BM_table_read_wide_tables` - Many columns (256, 512, 1024, 2048)
- `BM_table_read_nested_types` - STRUCT and LIST with varying depth
- `BM_table_read_column_count` - Varying column counts (8, 32, 64, 128, 256)

### 4. **Data Type Variations** (MODERATE)
**Purpose**: Ensure unpacking works efficiently for all types
- Focus on types that affect metadata size/complexity, not decode performance
- Nested types (STRUCT, LIST) are most interesting

**Suggested benchmarks**:
- `BM_table_read_all_types` - All basic types (similar to current, but simpler)
- `BM_table_read_nested_struct` - STRUCT with varying number of fields
- `BM_table_read_nested_list` - LIST with varying nesting depth
- `BM_table_read_mixed_nested` - Tables with mix of nested types

### 5. **Null Handling** (MODERATE)
**Purpose**: Test null mask reading and unpacking overhead
- Different null densities affect metadata size slightly
- Should test: 0%, 10%, 50%, 90% null density

**Suggested benchmarks**:
- `BM_table_read_null_density` - Same data type, varying null percentages

### 6. **Memory Allocation Patterns** (MODERATE)
**Purpose**: Test device buffer allocation overhead
- Small tables = fixed overhead dominates
- Large tables = allocation time becomes significant

**Suggested benchmarks**:
- `BM_table_read_small_tables` - Very small tables (1KB, 10KB, 100KB, 1MB)
- `BM_table_read_large_tables` - Very large tables (2GB, 4GB, 8GB)

### 7. **String Column Variations** (LOW PRIORITY)
**Purpose**: Test unpacking with variable-length data
- String columns have more complex metadata (offsets)
- Different string lengths = different metadata/data ratios

**Suggested benchmarks**:
- `BM_table_read_string_variations` - Short vs long strings
- `BM_table_read_string_column_ratio` - Tables with varying % of string columns

## Benchmarks to REMOVE or SIMPLIFY

### Remove/Simplify:
1. **Compression benchmarks** - Table format doesn't support compression
2. **Chunked reading** - Not applicable to table format
3. **"IO compression" benchmark** - Misleading name, table format has no compression
4. **Complex decoding benchmarks** - Table format doesn't decode, just unpacks

### Keep but Modify:
1. **Wide tables** - Keep but focus on metadata parsing overhead, not columnar decode
2. **Long strings** - Keep but simpler (just test string unpacking, not encoding)
3. **Structs** - Keep but focus on nested structure parsing

## Recommended Benchmark Structure

```
BM_table_read_io_paths
  - Compare FILEPATH, HOST_BUFFER, DEVICE_BUFFER
  - Single data type (INTEGRAL), fixed 64 columns
  - Varying data sizes: 512MB, 1GB, 2GB

BM_table_read_size_scaling
  - DEVICE_BUFFER only (to isolate unpack overhead)
  - Single data type, fixed columns
  - Sizes: 64MB, 256MB, 512MB, 1GB, 2GB, 4GB

BM_table_read_column_complexity
  - DEVICE_BUFFER, fixed data size
  - Varying column counts: 8, 32, 64, 128, 256, 512, 1024
  - Single data type (INTEGRAL)

BM_table_read_wide_tables
  - Many columns (256, 512, 1024, 2048)
  - DEVICE_BUFFER to focus on unpack overhead
  - Different data types: DECIMAL, STRING

BM_table_read_nested_types
  - STRUCT with varying depth (1, 2, 3 levels)
  - LIST with varying nesting depth
  - Mixed nested types

BM_table_read_all_types
  - All basic types (simpler than current)
  - DEVICE_BUFFER, fixed size/columns
  - Just verify unpack works for all types

BM_table_read_null_density
  - Varying null percentages: 0%, 10%, 50%, 90%
  - Single data type, fixed size

BM_table_read_small_tables
  - Very small: 1KB, 10KB, 100KB, 1MB
  - Test fixed overhead

BM_table_read_string_variations
  - Short strings (avg 32 bytes) vs long strings (avg 1KB, 4KB)
  - Test metadata overhead for offset arrays
```

## Key Differences from Parquet Benchmarks

| Aspect | Parquet | Table Format |
|--------|---------|--------------|
| **Compression** | Multiple types (SNAPPY, ZSTD) | None |
| **Encoding** | Dictionary, RLE, Delta, etc. | None (raw packed) |
| **Chunked Reading** | Row groups | Not applicable |
| **Main Bottleneck** | Decode + Decompress | IO + Unpack |
| **Zero-copy** | Partial (depends on encoding) | Full (always zero-copy unpack) |
| **Benchmark Focus** | Decode performance | IO bandwidth + metadata parsing |

## Implementation Notes

1. **Always use DEVICE_BUFFER for pure unpack benchmarks** - This isolates unpack performance from IO
2. **Use FILEPATH/HOST_BUFFER for IO benchmarks** - These test real-world scenarios
3. **Track both raw data size and serialized size** - Serialized size = header + metadata + data
4. **Measure peak memory** - Should be roughly equal to data size (zero-copy)
5. **Test metadata/data ratio** - Wide tables have more metadata relative to data
