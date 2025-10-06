# External Sorting Example

This example demonstrates external sorting using libcudf by:

1. **Generating random data**: Creates tables with configurable numbers of columns and rows
2. **Writing to parquet files**: Stores data across multiple parquet files to simulate external storage
3. **Multithreaded I/O**: Uses parallel reading and writing for better performance
4. **Sorting large datasets**: Combines all data and sorts using libcudf's optimized sorting algorithms

## Building

From the cudf cpp directory:
```bash
cd examples/external_sorting
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

## Usage

```bash
./sort [n_columns] [m_rows_per_file] [num_files] [output_dir]
```

### Arguments

- **n_columns**: Number of columns in each table (default: 5)
- **m_rows_per_file**: Number of rows per parquet file (default: 1,000,000)
- **num_files**: Number of parquet files to create (default: 4)
- **output_dir**: Directory to store parquet files (default: ./sort_data)

### Examples

```bash
# Default run: 5 columns, 4 files of 1M rows each (4M total rows)
./sort

# Custom run: 3 columns, 8 files of 500K rows each (4M total rows)
./sort 3 500000 8 /tmp/my_sort_test

# Large dataset: 10 columns, 10 files of 2M rows each (20M total rows)
./sort 10 2000000 10 /data/large_sort
```

## What it Does

### Phase 1: Data Generation and Storage
- Generates random tables with mixed data types (int32, float64, int64, float32, int16)
- Writes each table to a separate parquet file using Snappy compression
- Uses multithreaded writing for better I/O performance

### Phase 2: Data Reading
- Reads all parquet files back using multithreaded I/O
- Concatenates data from multiple threads efficiently

### Phase 3: Data Concatenation
- Combines all data into a single large table
- Reports total number of rows processed

### Phase 4: Sorting
- Sorts the combined dataset by the first column (ascending order)
- Uses libcudf's optimized sorting algorithms including:
  - Radix sort for integer types
  - Multi-path optimization based on data characteristics
  - Index-based sorting for memory efficiency

### Phase 5: Result Output
- Writes the sorted result to a parquet file
- Provides performance metrics for each phase

## Data Types

The example generates columns with different data types to demonstrate libcudf's type system:

- **INT32**: Primary sorting column with large integer values
- **FLOAT64**: Double precision floating point numbers
- **INT64**: Long integer values
- **FLOAT32**: Single precision floating point numbers  
- **INT16**: Short integer values

Each column type has different value ranges to create diverse datasets for testing.

## Performance Notes

- **Memory Management**: Uses RMM (Rapids Memory Manager) with pooled allocations for optimal GPU memory usage
- **Stream Management**: Utilizes CUDA stream pools for concurrent operations
- **I/O Optimization**: Leverages multithreaded parquet I/O for better disk/network throughput
- **Compression**: Uses Snappy compression for parquet files to reduce storage requirements

## Example Output

```
External Sorting Example
========================
Columns per table: 5
Rows per file: 1000000
Number of files: 4
Total rows: 4000000
Output directory: ./sort_data

Phase 1: Generating and writing 4 parquet files...
Generating table 1/4
Generating table 2/4
Generating table 3/4
Generating table 4/4
Data generation Elapsed Time: 1250ms

Writing parquet files Elapsed Time: 2100ms

Phase 2: Reading parquet files...
Reading parquet files Elapsed Time: 800ms

Phase 3: Concatenating data...
Total rows to concatenate: 4000000
Data concatenation Elapsed Time: 150ms

Phase 4: Sorting data...
Sorting by first column (ascending)
Computing sort order Elapsed Time: 320ms
Gathering sorted data Elapsed Time: 180ms

Phase 5: Writing sorted result...
Writing sorted result Elapsed Time: 750ms

External sorting completed successfully!
Summary:
  Input: 4 files × 1000000 rows × 5 columns
  Total processed: 4000000 rows
  Sorted result: ./sort_data/sorted_result.parquet
```

## Educational Value

This example demonstrates several key libcudf concepts:

- **Column Creation**: Using factory functions to create columns with different types
- **Table Management**: Working with tables as collections of columns
- **Memory Management**: Proper use of RMM for GPU memory allocation
- **I/O Operations**: Reading and writing parquet files with optimal settings
- **Sorting Algorithms**: Leveraging libcudf's high-performance sorting capabilities
- **Multithreading**: Using CUDA streams for concurrent operations
- **Data Concatenation**: Efficiently combining multiple tables

The code serves as a practical reference for building applications that need to process large datasets that don't fit in memory, demonstrating the external sorting pattern commonly used in big data processing.
