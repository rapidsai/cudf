# External Sorting Example

This example demonstrates external sorting using libcudf with a sample sort algorithm:

1. **Reading parquet files**: Reads existing parquet files from a specified directory individually (without concatenating to avoid memory limits)
2. **Individual sorting**: Sorts each table separately by the first column
3. **Splitter sampling**: Extracts equally spaced splitter values from each sorted table
4. **External sorting**: Demonstrates the sample sort algorithm suitable for datasets larger than available memory

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
./sort [input_dir] [num_files]
```

### Arguments

- **input_dir**: Directory containing parquet files to read (default: ./sort_data)
- **num_files**: Number of parquet files to read (default: 4)

The program expects parquet files named `data_0.parquet`, `data_1.parquet`, etc. in the input directory.

### Examples

```bash
# Default run: Read 4 files from ./sort_data
./sort

# Read 8 files from a specific directory
./sort /path/to/data 8

# Process large number of files
./sort /data/large_dataset 20
```

## Algorithm: Sample Sort for External Sorting

This example implements the initial phases of a sample sort algorithm, which is commonly used for external sorting:

### Phase 1: Data Reading
- Reads parquet files sequentially from the specified input directory
- **Crucially, does NOT concatenate the tables** - this avoids memory limitations
- Each table is kept separate to simulate external sorting constraints
- Validates that all expected files exist before processing

### Phase 2: Individual Sorting and Splitter Sampling
- Sorts each table individually by the first column using libcudf's optimized sorting
- Samples n equally spaced splitter values from each sorted table (where n = number of files)
- These splitters would be used in a full external sort to partition data across buckets
- Keeps memory usage bounded by processing one table at a time

## Sample Sort Algorithm Overview

The sample sort algorithm is designed for sorting datasets that don't fit in memory:

1. **Local Sort**: Sort each chunk/partition individually ✓ (implemented)
2. **Sample Collection**: Extract representative samples from each sorted chunk ✓ (implemented)  
3. **Global Splitter Selection**: Choose global splitters from all samples (future enhancement)
4. **Data Partitioning**: Redistribute data based on global splitters (future enhancement)
5. **Final Sort**: Sort each partition and concatenate results (future enhancement)

This example demonstrates steps 1 and 2, providing a foundation for a complete external sorting implementation.

## Input File Format

The program expects parquet files with the naming pattern:
- `data_0.parquet`
- `data_1.parquet`
- `data_2.parquet`
- etc.

All files should have the same schema (same column names and types).

## Performance Notes

- **Memory Management**: Uses RMM (Rapids Memory Manager) with pooled allocations for optimal GPU memory usage
- **Bounded Memory Usage**: Processes one table at a time to avoid memory limitations
- **Sequential I/O**: Reads parquet files sequentially for simplicity and reliability
- **Sorting Efficiency**: Uses libcudf's optimized sorting for each individual table
- **External Algorithm**: Designed for datasets larger than available memory

## Example Output

```
External Sorting Example (Sample Sort Algorithm)
================================================
Input directory: /path/to/data
Number of files: 4

Phase 1: Reading parquet files...
Reading: /path/to/data/data_0.parquet
Reading: /path/to/data/data_1.parquet
Reading: /path/to/data/data_2.parquet
Reading: /path/to/data/data_3.parquet
Reading parquet files Elapsed Time: 800ms

Phase 2: Sorting individual tables and sampling splitters...
Sorting table 0 (1000000 rows)
  Sampled 4 splitters from table 0
Sorting table 1 (1000000 rows)
  Sampled 4 splitters from table 1
Sorting table 2 (1000000 rows)
  Sampled 4 splitters from table 2
Sorting table 3 (1000000 rows)
  Sampled 4 splitters from table 3
Sorting individual tables and sampling Elapsed Time: 650ms

External sorting (sample sort) completed!
Summary:
  Input: 4 files from /path/to/data
  Total processed: 4000000 rows across 4 tables
  Algorithm: Sample sort with 4 splitters per table
  Memory usage: Each table processed individually (external sorting)
```

## Error Handling

The program performs several validation checks:

- Verifies the input directory exists and is accessible
- Checks that all required parquet files exist before processing
- Validates that the number of files parameter is positive
- Provides clear error messages for common issues

## Educational Value

This example demonstrates several key libcudf concepts and external sorting principles:

- **Parquet I/O**: Reading parquet files with optimal performance settings
- **Table Management**: Working with individual tables without concatenation
- **Memory Management**: Proper use of RMM for GPU memory allocation with bounded usage
- **Sorting Algorithms**: Leveraging libcudf's high-performance sorting capabilities
- **External Sorting**: Processing datasets larger than available memory
- **Sample Sort Algorithm**: Implementing the sampling phase of distributed sorting
- **Splitter Sampling**: Extracting representative values for data partitioning

The code serves as a foundation for building applications that need to sort large datasets stored across multiple parquet files using external sorting techniques. It demonstrates the critical concept of bounded memory usage and the sample sort algorithm commonly used in distributed big data processing systems.
