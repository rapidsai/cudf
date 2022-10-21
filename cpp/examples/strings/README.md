# libcudf C++ examples using strings columns

This C++ example demonstrates using libcudf APIs to access and create
strings columns.

The example source code loads a csv file and produces a redacted strings
column from the names column using the values from the visibilities column.

Four examples are included:
1. Using libcudf APIs to build the output
2. Using a simple custom kernel with dynamic memory
3. Using a custom kernel with pre-allocated device memory
4. Using a two-pass approach to improve performance

These examples are described in more detail in
https://developer.nvidia.com/blog/mastering-string-transformations-in-rapids-libcudf/

## Compile and execute

```bash
# Configure project
cmake -S . -B build/
# Build
cmake --build build/ --parallel $PARALLEL_LEVEL
# Execute
build/libcudf_apis names.csv
--OR--
build/custom_with_malloc names.csv
--OR--
build/custom_prealloc names.csv
--OR--
build/custom_optimized names.csv
```

If your machine does not come with a pre-built libcudf binary, expect the
first build to take some time, as it would build libcudf on the host machine.
It may be sped up by configuring the proper `PARALLEL_LEVEL` number.
