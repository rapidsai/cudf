# libcudf C++ examples using string transforms

This C++ example demonstrates using libcudf transform API to access and create
strings columns.

The example source code loads a csv file and produces a transformed column from the table using the values from the tables.

The following examples are included:
1. `int_output` - Using a transform to perform a fused checksum on two columns
2. `output` - Using a transform to get a substring from a kernel
3. `preallocated` - Using a transform kernel to output a string to a pre-allocated buffer
4. `output_public` - Performs same transformation on the CSV as `output` but uses alternative public APIs
4. `preallocated_public` - Performs same transformation on the CSV as `preallocated` but uses alternative public APIs

## Compile and execute

```bash
# Configure project
cmake -S . -B build/
# Build
cmake --build build/ --parallel $PARALLEL_LEVEL
# Execute
build/output info.csv
```

If your machine does not come with a pre-built libcudf binary, expect the
first build to take some time, as it would build libcudf on the host machine.
It may be sped up by configuring the proper `PARALLEL_LEVEL` number.
