# libcudf C++ example using distinct API

This C++ example demonstrates libcudf usage of drop-duplicates by  using distinct API.

The example source code loads a csv file that contains transactions of a Bank in a
single day, and processes the data with four keep options: `any`, `first`, `last`,
and `none`.
Keep option `any` stores the results of customers with any of the transaction.
Keep option `first` stores only the first transaction by a customer, if there are multiple.
Keep option `last` stores only the last transaction by a customer, if there are multiple.
Keep option `none` stores only customers with a single transaction on that day.
All the results are stored back in different csv files based on keep option.

## Compile and execute

```bash
# Configure project
cmake -S . -B build/
# Build
cmake --build build/ --parallel $PARALLEL_LEVEL
# Execute
build/drop_duplicates
```

If your machine does not come with a pre-built libcudf binary, expect the
first build to take some time, as it would build libcudf on the host machine.
It may be sped up by configuring the proper `PARALLEL_LEVEL` number.
