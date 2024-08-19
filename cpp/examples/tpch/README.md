# TPC-H Derived Examples

Implements TPC-H queries using `libcudf`. We leverage the data generator (wrapper around official TPC-H datagen) from [Apache Datafusion](https://github.com/apache/datafusion) for generating data in Parquet format.

## Requirements

- Rust
- [libcudf](https://github.com/rapidsai/cudf/blob/branch-24.08/CONTRIBUTING.md#setting-up-your-build-environment)

## Running Queries

1. Build the `libcudf` examples.
```bash
cd cudf/cpp/examples
./build.sh
```
The TPC-H query binaries would be built inside `tpch/build`.

2. Generate the dataset.
```bash
cd tpch/datagen
./datagen.sh [scale factor (1/10)]
```

The parquet files will be generated in `tpch/datagen/datafusion/benchmarks/data/tpch_sf[scale factor]`.

3. Set these environment variables for optimized runtimes.
```bash
export KVIKIO_COMPAT_MODE="on"
export LIBCUDF_CUFILE_POLICY="KVIKIO"
export CUDA_MODULE_LOADING="EAGER"
```

4. Execute the queries.
```bash
./tpch/build/tpch_q[query no] [path to dataset] [memory resource type (cuda/pool/managed/managed_pool)]
```

A parquet file named `q[query no].parquet` would be generated containing the results of the query.
