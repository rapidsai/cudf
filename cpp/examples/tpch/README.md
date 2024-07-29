# TPC-H Derived Examples

Implements TPC-H queries using `libcudf`. We leverage the data generator (wrapper around official TPC-H datagen) from [Apache Datafusion](https://github.com/apache/datafusion) for generating data in Parquet format.

## Requirements

- Rust

## Generating the Dataset

1. Clone the datafusion repository.
```bash
git clone git@github.com:JayjeetAtGithub/datafusion.git $HOME/datafusion
```

2. Run the data generator with a specific scale factor. The data will be placed in a `data/` subdirectory.
```bash
cd $HOME/datafusion/benchmarks/
./bench.sh data tpch

# for scale factor 10,
./bench.sh data tpch10
```

3. Parquet files named as `[tablename].parquet` will be generated inside the `data/tpch_sf1` directory.

## Running Queries

1. Build the libcudf examples.
```bash
cd $HOME/cudf/cpp/examples
./build.sh
```
The TPC-H query binaries would be built inside `tpch/build`.

2. Set these environment variables for optimized runtimes.
```bash
export KVIKIO_COMPAT_MODE="on"
export LIBCUDF_CUFILE_POLICY="KVIKIO"
export CUDA_MODULE_LOADING="EAGER"
```

3. Execute the queries.
```bash
./tpch/build/tpch_q[X] [path to dataset] [memory resource type (cuda/pool/managed/managed_pool)]
```

For example, for query 1,
```bash
./tpch/build/tpch_q1 $HOME/datafusion/benchmarks/data/tpch_sf1 pool
```

A parquet file named `q1.parquet` would be generated containing the results of the query.
