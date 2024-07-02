# TPC-H Inspired Examples

Implementing the TPC-H queries using `libcudf`. We leverage the data generator (wrapper around official TPC-H datagen) from [Apache Datafusion](https://github.com/apache/datafusion) for generating data in the form of Parquet files. 

## Requirements 

- Rust

## Generating the Dataset

1. Clone the datafusion repository.
```bash
git clone git@github.com:apache/datafusion.git
```

2. Run the data generator. The data will be placed in a `data/` subdirectory.
```bash
cd datafusion/benchmarks/
./bench.sh data tpch

# for scale factor 10,
./bench.sh data tpch10
```

## Running Queries

1. Build the examples.
```bash
cd cpp/examples
./build.sh
```
The TPC-H query binaries would be built inside `examples/tpch/build`.

2. Execute the queries.
```bash
./tpch/build/tpch_q1
```
A parquet file named `q1.parquet` would be generated holding the results of the query.
